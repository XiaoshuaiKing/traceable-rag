# -*- coding: utf-8 -*-
"""混合检索：向量通道 + 字面通道（字符 bigram BM25），RRF 融合 + 拒答闸门。

为什么需要字面通道
------------------
这份知识库只有 32 个父块（322 个子块），都是同一批个人资料，句子之间用词高度重叠。
汉语专有名词、术语、编号这类强信号，字面匹配几乎不会漏（单路 Recall@5 82.3%）。
向量通道负责同义改写和跨表述的问法，两路各自排名后用 RRF 融合，
不需要把两套分数量纲调到一起（融合后 R@5 83.8%，n=198 单块题）。

向量通道的两次升级
------------------
1. 多语 MiniLM → 中文 BGE-small-zh：块级 Recall@5 61.0% → 72.7%。
   语料是中文的，多语模型在中文短查询上区分度不够。
2. 检索单元从 500 字整块改为句子级子块、命中后仍返回父块（small-to-big）：
   72.7% → 80.5%。整块里混着多个主题会把块向量稀释，查询只对上其中一个
   主题时也拿不到高分；拆到句子级后匹配聚焦，而返回给 LLM 的仍是完整块。

拒答闸门
--------
只看向量距离时，一批「问法偏但资料里确实有」的题会被误杀。
改成斜边界 score = 距离 - COVER_SLOPE × 字面覆盖率，覆盖率高可以抵扣距离：
库内误拒 7.5%、库外误接受 5.5%，总准确 93.5%（库内 200 + 库外 200 题）。

注意：这里的距离是归一化向量上的 L2（落在 0~4），与早期 MiniLM 时代
（未归一化，量纲 0~30）完全不同，闸门参数不能跨版本复用。
"""
import math
from collections import Counter
from .evidence import assemble_evidence

# ---- 融合参数（由 400 题评测集扫参得到，见 eval/run_model_sweep.py）----
K_VEC = 20          # 向量通道召回数
K_BM = 20           # 字面通道召回数
RRF_K = 60          # RRF 平滑常数
W_VEC = 0.45        # 向量通道权重，1-W_VEC 给字面通道（BGE+子块后向量更可信）

# ---- 拒答闸门参数（见 eval/run_bge_eval.py）----
COVER_SLOPE = 1.25  # 覆盖率对距离的抵扣系数
MAX_SCORE = 0.40    # 抵扣后超过该值才拒答


def bigrams(text):
    """去掉空白后取相邻二字，兼顾中文短语与英文/数字片段。"""
    s = "".join(ch for ch in text if ch.strip())
    return [s[i:i + 2] for i in range(len(s) - 1)]


class BM25:
    """字符 bigram + BM25。语料小、不依赖分词，对中文专名/术语足够有效。"""

    def __init__(self, docs, k1=1.5, b=0.75):
        self.k1, self.b = k1, b
        self.grams = [bigrams(d) for d in docs]
        self.lens = [len(g) for g in self.grams]
        self.avg = sum(self.lens) / max(1, len(self.lens))
        self.tf, self.df = [], Counter()
        for g in self.grams:
            c = Counter(g)
            self.tf.append(c)
            for t in c:
                self.df[t] += 1
        self.n = len(self.grams)

    def scores(self, query):
        q = Counter(bigrams(query))
        out = []
        for i in range(self.n):
            s = 0.0
            tf, dl = self.tf[i], self.lens[i]
            for t, qn in q.items():
                f = tf.get(t, 0)
                if not f:
                    continue
                df = self.df.get(t, 0)
                idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
                s += idf * (f * (self.k1 + 1)) / (
                    f + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avg)))
            out.append(s)
        return out

    def rank(self, query):
        """仅返回正分候选，按分数降序；无匹配时返回空列表。"""
        sc = self.scores(query)
        return sorted((i for i in range(self.n) if sc[i] > 0), key=lambda i: -sc[i])


def rrf_merge(rankings, weights, k=RRF_K):
    """Reciprocal Rank Fusion：score = Σ w_i / (k + rank_i)。

    rankings: 每个通道一个「语料下标按相关性降序」的列表
    weights:  同长度权重列表
    """
    fused = {}
    for rank_list, w in zip(rankings, weights):
        for pos, idx in enumerate(rank_list, 1):
            fused[idx] = fused.get(idx, 0.0) + w / (k + pos)
    return sorted(fused, key=lambda i: -fused[i])


class HybridRetriever:
    def __init__(self, vector_store, k_vec=K_VEC, k_bm=K_BM, rrf_k=RRF_K,
                 w_vec=W_VEC, cover_slope=COVER_SLOPE, max_score=MAX_SCORE):
        self.store = vector_store
        self.k_vec, self.k_bm, self.rrf_k = k_vec, k_bm, rrf_k
        self.w_vec = w_vec
        self.cover_slope, self.max_score = cover_slope, max_score

        # 字面通道与覆盖度都要在「返回单元」上算：子块索引返回的是父块，
        # 所以语料取 store.corpus（父块全集），而不是底层的 322 个子块。
        if hasattr(vector_store, "corpus"):
            self.texts, self.metas = vector_store.corpus
        else:
            raw = vector_store._collection.get(include=["documents", "metadatas"])
            self.texts = list(raw["documents"])
            self.metas = list(raw["metadatas"])
        self.pos_of = {}
        self.pos_of_id = {}
        for i, t in enumerate(self.texts):
            self.pos_of.setdefault(t, i)
            meta = self.metas[i]
            if 'parent_id' in meta:
                self.pos_of_id[meta['parent_id']] = i
        self.bm25 = BM25(self.texts)
        self.grams = [set(bigrams(t)) for t in self.texts]

    # ---- 向量通道 ----
    def _vector_rank(self, query):
        hits = self.store.similarity_search_with_score(query, k=self.k_vec)
        order, dists = [], {}
        for doc, score in hits:
            i = self.pos_of_id.get(doc.metadata.get('parent_id'))
            if i is None and 'parent_id' not in doc.metadata:
                i = self.pos_of.get(doc.page_content)
            if i is None:
                continue
            order.append(i)
            dists[i] = float(score)
        return order, dists

    # ---- 字面通道 ----
    def _bm_rank(self, query):
        return self.bm25.rank(query)[:self.k_bm]

    def _coverage(self, query, idx):
        """查询里的二字片段，有多少比例出现在指定块里。"""
        q = set(bigrams(query))
        if not q:
            return 0.0
        return len(q & self.grams[idx]) / len(q)

    def gate(self, query):
        """返回 (是否拒答, 最近邻距离, top5 最大覆盖率, 抵扣后分数)。评测脚本复用同一判定。"""
        v_order, v_dist = self._vector_rank(query)
        if not v_dist:
            return True, float("inf"), 0.0, float("inf")
        nearest = v_dist[min(v_dist, key=v_dist.get)]
        cov = max((self._coverage(query, i) for i in v_order[:5]), default=0.0)
        score = nearest - self.cover_slope * cov
        return score > self.max_score, nearest, cov, score

    def retrieve_indices(self, query, k=5, apply_gate=True):
        """返回 (父块下标列表, 是否拒答)。search 与评测脚本共用这一条路径。

        单独抽出来是为了让评测量到的就是线上跑的排序，而不是脚本里另写一份
        等价逻辑 —— 两份实现迟早会漂移。

        apply_gate=False 用于单独度量召回：拒答是准入判断，混进召回统计会
        把「库内误拒」重复算成「检索没找到」，两个指标互相污染。
        """
        if not query.strip() or k <= 0:
            raise ValueError('query must be nonempty and k must be positive')
        v_order, v_dist = self._vector_rank(query)
        b_order = self._bm_rank(query)

        # 拒答闸门：距离远、字面也沾不上，才认定资料库里没有相关内容。
        # 只做准入判断，不参与排序。
        if apply_gate and self.max_score is not None:
            if not v_dist:
                return [], True
            nearest = v_dist[min(v_dist, key=v_dist.get)]
            cov = max((self._coverage(query, i) for i in v_order[:5]), default=0.0)
            if nearest - self.cover_slope * cov > self.max_score:
                return [], True

        fused = rrf_merge([v_order, b_order], [self.w_vec, 1 - self.w_vec], self.rrf_k)
        return fused[:k], False

    def search(self, query, k=5):
        picked, rejected = self.retrieve_indices(query, k=k)
        if rejected:
            return {"context": None, "sources": []}

        return assemble_evidence(self.texts, self.metas, picked)

    def describe(self):
        """每个块的原文与来源，供评测脚本核对。"""
        return [{"text": t, "source": m.get("source_file", "unknown_source")}
                for t, m in zip(self.texts, self.metas)]


_CACHE = {}


def search_documents(vector_store, query, k=5):
    """与 retriever_vec.search_documents 同签名，检索器按库缓存（BM25 只建一次）。"""
    key = id(vector_store)
    if key not in _CACHE:
        _CACHE[key] = HybridRetriever(vector_store, **getattr(vector_store, 'profile', {}))
    return _CACHE[key].search(query, k=k)
