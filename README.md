# 可追溯 RAG 问答与检索评测

本项目独立管理文档、切分快照、BGE模型、索引、检索、生成和评测。默认路径均相对于项目根目录；Embedding从项目内加载，不依赖其他学习项目或全局模型缓存。

当前已实现多格式解析、父子索引、BM25与RRF、规则闸门、来源编号、索引版本切换和固定参数评测。领域语料、Reranker、完整生成与引用语义评测仍需补充，见 `docs/TARGET.md`。

## 目录和文件职责

| 文件或目录 | 职责 |
|---|---|
| `app.py` | 命令行问答、只检索、结构化回答状态 |
| `download_embedding.py` | 检查项目内BGE；缺失时准备公开模型 |
| `requirements.txt` | Python依赖与已验证版本 |
| `.env.example` / `.env` | 生成接口配置样例与本地密钥；密钥不提交 |
| `rag/config.py` | 项目内路径、模型身份、切分和检索参数 |
| `rag/embedding.py` | 仅从项目内加载BGE，不隐式使用缓存 |
| `rag/loader.py` | 解析TXT/MD/PDF/DOCX，切父块并保留来源标识 |
| `rag/splitter.py` | 按句切子块并限制长度 |
| `rag/index.py` | 新建父子集合，验证后原子切换活动清单 |
| `rag/store.py` | 查询子块，按parent_id聚合并返回父块 |
| `rag/retriever_hybrid.py` | 字符BM25、RRF、规则闸门及证据返回 |
| `rag/retriever_vec.py` | 相同活动索引上的纯向量路径 |
| `rag/evidence.py` | 证据编号、来源与页码、引用ID校验 |
| `rag/prompts.py` | 仅依据证据回答的系统提示词 |
| `rag/llm_client.py` | 请求OpenAI兼容生成接口 |
| `rag/__init__.py` | 导出检索接口 |
| `models/bge-small-zh-v1.5/` | 本项目Embedding权重和Tokenizer（不入库，约96MB） |
| `data/docs/` | 原始资料目录，格式支持TXT/MD/PDF/DOCX（不入库，需自备） |
| `data/chunks.json` | 冻结评测快照，题集gold对应这里的块（不入库） |
| `data/index/` | 向量库和活动 `index_manifest.json`（不入库，可由语料重建） |
| `eval/qset.py` | 400题开发集与标注（随仓库分发，仅含题目与块下标） |
| `eval/run_release.py` | 固定参数三路对照、拒答与逐题结果 |
| `eval/datasets/challenge_v1.json` | 64题诊断集、证据及语料哈希（不入库，含语料原文） |
| `eval/results/` | 本项目当前评测结果（不入库，含语料原文） |
| `tests/test_regression.py` | 索引、检索、引用、回答状态回归 |
| `tests/test_retrieval_boundaries.py` | 纯向量阈值隔离、边界过滤、BM25零分候选与RRF回归 |
| `tests/integration_lifecycle.py` | 四格式解析、重复构建、修改/删除验证 |
| `docs/TARGET.md` | 本项目的分阶段深化清单 |
| `docs/EVALUATION_PROTOCOL.md` | 语料质量、消融顺序、完整指标口径 |
| `docs/INTERVIEW.md` / `docs/RELEASE.md` | 当前原理、实现范围、成绩与限制 |

## 数据流和阅读顺序

```text
data/docs → loader父块 → splitter子块 → 项目内BGE编码
          → index创建父子集合 → 校验 → 清单原子切换

问题 → store子块召回/父块聚合 + 字符BM25
     → RRF → 拒答闸门 → evidence证据编号
     → app组装消息 → llm_client生成 → 引用ID检查
```

先看一题 `eval/qset.py` 的标注结构，再按 config/loader → splitter/index → store → retriever_hybrid → evidence/app → run_release 的顺序阅读。先理解“资料、正确证据、模型输入”三者，再逐段分析检索和生成。

## 运行

已在 Windows、Python 3.10、Chroma 1.5.9 验证。Embedding 权重不入库，首次运行用 `download_embedding.py` 拉取公开模型到 `models/`，之后日常运行不再下载。

```powershell
# 连续问答：回答后继续输入下一个问题，输入 quit 退出
python app.py

# 可选的单次检索检查与回归测试
python app.py -q "Agent 有哪四个核心部件？" --retrieve-only
python -m unittest discover -s tests -v
```

完整安装、构建和评测：

```powershell
pip install -r requirements.txt
python download_embedding.py
python -m rag.index --from-snapshot
python -m eval.run_release --dataset development
python -m eval.run_release --dataset challenge
python tests/integration_lifecycle.py
```

生成问答需要在项目内 `.env` 填写模型接口。日常连续提问运行 `python app.py`，每次回答后会继续等待输入，输入 `quit` 退出。`-q "问题"` 明确表示单次模式，处理后退出；`python app.py -q "问题" --json` 用于单次结构化输出，`--retrieve-only`只检索、不调用LLM。

### 检索阈值与候选规则

纯向量入口使用独立的 `rag/config.py` 中 `VECTOR_DISTANCE_THRESHOLD`，只比较原始 Chroma L2 距离；默认 `None`，表示尚未独立校准，不启用距离拒答。需要过滤时应针对当前模型、归一化和索引在开发集校准数值；距离等于阈值时保留候选。它不使用混合闸门的 `max_score=0.4`，也不沿用旧模型的 `13.5`。

混合闸门仍使用 `distance - cover_slope * coverage > max_score` 判断拒答。BM25只让分数大于0的文档进入候选和RRF，无匹配时返回空列表，不以零分文档补齐TopK。向量候选仍可独立进入融合。

## 文档更新和索引

`python -m rag.index`按 `data/docs/`当前内容重建；`python -m rag.index --from-snapshot`返回固定评测快照。普通构建不覆盖 `chunks.json`，新活动语料随索引清单切换。活动语料与gold不一致时评测会停止。

相同输入和配置复用版本；修改或删除后先写新集合，条数校验成功后才切换清单。失败保留活动版本。按单写者使用，尚未实现并发构建锁、按文档增量重算或自动版本清理。

索引记录模型语义标识，加载路径单独由项目位置确定，搬迁项目不会改变该标识。默认无需外部目录；可通过 `RAG_MODEL_DIR`等环境变量显式覆盖。

## 当前实测与边界

下表保留基础版本成绩；2026-11-05的阈值隔离与BM25候选修复、最新逐题结果及回归记录见 [检索修复说明](docs/RETRIEVAL_FIX_20261105.md)。

冻结资料4份、32父块、322子块。开发集200库内/200库外；块级仅统计198道fact/para，统一top5和MRR@5。

| 方法 | 块级R@1 | 块级R@5 | MRR@5 |
|---|---:|---:|---:|
| BGE子块向量 | 52.5% | 79.3% | 0.631 |
| 字符BM25 | 61.1% | 82.3% | 0.694 |
| RRF，向量权重0.45 | 58.6% | 83.8% | 0.685 |

开发集闸门FRR7.5%、FAR5.5%；这是曾用于调参的开发集表现，不是独立测试或最终回答正确率。

64道助手编写诊断题（32内/32外）上：RRF R@5为84.4%、BM25为90.6%，闸门FRR62.5%、FAR28.1%。长问法和同领域无答案题暴露了泛化问题；这批题今后只作诊断/回归，不能调参后重新称为未知测试。

引用ID存在不代表证据支持主张，尚未完成真实生成Correctness、Faithfulness和人工引用精确率评测。回答状态区分正常回答、检索拒答、证据不足、引用无效和生成错误，API异常不当作资料拒答。

后续约500个高质量有效块先验证完整链路，再按覆盖扩展至1000—2000，约5000仅作按需规模实验。按Dense → BM25 → Hybrid+RRF → Reranker逐步消融；父子索引和Query Rewrite按诊断验证。质量、覆盖和可信评测优先，详见评测协议。

## 数据与隐私边界

语料、索引与评测产物均来自私有文档，不随仓库分发。仓库只保留代码、协议文档与脱敏后的题集：

- **不提交**：`data/docs/`（原始资料）、`data/chunks.json`（逐块文本即原文）、`data/index/`（向量库）、`eval/datasets/` 与 `eval/results/`（含证据原文与逐题结果）、`models/`（模型权重）
- **提交**：`rag/`、`tests/`、`eval/qset.py`（仅题目文本与块下标，不含答案原文）、`docs/`
- `.env` 一律不入库，`LLM_API_KEY` 由使用者自行填写

因此克隆后可读通全部代码路径，但要复现上表成绩仍需自备语料重建索引：把资料放入 `data/docs/` → `python download_embedding.py` → `python -m rag.index`。题集的块下标对应本地语料的固定快照，换语料后必须重新标注；`run_release.py` 在 gold 与活动语料不一致时会直接停止，而不是静默给出错误成绩。
