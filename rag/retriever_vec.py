# -*- coding: utf-8 -*-
"""纯向量基线，可选按原始 Chroma L2 距离过滤。

VECTOR_DISTANCE_THRESHOLD 默认 None：未独立校准时只返回最近邻，不做距离拒答。
不能使用混合闸门 distance - cover_slope * coverage 的 max_score，
也不能沿用旧模型的 13.5；这些阈值与当前原始距离不是同一标尺。
"""

from . import config


def search_documents(vector_store, query, k=2, verbose=False):
    hits = vector_store.similarity_search_with_score(query, k=k)
    threshold = getattr(vector_store, 'distance_threshold', config.VECTOR_DISTANCE_THRESHOLD)

    kept = []
    for doc, score in hits:
        if verbose:
            print(f"原始 L2 距离：{score:.2f}")
        if threshold is None or score <= threshold:
            kept.append(doc)

    if verbose:
        print(f"过滤后数量：{len(kept)}")

    if not kept:
        return {"context": None, "sources": []}

    from .evidence import assemble_evidence
    return assemble_evidence([d.page_content for d in kept], [d.metadata for d in kept], range(len(kept)))
