# -*- coding: utf-8 -*-
"""可信 RAG 问答系统 —— 检索层。

两种检索器，接口一致，可互换：
    retriever_vec.search_documents      纯向量 + 固定阈值（基线）
    retriever_hybrid.search_documents   向量 + 字符 BM25 双通道 RRF 融合 + 拒答闸门（线上）

索引层两套并存：
    rag/index.py      子块（句子级）+ 父块（500 字）双 collection
    rag/store.py      ChildChromaStore 查子块、返回父块，接口与单层一致
"""
from .retriever_hybrid import HybridRetriever, search_documents as search_hybrid
from .retriever_vec import search_documents as search_vector

__all__ = ["HybridRetriever", "search_hybrid", "search_vector"]
