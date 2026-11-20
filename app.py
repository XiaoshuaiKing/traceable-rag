# -*- coding: utf-8 -*-
"""命令行入口。

    python app.py                          # 交互提问
    python app.py -q "Agent 有哪四个核心部件？"   # 单次提问
    python app.py -q "..." --retriever vector    # 用基线检索器对照
    python app.py --rebuild                # 重建向量库
"""
import argparse
import sys
import json
import time
from rag.evidence import REFUSAL, validate_citations

from rag import search_hybrid, search_vector
from rag.config import CHROMA_DIR
from rag.llm_client import call_llm
from rag.loader import init_vector_store
from rag.prompts import SYSTEM_PROMPT

RETRIEVERS = {"hybrid": search_hybrid, "vector": search_vector}         # 用字典来调用函数省去了if-else

# 检索证据、构造模型输入、请求回答、检查引用。
def answer(vector_store, query, retriever="hybrid", k=5, show_context=False):
    """检索 → 组装 prompt → 调模型。检索为空就直接拒答，不浪费一次 LLM 调用。"""
    doc_info = RETRIEVERS[retriever](vector_store, query, k=k)
    context, sources = doc_info["context"], doc_info["sources"]

    evidence = doc_info.get('evidence', [])
    if not context:
        return {"answer": None, "sources": [], "rejected": True, "context": None,
                'status': 'retrieval_rejected', 'evidence': []}

    if show_context:
        print(f"\n[检索到 {len(sources)} 份资料]\n{context[:800]}\n")

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"参考资料：\n{context}\n\n请根据上述资料回答问题：{query}"},
    ]
    for attempt in range(3):
        try:
            text = call_llm(messages)
            if text.strip() == REFUSAL:
                return {'answer': text, 'sources': sources, 'rejected': True, 'context': context,
                        'status': 'insufficient_evidence', 'evidence': evidence}
            citations = validate_citations(text, evidence)
            return {"answer": text if citations['valid'] else None, "sources": sources,
                    "rejected": False, "context": context, 'evidence': evidence,
                    'status': 'answered' if citations['valid'] else 'citation_invalid',
                    'citations': citations}
        except Exception as error:
            print(f"第 {attempt + 1} 次调用失败：{error}")
            if isinstance(error, ValueError):
                break
            if attempt < 2:
                time.sleep(0.5 * (2 ** attempt))
    return {"answer": None, "sources": sources, "rejected": False, "context": context,
            'status': 'generation_error', 'evidence': evidence}

# 用 while True 接收问题，每次调用 answer()
def interactive(vector_store, retriever="hybrid"):
    print("输入问题开始提问，quit 退出")
    while True:
        query = input("\n请输入想问的问题：").strip()
        if not query:
            print("请勿输入空字符")
            continue
        if query.lower() == "quit":
            print("已退出")
            break

        result = answer(vector_store, query, retriever=retriever)
        if result["rejected"]:
            print("资料库中没有足够依据回答这个问题")
            continue
        if result['status'] != 'answered':
            print(f"回答未通过检查：{result['status']}")
            continue
        print(f"\n检索到的资料来源：{result['sources']}")
        print(f"\n模型回答：\n{result['answer']}")


def main():
    parser = argparse.ArgumentParser(description="可信 RAG 问答系统")
    parser.add_argument("-q", "--query", help="单次提问，不给则进入交互模式")
    parser.add_argument("--retriever", choices=list(RETRIEVERS), default="hybrid",
                        help="hybrid=向量+BM25 融合（默认），vector=纯向量基线")
    parser.add_argument("-k", type=int, default=5, help="返回知识块数，默认 5")
    parser.add_argument("--rebuild", action="store_true", help="重建向量库")
    parser.add_argument("--show-context", action="store_true", help="打印召回原文")
    parser.add_argument('--retrieve-only', action='store_true', help='只检索，不调用模型或消耗 API')
    parser.add_argument('--json', action='store_true', help='以 JSON 输出单次结果')
    args = parser.parse_args()

    if args.k <= 0 or (args.query is not None and not args.query.strip()):
        parser.error('k 必须为正数，问题不能为空')
    try:
        vector_store = init_vector_store(rebuild=args.rebuild)
    except Exception as exc:
        print(f'初始化失败：{exc}', file=sys.stderr)
        return 1
    if vector_store is None:
        print("知识库初始化失败，程序结束")
        return 1

    if not args.query:
        if args.retrieve_only or args.json:
            parser.error('--retrieve-only / --json 需要 -q')
        interactive(vector_store, retriever=args.retriever)
        return 0

    if args.retrieve_only:
        result = RETRIEVERS[args.retriever](vector_store, args.query, k=args.k)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    result = answer(vector_store, args.query, retriever=args.retriever,
                    k=args.k, show_context=args.show_context)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result['status'] in ('answered', 'retrieval_rejected', 'insufficient_evidence') else 2
    if result["rejected"]:
        print("资料库中没有足够依据回答这个问题")
        return 0
    print(f"资料来源：{result['sources']}")
    if result['status'] != 'answered':
        print(f"回答未通过检查：{result['status']}")
        return 2
    print(f"\n回答：\n{result['answer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
