"""Frozen configuration evaluation. Never selects parameters on the evaluated set."""
import argparse
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from rag.config import CHUNKS_JSON, PROJECT_ROOT
from rag.index import digest
from rag.loader import init_vector_store
from rag.retriever_hybrid import HybridRetriever
from eval.qset import IN_SCOPE, OUT_SCOPE


def metrics(ranks, k=5):
    if not ranks:
        return {'n': 0}
    return {'n': len(ranks), **{f'R@{i}': sum(r is not None and r <= i for r in ranks) / len(ranks) for i in (1, 3, 5)},
            f'MRR@{k}': sum(1 / r for r in ranks if r is not None and r <= k) / len(ranks)}


def rank_of(order, gold):
    return next((i for i, p in enumerate(order[:5], 1) if p in gold), None)


def load_dataset(name):
    chunks = json.loads(CHUNKS_JSON.read_text(encoding='utf-8'))
    if name == 'development':
        rows = [{'id': f'in-{i}', 'question': q, 'source': s, 'gold': [g], 'kind': kind, 'answerable': True}
                for i, (q, s, g, kind) in enumerate(IN_SCOPE)]
        rows += [{'id': f'out-{i}', 'question': q, 'source': None, 'gold': [], 'kind': kind, 'answerable': False}
                 for i, (q, kind) in enumerate(OUT_SCOPE)]
        return chunks, rows, 'Development set reused in historical tuning; not an independent test.'
    data = json.loads((Path(__file__).parent / 'datasets/challenge_v1.json').read_text(encoding='utf-8'))
    if data['chunks_hash'] != digest(chunks):
        raise ValueError('Challenge annotations do not match the corpus snapshot')
    return chunks, data['rows'], data['description']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['development', 'challenge'], default='development')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    chunks, rows, note = load_dataset(args.dataset)
    store = init_vector_store()
    ret = HybridRetriever(store, **store.profile)
    if len(ret.texts) != len(chunks) or any(t != c['text'] or m['source_file'] != c['source']
            for t, m, c in zip(ret.texts, ret.metas, chunks)):
        raise ValueError('Active corpus differs from benchmark; rebuild --from-snapshot or annotate a new dataset')
    # Warm encoder/index once, then time each complete retrieval including encoding.
    ret.retrieve_indices('检索预热', apply_gate=False)
    detail = []
    ranks = {method: [] for method in ('vector', 'bm25', 'hybrid')}
    source_ranks = {method: [] for method in ranks}
    latencies = []
    for row in rows:
        q = row['question']
        start = time.perf_counter()
        vo, vd = ret._vector_rank(q)
        from rag.retriever_hybrid import rrf_merge
        bo = ret._bm_rank(q)
        ho = rrf_merge([vo, bo], [ret.w_vec, 1-ret.w_vec], ret.rrf_k)[:5]
        cov = max((ret._coverage(q, i) for i in vo[:5]), default=0)
        nearest = min(vd.values(), default=float('inf'))
        score = nearest - ret.cover_slope * cov
        rejected = score > ret.max_score
        elapsed = (time.perf_counter() - start) * 1000
        latencies.append(elapsed)
        # Verify the public production path agrees; timing above avoids double encoding.
        production, public_rejected = ret.retrieve_indices(q, k=5)
        if public_rejected != rejected or (not rejected and production != ho):
            raise AssertionError('Evaluation and production retrieval paths diverged')
        result = {**row, 'orders': {'vector': vo[:5], 'bm25': bo[:5], 'hybrid': ho},
                  'rejected': rejected, 'nearest_distance': nearest, 'coverage': cov,
                  'gate_score': score, 'retrieval_ms': elapsed}
        if row['answerable']:
            for method, order in result['orders'].items():
                if row['kind'] != 'cross':
                    ranks[method].append(rank_of(order, row['gold']))
                source_ranks[method].append(next((i for i, p in enumerate(order, 1) if chunks[p]['source'] == row['source']), None))
        detail.append(result)
    inside = [r for r in detail if r['answerable']]
    outside = [r for r in detail if not r['answerable']]
    fn = sum(r['rejected'] for r in inside)
    fp = sum(not r['rejected'] for r in outside)
    source_hash = digest({str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((PROJECT_ROOT / 'rag').glob('*.py'))})
    packages = {p: importlib.metadata.version(p) for p in ('chromadb', 'sentence-transformers', 'numpy')}
    result = {'dataset': args.dataset, 'dataset_note': note, 'dataset_hash': digest(rows),
              'chunks_hash': digest(chunks), 'source_hash': source_hash, 'packages': packages,
              'created_utc': datetime.now(timezone.utc).isoformat(), 'index_version': store.version,
              'profile': store.profile, 'counts': dict(Counter(r['kind'] for r in rows)),
              'chunk_level': {m: metrics(rs) for m, rs in ranks.items()},
              'source_level': {m: metrics(rs) for m, rs in source_ranks.items()},
              'gate': {'in_n': len(inside), 'out_n': len(outside), 'FN': fn, 'FP': fp,
                       'FRR': fn/len(inside), 'FAR': fp/len(outside),
                       'accuracy': (len(rows)-fn-fp)/len(rows)},
              'latency_ms': {'includes_query_embedding': True, 'warm_p50': statistics.median(latencies),
                             'warm_p95': sorted(latencies)[max(0, __import__('math').ceil(.95*len(latencies))-1)]},
              'detail': detail}
    output = args.output or PROJECT_ROOT / 'eval/results' / f'release_{args.dataset}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k != 'detail'}, ensure_ascii=False, indent=2))
    print(f'Result: {output}')


if __name__ == '__main__':
    main()
