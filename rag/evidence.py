"""Evidence formatting and citation identifier validation (not entailment scoring)."""
import re

REF = re.compile(r'\[S(\d+)\]')
REFUSAL = '资料中没有相关信息。'


def assemble_evidence(texts, metadatas, indices):
    evidence = []
    for n, i in enumerate(indices, 1):
        meta = metadatas[i]
        evidence.append({'id': f'S{n}', 'chunk_id': meta.get('chunk_id', f'legacy:{i}'),
                         'source': meta.get('source_path', meta.get('source_file', 'unknown')),
                         'page': meta.get('page', -1), 'text': texts[i]})
    context = '\n\n'.join(f"[{e['id']}] 来源：{e['source']}；页码：{e['page'] + 1 if e['page'] >= 0 else '未提供'}\n{e['text']}" for e in evidence)
    return {'context': context or None, 'sources': list(dict.fromkeys(e['source'] for e in evidence)),
            'evidence': evidence}


def validate_citations(answer, evidence):
    cited = {f'S{m}' for m in REF.findall(answer)}
    valid = {e['id'] for e in evidence}
    invalid = sorted(cited - valid)
    return {'valid': bool(cited) and not invalid, 'cited': sorted(cited), 'invalid': invalid,
            'scope': 'identifier validation only; evidence support needs semantic/human review'}
