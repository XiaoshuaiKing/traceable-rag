"""Versioned, idempotent parent/child indexes with atomic manifest activation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import uuid

from . import config
from .splitter import split_sentences


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def index_spec(chunks):
    return {
        'schema': config.INDEX_SCHEMA, 'model': config.EMBED_MODEL_BGE,
        'normalize_embeddings': True, 'distance': 'l2',
        'child_min_len': config.CHILD_MIN_LEN, 'child_max_len': config.CHILD_MAX_LEN,
        'parent_chunk_size': config.CHUNK_SIZE, 'parent_overlap': config.CHUNK_OVERLAP,
        'chunks_hash': digest(chunks),
    }


def build_from_chunks(chunks, rebuild_parent=True, verbose=True, *, client=None, model=None):
    """Build both levels; unchanged inputs reuse the active collections.

    The compatibility argument cannot create mixed versions. A failed staging
    build leaves the active manifest unchanged. Historical indexes are retained.
    """
    if not chunks or any(not c.get('text', '').strip() or not c.get('source') for c in chunks):
        raise ValueError('Cannot activate an empty or malformed corpus')
    spec = index_spec(chunks)
    if client is None:
        import chromadb
        client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    manifest_path = config.INDEX_MANIFEST
    if manifest_path.exists():
        active = json.loads(manifest_path.read_text(encoding='utf-8'))
        if active.get('spec') == spec:
            pcol = client.get_collection(active['parent_collection'], embedding_function=None)
            ccol = client.get_collection(active['child_collection'], embedding_function=None)
            if pcol.count() != active['parents'] or ccol.count() != active['children']:
                raise ValueError('Active index is incomplete; restore a valid manifest/index pair')
            if verbose:
                print(f"Index unchanged: {active['version']} ({pcol.count()} parents, {ccol.count()} children)")
            return pcol, ccol
    if model is None:
        from .embedding import load_embedding_model
        model = load_embedding_model()
    texts = [c['text'] for c in chunks]
    kids, owners = [], []
    for pid, text in enumerate(texts):
        parts = [p for p in split_sentences(text, config.CHILD_MIN_LEN, config.CHILD_MAX_LEN) if p.strip()]
        kids.extend(parts)
        owners.extend([pid] * len(parts))
    pe = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    ce = model.encode(kids, normalize_embeddings=True, show_progress_bar=False)
    version = digest(spec)[:12] + '_' + uuid.uuid4().hex[:8]
    pname, cname = 'rag_parent_' + version, 'rag_child_' + version
    pcol = client.create_collection(pname, embedding_function=None, metadata={'hnsw:space': 'l2'})
    ccol = client.create_collection(cname, embedding_function=None, metadata={'hnsw:space': 'l2'})
    def metadata(i):
        c = chunks[i]
        return {'parent_id': i, 'chunk_id': c.get('chunk_id', 'chunk_' + digest(c)[:24]),
                'source_file': c['source'], 'source_path': c.get('source_path', c['source']),
                'page': c.get('page', -1)}
    pcol.add(ids=[f'p{i}' for i in range(len(texts))], documents=texts,
             embeddings=[e.tolist() for e in pe], metadatas=[metadata(i) for i in range(len(texts))])
    ccol.add(ids=[f'c{i}' for i in range(len(kids))], documents=kids,
             embeddings=[e.tolist() for e in ce],
             metadatas=[{**metadata(pid), 'parent_text': texts[pid]} for pid in owners])
    if pcol.count() != len(texts) or ccol.count() != len(kids):
        raise ValueError('Incomplete staging index; active manifest was not changed')
    profile = {'w_vec': config.FUSION_WEIGHT, 'cover_slope': config.GATE_SLOPE,
               'max_score': config.GATE_THRESHOLD, 'calibration': 'development-400'}
    manifest = {'version': version, 'spec': spec, 'parent_collection': pname,
                'child_collection': cname, 'parents': len(texts), 'children': len(kids),
                'chunks': chunks, 'profile': profile}
    atomic_json(manifest_path, manifest)
    if verbose:
        print(f'Activated {version}: {len(texts)} parents, {len(kids)} children')
    return pcol, ccol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--from-snapshot', action='store_true', help='Build the frozen benchmark corpus')
    parser.add_argument('--rebuild-parent', action='store_true', help='Compatibility flag; both levels always stay consistent')
    args = parser.parse_args()
    if args.from_snapshot:
        chunks = json.loads(config.CHUNKS_JSON.read_text(encoding='utf-8'))
    else:
        from .loader import load_documents, prepare_chunks
        chunks = prepare_chunks(load_documents())
    build_from_chunks(chunks)


if __name__ == '__main__':
    main()
