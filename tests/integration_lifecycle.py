"""Real Chroma lifecycle smoke test; runs offline with a cached BGE model."""
from pathlib import Path
import json
import os
import sys
import tempfile
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
from rag.embedding import load_embedding_model
import chromadb
from rag import config, loader
from rag.index import build_from_chunks


def main():
    model = load_embedding_model()
    # Exercise actual four-format parsers first; keep the benchmark index intact.
    chunks = loader.prepare_chunks(loader.load_documents())
    assert {Path(c['source']).suffix for c in chunks} == {'.txt','.md','.pdf','.docx'}
    with tempfile.TemporaryDirectory(prefix='rag_lifecycle_') as temp:
        manifest = Path(temp) / 'index_manifest.json'
        client = chromadb.PersistentClient(path=str(Path(temp) / 'db'))
        with patch.object(config, 'INDEX_MANIFEST', manifest):
            p,c = build_from_chunks(chunks, client=client, model=model)
            first = manifest.read_bytes()
            p2,c2 = build_from_chunks(chunks, client=client, model=model)
            assert first == manifest.read_bytes() and p.name == p2.name
            # Remove every chunk of one source and replace a remaining document chunk.
            remove = chunks[0]['source']
            updated = [dict(x) for x in chunks if x['source'] != remove]
            updated[0]['text'] += '\n生命周期更新校验专用标记。'
            updated[0].pop('chunk_id',None)
            newp,newc = build_from_chunks(updated, client=client, model=model)
            active = json.loads(manifest.read_text(encoding='utf-8'))
            assert active['parent_collection'] != p.name
            docs = newp.get(include=['documents','metadatas'])
            assert all(m['source_file'] != remove for m in docs['metadatas'])
            assert any('生命周期更新校验专用标记' in t for t in docs['documents'])
            assert p.count() == len(chunks)  # Previous version retained.
            print(json.dumps({'formats':4, 'initial_parents':len(chunks),
                              'updated_parents':newp.count(), 'idempotent':True,
                              'deletion_and_update':True, 'previous_version_retained':True},ensure_ascii=False))
        # Chroma keeps native handles alive; its temp directory is kept for this
        # process lifetime and cleaned after Python exits by the OS temp policy.
        # Explicit cleanup may raise PermissionError on Windows, so close the
        # system before TemporaryDirectory exits.
        client._system.stop()


if __name__ == '__main__': main()
