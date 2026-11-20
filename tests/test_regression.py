import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rag import config
from rag.index import build_from_chunks
from rag.evidence import assemble_evidence, validate_citations, REFUSAL
from rag.retriever_hybrid import HybridRetriever
from rag.store import Doc
from rag.splitter import split_sentences


class Vector(list):
    def tolist(self): return list(self)


class Model:
    def encode(self, texts, **kwargs):
        assert kwargs['normalize_embeddings']
        return [Vector([1., 0.]) for _ in texts]


class Collection:
    def add(self, **kwargs): self.rows = kwargs
    def count(self): return len(self.rows['ids']) if hasattr(self, 'rows') else 0


class Client:
    def __init__(self): self.collections = {}; self.fail = False
    def create_collection(self, name, **kwargs):
        if self.fail: raise RuntimeError('simulated write failure')
        self.collections[name] = Collection()
        return self.collections[name]
    def get_collection(self, name, **kwargs): return self.collections[name]


class IndexTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.manifest = Path(self.tmp.name) / 'index_manifest.json'
        self.patcher = patch.object(config, 'INDEX_MANIFEST', self.manifest)
        self.patcher.start()
        self.client = Client()
        self.chunks = [{'text': '短句也不能在建库时悄悄丢失。', 'source': 'a.txt'},
                       {'text': '文档删除后不应再次出现在当前索引。', 'source': 'b.txt'}]
    def tearDown(self): self.patcher.stop(); self.tmp.cleanup()
    def build(self, chunks=None, **kwargs):
        return build_from_chunks(self.chunks if chunks is None else chunks, client=self.client, model=Model(), verbose=False, **kwargs)
    def test_default_and_compatibility_path(self):
        p,c = self.build(rebuild_parent=False)
        self.assertEqual(p.count(), 2); self.assertEqual(c.count(), 2)
    def test_idempotent_build(self):
        self.build(); before = self.manifest.read_bytes(); self.build()
        self.assertEqual(before, self.manifest.read_bytes())
        self.assertEqual(len(self.client.collections), 2)
    def test_delete_activates_reduced_corpus(self):
        self.build(); self.build(self.chunks[:1])
        m = json.loads(self.manifest.read_text(encoding='utf-8'))
        self.assertEqual(m['parents'], 1)
        self.assertEqual([c['source'] for c in m['chunks']], ['a.txt'])
    def test_failed_build_preserves_active_manifest(self):
        self.build(); before = self.manifest.read_bytes(); self.client.fail = True
        with self.assertRaises(RuntimeError): self.build(self.chunks[:1])
        self.assertEqual(before, self.manifest.read_bytes())
    def test_empty_build_preserves_active(self):
        self.build(); before = self.manifest.read_bytes()
        with self.assertRaises(ValueError): self.build([])
        self.assertEqual(before, self.manifest.read_bytes())
    def test_profile_bound_to_index(self):
        self.build(); m = json.loads(self.manifest.read_text(encoding='utf-8'))
        self.assertTrue(m['spec']['normalize_embeddings'])
        self.assertEqual(m['spec']['model'], config.EMBED_MODEL_BGE)
        self.assertEqual(m['profile']['max_score'], .4)


class RetrievalTests(unittest.TestCase):
    def setUp(self):
        class Store:
            corpus = (['相同的内容', '相同的内容'], [{'parent_id': 0, 'source_file':'a.txt'}, {'parent_id':1, 'source_file':'b.txt'}])
            def similarity_search_with_score(self, query, k): return [(Doc('相同的内容', {'parent_id':1}), .1)]
        self.ret = HybridRetriever(Store())
    def test_identical_text_keeps_correct_identity(self):
        self.assertEqual(self.ret._vector_rank('内容')[0], [1])
    def test_invalid_query_or_k(self):
        for q,k in [('',5), ('问题',0)]:
            with self.assertRaises(ValueError): self.ret.retrieve_indices(q,k=k)
    def test_empty_neighbors_rejected(self):
        with patch.object(self.ret, '_vector_rank', return_value=([],{})):
            self.assertTrue(self.ret.retrieve_indices('问题')[1])
    def test_citation_mapping(self):
        result = assemble_evidence(self.ret.texts, self.ret.metas, [1])
        self.assertIn('[S1] 来源：b.txt', result['context'])
        self.assertTrue(validate_citations('结论[S1]', result['evidence'])['valid'])
        self.assertFalse(validate_citations('结论[S2]', result['evidence'])['valid'])
        self.assertFalse(validate_citations('没有编号的回答', result['evidence'])['valid'])
    def test_sentence_hard_limit(self):
        original = '连续没有标点的长句' * 100
        parts = split_sentences(original)
        self.assertTrue(all(len(p)<=120 for p in parts))
        self.assertEqual(''.join(parts), original)


class AnswerTests(unittest.TestCase):
    def run_answer(self, reply=None, error=None):
        import app
        info = assemble_evidence(['有依据的事实'], [{'source_file':'a.txt'}], [0])
        with patch.dict(app.RETRIEVERS, {'hybrid': lambda *a,**kw: info}), patch.object(app, 'call_llm', return_value=reply, side_effect=error):
            return app.answer(None, '问题')
    def test_valid_answer(self): self.assertEqual(self.run_answer('有依据的事实[S1]')['status'], 'answered')
    def test_model_refusal(self): self.assertTrue(self.run_answer(REFUSAL)['rejected'])
    def test_fabricated_citation(self): self.assertEqual(self.run_answer('结论[S9]')['status'], 'citation_invalid')
    def test_generation_failure_distinguished(self):
        r = self.run_answer(error=ValueError('missing config'))
        self.assertEqual(r['status'], 'generation_error'); self.assertFalse(r['rejected'])


if __name__ == '__main__': unittest.main()
