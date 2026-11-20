"""Regression coverage for raw-distance filtering and lexical candidate admission."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rag import config
from rag.retriever_hybrid import BM25, HybridRetriever
from rag.retriever_vec import search_documents
from rag.store import ChildChromaStore, Doc, open_child_store


class VectorDistanceTests(unittest.TestCase):
    def store(self, **attributes):
        hits = [(Doc('第一条证据', {'source_file': 'a.txt'}), .4),
                (Doc('第二条证据', {'source_file': 'b.txt'}), 1.2),
                (Doc('第三条证据', {'source_file': 'c.txt'}), 1.21)]
        return SimpleNamespace(similarity_search_with_score=lambda query, k: hits[:k],
                               **attributes)

    def test_default_does_not_use_uncalibrated_or_legacy_threshold(self):
        hits = [(Doc('向量候选', {'source_file': 'a.txt'}), 20.)]
        store = SimpleNamespace(similarity_search_with_score=lambda query, k: hits)
        result = search_documents(store, '问题')
        self.assertEqual(result['sources'], ['a.txt'])

    def test_store_raw_threshold_is_independent_of_hybrid_gate(self):
        store = ChildChromaStore(None, None)
        store.profile['max_score'] = .01
        result = search_documents(self.store(distance_threshold=store.distance_threshold,
                                              profile=store.profile), '问题', k=3)
        self.assertEqual(result['sources'], ['a.txt', 'b.txt', 'c.txt'])

    def test_explicit_raw_threshold_keeps_boundary_and_filters_above(self):
        result = search_documents(self.store(distance_threshold=1.2), '问题', k=3)
        self.assertEqual(result['sources'], ['a.txt', 'b.txt'])

    def test_explicit_none_disables_raw_distance_filter(self):
        result = search_documents(self.store(distance_threshold=None), '问题', k=3)
        self.assertEqual(len(result['evidence']), 3)

    def test_zero_threshold_is_not_treated_as_disabled(self):
        store = SimpleNamespace(distance_threshold=0., similarity_search_with_score=lambda query, k: [
            (Doc('精确匹配', {'source_file': 'exact.txt'}), 0.),
            (Doc('非零距离', {'source_file': 'other.txt'}), .01)])
        self.assertEqual(search_documents(store, '问题')['sources'], ['exact.txt'])

    def test_no_neighbors_or_all_filtered_returns_empty(self):
        self.assertIsNone(search_documents(self.store(distance_threshold=.1), '问题')['context'])
        empty = SimpleNamespace(similarity_search_with_score=lambda query, k: [])
        self.assertEqual(search_documents(empty, '问题')['sources'], [])

    def test_open_store_does_not_restore_gate_as_raw_distance_threshold(self):
        # Use a persisted profile with a deliberately different max_score so the
        # manifest-loading path cannot silently reintroduce the unit mismatch.
        manifest = {'spec': {'model': config.EMBED_MODEL_BGE, 'distance': 'l2',
                             'normalize_embeddings': True},
                    'parent_collection': 'parents', 'child_collection': 'children',
                    'parents': 2, 'children': 2, 'version': 'fixture',
                    'profile': {'w_vec': .45, 'cover_slope': 1.25, 'max_score': .01}}
        collection = SimpleNamespace(count=lambda: 2)
        client = SimpleNamespace(get_collection=lambda *args, **kwargs: collection)
        chroma = SimpleNamespace(PersistentClient=lambda **kwargs: client)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'index_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            with patch.dict(sys.modules, {'chromadb': chroma}), \
                    patch('rag.store.load_embedding_model', return_value=object()), \
                    patch.object(config, 'VECTOR_DISTANCE_THRESHOLD', 1.2, create=True):
                store = open_child_store(directory)
        self.assertEqual(store.profile['max_score'], .01)
        self.assertEqual(store.distance_threshold, 1.2)


class BM25CandidateTests(unittest.TestCase):
    def test_no_matching_bigrams_returns_no_candidates(self):
        ranker = BM25(['苹果香蕉', '机器学习', ''])
        self.assertEqual(ranker.scores('量子计算'), [0., 0., 0.])
        self.assertEqual(ranker.rank('量子计算'), [])

    def test_empty_and_single_character_queries_have_no_bigrams(self):
        ranker = BM25(['苹果', '香蕉'])
        for query in ('', ' ', '苹'):
            with self.subTest(query=query):
                self.assertEqual(ranker.rank(query), [])

    def test_positive_candidates_only_in_descending_order(self):
        ranker = BM25(['苹果香蕉', '机器学习', '苹果苹果', ''])
        scores = ranker.scores('苹果')
        ranked = ranker.rank('苹果')
        self.assertEqual(set(ranked), {0, 2})
        self.assertTrue(all(scores[index] > 0 for index in ranked))
        self.assertEqual([scores[index] for index in ranked], sorted((scores[0], scores[2]), reverse=True))

    def test_positive_ties_keep_corpus_order(self):
        self.assertEqual(BM25(['苹果', '香蕉', '苹果']).rank('苹果'), [0, 2])

    def test_empty_corpus(self):
        self.assertEqual(BM25([]).rank('苹果'), [])

    def test_hybrid_does_not_fill_bm25_topk_with_unmatched_documents(self):
        class Store:
            corpus = (['向量候选', '苹果知识', '完全无关'],
                      [{'parent_id': i, 'source_file': f'{i}.txt'} for i in range(3)])

            def similarity_search_with_score(self, query, k):
                return [(Doc('向量候选', self.corpus[1][0]), .8)]

        retriever = HybridRetriever(Store(), max_score=None)
        self.assertEqual(retriever._bm_rank('苹果'), [1])
        picked, rejected = retriever.retrieve_indices('苹果', k=5, apply_gate=False)
        self.assertFalse(rejected)
        self.assertEqual(set(picked), {0, 1})
        self.assertNotIn(2, picked)

    def test_no_lexical_match_leaves_vector_ranking_intact(self):
        class Store:
            corpus = (['向量候选', '完全无关'],
                      [{'parent_id': 0}, {'parent_id': 1}])

            def similarity_search_with_score(self, query, k):
                return [(Doc('向量候选', {'parent_id': 0}), .8)]

        retriever = HybridRetriever(Store(), max_score=None)
        self.assertEqual(retriever._bm_rank('量子计算'), [])
        self.assertEqual(retriever.retrieve_indices('量子计算', apply_gate=False), ([0], False))


if __name__ == '__main__':
    unittest.main()
