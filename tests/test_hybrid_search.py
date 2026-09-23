import asyncio
from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from swiss_court_assistant.fts import KeywordIndex, broad_terms, match_expr
from swiss_court_assistant.server.corpus import Corpus, Passage
from swiss_court_assistant.server.search_ranking import (
    DEPLOYED_RANK_CONFIG, RankConfig, decision_text, fuse_decisions, rank_decisions, select_decisions,
)


def hit(did, logit=1, court='bge', lang='de', chunk=None):
    return Passage(chunk or did + '#0', did, 'body', [], 0, 12, 'original text', did, court, None, lang,
                   court=court, score=logit, reranker_score=logit)


class HybridHelpers(unittest.TestCase):
    def test_rrf_counts_decision_once_per_source(self):
        a, b = hit('a'), hit('b')
        pools = {'dense': [a, replace(a, chunk_id='a#1'), b], 'bm25': [b, a]}
        fused = fuse_decisions(pools, lambda x: x)
        self.assertEqual(len(fused), 2)
        scores = {p.decision_id: p.score for p, _ in fused}
        self.assertAlmostEqual(scores['a'], scores['b'])
        self.assertEqual(set(fused[0][0].retrieval_sources), {'dense', 'bm25'})

    def test_fusion_keeps_each_source_and_alias_deduplication(self):
        pools = {'dense': [hit(f'd{i}') for i in range(20)], 'bm25': [hit('lexical')],
                 'bge': [hit('twin'), hit('leading')]}
        results = fuse_decisions(pools, lambda did: 'd0' if did == 'twin' else did, limit=12, floor=2)
        ids = [h.decision_id for h, _ in results]
        self.assertEqual(len(ids), 12)
        self.assertIn('lexical', ids)
        self.assertIn('leading', ids)
        self.assertNotIn('twin', ids)

    def test_headnote_input_is_bounded_without_changing_evidence(self):
        p = hit('a')
        s = SimpleNamespace(docket='BGE 1 I 1', court_label='Federal Court', title='Title',
                            legal_area='Civil', regeste='Authoritative headnote')
        text = decision_text(s, [p, replace(p, chunk_id='another')])
        self.assertIn('Authoritative headnote', text)
        self.assertEqual(text.count('original text'), 1)
        self.assertEqual(p.text, 'original text')
        self.assertEqual((p.char_start, p.char_end), (0, 12))
        s.regeste = 'x' * 20000
        self.assertLessEqual(len(decision_text(s, [p])), 6500)

    def test_popularity_cap_and_fallback(self):
        relevant = hit('relevant', 5, 'ti', 'it')
        popular = replace(hit('popular', 1), cited_by=100000000)
        ranked = rank_decisions([popular, relevant], 'it', RankConfig())
        self.assertEqual(ranked[0].decision_id, 'relevant')
        self.assertEqual(ranked[1].score, 3)
        fallback = replace(popular, reranker_score=None, score=.03, fusion_score=.03)
        self.assertEqual(rank_decisions([fallback], 'de', RankConfig())[0].score, .03)

    def test_headnote_and_best_passage_blend(self):
        p = replace(hit('a', 4), passage_score=-2)
        zero = RankConfig(citations=0, bge=0, bger=0, native_language=0, headnote_weight=0)
        self.assertEqual(rank_decisions([p], None, zero)[0].score, -2)
        self.assertEqual(rank_decisions([p], None, replace(zero, headnote_weight=.5))[0].score, 1)
        self.assertEqual(rank_decisions([p], None, replace(zero, headnote_weight=1))[0].score, 4)

    def test_authority_uses_judgment_not_passage_export(self):
        p = replace(hit('bger_docket', 1, 'bger'), authority_id='bge_reporter', authority_court='bge')
        self.assertEqual(rank_decisions([p], None, RankConfig())[0].score, 1.5)
        self.assertEqual(p.decision_id, 'bger_docket')  # citation source/offsets remain the original

    def test_native_relevance_anchors_survive_authority(self):
        italian = [hit('ti1', 6, 'ti', 'it'), hit('ti2', 5, 'ti', 'it')]
        popular = [replace(hit(f'popular{i}', 4, 'bge', 'de'), cited_by=100000) for i in range(10)]
        selected = select_decisions(italian + popular, 5, 'it', RankConfig(), lambda x: x)
        self.assertIn('ti2', [h.decision_id for h in selected])
        self.assertEqual(len(selected), 5)
        self.assertEqual([h.score for h in selected], sorted([h.score for h in selected], reverse=True))

    def test_unrelated_bge_not_reserved(self):
        hits = [hit(str(i), 10-i, 'ti', 'it') for i in range(5)] + [hit('irrelevant', -10, 'bge', 'it')]
        self.assertNotIn('irrelevant', [h.decision_id for h in select_decisions(hits, 3, 'it', RankConfig(), lambda x: x)])


class Lexical(unittest.TestCase):
    def test_broad_is_safe_and_does_not_change_exact_search(self):
        self.assertEqual(match_expr('"Art. 41 OR" Haftung'), '"Art. 41 OR" "Haftung"')
        self.assertIn('or', broad_terms('What does Art. 41 OR say about Schadenersatz?'))
        self.assertNotIn('what', broad_terms('What is Haftung?'))
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fts.sqlite'
            con = sqlite3.connect(path)
            con.execute("CREATE VIRTUAL TABLE passages USING fts5(text, content='')")
            con.executemany('INSERT INTO passages(rowid, text) VALUES (?, ?)',
                            [(1, 'Tierhalter Haftung Schaden'), (2, 'Haftung Arbeitgeber')])
            con.commit()
            con.close()
            index = KeywordIndex(path)
            self.assertEqual([r[0] for r in index.search('Tierhalter Arbeitgeber')], [])
            self.assertEqual({r[0] for r in index.broad_search('Tierhalter Arbeitgeber')}, {1, 2})
            self.assertEqual({r[0] for r in index.broad_search('Tierhalter OR " NOT * Arbeitgeber')}, {1, 2})
            index.broad_search('Haftung', timeout=-1)  # cancellation must not poison the connection
            self.assertTrue(index.search('Haftung'))
            index.con.close()


class Pipeline(unittest.IsolatedAsyncioTestCase):
    async def test_live_default_uses_frozen_hybrid_configuration(self):
        import os
        with patch.dict(os.environ, {}, clear=True), patch.object(Corpus, '_open'):
            c = Corpus(Path('unused'), Path('unused'), SimpleNamespace())
        try:
            self.assertEqual(c.search_strategy, 'hybrid')
            self.assertEqual(asdict(c.hybrid_config), {
                'citations': .5, 'bge': 1, 'bger': 0, 'authority_cap': 5,
                'native_language': .25, 'bge_margin': 2, 'headnote_weight': 1})
            self.assertEqual(c.hybrid_config, DEPLOYED_RANK_CONFIG)
            c._hybrid_search = AsyncMock(return_value=[])
            queries = {'de': 'Haftung', 'fr': 'responsabilité', 'it': 'responsabilità'}
            await c.semantic_search_by_language(queries, query_language='it')
            c._hybrid_search.assert_awaited_once_with(queries, 12, None, 'it', None)
        finally:
            c._pool.shutdown()
            c._knn_pool.shutdown()

    async def test_hybrid_respects_filters_language_and_real_passages(self):
        c = object.__new__(Corpus)
        c.citations = None
        c.hybrid_config = RankConfig()
        c.statute_links = c.ranker = c.translator = None
        c.embed_model = 'fake'
        c.vdb = SimpleNamespace(encode=lambda *a: None)
        c._call = AsyncMock(side_effect=lambda fn, *a, **kw: fn(*a, **kw))
        c.facets = SimpleNamespace(decision_mask=lambda **kw: np.array([True, False]), rows=lambda m: m)
        c._bge_headnote_rows = lambda: np.array([True, False])
        summary = SimpleNamespace(docket='real docket', court_label='court', court='ti', title=None,
                                  legal_area=None, regeste='headnote')
        c.decisions = SimpleNamespace(find=lambda d: [d], summary=lambda d: summary)
        seen = []
        def knn(v, lang, rows):
            seen.append(rows.tolist())
            return [{'passage': hit(lang, court='ti', lang=lang)}] if rows.any() else []
        c._knn = knn
        c._passages = lambda rows: [r['passage'] for r in rows]
        c._lexical_candidates = lambda query, where: [hit('lex-it', court='ti', lang='it'), hit('lex-de')]
        async def rerank(q, hits, texts=None):
            if texts is not None:
                self.assertTrue(all('headnote' in t for t in texts))
            self.assertTrue(all(p.text == 'original text' for p in hits))
            return hits
        c._rerank = rerank
        debug = {}
        with ThreadPoolExecutor(3) as pool:
            c._knn_pool = pool
            hits = await c._hybrid_search({'it': 'domanda'}, 10, np.array([False, True]), 'it', debug)
        self.assertNotIn('lex-de', [h.decision_id for h in hits])
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen.count([False, False]), 2)
        self.assertEqual(seen.count([False, True]), 1)
        self.assertTrue(debug['rerank_complete'])
        self.assertEqual(debug['pools']['it']['bm25'], 1)


class Split(unittest.TestCase):
    def test_related_gold_cannot_cross_split(self):
        folder = Path(__file__).parents[1] / 'agent-eval'
        sys.path.insert(0, str(folder))
        try:
            from search_experiments import grouped_split
            qs = [{'id': str(i), 'tags': ['de'], 'relevant': [{'decision_id': f'd{i // 2}'}]} for i in range(20)]
            s = grouped_split(qs, lambda d: d)
            for i in range(0, 20, 2):
                self.assertEqual(str(i) in s['dev'], str(i+1) in s['dev'])
            self.assertFalse(set(s['dev']) & set(s['validation']))
            self.assertEqual(set(s['dev']) | set(s['validation']), {str(i) for i in range(20)})
        finally:
            sys.path.pop(0)


if __name__ == '__main__':
    unittest.main()
