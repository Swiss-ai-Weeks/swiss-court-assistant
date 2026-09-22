"""Run with: uv run python -m unittest discover -s tests -v"""
import asyncio
import importlib.util
import math
import sqlite3
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import numpy as np
import polars as pl

from swiss_court_assistant.identity import canonical_id, docket_key, exact_reference, own_dockets
from swiss_court_assistant.index import prepare
from swiss_court_assistant.sampling import sample
from swiss_court_assistant.server.citations import CitationIndex
from swiss_court_assistant.server.corpus import Corpus, Passage, _diverse, _select
from swiss_court_assistant.server.decisions import SqliteDecisionStore, _Identities
from swiss_court_assistant.server.language import detect_language, language_matches


def hit(did, score=1, court='bge', language='de', chunk=None):
    return Passage(chunk or did + '#0', did, 'body', [], 0, 10, 'a passage', did, court, None,
                   language, court=court, score=score, reranker_score=score)


class Identities(unittest.TestCase):
    def test_canonical_and_exact(self):
        for s in ('bge_122 V 157', 'BGE 122 V 157', 'bge_BGE_122_V_157', 'ATF 122 V 157'):
            self.assertEqual(canonical_id(s), 'bge_BGE_122_V_157')
        self.assertEqual(canonical_id('bge_113 Ib 420'), 'bge_BGE_113_Ib_420')
        self.assertEqual(canonical_id('bge_historical_1_I_3'), 'bge_BGE_1_I_3')
        self.assertEqual(canonical_id('bge_unknown'), 'bge_unknown')
        self.assertTrue(exact_reference('BVGE 2013/10'))
        self.assertFalse(exact_reference('Haftung nach Art. 41 OR'))
        self.assertEqual(docket_key('bger_1A.122_2005'), docket_key('1A.122/2005'))

    def test_heading_not_cited_dockets(self):
        text = 'Regeste\nmentions 4C.100/2000\nUrteilskopf\n132 II 408\n1A.122/2005 / 1P.288/2005\nRegeste\nLaw\nSachverhalt\nsee 4A_123/2020'
        self.assertEqual(own_dockets(text), {'1A.122/2005', '1P.288/2005'})
        store = _Identities()
        store._identities([('bge_132 II 408', 'BGE 132 II 408')], [('bge_132 II 408', 'bge', text)])
        for ref in ('1A.122/2005', 'bger_1A.122_2005', 'BGE 132 II 408', 'bge_BGE_132_II_408'):
            self.assertEqual(store.find(ref), ['bge_BGE_132_II_408'])
        self.assertEqual(store.find('4A_123/2020'), [])
        self.assertEqual(store.physical_id('BGE 132 II 408'), 'bge_132 II 408')

    def test_build_ids_preserve_original_paragraph_join(self):
        df = pl.DataFrame({'decision_id': ['bge_122 V 157', 'bge_BGE_122_V_157', 'bge_unknown'],
                           'court': ['bge'] * 3, 'canton': ['CH'] * 3,
                           'docket_number': ['BGE 122 V 157', 'BGE 122 V 157', None],
                           'decision_date': ['1996-01-01'] * 3})
        out = prepare(df)
        self.assertEqual(out['decision_id'].to_list(), ['bge_BGE_122_V_157', 'bge_unknown'])
        self.assertEqual(out['source_decision_id'][0], 'bge_122 V 157')
        self.assertEqual(prepare(out)['source_decision_id'][0], 'bge_122 V 157')

    def test_chunk_paragraph_annotations_survive_canonicalisation(self):
        from swiss_court_assistant import chunking
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            text = 'Das ist der Anfang dieser Erwägung und ihr weiterer Text.'
            docs = pl.DataFrame({'decision_id': ['bge_BGE_122_V_157'],
                                 'source_decision_id': ['bge_122 V 157'], 'language': ['de'],
                                 'full_text': [text], 'regeste': ['']})
            docs.write_parquet(path / 'docs.parquet')
            pl.DataFrame({'decision_id': ['bge_122 V 157'], 'e_number': ['2.1'],
                          'text': [text]}).write_parquet(path / 'paras.parquet')
            with patch.object(chunking, 'PARAGRAPHS', str(path / 'paras.parquet')):
                chunks = chunking.build_chunks(path / 'docs.parquet', workers=1)
            self.assertEqual(chunks['chunk_id'][0], 'bge_BGE_122_V_157#0')
            self.assertEqual(chunks['erwaegungen'][0].to_list(), ['2.1'])

    def test_sampling_retains_old_bge_outside_date_filter(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'fixture.parquet'
            pl.DataFrame({'decision_id': ['bge_50 I 10', 'cantonal'], 'court': ['bge', 'zh'],
                          'canton': ['CH', 'ZH'], 'branch': ['zivil'] * 2, 'language': ['de'] * 2,
                          'decision_date': ['1924-01-01', '2020-01-01'], 'has_full_text': [True] * 2,
                          'text_length': [100, 600], 'content_hash': [None, None]}).write_parquet(path)
            subset, _ = sample(0, 0, 42, str(path), where=pl.col('decision_date') >= '2000', fraction=0.01)
            self.assertIn('bge_BGE_50_I_10', subset['decision_id'])

    def test_graph_counts_unique_sources_not_sum_of_twins(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'graph.sqlite'
            con = sqlite3.connect(path)
            con.executescript('CREATE TABLE decisions (decision_id TEXT, court TEXT, docket TEXT, date TEXT);'
                              'CREATE TABLE counts (decision_id TEXT, cited_by INT, cites INT);'
                              'CREATE TABLE edges (source TEXT, target TEXT);')
            ids = ['bge_122 V 157', 'bge_BGE_122_V_157']
            con.executemany('INSERT INTO decisions VALUES (?, ?, ?, ?)', [(i, 'bge', 'BGE 122 V 157', None) for i in ids])
            con.executemany('INSERT INTO edges VALUES (?, ?)', [('source1', ids[0]), ('source1', ids[1]),
                                                               ('source2', ids[1]), (ids[0], ids[1])])
            con.commit()
            graph = CitationIndex(path)
            self.assertEqual(graph.counts(ids[0]), (2, 0))
            self.assertEqual(graph.cited_by_counts(ids), dict.fromkeys(ids, 2))
            self.assertEqual({c.decision_id for c in graph.cited_by(ids[1])}, {'source1', 'source2'})
            con.close()


class Languages(unittest.TestCase):
    def test_queries(self):
        for q, lang in [('Hundebiss', 'de'), ('Art. 41 OR Schadenersatz ausservertragliche Haftung', 'de'),
                        ('Notwehr Angriff straflos Strafrecht', 'de'), ('Art. 41 CO', 'fr'),
                        ('What does Art. 41 OR say about liability?', 'en'),
                        ('responsabilità civile danno morale', 'it'), ('ricorso tardivo restituzione termine', 'it'),
                        ('responsabilité civile dommage', 'fr')]:
            with self.subTest(q=q):
                self.assertEqual(detect_language(q), lang)
        self.assertEqual(detect_language('XYZ', default=''), '')

    def test_mixed_spanish_drift(self):
        german = 'Der Halter eines Tieres haftet für den Schaden. Er kann sich durch den Nachweis der gebotenen Sorgfalt entlasten.'
        spanish = ' Art. 56 al. 1 CO establece la responsabilidad del titular de un animal por los daños que causa.'
        self.assertTrue(language_matches(german, 'de'))
        self.assertFalse(language_matches(german + spanish, 'de'))


class Ranking(unittest.IsolatedAsyncioTestCase):
    async def test_logit_priors_and_fallback(self):
        c = object.__new__(Corpus)
        c.citations = None
        c.authority_weights = (0.5, 1, 0.5)
        result = await c._authority_rank([hit('cantonal', court='zh'), hit('bger', court='bger'), hit('bge')])
        self.assertEqual([p.decision_id for p in result], ['bge', 'bger', 'cantonal'])
        popular = replace(hit('bge'), cited_by=99)
        self.assertAlmostEqual((await c._authority_rank([popular]))[0].score, 2 + 0.5 * math.log(100))
        fallback = replace(popular, reranker_score=None, score=0.3)
        self.assertEqual((await c._authority_rank([fallback]))[0].score, 0.3)

    async def test_multilingual_pipeline_executes_and_intersects_filters(self):
        c = object.__new__(Corpus)
        c.citations = None
        c.authority_weights = (0.5, 1, 0.5)
        c.embed_model = 'fake'
        c.decisions = SimpleNamespace(find=lambda did: [did])
        c.vdb = SimpleNamespace(encode=lambda *a: None)
        c._call = AsyncMock(return_value=np.array([1.0]))
        c.facets = SimpleNamespace(decision_mask=lambda **kw: np.array([True, False]), rows=lambda mask: mask)
        calls, reranked = [], []
        def knn(vector, lang, rows):
            calls.append((lang, rows.tolist()))
            return [{'id': lang + '1', 'passage': hit(lang, court='zh', language=lang)}] if rows.any() else []
        c._knn = knn
        c._passages = lambda rows: [r['passage'] for r in rows]
        async def rerank(q, hits):
            reranked.append(hits)
            return hits
        c._rerank = rerank
        with ThreadPoolExecutor(3) as pool:
            c._knn_pool = pool
            found = await c.semantic_search_by_language({'de': 'Frage', 'fr': 'question', 'it': 'domanda'},
                                                         where=np.array([False, True]))
        self.assertEqual(len(found), 3)
        self.assertEqual(len(calls), 6)
        self.assertEqual(sum(rows == [False, False] for _, rows in calls), 3)
        self.assertEqual(len(reranked), 3)

    async def test_reservations_sorted_and_unique(self):
        groups = [[hit('de-top', 6, 'zh'), hit('bge_122 V 157', -3), hit('de-other', 5, 'zh')],
                  [hit('fr-top', 7, 'vd', 'fr'), hit('fr-bge', -4, 'bge', 'fr')],
                  [hit('it-top', 10, 'ti', 'it'), hit('it-bge', -5, 'bge', 'it')]]
        selected = _select(groups, 6)
        self.assertEqual(selected[0].decision_id, 'it-top')
        self.assertEqual(sum(p.court == 'bge' for p in selected), 3)
        self.assertEqual(len(_diverse([hit('bge_122 V 157'), hit('bge_BGE_122_V_157')], 12, 1)), 1)
        self.assertEqual(_diverse(groups[0], 0), [])
        same_judgment = [hit('bge_BGE_132_II_408'), hit('bger_1A.122_2005')]
        self.assertEqual(len(_select([same_judgment], 10, identity=lambda _: 'same')), 1)

    async def test_exact_missing_never_semantic_fallback(self):
        c = object.__new__(Corpus)
        c.decisions = SimpleNamespace(find=lambda _: [])
        c.citations = None
        c.semantic_search_by_language = AsyncMock()
        self.assertEqual(await c.search_decisions('BGE 115 IV 162'), [])
        c.semantic_search_by_language.assert_not_called()

    async def test_endpoint(self):
        from swiss_court_assistant.server.app import app, services
        from swiss_court_assistant.server.search_ranking import RankConfig
        corpus = SimpleNamespace(search_decisions=AsyncMock(return_value=[hit('bge_BGE_122_V_157')]),
                                 authority_weights=(0.5, 1, 0.5), hybrid_config=RankConfig())
        app.dependency_overrides[services] = lambda: SimpleNamespace(agent=SimpleNamespace(corpus=corpus))
        try:
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url='http://test') as client:
                r = await client.get('/api/search', params={'q': 'Hundebiss'})
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.json()['results'][0]['decision_id'], 'bge_BGE_122_V_157')
                self.assertEqual((await client.get('/api/search', params={'q': '   '})).status_code, 422)
                self.assertEqual((await client.get('/api/search', params={'q': 'x', 'k': 0})).status_code, 422)
                r = await client.get('/api/search', params={'q': 'x', 'strategy': 'hybrid', 'debug': True})
                self.assertEqual(r.json()['strategy'], 'hybrid')
                self.assertEqual(r.json()['diagnostics'], {})
                self.assertIsInstance(r.json()['weights'], dict)
                self.assertEqual((await client.get('/api/search', params={
                    'q': 'x', 'strategy': 'hybrid', 'baseline': True})).status_code, 422)
                self.assertEqual((await client.get('/api/search', params={
                    'q': 'x', 'strategy': 'unknown'})).status_code, 422)
        finally:
            app.dependency_overrides.clear()


class Metrics(unittest.TestCase):
    def test_rank_aware_ndcg_and_compatibility(self):
        spec = importlib.util.spec_from_file_location('ocl', Path(__file__).parents[1] / 'agent-eval/opencaselaw.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        m = module.metrics(['miss', 'gold'], {'gold': 3})
        self.assertEqual(m['hit@1'], 0)
        self.assertEqual(m['mrr@10'], 0.5)
        self.assertAlmostEqual(m['ndcg@10'], 1 / math.log2(3))
        self.assertEqual(m['opencaselaw_ndcg@10'], 1)


if __name__ == '__main__':
    unittest.main()
