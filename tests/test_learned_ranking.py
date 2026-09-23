import asyncio
from dataclasses import replace
import json
import sqlite3
import tempfile
from pathlib import Path
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from swiss_court_assistant.server.corpus import Passage
from swiss_court_assistant.server.query_translation import QueryTranslator, sanitize
from swiss_court_assistant.server.search_ranking import (
    DEPLOYED_RANK_CONFIG, FEATURES, LinearRanker, candidate_features, learned_select,
)
from swiss_court_assistant.statute_links import StatuteLinks, canon, query_articles


def hit(did, logit=1.0, court='bge', lang='de', cited=0, sources=('dense',)):
    return Passage(did + '#0', did, 'body', [], 0, 12, 'text', did, court, None, lang, court=court,
                   score=logit, reranker_score=logit, passage_score=logit, fusion_score=0.01,
                   cited_by=cited, retrieval_sources=sources)


class StatuteReferences(unittest.TestCase):
    def test_codes_fold_to_german_form(self):
        self.assertEqual(canon('CO'), 'OR')
        self.assertEqual(canon('Cst.'), 'BV')
        self.assertEqual(canon('LAINF'), 'UVG')
        self.assertEqual(canon('EMRK'), 'EMRK')

    def test_query_articles_across_languages(self):
        self.assertEqual(query_articles('Art. 41 OR Haftpflicht'), {('OR', '41')})
        self.assertEqual(query_articles("art. 336c al. 1 let. c CO"), {('OR', '336c')})
        self.assertEqual(query_articles('Kündigung wegen Krankheit'), set())

    def test_most_cited_orders_by_citations_within_language(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'links.sqlite'
            con = sqlite3.connect(path)
            con.executescript("""
                CREATE TABLE links (decision_id TEXT, law TEXT, article TEXT, mentions INTEGER,
                                    language TEXT, court TEXT, cited_by INTEGER);
                CREATE TABLE articles (law TEXT, article TEXT, decisions INTEGER);
                CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
                INSERT INTO meta VALUES ('decisions', '1000');
                INSERT INTO articles VALUES ('OR', '41', 3);
                INSERT INTO links VALUES ('a', 'OR', '41', 1, 'de', 'bge', 5), ('b', 'OR', '41', 1, 'de', 'bge', 50),
                                         ('c', 'OR', '41', 1, 'fr', 'bge', 500), ('a', 'ZGB', '8', 2, 'de', 'bge', 5);
            """)
            con.commit()
            con.close()
            links = StatuteLinks(path)
            self.assertEqual(links.most_cited({('OR', '41')}, 'de'), ['b', 'a'])
            self.assertEqual(links.articles_of(['a', 'x']), {'a': {('OR', '41'), ('ZGB', '8')}, 'x': set()})
            self.assertGreater(links.idf({('OR', '41')})[('OR', '41')], links.idf({('ZGB', '9')})[('ZGB', '9')] - 10)


class LearnedRanking(unittest.TestCase):
    def features(self, hits, explicit=frozenset(), articles=None):
        articles = articles or {}
        return candidate_features(hits, 'de', DEPLOYED_RANK_CONFIG, articles,
                                  lambda arts: {a: 1.0 for a in arts}, set(explicit))

    def test_features_cover_ranker_inputs(self):
        f = self.features([hit('a', sources=('dense', 'statute'))], {('OR', '41')}, {'a': {('OR', '41')}})['a']
        self.assertEqual(set(f), set(FEATURES))
        self.assertEqual((f['src_statute'], f['explicit_article'], f['native']), (1.0, 1.0, 1.0))
        self.assertAlmostEqual(f['prf_articles'], 1.0)

    def test_prf_profile_rewards_articles_of_top_judgments(self):
        hits = [hit('top', 5), hit('shares', 0), hit('other', 0), hit('none', -1)]
        arts = {'top': {('OR', '336c')}, 'shares': {('OR', '336c')}, 'other': {('BGG', '42')}}
        f = self.features(hits, articles=arts)
        # The article of the top judgment weighs more than one only a lower-ranked judgment cites.
        self.assertGreater(f['shares']['prf_articles'], f['other']['prf_articles'])
        self.assertEqual(f['none']['prf_articles'], 0)

    def test_learned_select_scores_and_deduplicates_aliases(self):
        ranker = LinearRanker(('rerank', 'logcit'), (0.0, 0.0), (1.0, 1.0), (1.0, 1.0))
        hits = [hit('bge_twin', 1, cited=0), hit('bger_twin', 3, cited=0), hit('cited', 1, cited=100)]
        f = self.features(hits)
        ident = {'bge_twin': 'X', 'bger_twin': 'X', 'cited': 'cited'}.__getitem__
        chosen = learned_select(hits, 10, f, ranker, ident)
        self.assertEqual([h.decision_id for h in chosen], ['cited', 'bger_twin'])

    def test_ranker_file_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'ranker.json'
            path.write_text(json.dumps({'features': ['rerank'], 'mean': [1.0], 'scale': [2.0], 'weights': [3.0]}))
            self.assertAlmostEqual(LinearRanker.load(path).score({'rerank': 5.0}), 6.0)


class Translation(unittest.TestCase):
    def run_with(self, content):
        response = MagicMock()
        response.json.return_value = {'choices': [{'message': {'content': content}}]}
        response.raise_for_status.return_value = None
        client = MagicMock()
        client.__aenter__ = AsyncMock(return_value=client)
        client.__aexit__ = AsyncMock(return_value=None)
        client.post = AsyncMock(return_value=response)
        t = QueryTranslator()
        t._model = 'm'
        with patch('swiss_court_assistant.server.query_translation.httpx.AsyncClient', return_value=client):
            first = asyncio.run(t.translate('Hundebiss'))
            second = asyncio.run(t.translate('Hundebiss'))
        return first, second, client.post.await_count

    def test_three_lines_parsed_and_cached(self):
        first, second, calls = self.run_with('de: Hundebiss\nfr: morsure de chien\nit: morso di cane')
        self.assertEqual(first, {'de': 'Hundebiss', 'fr': 'morsure de chien', 'it': 'morso di cane'})
        self.assertEqual((second, calls), (first, 1))

    def test_incomplete_answer_falls_back(self):
        first, _, calls = self.run_with('de: Hundebiss\nfr: morsure de chien')
        self.assertIsNone(first)
        self.assertEqual(calls, 2)  # failures are not cached

    def test_sanitize_drops_invented_statutes_and_keeps_named_ones(self):
        self.assertEqual(sanitize('Raub Gewalt Drohung', 'Raub Gewalt Drohung OR § 143 ZGB'), 'Raub Gewalt Drohung')
        self.assertEqual(sanitize('Art. 41 OR Haftpflicht', 'Art. 41 CO Dommages-intérêts'),
                         'Art. 41 CO Dommages-intérêts')
        self.assertEqual(sanitize('Art. 41 OR', 'Art. 41 DTF Risarcimento'), 'Art. 41 OR Risarcimento')
        self.assertEqual(sanitize('LAA liaison', 'UVG Verbindung'), 'UVG Verbindung')


if __name__ == '__main__':
    unittest.main()
