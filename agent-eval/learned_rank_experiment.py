"""Offline learned linear ranker over cached hybrid candidates (fit on dev only, score validation once).

Features per (query, candidate judgment) come from the cached diagnostics plus the statute-reference
graph (data/raw/graph/statute_references.parquet). No gold IDs reach the features; labels are used only
for the dev fit. A ListNet-style softmax loss with L2 keeps the model small (one weight per feature).

uv run python agent-eval/learned_rank_experiment.py --dev data/eval/opencaselaw/hybrid/dev-v3.json \
    --eval data/eval/opencaselaw/hybrid/validation-raw.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pickle
from pathlib import Path

import numpy as np
import polars as pl
import torch

from opencaselaw import aggregate, metrics
from search_experiments import cached_rows
from swiss_court_assistant.server.mentions import _MENTION
from swiss_court_assistant.server.search_ranking import RankConfig, rank_decisions
from swiss_court_assistant.statutes import GLOSSARY

PARAMS = Path('data/eval/opencaselaw/hybrid/parameters.json')
LINKS = Path('data/eval/opencaselaw/hybrid/statute-links.pkl')

CANON = {f.rstrip('.').upper(): row['de'][0].upper() for row in GLOSSARY for forms in row.values() for f in forms}
CANON.update({'CEDH': 'EMRK', 'CEDU': 'EMRK', 'LTAF': 'VGG', 'OJ': 'OG', 'AUG': 'AIG', 'LETR': 'AIG',
              'LSTR': 'AIG', 'LSTRI': 'AIG', 'LEI': 'AIG', 'LEF': 'SCHKG', 'LAINF': 'UVG', 'CST': 'BV', 'COST': 'BV'})


def canon(code: str) -> str:
    c = code.rstrip('.').upper()
    return CANON.get(c, c)


def load_links():
    if LINKS.exists():
        return pickle.loads(LINKS.read_bytes())
    df = pl.read_parquet('data/raw/graph/statute_references.parquet')
    df = df.with_columns(pl.col('law_code').map_elements(canon, return_dtype=pl.String).alias('law'))
    df = df.group_by(['decision_id', 'law', 'article']).agg(pl.col('mention_count').sum())
    by_dec = collections.defaultdict(dict)
    df_count = collections.Counter()
    for d, law, art, m in df.iter_rows():
        by_dec[d][(law, art)] = m
        df_count[(law, art)] += 1
    out = (dict(by_dec), dict(df_count), len(by_dec))
    LINKS.write_bytes(pickle.dumps(out))
    return out


NAMES = ['rerank', 'passage', 'fusion', 'logcit', 'bge', 'bger', 'native', 'explicit_article',
         'prf_articles', 'src_dense', 'src_bm25', 'src_bge', 'src_headnotes', 'frozen_score',
         'rerank_gap', 'n_sources', 'pool_citations']


def _graph():
    import sqlite3
    return sqlite3.connect('file:data/graph/corpus.citations.sqlite?mode=ro', uri=True)


def build(report, links, prf_depth=20, cite_depth=30):
    by_dec, df_count, n_docs = links
    graph = _graph()
    physical_to_identity = {}
    config = RankConfig(**json.loads(PARAMS.read_text())['parameters'])
    raw = {r['id']: {c['decision_id']: c for c in (r['search'].get('diagnostics') or {}).get('candidates', [])}
           for r in report['per_query'] if r['status'] == 'ok'}
    out = []
    for row, hits, ids, lang in cached_rows(report):
        if not hits:
            out.append((row, None, None, None))
            continue
        explicit = {(canon(m.group('code')), m.group('num').replace(' ', '')) for m in _MENTION.finditer(row['query'])}
        ranked = rank_decisions(hits, lang, config)
        frozen = {h.decision_id: h.score for h in ranked}
        profile = collections.Counter()
        for r, h in enumerate(ranked[:prf_depth]):
            for art in by_dec.get(h.decision_id, {}):
                profile[art] += math.log(n_docs / (1 + df_count.get(art, 0))) / (r + 1)
        norm = sum(profile.values()) or 1.0
        # Cross-result citation evidence: how strongly the query's own top judgments cite each candidate.
        identity = {**ids}
        order = list(dict.fromkeys(ids[h.decision_id] for h in ranked))[:cite_depth]
        physical = collections.defaultdict(list)
        for h in hits:
            physical[ids[h.decision_id]].append(h.decision_id)
        by_physical = {p: ids[p] for p in ids}
        cited = collections.Counter()
        for r, did in enumerate(order):
            targets = {t for p in physical[did] for (t,) in graph.execute('SELECT target FROM edges WHERE source = ?', (p,))}
            for t in {by_physical.get(t, t) for t in targets} - {did}:
                cited[t] += 1 / math.sqrt(1 + r)
        best = {}
        for h in hits:  # one row per judgment identity: keep the best-scoring physical record
            did = ids[h.decision_id]
            if did not in best or frozen[h.decision_id] > frozen[best[did].decision_id]:
                best[did] = h
        feats, dids = [], []
        for did, h in best.items():
            arts = by_dec.get(h.decision_id, {})
            src = set(raw[row['id']][h.decision_id]['retrieval_sources'])
            court = h.authority_court or h.court
            feats.append([h.reranker_score, h.passage_score or 0.0, h.fusion_score * 100,
                          math.log1p(h.cited_by), court == 'bge', court == 'bger', h.language == lang,
                          bool(explicit & arts.keys()), sum(profile[a] for a in arts) / norm,
                          'dense' in src, 'bm25' in src, 'bge' in src, 'bge_headnotes' in src, frozen[h.decision_id]])
            dids.append(did)
        feats = np.array(feats, dtype=np.float32)
        # Query-relative reranker score: how far below this query's best candidate it is.
        gap = feats[:, 0] - feats[:, 0].max()
        n_sources = feats[:, 9:13].sum(1)
        pool_citations = np.array([sum(cited.get(p, 0.0) for p in {d, *physical[d]}) for d in dids])
        feats = np.column_stack([feats, gap, n_sources, np.log1p(pool_citations)]).astype(np.float32)
        labels = [row['available_gold'].get(d, 0) for d in dids]
        out.append((row, dids, feats, np.array(labels, dtype=np.float32)))
    return out


def fit(data, cols, l2, epochs=400):
    xs = [torch.tensor(f[:, cols]) for _, _, f, y in data if f is not None and y.sum() > 0]
    ys = [torch.tensor(y) for _, _, f, y in data if f is not None and y.sum() > 0]
    mu = torch.cat(xs).mean(0)
    sd = torch.cat(xs).std(0) + 1e-6
    w = torch.zeros(len(cols), requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(epochs):
        opt.zero_grad()
        loss = 0
        for x, y in zip(xs, ys):
            s = ((x - mu) / sd) @ w
            target = torch.softmax(torch.where(y > 0, y, torch.tensor(-1e4)), 0)
            loss = loss - (target * torch.log_softmax(s, 0)).sum()
        loss = loss / len(xs) + l2 * (w ** 2).sum()
        loss.backward()
        opt.step()
    return w.detach(), mu, sd


def score(data, cols, model):
    w, mu, sd = model
    rows = []
    for row, dids, f, _ in data:
        if f is None:
            ranked = row['ranked_ids']
        else:
            s = ((torch.tensor(f[:, cols]) - mu) / sd) @ w
            ranked = [dids[i] for i in torch.argsort(s, descending=True).tolist()[:10]]
        rows.append({'id': row['id'], 'language': row['language'], **metrics(ranked, row['available_gold'])})
    return aggregate(rows)


def folds(data, split, k=5):
    """Group-disjoint folds inside dev, using the frozen split's related-judgment components."""
    group = {qid: i for i, g in enumerate(split['groups']) for qid in g}
    order = sorted({group[row['id']] for row, *_ in data})
    fold_of = {g: i % k for i, g in enumerate(order)}
    return [[d for d in data if fold_of[group[d[0]['id']]] == f] for f in range(k)]


def cross_validate(data, cols, l2, split):
    parts = folds(data, split)
    rows = []
    for i, test in enumerate(parts):
        train = [d for j, p in enumerate(parts) if j != i for d in p]
        model = fit(train, cols, l2)
        rows.append((score(test, cols, model), len(test)))
    n = sum(c for _, c in rows)
    return {k: sum(m[k] * c for m, c in rows) / n for k in rows[0][0]}


def fmt(m):
    return ' '.join(f"{k}={m[k]:.3f}" for k in ('hit@1', 'hit@10', 'mrr@10', 'recall@10', 'ndcg@10'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dev', type=Path, required=True)
    p.add_argument('--eval', type=Path, nargs='*', default=[])
    p.add_argument('--l2', type=float, default=0.01)
    p.add_argument('--split', type=Path, default=Path('agent-eval/fixtures/opencaselaw-split.json'))
    p.add_argument('--cv', action='store_true', help='select on grouped dev cross-validation only')
    args = p.parse_args()
    links = load_links()
    dev = build(json.loads(args.dev.read_text()), links)
    evals = {path.name: build(json.loads(path.read_text()), links) for path in args.eval}
    base = [NAMES.index(n) for n in ('frozen_score',)]
    sets = {
        'frozen': base,
        'signals': [NAMES.index(n) for n in NAMES if n not in ('frozen_score', 'explicit_article', 'prf_articles')],
        'signals+statutes': [NAMES.index(n) for n in NAMES if n != 'frozen_score'],
        'frozen+statutes': [NAMES.index(n) for n in ('frozen_score', 'explicit_article', 'prf_articles')],
    }
    if args.cv:
        split = json.loads(args.split.read_text())
        idx = lambda *names: [NAMES.index(n) for n in names]
        core = ['rerank', 'passage', 'fusion', 'logcit', 'bge', 'bger', 'native']
        srcs = ['src_dense', 'src_bm25', 'src_bge', 'src_headnotes']
        variants = {
            'frozen': idx('frozen_score'),
            'core': idx(*core),
            'core+src': idx(*core, *srcs),
            'core+src+statutes': idx(*core, *srcs, 'explicit_article', 'prf_articles'),
            'core+src+explicit': idx(*core, *srcs, 'explicit_article'),
            'core+src+statutes+cites': idx(*core, *srcs, 'explicit_article', 'prf_articles', 'pool_citations'),
            'core+src+cites': idx(*core, *srcs, 'pool_citations'),
            'all': idx(*[n for n in NAMES if n != 'frozen_score']),
        }
        for l2 in (0.001, 0.01):
            for name, cols in variants.items():
                print(f'l2={l2:<6} {name:20} cv {fmt(cross_validate(dev, cols, l2, split))}', flush=True)
        return
    for name, cols in sets.items():
        model = fit(dev, cols, args.l2)
        line = f'{name:18} dev {fmt(score(dev, cols, model))}'
        print(line)
        for ename, data in evals.items():
            print(f'{"":18} {ename} {fmt(score(data, cols, model))}')
        print(f'{"":18} weights', {NAMES[c]: round(float(x), 3) for c, x in zip(cols, model[0])})


if __name__ == '__main__':
    main()
