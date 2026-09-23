"""Fit the serving ranker (search_ranking.LinearRanker) on a development capture; replay any capture with it.

Features come from the capture's diagnostics, computed by serving code (search_ranking.candidate_features),
so the fit and the replay rank exactly what the live API ranks. `fit` accepts only a development-partition
cache; freeze its output, then `score` validation / other benchmarks once.

uv run python agent-eval/fit_ranker.py fit --cache data/eval/opencaselaw/ranker/dev.json \
    --output src/swiss_court_assistant/server/ranker.json
uv run python agent-eval/fit_ranker.py score --cache data/eval/opencaselaw/ranker/validation.json \
    --ranker src/swiss_court_assistant/server/ranker.json --output data/eval/opencaselaw/ranker/validation-scored.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from opencaselaw import aggregate, metrics
from swiss_court_assistant.server.corpus import Passage
from swiss_court_assistant.server.search_ranking import FEATURES, LinearRanker, learned_select


def load(path: Path):
    report = json.loads(path.read_text())
    s = report['summary']
    if s['errors'] or s['completed'] != s['total']:
        raise SystemExit(f'{path}: cache must be complete and error-free')
    rows = []
    for row in report['per_query']:
        if row['status'] != 'ok':
            continue
        cands = (row['search'].get('diagnostics') or {}).get('candidates', [])
        if cands and any(c.get('features') is None for c in cands):
            raise SystemExit(f'{path}: capture has no serving features; recapture with the current pipeline')
        rows.append((row, cands))
    return report, rows


def fit(rows, l2: float, epochs: int = 400) -> LinearRanker:
    xs, ys = [], []
    for row, cands in rows:
        if not cands:
            continue
        # One row per judgment: its record (e.g. BGE vs BGer export) the reranker scored highest.
        best = {}
        for c in cands:
            if c['identity'] not in best or c['features']['rerank'] > best[c['identity']]['features']['rerank']:
                best[c['identity']] = c
        y = torch.tensor([float(row['available_gold'].get(i, 0)) for i in best])
        if y.sum() == 0:
            continue  # the gold is not in the pool: nothing to learn from this query's ordering
        xs.append(torch.tensor([[c['features'][n] for n in FEATURES] for c in best.values()], dtype=torch.float32))
        ys.append(y)
    allx = torch.cat(xs)
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    w = torch.zeros(len(FEATURES), requires_grad=True)
    opt = torch.optim.Adam([w], lr=0.05)
    for _ in range(epochs):
        opt.zero_grad()
        loss = sum(-(torch.softmax(torch.where(y > 0, y, torch.tensor(-1e4)), 0)
                     * torch.log_softmax(((x - mu) / sd) @ w, 0)).sum() for x, y in zip(xs, ys))
        loss = loss / len(xs) + l2 * (w ** 2).sum()
        loss.backward()
        opt.step()
    return LinearRanker(FEATURES, tuple(mu.tolist()), tuple(sd.tolist()), tuple(w.detach().tolist()))


def replay(rows, ranker: LinearRanker):
    scored = []
    for row, cands in rows:
        if cands:
            hits = [Passage(c['chunk_id'], c['decision_id'], 'body', [], None, None, '', '', '', None, c['language'])
                    for c in cands]
            ident = {c['decision_id']: c['identity'] for c in cands}
            features = {c['decision_id']: c['features'] for c in cands}
            ranked = [ident[h.decision_id] for h in learned_select(hits, 10, features, ranker, ident.__getitem__)]
        else:  # exact docket/reporter lookups bypass ranking
            ranked = row['ranked_ids']
        scored.append({'id': row['id'], 'language': row['language'], 'tags': row.get('tags', []),
                       'ranked_ids': ranked, 'available_gold': row['available_gold'],
                       'missing_gold': row['missing_gold'], **metrics(ranked, row['available_gold'])})
    return scored


def cross_validate(rows, split, l2, k=5):
    """Group-disjoint k-fold inside dev (the frozen split's related-judgment components)."""
    group = {qid: i for i, g in enumerate(split['groups']) for qid in g}
    order = sorted({group[row['id']] for row, _ in rows})
    fold = {g: i % k for i, g in enumerate(order)}
    scored = []
    for f in range(k):
        train = [r for r in rows if fold[group[r[0]['id']]] != f]
        test = [r for r in rows if fold[group[r[0]['id']]] == f]
        scored += replay(test, fit(train, l2))
    return aggregate(scored)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    f = sub.add_parser('fit')
    f.add_argument('--cache', type=Path, required=True)
    f.add_argument('--output', type=Path, required=True)
    f.add_argument('--l2', type=float, default=0.001)
    c = sub.add_parser('cv')
    c.add_argument('--cache', type=Path, required=True)
    c.add_argument('--split', type=Path, default=Path('agent-eval/fixtures/opencaselaw-split.json'))
    s = sub.add_parser('score')
    s.add_argument('--cache', type=Path, required=True)
    s.add_argument('--ranker', type=Path, required=True)
    s.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report, rows = load(args.cache)
    if args.command == 'cv':
        if report['config'].get('partition') != 'dev':
            raise SystemExit('cv accepts only a development-partition cache')
        split = json.loads(args.split.read_text())
        for l2 in (0.001, 0.01, 0.05):
            m = cross_validate(rows, split, l2)
            print(f'l2={l2:<6}', ' '.join(f'{k}={m[k]:.3f}' for k in ('hit@1', 'hit@10', 'mrr@10', 'recall@10', 'ndcg@10')))
        return
    if args.command == 'fit':
        if report['config'].get('partition') != 'dev':
            raise SystemExit('fit accepts only a development-partition cache')
        ranker = fit(rows, args.l2)
        out = {'features': ranker.features, 'mean': ranker.mean, 'scale': ranker.scale, 'weights': ranker.weights,
               'l2': args.l2, 'fitted_on': str(args.cache),
               'cache_sha256': hashlib.sha256(args.cache.read_bytes()).hexdigest(),
               'split_sha256': report['config']['split_sha256'],
               'retrieval_source_sha256': report['config']['retrieval_source_sha256'],
               'development': aggregate(replay(rows, ranker))}
        print(json.dumps({'weights': dict(zip(ranker.features, [round(x, 3) for x in ranker.weights])),
                          'development (in-sample)': out['development']}, indent=2))
    else:
        chosen = json.loads(args.ranker.read_text())
        if chosen['retrieval_source_sha256'] != report['config']['retrieval_source_sha256']:
            raise SystemExit('Candidate pipeline changed since the ranker was fitted; recapture first')
        scored = replay(rows, LinearRanker.load(args.ranker))
        out = {'ranker': str(args.ranker), 'ranker_sha256': hashlib.sha256(args.ranker.read_bytes()).hexdigest(),
               'config': report['config'], 'cache_sha256': hashlib.sha256(args.cache.read_bytes()).hexdigest(),
               'n': len(scored), 'metrics': aggregate(scored), 'per_query': scored,
               'by_language': {l: {'n': len(g), **aggregate(g)} for l in ('de', 'fr', 'it')
                               if (g := [r for r in scored if r['language'] == l])}}
        print(json.dumps({k: out[k] for k in ('n', 'metrics', 'by_language')}, indent=2))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
