"""Freeze related-query groups before tuning; replay cached candidate scores without model calls.

No benchmark IDs or relevance labels are sent to the app. Validation is deliberately not accepted
by `tune`: freeze parameters on development, then use `score` on the validation cache once.
This public benchmark has been inspected before, so this is a tuning holdout, not an unseen test set.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import random

from opencaselaw import aggregate, load_queries, metrics
from swiss_court_assistant.server.corpus import Passage
from swiss_court_assistant.server.decisions import SqliteDecisionStore
from swiss_court_assistant.server.search_ranking import RankConfig, select_decisions


def grouped_split(queries, resolve, seed=42):
    parent = {q['id']: q['id'] for q in queries}
    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x
    seen = {}
    for q in queries:
        for rel in q['relevant']:
            did = resolve(rel['decision_id'])
            if did in seen:
                parent[root(q['id'])] = root(seen[did])
            seen[did] = q['id']
    groups = defaultdict(list)
    for q in queries:
        groups[root(q['id'])].append(q['id'])
    groups = sorted(sorted(g) for g in groups.values())
    language = {q['id']: next((t for t in q.get('tags', []) if t in ('de', 'fr', 'it')), 'other') for q in queries}
    total = Counter(language.values())
    rng = random.Random(seed)
    best = None
    # Balance group-disjoint validation by language and size, using labels for grouping only,
    # never model performance. Resolve docket/BGE aliases before forming connected components.
    for _ in range(1000):
        order = list(groups)
        rng.shuffle(order)
        chosen = []
        for group in order:
            if len(chosen) >= round(len(queries) * .3):
                break
            chosen.extend(group)
        counts = Counter(language[i] for i in chosen)
        cost = sum((counts[l] - .3 * n) ** 2 / max(n, 1) for l, n in total.items())
        candidate = (cost, sorted(chosen))
        if best is None or candidate < best:
            best = candidate
    validation = best[1]
    return {'seed': seed, 'groups': groups, 'dev': sorted(set(parent) - set(validation)),
            'validation': validation, 'validation_languages': dict(Counter(language[i] for i in validation)),
            'note': 'Components sharing a canonical gold judgment never cross the split. Previously inspected public benchmark.'}


def cached_rows(report):
    rows = []
    for row in report['per_query']:
        if row['status'] != 'ok':
            continue
        diag = row['search'].get('diagnostics') or {}
        hits, identities = [], {}
        for c in diag.get('candidates', []):
            if c['reranker_score'] is None:
                raise ValueError('Cannot calibrate a cache with missing reranker logits')
            identities[c['decision_id']] = c['identity']
            hits.append(Passage(c['chunk_id'], c['decision_id'], 'body', [], None, None, '', '', '', None,
                                c['language'], court=c['court'], cited_by=c['cited_by'],
                                authority_id=c.get('authority_id'), authority_court=c.get('authority_court'),
                                reranker_score=c['reranker_score'], passage_score=c.get('passage_score'),
                                fusion_score=c['fusion_score']))
        rows.append((row, hits, identities, diag.get('query_language')))
    return rows


def replay(rows, config):
    scored = []
    for row, hits, identities, language in rows:
        if hits:
            key = lambda did: identities[did]
            ranked = [key(h.decision_id) for h in select_decisions(hits, 10, language, config, key)]
        else:  # Exact lookups are resolved without a reranker and stay fixed.
            ranked = row['ranked_ids']
        scored.append({'id': row['id'], 'language': row['language'], 'ranked_ids': ranked,
                       'available_gold': row['available_gold'], 'missing_gold': row['missing_gold'],
                       **metrics(ranked, row['available_gold'])})
    return scored


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    s = sub.add_parser('split')
    s.add_argument('--golden', type=Path, required=True)
    s.add_argument('--db', type=Path, default=Path('data/vectordb/corpus.sqlite'))
    s.add_argument('--output', type=Path, required=True)
    for cmd in ('tune', 'score'):
        s = sub.add_parser(cmd)
        s.add_argument('--cache', type=Path, required=True)
        s.add_argument('--output', type=Path, required=True)
        if cmd == 'tune':
            s.add_argument('--control', type=Path, required=True)
        else:
            s.add_argument('--parameters', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'split':
        if args.output.exists():
            raise SystemExit('Split already exists; do not silently reshuffle after inspecting scores')
        store = SqliteDecisionStore(args.db)
        out = grouped_split(load_queries(args.golden), lambda d: next(iter(store.find(d)), d))
        out['golden_sha256'] = hashlib.sha256(args.golden.read_bytes()).hexdigest()
    else:
        report = json.loads(args.cache.read_text())
        if report['summary']['errors'] or report['summary']['completed'] != report['summary']['total']:
            raise SystemExit('Cache must be complete and error-free')
        if not report['config'].get('capture_candidates'):
            raise SystemExit('Use --capture-candidates; final top-10 lists cannot be used for calibration')
        rows = cached_rows(report)
        if args.command == 'tune':
            if report['config'].get('partition') != 'dev':
                raise SystemExit('Tune accepts only a development-partition cache')
            ids = {row[0]['id'] for row in rows}
            control = [r for r in json.loads(args.control.read_text())['per_query'] if r['id'] in ids and r['status'] == 'ok']
            if len(control) != len(rows):
                raise SystemExit('Control must cover the same scored queries')
            baseline = aggregate(control)
            trials = []
            for a, b, c, cap, native, headnote in itertools.product(
                    (0, .15, .3, .5), (0, .5, 1), (0, .5), (2, 3, 5), (0, .25), (0, .5, 1)):
                config = RankConfig(a, b, c, cap, native, headnote_weight=headnote)
                scores = aggregate(replay(rows, config))
                trials.append({'parameters': asdict(config), 'metrics': scores})
            safe = [t for t in trials if t['metrics']['hit@10'] >= baseline['hit@10']
                    and t['metrics']['recall@10'] >= baseline['recall@10']]
            eligible = safe or trials
            best = max(eligible, key=lambda t: (t['metrics']['hit@1'], t['metrics']['ndcg@10'],
                       t['metrics']['recall@10'], -sum(t['parameters'].values())))
            out = {'parameters': best['parameters'], 'development': best['metrics'], 'control': baseline,
                   'passes_dev_guardrails': bool(safe), 'n': len(rows), 'trials': trials,
                   'cache_sha256': hashlib.sha256(args.cache.read_bytes()).hexdigest(),
                   'split_sha256': report['config']['split_sha256'],
                   'retrieval_source_sha256': report['config']['retrieval_source_sha256']}
        else:
            chosen = json.loads(args.parameters.read_text())
            if report['config'].get('split_sha256') and chosen['split_sha256'] != report['config']['split_sha256']:
                raise SystemExit('Parameters and cache use different split manifests')
            if chosen['retrieval_source_sha256'] != report['config']['retrieval_source_sha256']:
                raise SystemExit('Candidate pipeline changed; capture a consistent cache before scoring')
            params = chosen['parameters']
            scored = replay(rows, RankConfig(**params))
            out = {'parameters': params, 'config': report['config'],
                   'cache_sha256': hashlib.sha256(args.cache.read_bytes()).hexdigest(),
                   'parameters_sha256': hashlib.sha256(args.parameters.read_bytes()).hexdigest(),
                   'n': len(scored), 'metrics': aggregate(scored), 'per_query': scored,
                   'by_language': {l: {'n': len(group), **aggregate(group)} for l in ('de', 'fr', 'it')
                                   if (group := [r for r in scored if r['language'] == l])}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k not in ('trials', 'per_query', 'groups')}, indent=2))


if __name__ == '__main__':
    main()
