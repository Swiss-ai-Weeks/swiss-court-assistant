"""Reproducible search-only OpenCaseLaw evaluation; no LLM judge or benchmark-aware retrieval.

uv run python agent-eval/opencaselaw.py --repo /tmp/opencaselaw-benchmark --output data/eval/ocl-current.json
uv run python agent-eval/opencaselaw.py --repo /tmp/opencaselaw-benchmark --baseline --output data/eval/ocl-baseline.json

Uses the local read-only decision index for corpus coverage and aliases, and /api/search for rankings.
JSON reports are checkpointed after every query; --resume skips successful rows, not errors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import time
from pathlib import Path

import httpx

from swiss_court_assistant.server.decisions import SqliteDecisionStore
from swiss_court_assistant.server.language import detect_language


def dcg(grades):
    return sum((2 ** g - 1) / math.log2(rank + 2) for rank, g in enumerate(grades))


def metrics(ranked: list[str], grades: dict[str, int], k: int = 10) -> dict:
    ranked = list(dict.fromkeys(ranked))[:k]
    found = {rid: i + 1 for i, rid in enumerate(ranked) if rid in grades}
    ideal = dcg(sorted(grades.values(), reverse=True)[:k])
    return {"hit@1": float(bool(ranked) and ranked[0] in grades), "hit@10": float(bool(found)),
            "mrr@10": 1 / min(found.values()) if found else 0.0,
            "recall@10": len(found) / len(grades) if grades else 0.0,
            "ndcg@10": dcg([grades.get(r, 0) for r in ranked]) / ideal if ideal else 0.0,
            # OpenCaseLaw's pinned script removes non-relevant ranks before computing DCG.
            # Preserve this as an explicitly labelled compatibility metric, not standard nDCG.
            "opencaselaw_ndcg@10": dcg([grades[r] for r in ranked if r in grades]) / ideal if ideal else 0.0,
            "matched_ranks": found}


def aggregate(rows):
    return {key: statistics.mean(r[key] for r in rows) if rows else None
            for key in ("hit@1", "hit@10", "mrr@10", "recall@10", "ndcg@10", "opencaselaw_ndcg@10")}


def load_queries(path):
    if path.suffix == '.jsonl':
        return [{"id": q["q_id"], "query": q["q_text"], "tags": [q["q_lang"], q["difficulty"]],
                 "relevant": [{"decision_id": q["target_decision_id"], "grade": 3}]}
                for line in path.read_text().splitlines() if line.strip() for q in [json.loads(line)]]
    return json.loads(path.read_text())["queries"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--golden', default='benchmarks/search_relevance_golden.json')
    p.add_argument('--db', type=Path, default=Path('data/vectordb/corpus.sqlite'))
    p.add_argument('--url', default='http://localhost:8090')
    p.add_argument('--baseline', action='store_true')
    p.add_argument('--strategy', choices=['legacy', 'hybrid'])
    p.add_argument('--capture-candidates', action='store_true')
    p.add_argument('--split', type=Path, help='Frozen topic-grouped split manifest')
    p.add_argument('--partition', choices=['dev', 'validation'], default='dev')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--limit', type=int)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    golden = args.repo / args.golden
    queries = load_queries(golden)
    if args.split:
        split = json.loads(args.split.read_text())
        if split['golden_sha256'] != hashlib.sha256(golden.read_bytes()).hexdigest():
            raise SystemExit('Split belongs to a different benchmark file')
        queries = [q for q in queries if q['id'] in split[args.partition]]
    queries = queries[:args.limit]
    if not queries:
        raise SystemExit('No queries selected')
    store = SqliteDecisionStore(args.db)
    from swiss_court_assistant import identity
    source = Path(identity.__file__).parent
    implementation = hashlib.sha256(b''.join((source / name).read_bytes() for name in (
        'identity.py', 'fts.py', 'server/corpus.py', 'server/search_ranking.py', 'server/decisions.py',
        'server/citations.py', 'server/language.py'))).hexdigest()
    config = {"retrieval_source_sha256": implementation, "benchmark_commit": subprocess.check_output(['git', '-C', str(args.repo), 'rev-parse', 'HEAD'], text=True).strip(),
              "golden": args.golden, "golden_sha256": hashlib.sha256(golden.read_bytes()).hexdigest(),
              "baseline": args.baseline, "strategy": args.strategy, "capture_candidates": args.capture_candidates,
              "partition": args.partition if args.split else None,
              "split_sha256": hashlib.sha256(args.split.read_bytes()).hexdigest() if args.split else None,
              "db": str(args.db.resolve()), "decisions": len(store),
              "url": args.url, "k": 10}
    report = {"config": config, "per_query": []}
    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text())
        if report['config'] != config:
            raise SystemExit('Cannot resume a report from a different configuration/corpus')
    done = {r['id']: r for r in report['per_query'] if r['status'] != 'error'}

    def resolve(ref):
        found = store.find(ref)
        return found[0] if found else None

    def save():
        rows = report['per_query']
        ok = [r for r in rows if r['status'] == 'ok']
        report['summary'] = {"total": len(queries), "completed": len(rows), "evaluated": len(ok),
                             "skipped": sum(r['status'] == 'skipped_no_relevant_docs_in_db' for r in rows),
                             "errors": sum(r['status'] == 'error' for r in rows),
                             "available_gold_ids": len({d for r in rows for d in r['resolved_gold']}),
                             "missing_gold_ids": len({d for r in rows for d in r['missing_gold']}),
                             **aggregate(ok),
                             "by_language": {lang: {"n": len(group), **aggregate(group)}
                                             for lang in ('de', 'fr', 'it', 'en')
                                             if (group := [r for r in ok if r['language'] == lang])}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temp = args.output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2, ensure_ascii=False))
        temp.replace(args.output)

    with httpx.Client(timeout=180) as client:
        health = client.get(args.url + '/api/health')
        health.raise_for_status()
        if health.json().get('decisions') != len(store):
            raise SystemExit('The API and the coverage index have different decision counts; restart the app or use its matching --db')
        for q in queries:
            if q['id'] in done:
                continue
            available = {}
            aliases, missing = {}, []
            for rel in q['relevant']:
                rid = resolve(rel['decision_id'])
                if rid:
                    available[rid] = max(available.get(rid, 0), rel['grade'])
                    aliases[rel['decision_id']] = rid
                else:
                    missing.append(rel['decision_id'])
            row = {"id": q['id'], "query": q['query'], "tags": q.get('tags', []),
                   "language": next((t for t in q.get('tags', []) if t in ('de', 'fr', 'it')), detect_language(q['query'])),
                   "available_gold": available, "resolved_gold": aliases, "missing_gold": missing}
            if not available:
                row['status'] = 'skipped_no_relevant_docs_in_db'
            else:
                start = time.perf_counter()
                try:
                    params = {'q': q['query'], 'k': 10, 'baseline': str(args.baseline).lower(),
                              'debug': str(args.capture_candidates).lower()}
                    if args.strategy:
                        params['strategy'] = args.strategy
                    response = client.get(args.url + '/api/search', params=params)
                    response.raise_for_status()
                    result = response.json()
                    if args.strategy and result.get('strategy') != args.strategy:
                        raise ValueError('Server did not apply the requested search strategy')
                    diagnostics = result.get('diagnostics') or {}
                    if diagnostics.get('rerank_complete') is False:
                        raise ValueError('Reranker unavailable/incomplete; do not compare degraded candidate-cache scores')
                    ranked = list(dict.fromkeys(resolve(h['decision_id']) or h['decision_id'] for h in result['results']))
                    row.update(status='ok', ranked_ids=ranked, search=result, **metrics(ranked, available))
                    if 'candidates' in diagnostics:
                        pool_ids = {c['identity'] for c in diagnostics['candidates']}
                        row['pool_recall'] = len(pool_ids & available.keys()) / len(available)
                        row['pool_hit'] = bool(pool_ids & available.keys())
                    all_grades = dict(available)
                    all_grades.update({r['decision_id']: r['grade'] for r in q['relevant'] if r['decision_id'] in missing})
                    row['including_missing_gold'] = metrics(ranked, all_grades)
                except (httpx.HTTPError, ValueError, KeyError) as e:
                    row.update(status='error', error=str(e))
                row['seconds'] = round(time.perf_counter() - start, 3)
            report['per_query'] = [r for r in report['per_query'] if r['id'] != q['id']] + [row]
            save()
            print(q['id'], row['status'], 'hit@1=', row.get('hit@1'), 'hit@10=', row.get('hit@10'), flush=True)
    save()
    print(json.dumps(report['summary'], indent=2))
    if report['summary']['errors']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
