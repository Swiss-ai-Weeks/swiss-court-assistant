"""Offline LLM re-ranking experiment over cached hybrid candidates (no app changes, no gold in prompts).

Takes a --capture-candidates cache, rebuilds the frozen hybrid ranking, and asks the local LLM NIM
to judge the top-N decisions for each query. Two modes:
  pointwise: one yes/no call per (query, decision); score = logP(yes) - logP(no)
  listwise:  one call per query listing N decision cards; the model returns an ordering

LLM outputs are cached on disk, so tuning the fusion weight replays without model calls.

uv run python agent-eval/llm_rerank_experiment.py collect --cache data/eval/opencaselaw/hybrid/dev-v3.json \
    --mode pointwise --llm-cache data/eval/opencaselaw/hybrid/llm-pointwise.json
uv run python agent-eval/llm_rerank_experiment.py tune --cache ... --mode pointwise --llm-cache ...
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path

import httpx

from opencaselaw import aggregate, metrics
from search_experiments import cached_rows
from swiss_court_assistant.server.decisions import SqliteDecisionStore
from swiss_court_assistant.server.search_ranking import RankConfig, select_decisions

LLM = 'http://localhost:9100/v1/chat/completions'
MODEL = 'nvidia/nemotron-3.5-lightning'
PARAMS = Path('data/eval/opencaselaw/hybrid/parameters.json')

POINTWISE = """You are a Swiss legal research assistant judging search results.
Search query: {query}

Court decision:
{card}

Is this decision one of the most relevant court decisions for this search query, i.e. would a Swiss lawyer typing the query want to find it among the first results? Leading cases (BGE) that establish the rule the query is about count as highly relevant. Answer only Yes or No."""

LISTWISE = """You are a Swiss legal research assistant ranking search results.
Search query: {query}

Candidate court decisions:
{cards}

Rank the candidates by how relevant they are to the search query: which decisions would a Swiss lawyer typing this query most want to find? Prefer decisions that squarely address the query's legal issue; leading cases (BGE) establishing the relevant rule are especially valuable. Answer with the candidate numbers of the 10 most relevant decisions, most relevant first, separated by commas (e.g. 4,1,12,...). Numbers only."""


def card(store, con, cand, max_chars=700):
    s = store.summary(cand['identity'])
    head = ' | '.join(x for x in (s.docket, s.court_label, s.date, s.language) if x) if s else cand['identity']
    parts = [head]
    if s and s.title:
        parts.append('Title: ' + s.title[:200])
    body = (s.regeste or '').strip() if s else ''
    if body:
        parts.append('Headnote: ' + ' '.join(body.split())[:max_chars])
    else:
        row = con.execute('SELECT text FROM chunks WHERE chunk_id = ?', [cand['chunk_id']]).fetchone()
        if row:
            parts.append('Passage: ' + ' '.join(row[0].split())[:max_chars])
    return '\n'.join(parts)


def top_candidates(rows, n, config):
    """(row, top-n Passages in frozen hybrid order, identity map) per cached query with a reranked pool."""
    out = []
    for row, hits, ids, lang in rows:
        if not hits:
            continue
        key = lambda did: ids[did]
        out.append((row, select_decisions(hits, n, lang, config, key), ids))
    return out


async def collect(args):
    report = json.loads(args.cache.read_text())
    config = RankConfig(**json.loads(PARAMS.read_text())['parameters'])
    rows = cached_rows(report)
    raw = {c['decision_id']: c for r in report['per_query'] if r['status'] == 'ok'
           for c in (r['search'].get('diagnostics') or {}).get('candidates', [])}
    store = SqliteDecisionStore(args.db)
    con = sqlite3.connect(f'file:{args.db}?mode=ro', uri=True)
    cache = json.loads(args.llm_cache.read_text()) if args.llm_cache.exists() else {}
    sem = asyncio.Semaphore(args.concurrency)

    async def ask(client, prompt, max_tokens, logprobs):
        body = {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
                'max_tokens': max_tokens, 'chat_template_kwargs': {'enable_thinking': False}}
        if logprobs:
            body.update(logprobs=True, top_logprobs=10)
        async with sem:
            for attempt in range(3):
                try:
                    r = await client.post(LLM, json=body)
                    r.raise_for_status()
                    return r.json()['choices'][0]
                except httpx.HTTPError:
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2)

    async def pointwise(client, query, cand):
        k = hashlib.sha256(f'p|{query}|{cand["identity"]}'.encode()).hexdigest()
        if k in cache:
            return
        choice = await ask(client, POINTWISE.format(query=query, card=card(store, con, cand)), 1, True)
        top = choice['logprobs']['content'][0]['top_logprobs']
        yes = max((t['logprob'] for t in top if t['token'].strip().lower() == 'yes'), default=-30.0)
        no = max((t['logprob'] for t in top if t['token'].strip().lower() == 'no'), default=-30.0)
        cache[k] = yes - no

    async def listwise(client, query, cands):
        k = hashlib.sha256(('l|' + query + '|' + '|'.join(c['identity'] for c in cands)).encode()).hexdigest()
        if k in cache:
            return
        cards = '\n\n'.join(f'[{i + 1}] ' + card(store, con, c, 450) for i, c in enumerate(cands))
        choice = await ask(client, LISTWISE.format(query=query, cards=cards), 80, False)
        order = []
        for x in re.findall(r'\d+', choice['message']['content'] or ''):
            i = int(x) - 1
            if 0 <= i < len(cands) and cands[i]['identity'] not in order:
                order.append(cands[i]['identity'])
        cache[k] = order

    tops = top_candidates(rows, args.n, config)
    async with httpx.AsyncClient(timeout=120) as client:
        tasks = []
        for row, top, ids in tops:
            cands = [dict(raw[h.decision_id], identity=ids[h.decision_id]) for h in top]
            if args.mode == 'pointwise':
                tasks += [pointwise(client, row['query'], c) for c in cands]
            else:
                tasks.append(listwise(client, row['query'], cands))
        for i in range(0, len(tasks), 64):
            await asyncio.gather(*tasks[i:i + 64])
            args.llm_cache.write_text(json.dumps(cache, ensure_ascii=False))
            print(f'{min(i + 64, len(tasks))}/{len(tasks)}', flush=True)


def rescore(rows, cache, mode, n, weight, config, decay=15):
    scored = []
    for row, top, ids in top_candidates(rows, n, config):
        base = {ids[h.decision_id]: h.score for h in top}
        if mode == 'pointwise':
            llm = {}
            for did in base:
                k = hashlib.sha256(f'p|{row["query"]}|{did}'.encode()).hexdigest()
                llm[did] = cache.get(k, 0.0)
            final = {d: base[d] + weight * llm[d] for d in base}
        else:
            k = hashlib.sha256(('l|' + row['query'] + '|' + '|'.join(ids[h.decision_id] for h in top)).encode()).hexdigest()
            order = cache.get(k, [])
            final = {d: base[d] + weight * max(0.0, 1 - order.index(d) / decay) if d in order else base[d]
                     for d in base}
        ranked = sorted(final, key=lambda d: -final[d])
        scored.append({'id': row['id'], 'language': row['language'], **metrics(ranked, row['available_gold'])})
    return scored


def evaluate(args):
    report = json.loads(args.cache.read_text())
    config = RankConfig(**json.loads(PARAMS.read_text())['parameters'])
    rows = cached_rows(report)
    cache = json.loads(args.llm_cache.read_text())
    # Exact lookups (no reranked pool) keep their fixed ranking; include them so totals match the report.
    fixed = [dict(id=r['id'], language=r['language'], **metrics(r['ranked_ids'], r['available_gold']))
             for r, hits, _, _ in rows if not hits]
    weights = [0, .25, .5, 1, 1.5, 2, 3, 4, 6, 8, 12, 1000] if args.weights is None else args.weights
    results = []
    for w in weights:
        m = aggregate(rescore(rows, cache, args.mode, args.n, w, config) + fixed)
        results.append({'weight': w, **m})
        print(f"w={w:<6} hit@1={m['hit@1']:.3f} hit@10={m['hit@10']:.3f} mrr={m['mrr@10']:.3f} "
              f"recall@10={m['recall@10']:.3f} ndcg={m['ndcg@10']:.3f} n={len(rows)}")
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=['collect', 'tune'])
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--llm-cache', type=Path, required=True)
    p.add_argument('--mode', choices=['pointwise', 'listwise'], required=True)
    p.add_argument('--n', type=int, default=20)
    p.add_argument('--db', type=Path, default=Path('data/vectordb/corpus.sqlite'))
    p.add_argument('--concurrency', type=int, default=16)
    p.add_argument('--weights', type=float, nargs='*')
    args = p.parse_args()
    if args.command == 'collect':
        asyncio.run(collect(args))
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
