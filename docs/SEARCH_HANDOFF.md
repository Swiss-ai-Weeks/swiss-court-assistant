# Retrieval optimization handoff — paused 2026-09-22

User requested a pause/summary because assistant credit is running low. Background hybrid cross-lingual evaluation and its supervising shell were stopped deliberately. The app on port 8090 was NOT stopped. No more experiments should run without resuming this work.

## Current results

Main OpenCaseLaw benchmark, pinned commit `8523191c11fc71baeca49feaf8c74e554b513cf2` in `/tmp/opencaselaw-benchmark`:

| Metric | Live legacy control | Frozen hybrid replay |
|---|---:|---:|
| hit@1 across all 100 queries (unavailable counted as miss) | 32% | 40% |
| hit@1 among 99 scored | 32.3% | 40.4% |
| hit@10 among 99 scored | 61.6% | 62.6% |
| recall@10 | 35.6% | 38.2% |
| MRR@10 | .421 | .483 |
| standard nDCG@10 | .330 | .366 |
| OpenCaseLaw-compatible nDCG@10 | .418 | .446 |

99 scored / 1 skipped / 0 errors; unchanged gold coverage 217/246. q009 is unavailable. OpenCaseLaw published plain search: 33/100 hit@1, 49.6% recall@10, .470 MRR, .525 upstream-compatible nDCG. Their reranked MRR .65 is another configuration; corresponding hit@1 not obtained. Do not claim we surpassed all their metrics.

Frozen related-judgment split: 70 development / 30 validation; 69 development queries scored. Shared canonical gold judgments/docket aliases cannot cross the split. This is not a completely subject-disjoint split, and this public benchmark was inspected earlier; it is a coefficient-fitting holdout, not an unseen private test.

Validation (30 queries):
- hit@1: 10/30 (33.3%) -> 14/30 (46.7%); five wins, one loss.
- hit@10: 26/30 -> 25/30 (one regression).
- recall@10: 47.0% -> 51.7%.
- MRR: .510 -> .610.

Main benchmark includes development queries; do not call its full score independent validation. Italian keyword subset remains weak: hit@1 stays 1/7, hit@10 drops 3/7 -> 2/7. German hit@1 rises 35.5% -> 43.4%; French 25% -> 37.5%.

Full 150-query LEGACY cross-lingual benchmark is complete: hit@1 66.7%, hit@10 82.7%, MRR .722, no skips/errors. OpenCaseLaw hit@10: 83.3%.

Hybrid cross-lingual capture is PAUSED at 82/150, zero errors at checkpoint. Its raw provisional-weight metrics are NOT the frozen-parameter result. Finish and replay before interpreting the score.

All of the above are SEARCH metrics. Original answer-citation benchmark remains 7.1% first-citation hits (6/85), compared cautiously with OpenCaseLaw plain-search 33%. Full answer-citation benchmark has not been rerun; do not advertise 40% as chat citation performance or legal accuracy.

## Deployment state — important

- App still defaults to `SCA_SEARCH_STRATEGY=legacy`.
- `GET /api/search?strategy=hybrid&debug=true` captures the experimental candidates, but currently uses PROVISIONAL `RankConfig()` defaults, not the calibrated weights!
- The 40% result is an offline replay of live cached model scores using shared serving/offline ranking code and frozen coefficients. It is not yet a live-default result.
- Frozen coefficients in `data/eval/opencaselaw/hybrid/parameters.json`:
  `citations=.5, bge=1, bger=0, authority_cap=5, native_language=.25, bge_margin=2, headnote_weight=1`.
- Do not edit retrieval source files before completing/resuming the pending cache: evaluator resume and replay check source fingerprints.
- Source changes are uncommitted. Preserve earlier work and unrelated `.auto/` material. Evaluation data are gitignored.

## Implemented

Earlier work completed BGE backfill, embeddings, vector/facet export, graph rebuild, and app reload. Added 23,697 decisions / 241,329 passages; current index has 732,931 physical decision rows and 12,588,877 vectors. No backfill job remains pending.

Existing fixes: canonical BGE/docket identities, safe own-heading aliases, twin suppression and graph deduplication, exact-reference abstention, language safeguards, search API, reproducible OpenCaseLaw scorer.

New hybrid work:
- `server/search_ranking.py`: document-level reciprocal-rank fusion, bounded authority, native-language relevance anchors, near-relevance BGE reservation, shared replay/serving functions.
- `fts.py`: escaped broad-token AND then OR BM25 retrieval with four-second deadline; separate exact keyword tool unchanged.
- `server/corpus.py`: dense + BGE-only + BGE-Regeste-only + BM25 pools; source floors retain original default 40-candidate pools; up to 160 judgments per language. Uses existing vectors, no rebuild.
- Reranker input uses preferred canonical judgment metadata/Regeste plus two genuine matching passages, at most 6,500 characters. Actual quoted passages retain original IDs/text/offsets; composite text is never exposed as fabricated evidence.
- Authority belongs to canonical published judgment, even if best evidence comes from BGer export. Counts use preferred record, not duplicate sums.
- Passage/headnote blending tested; selected weight is 1 (document relevance). Best passage still chosen separately for citation evidence.
- `server/decisions.py`: summary reads avoid loading full judgment text unnecessarily.
- Agent passes turn language into hybrid retrieval.
- `agent-eval/search_experiments.py`: frozen alias-aware grouping, development-only tuning, cached replay with fingerprint checks.
- `agent-eval/compare_search.py`: matched-population/gold-coverage comparison, duplicate/mixed-parameter guards.
- `agent-eval/fixtures/opencaselaw-split.json`: committed-intent frozen split.
- 25 unit tests pass; compile checks and git diff whitespace checks passed. New comparison helper has been executed successfully, but has no dedicated unit test yet.

Three development variants: initial hybrid 60-doc pool calibrated hit@1 30.4%; widened 120-doc pool/blending 34.8%; canonical authority plus BGE-headnote pool 37.7%. Legacy development control 31.9%. Final 432-configuration sweep used development only; coefficients frozen before validation/cross-lingual. No gold IDs in serving logic.

## Additional issues discovered, NOT fixed

Five legacy live-answer regression checks completed under `data/eval/opencaselaw/hybrid/chat-control/latest/`:
- No stream or tool errors; German Hundebiss/Art.41/Notwehr and Italian deadline question used intended languages.
- Bare missing reference `BGE 999 I 999` correctly abstained with no citations, BUT answered Italian: `Nessuna decisione corrispondente trovata nel corpus.` Expected German. Likely `language.py` treats Roman numeral `I` as Italian function word; investigate exact-reference detection before stopword scoring. This is not confirmed/fixed yet.
- Hundebiss had one unverified quotation among two sources; inspect transcript before claiming all grounding checks pass.
- `--no-judge` runner prints FAIL/rubric 0% for unjudged cases; distinguish absent legal judgment from mechanical language/citation checks.
- These chat checks used LEGACY, not the calibrated hybrid. Italian answer took ~131 seconds; performance deserves attention.
- Hybrid serial development candidate capture averaged roughly six seconds/search (precise latency can be calculated from raw reports). Agent's three distinct translated queries may cost more. Both document and passage reranking currently run over all candidates even though selected headnote weight is 1. Optimize only with equivalence tests and live benchmark verification, not silently.
- Cached short-query LLM expansion was NOT implemented.

## Key artifacts

`docs/OPENCASELAW.md` records completed backfill and main hybrid results/protocol.

Under `data/eval/opencaselaw/`:
- `post-backfill-current.json`, `post-backfill-baseline.json`, `post-backfill-cross-lingual.json` — completed live legacy/control evaluations.
- `hybrid/dev.json`, `hybrid/pilot-parameters.json` — first variant.
- `hybrid/dev-v2.json`, `hybrid/v2-parameters.json` — second variant.
- `hybrid/dev-v3.json`, `hybrid/parameters.json` — final development capture and frozen coefficients.
- `hybrid/dev-scored.json`, `hybrid/validation-raw.json`, `hybrid/validation-scored.json`.
- `hybrid/main-comparison.json`, `hybrid/validation-comparison.json` — headline comparisons.
- `hybrid/cross-lingual-raw.json` — paused checkpoint, 82/150.
- `hybrid/cross-lingual-scored.json` — NOT yet produced.
- `hybrid/chat-control/latest/results.json` and `transcripts/*.json` — legacy chat checks.

Log of paused evaluation: `/tmp/sca-hybrid-validation.log`.

## Resume commands (before editing retrieval source)

```bash
uv run python agent-eval/opencaselaw.py --repo /tmp/opencaselaw-benchmark \
  --strategy hybrid --capture-candidates --resume \
  --golden benchmarks/swiss_legal_rag_bench/cross_lingual_v1.jsonl \
  --output data/eval/opencaselaw/hybrid/cross-lingual-raw.json

uv run python agent-eval/search_experiments.py score \
  --cache data/eval/opencaselaw/hybrid/cross-lingual-raw.json \
  --parameters data/eval/opencaselaw/hybrid/parameters.json \
  --output data/eval/opencaselaw/hybrid/cross-lingual-scored.json

uv run python agent-eval/compare_search.py \
  --control data/eval/opencaselaw/post-backfill-cross-lingual.json \
  --reports data/eval/opencaselaw/hybrid/cross-lingual-scored.json \
  --output data/eval/opencaselaw/hybrid/cross-lingual-comparison.json
```

Then decide whether cross-lingual quality/Italian trade-offs justify promotion. If promoting, explicitly install frozen weights, verify live API matches cached rankings, enable hybrid default for agent, and rerun answer language/grounding regressions. Fix missing-exact-reference language bug with tests after finishing the fingerprint-sensitive evaluation. Do not retune on validation/cross-lingual and continue calling them untouched holdouts.
