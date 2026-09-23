# OpenCaseLaw evaluation

## Two different measurements

The September 21 chat run scored **answer citations**, not search results: 6/85
queries had a gold first citation (**hit@1 7.1%**). OpenCaseLaw's published plain
search has **hit@1 33%**, over 100 queries. This is the preferred comparison for
that run, with the different output types and populations disclosed. It is not
a measure of whether the explanations of Art. 56 OR, Art. 337 OR, etc. are legally
correct. MRR over 1–5 citations is not equivalent to MRR over ten search results.

`GET /api/search?q=...&k=10` now provides a separate, search-only measurement.
It returns distinct ranked decision IDs, court, language, citation count, raw
reranker logit, final score, and elapsed time. It uses the same multilingual
retrieval/reranking implementation as the agent, but sends the original query
through the multilingual encoder rather than asking an LLM to translate or expand
it. `language=it` restricts searches to Italian for diagnostics; by default all
three corpus languages are searched. Bare exact references return matches or an
empty list, never unrelated semantic substitutes.

`baseline=true` disables the BGE candidate expansion and authority priors. It is a
controlled relevance-only search ablation, **not a replay of the old chat agent**.
Both modes share exact lookup, canonical identity resolution, and decision-level
output. Three-language reservations are selected before globally sorting results;
language order must not make German results appear first on Italian queries.

## Reproduce

```bash
git clone https://github.com/jonashertner/opencaselaw.git /tmp/opencaselaw-benchmark
git -C /tmp/opencaselaw-benchmark checkout 8523191c11fc71baeca49feaf8c74e554b513cf2
uv run python -m unittest discover -s tests -v

# Answer-language/exact-lookup regressions (separate from retrieval scoring):
uv run python agent-eval/run.py --cases agent-eval/cases/search-regressions.yaml --no-judge

# With the app running locally on 8090:
uv run python agent-eval/opencaselaw.py --repo /tmp/opencaselaw-benchmark \
  --baseline --output data/eval/opencaselaw/baseline.json
uv run python agent-eval/opencaselaw.py --repo /tmp/opencaselaw-benchmark \
  --output data/eval/opencaselaw/current.json
uv run python agent-eval/opencaselaw.py --repo /tmp/opencaselaw-benchmark \
  --golden benchmarks/swiss_legal_rag_bench/cross_lingual_v1.jsonl \
  --output data/eval/opencaselaw/cross-lingual-current.json
```

Reports are checkpointed after each query. Add `--resume` after a network/session
failure; failed requests are retried, not silently scored as misses. Use `--url`
for a different app and `--db` for its matching local corpus index. Do not use a
local coverage index from another deployment. Each report contains benchmark
commit/hash, raw ranked results, available/missing gold IDs, alias resolution,
skipped queries, and errors. No benchmark relevance data enters the server.

Queries with no available gold decisions are skipped explicitly. Available-gold
recall/nDCG follow the upstream benchmark convention; per-query
`including_missing_gold` retains the full gold denominator too. The report keeps
standard rank-aware `ndcg@10` separate from `opencaselaw_ndcg@10`: at the pinned
commit, the upstream script removes non-relevant ranks before computing DCG.
Only the latter is numerically compatible with their published nDCG.

## Ranking and identity

Default priors on the reranker logit scale:

```
score = logit + 0.5 * log1p(cited_by) + 1.0 * is_bge + 0.5 * is_bger
```

Tune at startup with `SCA_AUTHORITY_CITATIONS`, `SCA_AUTHORITY_BGE`, and
`SCA_AUTHORITY_BGER`. These are initial coefficients, not a claim of an optimum.
Priors are not applied to cosine scores when reranking is unavailable. Extra BGE
KNN candidates respect all explicit filters. The agent gets up to 12 distinct
decisions, with a BGE reserved per searched language when available.

BGE identities are normalised centrally (`bge_122 V 157` →
`bge_BGE_122_V_157`, preserving suffixes such as `Ib`). Docket aliases are taken
only from the judgment's own heading, **not from citations in its reasoning**.
Old index rows remain readable through canonical aliases. New ingestion preserves
the original ID for joining paragraph annotations, then writes canonical decision
and chunk IDs. Citation graph builds canonicalise edges, remove twin self-edges,
and count distinct neighbours. The runtime reader supports old graph twins too,
without double-counting a judgment that cites both exports.

Sampling includes all usable published BGE, even before a `--since` cutoff. This
changes corpus composition intentionally; the requested fraction is no longer an
exact total-size constraint. Existing indexes require backfilling/rebuilding and
embedding missing BGE; changing Python sampling code alone does not add vectors.

```bash
# Resumable append-only backfill; preserves existing judgments and sampling of other courts.
# Includes embedding, matrix/facet export, and an atomic citation-graph rebuild.
uv run python -m swiss_court_assistant.index backfill-bge --batch 8192
# Restart/reload the app AFTER it completes, then rerun the benchmark.
```

Plan a maintenance window: until embedding/export finishes, a restarted app may
fall back to sqlite-vec and lack matrix-based filters. At ~150 passages/s the
241k-passage backfill takes tens of minutes, not seconds. `--skip-embed` deliberately
leaves pending vectors; rerunning without it drains the queue. Historical court
code `bge_historical` is normalised to `bge`. Full `build --courts bge,bge_historical`
also replaces physical legacy twin rows; append-only backfill preserves them and
the runtime suppresses discarded twins and deduplicates docket/BGE equivalents.

## First measured search run (existing 709,234-decision index)

100 queries completed in each mode; **95 scored, 5 skipped, 0 errors**. Canonical
identity and heading aliases resolved 174 of 246 distinct gold IDs (72 missing).
This coverage differs from the original API-ID-only check because aliases now
resolve additional references; no decisions were added for this measurement.

| Search mode | hit@1 | hit@10 | recall@10 | MRR@10 | standard nDCG@10 | upstream-compatible nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Relevance-only ablation | 15.8% | 35.8% | 22.7% | .210 | .174 | .240 |
| BGE candidates + authority | **29.5%** | **57.9%** | **36.6%** | **.393** | **.332** | **.429** |

OpenCaseLaw plain search reports hit@1 33% and hit@10-sized-list MRR .470, but on
all 100 queries and a larger corpus; it is not an identical scored population.
Italian remained 3/7 hit@10 in both search modes. Inspection confirmed that Italian
queries do retrieve Italian/Ticino decisions, while authority can crowd them out.
The benchmark gap is reduced, not eliminated. These are the first run, before the
backfill and final compatibility guards; use the post-backfill reports for the
final deployed configuration.

### Cross-lingual and live-answer checks

The initial cross-lingual run processed all 150 queries: **129 scored, 21 skipped,
0 errors** (43 of 50 target judgments available). It reached **hit@1 67.4%, hit@10
83.7%, MRR@10 .728**. Italian was 34/43 hit@10. OpenCaseLaw's published .630 MRR /
.833 hit@10 covers a different corpus/population; the skips preclude a superiority
claim. Raw report: `data/eval/opencaselaw/cross-lingual-current.json`.

Five live chat smoke checks had no stream errors and all passed the language
check. `Hundebiss` cited BGE 131 III 115 first; the Art. 41 OR and Notwehr queries
answered in German; `ricorso tardivo restituzione termine` answered in Italian.
The then-absent BGE 115 IV 162 returned a German no-match message with no sources.
That BGE is subsequently included in the backfill; the durable missing-reference
regression uses the non-existent `BGE 999 I 999` instead. Raw smoke report:
`data/eval/opencaselaw/chat-smoke.json`.

### Completed backfill and measured search results

Embedding, vector/facet export, graph rebuild, reload, and all three evaluations
**completed**. The append phase added **23,697 decisions / 241,329 passages**.
The corpus now has **732,931 physical decision rows / 12,588,877 vectors** and
**217/246** distinct gold IDs. q009, “Auslieferung an Rumänien”, is the sole query
with no available gold judgment. Main runs: **99 scored / 1 skipped / 0 errors**.

| Post-backfill search mode | hit@1 | hit@10 | recall@10 | MRR@10 | standard nDCG@10 | upstream-compatible nDCG@10 |
|---|---:|---:|---:|---:|---:|---:|
| Relevance-only ablation | 19.2% | 36.4% | 22.1% | .237 | .190 | .248 |
| BGE candidates + authority (`legacy`) | **32.3%** | **61.6%** | **35.6%** | **.421** | **.330** | **.418** |

Including the unavailable query as a miss gives **32/100 hit@1**, versus the
published OpenCaseLaw plain-search **33/100**. Their recall@10 is 49.6%, MRR .470,
and upstream-compatible nDCG .525; corpus and gold-coverage differences remain.
The published reranked MRR .65 is a different configuration, not its hit@1.

The completed cross-lingual run scores **all 150 queries, no skips or errors**:
**66.7% hit@1, 82.7% hit@10, .722 MRR@10**. OpenCaseLaw reports 83.3% hit@10.
Do not substitute the earlier 129-query coverage-filtered result for this run.

Reports:
- `data/eval/opencaselaw/post-backfill-baseline.json`
- `data/eval/opencaselaw/post-backfill-current.json`
- `data/eval/opencaselaw/post-backfill-cross-lingual.json`

These are search measurements. The full answer-citation benchmark has **not**
been rerun, so the original 7.1% answer-citation hit@1 remains the measured chat
result; neither search gains nor a handful of successful chat examples replace it.

## Hybrid / headnote experiment

`strategy=hybrid` combines existing dense, BGE-only, BGE-headnote-only, and FTS5
BM25 candidates. Document-level reciprocal-rank fusion deduplicates canonical
judgments. A floor of 40 judgments per source and a 160-judgment per-language
budget retain the original dense/BGE pools instead of sacrificing their recall.
Headnote masks use existing vectors; no embedding or index rebuild is required.
The broader FTS query uses escaped tokens, AND then OR, and a four-second deadline;
the agent's separate exact-phrase keyword tool is unchanged.

Nemotron reranks each judgment using its preferred canonical record's Regeste,
metadata, and two matching passages (bounded to 6,500 characters). The best actual
passage is selected separately for quoting: concatenated reranker input is never
returned as a fabricated citable passage. Authority comes from the canonical
published judgment, not whichever BGer/BGE export supplied that passage. Citation
counts use the preferred record, never a sum of potentially overlapping twins.

### Calibration protocol

`agent-eval/fixtures/opencaselaw-split.json` freezes **70 development / 30 validation
queries**. Connected components sharing a gold judgment, including resolved docket
aliases, cannot cross the split. This is related-judgment grouping, not a guarantee
that every legal subject is disjoint. The public benchmark had already been
inspected earlier: validation is held out from coefficient fitting, **not an unseen
private test set**. q009 is in development, leaving 69 scored development queries.

Three development variants were tried before running validation: an initial
60-document hybrid pool, a wider 120-document pool with passage/headnote blending,
and the final 160-document pool with direct headnote retrieval and canonical
judgment authority. Their calibrated development hit@1 was 30.4%, 34.8%, and 37.7%,
respectively, versus the frozen legacy control's 31.9%.

The final 432-configuration development sweep optimized hit@1, then standard
nDCG, subject to not reducing development hit@10 or recall@10. It chose:

```
score = document_logit + min(5, 0.5 * log1p(cited_by) + 1.0 * is_bge)
        + 0.25 * matches_query_language
```

The extra BGer bonus is zero. Passage/headnote blends were tested, but the selected
headnote weight is 1: the passage score chooses evidence, not the final judgment
rank. Selection also reserves language relevance anchors and a BGE only when its
relevance is within two logits of that language's best result. Authority is bounded,
not merely a tie-breaker. No benchmark judgment IDs enter serving logic.

Coefficients were frozen **before** the validation and cross-lingual runs. Raw live
candidate pools/logits are cached; `search_experiments.py score` replays the same
shared ranking functions with those frozen coefficients. Raw cache report metrics
use the provisional serving weights and are **not** the calibrated results below.

### Main benchmark results (frozen-parameter replay)

| Population / metric | Legacy control | Hybrid |
|---|---:|---:|
| Validation hit@1, 30 queries | 33.3% (10/30) | **46.7% (14/30)** |
| Validation hit@10 | 86.7% (26/30) | 83.3% (25/30) |
| Validation recall@10 | 47.0% | **51.7%** |
| Validation MRR@10 | .510 | **.610** |
| Full benchmark hit@1, unavailable query counted as miss | 32/100 | **40/100** |
| Full benchmark hit@1, 99 scored | 32.3% | **40.4%** |
| Full benchmark hit@10, 99 scored | 61.6% | **62.6%** |
| Full benchmark recall@10, 99 scored | 35.6% | **38.2%** |
| Full benchmark MRR@10, 99 scored | .421 | **.483** |
| Full benchmark standard nDCG@10 | .330 | **.366** |
| Full benchmark upstream-compatible nDCG@10 | .418 | **.446** |

The full benchmark contains the tuning queries and is not independent validation.
Validation has five first-place wins and one loss; 30 queries is too small for a
strong generalization claim. There are 99 scored queries, one unavailable query,
and no evaluation errors in the main candidate captures. Coverage is unchanged.

This clears the 40%-hit@1 target, but not the 50%-recall target. OpenCaseLaw plain
search reports 33/100 hit@1 and 49.6% recall@10. We have not surpassed all its metrics,
nor compared against its reranked configuration's hit@1. A gold-list miss is also
not automatically a substantively incorrect legal result.

**Known regression:** the seven Italian keyword queries retain 1/7 hit@1, but
hit@10 falls from 3/7 to 2/7. Native relevance anchors do not solve every cantonal
case-selection failure. German hit@1 rises from 35.5% to 43.4%; French from 25% to
37.5%. Language correctness of generated prose is a separate check.

Reports are under `data/eval/opencaselaw/hybrid/`: `parameters.json`,
`dev-v3.json`, `dev-scored.json`, `validation-raw.json`, `validation-scored.json`,
`main-comparison.json`, and `validation-comparison.json`.

### Reproduce candidate capture and held-out scoring

```bash
# The committed split is already frozen; do not reshuffle it after inspecting results.
uv run python agent-eval/opencaselaw.py --repo /tmp/opencaselaw-benchmark \
  --strategy hybrid --capture-candidates \
  --split agent-eval/fixtures/opencaselaw-split.json --partition dev \
  --output data/eval/opencaselaw/hybrid/dev-v3.json
uv run python agent-eval/search_experiments.py tune \
  --cache data/eval/opencaselaw/hybrid/dev-v3.json \
  --control data/eval/opencaselaw/post-backfill-current.json \
  --output data/eval/opencaselaw/hybrid/parameters.json

uv run python agent-eval/opencaselaw.py --repo /tmp/opencaselaw-benchmark \
  --strategy hybrid --capture-candidates \
  --split agent-eval/fixtures/opencaselaw-split.json --partition validation \
  --output data/eval/opencaselaw/hybrid/validation-raw.json
uv run python agent-eval/search_experiments.py score \
  --cache data/eval/opencaselaw/hybrid/validation-raw.json \
  --parameters data/eval/opencaselaw/hybrid/parameters.json \
  --output data/eval/opencaselaw/hybrid/validation-scored.json
uv run python agent-eval/compare_search.py \
  --control data/eval/opencaselaw/post-backfill-current.json \
  --reports data/eval/opencaselaw/hybrid/validation-scored.json --subset \
  --output data/eval/opencaselaw/hybrid/validation-comparison.json
```

Use the same `score` command on the development cache, then pass both scored files
to `compare_search.py` without `--subset` for the full 99-query comparison. The
comparison refuses mismatched benchmark hashes, gold coverage, duplicate queries,
or mixed parameter sets. Candidate capture fails explicitly on reranker outages;
a fallback ranking must not silently become a calibration result.

## Learned ranker + statute-graph pool (2026-09-22)

What OpenCaseLaw's [methodology](https://opencaselaw.ch/methodology.html) does that we did not:
a statute-graph candidate pool, cross-result citation evidence, LLM query analysis and a Haiku
re-rank of the top 15 (their MRR .470 → .647). Their gold labels come from the citation graph
(`generate_golden_queries.py`: top-cited decisions per statute article); 188/290 gold are BGE.

Measured offline on the frozen dev capture before touching serving code:

* **LLM re-ranking with the local Nemotron (3B active)** does not transfer: pointwise yes/no
  alone gives dev hit@1 17%, listwise 22%; blended with the hybrid score, no gain
  (`agent-eval/llm_rerank_experiment.py`). The model's notion of "relevant" is not the
  citation-graph gold.
* **Signals that separate gold**: citation count (gold mean log1p 4.7 vs 1.7), fusion score,
  native language, which pools found it, and whether the decision cites the article the query
  names. A linear combination fitted on dev beats the hand formula; cross-result citation
  evidence added nothing on top.
* **Statute pool**: `data/raw/graph/statute_references.parquet` → `data/graph/corpus.statutes.sqlite`
  (`python -m swiss_court_assistant.statute_links build`, 45 s). For an article named in the
  query, the 40 most-cited decisions per language citing it join the RRF pools ("Art. 41 OR":
  the three q017 gold judgments are among the top 10 German ones; the old pools had none).
* **LLM query translation** (`translate=true`, off by default): hurts. Main dev CV hit@1
  .449 → .435, recall@10 .418 → .369; cross-lingual hit@1 82% → 60%. Untreated, the model also
  invented statutes ("Raub Gewalt Drohung OR § 143 ZGB"); `query_translation.sanitize` strips them.

Serving: `search_ranking.candidate_features` computes 14 features, written to the diagnostics;
`LinearRanker` (weights in `server/ranker.json`, fitted by `agent-eval/fit_ranker.py fit` on the
dev partition only, l2 chosen by grouped CV inside dev) orders the judgments. `SCA_RANKER=off`
restores the hand formula; so does a reranker outage.

| Population / metric | Hand weights (same pools) | **Learned ranker** | OpenCaseLaw plain | OpenCaseLaw + Haiku |
|---|---:|---:|---:|---:|
| Validation (30, scored once) hit@1 | 46.7% | **53.3%** | – | – |
| Validation hit@10 | 83.3% | **90.0%** | – | – |
| Validation MRR@10 | .610 | **.652** | – | – |
| Validation recall@10 | 51.7% | **59.0%** | – | – |
| Full 100: hit@1 (unavailable = miss) | – | **46/100** | 33/100 | ≈57/100 |
| Full 99 scored: MRR@10 | – | .557 | .470 | ≈.647 |
| Full 99: recall@10 | – | 49.5% | 49.6% | – |
| Full 99: upstream-compatible nDCG@10 | – | **.568** | .525 | – |
| Cross-lingual 150: hit@1 | **82.0%** | 76.0% | – | – |
| Cross-lingual hit@10 | 89.3% | **96.0%** | 83.3% | – |
| Cross-lingual MRR@10 | **.852** | .836 | – | – |

Caveats: the full-100 rows include the 70 dev queries the ranker was fitted on (dev grouped-CV
estimate: hit@1 44.9%, MRR .512, recall@10 41.8%); only validation and cross-lingual are held
out, and this public benchmark was inspected before. OpenCaseLaw tuned on the same 100 queries.
Recall@10 is a tie, not a win. Italian stays weak (0/7 hit@1, 5/7 hit@10): its gold is mostly
Ticino cantonal decisions, which a ranker fitted on German dev queries ranks below cited BGEs.
The learned ranker trades cross-lingual hit@1 (-6 pt) for hit@10 (+6.7 pt). The agent now
uses this ranker too; answer quality with it has not been re-evaluated.

Reports: `data/eval/opencaselaw/final/` (`dev.json`, `validation.json`, `cross-lingual*.json`,
`*-scored.json`, `ranker.json`); dev ablations in `data/eval/opencaselaw/ranker/`.

```bash
O=data/eval/opencaselaw/final
S="--repo /tmp/opencaselaw-benchmark --strategy hybrid --capture-candidates --split agent-eval/fixtures/opencaselaw-split.json"
uv run python agent-eval/opencaselaw.py $S --partition dev --output $O/dev.json
uv run python agent-eval/fit_ranker.py cv --cache $O/dev.json
uv run python agent-eval/fit_ranker.py fit --cache $O/dev.json --l2 0.001 --output $O/ranker.json
uv run python agent-eval/opencaselaw.py $S --partition validation --output $O/validation.json
uv run python agent-eval/fit_ranker.py score --cache $O/validation.json --ranker $O/ranker.json --output $O/validation-scored.json
```
