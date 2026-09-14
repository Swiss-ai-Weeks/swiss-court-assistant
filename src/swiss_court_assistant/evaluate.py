"""Evaluate retrieval systems on the multilingual known-item test set.

Each base retriever's top-k chunk ranking is cached under data/eval/runs/, so
fusion and rerank variants reuse it. Metrics are computed at decision level
(chunks collapsed to their decision, first occurrence wins) and at passage
level (the gold chunk itself), broken down by query language, decision
language, period, question style and same- vs cross-language.

Usage:
    uv run python -m swiss_court_assistant.evaluate data/subset/decisions_50k_seed42.chunks.parquet \
        --systems bm25 bge-m3 qwen3-4b hybrid hybrid+rerank
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from swiss_court_assistant import retrieval as R

RUNS = Path("data/eval/runs")
K = 100
BREAKDOWNS = ["query_lang", "doc_language", "period", "style", "cross_lingual"]


def base_run(name: str, chunks_path: Path, q: pl.DataFrame, device: str) -> np.ndarray:
    path = RUNS / f"{name}.npy"
    if path.exists():
        return np.load(path)
    queries, langs = q["query"].to_list(), q["query_lang"].to_list()
    if name == "bm25":
        idx, _ = R.BM25Retriever(chunks_path).search(queries, langs, K)
    else:
        idx, _ = R.DenseRetriever(chunks_path, name, device).search(queries, langs, K)
    RUNS.mkdir(parents=True, exist_ok=True)
    np.save(path, idx)
    return idx


def run_system(system: str, chunks_path: Path, q: pl.DataFrame, texts: list[str],
               device: str, dense: list[str] | None = None) -> tuple[str, np.ndarray]:
    """Return (label, ranking). "hybrid" fuses BM25 with the dense indexes in `dense`
    (default: every one built so far); the label names the components so cached
    reranks never go stale."""
    if system in ("bm25", *R.DENSE_MODELS):
        return system, base_run(system, chunks_path, q, device)
    if system.startswith("hybrid"):
        built = [m for m in R.DENSE_MODELS if (R.index_dir(chunks_path) / f"{m}.npy").exists()]
        comps = ["bm25", *(dense or built)]
        runs = {n: base_run(n, chunks_path, q, device) for n in comps}
        if "+rerank" in system:
            # "hybrid+rerank" = bge reranker, "hybrid+rerank-<key>" = R.RERANKERS[key].
            # Rerank the union of candidates, not a fused top-k (higher gold recall).
            key = system.split("+rerank-")[1] if "+rerank-" in system else "bge"
            label = f"rerank{'' if key == 'bge' else '-' + key}({'+'.join(comps)})"
            path = RUNS / f"{label}.npy"
            if not path.exists():
                pools = R.candidate_pool(runs)
                np.save(path, R.Reranker(key, device=device).rerank(q["query"].to_list(), pools, texts))
            return label, np.load(path)
        return f"hybrid({'+'.join(comps)})", R.rrf(list(runs.values()), K)
    raise ValueError(system)


def metrics(idx: np.ndarray, q: pl.DataFrame, chunk_doc: np.ndarray, chunk_ids: np.ndarray) -> pl.DataFrame:
    gold_doc, gold_chunk = q["gold_decision_id"].to_list(), q["gold_chunk_id"].to_list()
    rows = []
    for qi in range(idx.shape[0]):
        ranked = idx[qi][idx[qi] >= 0]
        docs = list(dict.fromkeys(chunk_doc[ranked]))  # dedupe, keep order
        rank = docs.index(gold_doc[qi]) + 1 if gold_doc[qi] in docs else None
        crank = next((r + 1 for r, c in enumerate(chunk_ids[ranked[:10]]) if c == gold_chunk[qi]), None)
        rows.append({
            "hit@1": float(rank == 1), "hit@10": float(rank is not None and rank <= 10),
            "hit@50": float(rank is not None and rank <= 50),
            "mrr@10": 1 / rank if rank and rank <= 10 else 0.0,
            "ndcg@10": 1 / np.log2(rank + 1) if rank and rank <= 10 else 0.0,
            "passage_hit@10": float(crank is not None),
        })
    return q.hstack(pl.DataFrame(rows))


METRIC_COLS = ["hit@1", "hit@10", "hit@50", "mrr@10", "ndcg@10", "passage_hit@10"]


def summarize(scored: pl.DataFrame) -> dict:
    agg = [pl.col(c).mean().round(3) for c in METRIC_COLS] + [pl.len().alias("n")]
    out = {"overall": scored.select(agg).to_dicts()[0]}
    for b in BREAKDOWNS:
        out[b] = {str(r[b]): {k: v for k, v in r.items() if k != b}
                  for r in scored.group_by(b).agg(agg).sort(b).iter_rows(named=True)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("chunks", type=Path)
    ap.add_argument("--queries", type=Path, default=Path("data/eval/queries.parquet"))
    ap.add_argument("--systems", nargs="+", default=["bm25", "bge-m3", "hybrid"])
    ap.add_argument("--dense", nargs="+", choices=list(R.DENSE_MODELS),
                    help="dense indexes the hybrid uses (default: every one built)")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    q = pl.read_parquet(args.queries).with_columns(
        cross_lingual=pl.col("query_lang") != pl.col("doc_language")
    )
    ch = pl.read_parquet(args.chunks, columns=["chunk_id", "decision_id", "text"])
    chunk_doc, chunk_ids, texts = ch["decision_id"].to_numpy(), ch["chunk_id"].to_numpy(), ch["text"].to_list()

    # Merge with earlier runs so the printed table compares every system so far.
    out = Path("data/eval/results.json")
    results = json.loads(out.read_text()) if out.exists() else {}
    for system in args.systems:
        label, idx = run_system(system, args.chunks, q, texts, args.device, args.dense)
        results[label] = summarize(metrics(idx, q, chunk_doc, chunk_ids))

    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}\n")
    systems = list(results)
    for view in ["overall", *BREAKDOWNS]:
        print(f"== {view}  (mrr@10 / hit@10)")
        keys = ["all"] if view == "overall" else list(results[systems[0]][view])
        print(f"{'system':16s}" + "".join(f"{k:>14s}" for k in keys))
        for s in systems:
            cells = [results[s]["overall"]] if view == "overall" else [results[s][view][k] for k in keys]
            print(f"{s:16s}" + "".join(f"{c['mrr@10']:>7.3f}/{c['hit@10']:.2f}" for c in cells))
        print()


if __name__ == "__main__":
    main()
