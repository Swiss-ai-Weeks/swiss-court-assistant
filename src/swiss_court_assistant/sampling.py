"""Build a stratified, reproducible subset of the voilaj/swiss-caselaw corpus.

The subset keeps the joint distribution of the full corpus over
language x canton x legal branch x period, so retrieval quality measured on
it transfers to the full dataset. Rare strata (Romansh, small cantons, old
decisions) get a minimum floor so they are not sampled away entirely.

Usage:
    uv run python -m swiss_court_assistant.sampling --n 50000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

RAW_GLOB = "data/raw/data/*.parquet"
OUT_DIR = Path("data/subset")

# "jurisdiction" = canton for cantonal courts, the court itself for federal
# ones (canton "CH" lumps BGer, BVGer, BGE leading cases, ... together).
STRATA = ["language", "jurisdiction", "branch", "period"]
MIN_TEXT_CHARS = 500  # below this the text is usually a failed PDF extraction


def period_expr() -> pl.Expr:
    year = pl.col("decision_date").str.slice(0, 4).cast(pl.Int32, strict=False)
    return (
        pl.when(year.is_null()).then(pl.lit("unknown"))
        .when(year < 1954).then(pl.lit("1875-1953"))  # BGE historical, OCR era
        .when(year < 2000).then(pl.lit("1954-1999"))
        .when(year < 2010).then(pl.lit("2000-2009"))
        .when(year < 2020).then(pl.lit("2010-2019"))
        .otherwise(pl.lit("2020+"))
    )


def load_candidates(raw_glob: str = RAW_GLOB) -> pl.LazyFrame:
    """All usable decisions, one row per distinct text, with stratum keys."""
    return (
        pl.scan_parquet(raw_glob)
        .filter(pl.col("has_full_text") & (pl.col("text_length") >= MIN_TEXT_CHARS))
        # GE/VD publish one ruling under two identifiers: keep one copy per text.
        .sort("text_length", descending=True)
        .unique(subset="content_hash", keep="first", maintain_order=False)
        .with_columns(
            period=period_expr(),
            branch=pl.col("branch").fill_null("unknown"),
            jurisdiction=pl.when(pl.col("canton") == "CH")
            .then(pl.col("court"))
            .otherwise(pl.col("canton")),
        )
    )


def allocate(counts: pl.DataFrame, n: int, floor: int) -> pl.DataFrame:
    """Proportional allocation per stratum with a minimum floor.

    Each stratum gets max(floor, round(n * share)), capped at its size. The
    proportional part is scaled down so the floors don't push the total far
    past n.
    """
    total = counts["count"].sum()
    floors = counts["count"].clip(upper_bound=floor)
    budget = max(n - int(floors.sum()), 0)
    return counts.with_columns(
        take=pl.max_horizontal(
            floors, (pl.col("count") / total * budget).round().cast(pl.Int64)
        ).clip(upper_bound=pl.col("count"))
    )


def sample(n: int, floor: int, seed: int, raw_glob: str = RAW_GLOB) -> tuple[pl.DataFrame, pl.DataFrame]:
    cand = load_candidates(raw_glob)
    keys = cand.select(["decision_id", *STRATA]).collect()
    counts = keys.group_by(STRATA).len("count")
    plan = allocate(counts, n, floor)

    # Rank inside each stratum by a seeded hash: deterministic and order-independent.
    chosen = (
        keys.with_columns(_r=pl.col("decision_id").hash(seed).rank("ordinal").over(STRATA))
        .join(plan.select([*STRATA, "take"]), on=STRATA)
        .filter(pl.col("_r") <= pl.col("take"))
        .select("decision_id")
    )
    subset = cand.join(chosen.lazy(), on="decision_id", how="semi").collect()
    return subset, keys


def distribution_report(full: pl.DataFrame, subset: pl.DataFrame) -> dict:
    report = {}
    for col in STRATA:
        f = full[col].value_counts().with_columns(full=pl.col("count") / full.height)
        s = subset[col].value_counts().with_columns(subset=pl.col("count") / subset.height)
        joined = f.select(col, "full").join(s.select(col, "subset"), on=col, how="left").fill_null(0)
        report[col] = {
            str(r[col]): {"full": round(r["full"], 4), "subset": round(r["subset"], 4)}
            for r in joined.sort("full", descending=True).iter_rows(named=True)
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50_000, help="target subset size")
    ap.add_argument("--floor", type=int, default=3, help="minimum rows per non-empty stratum")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    subset, full_keys = sample(args.n, args.floor, args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"decisions_{args.n // 1000}k_seed{args.seed}.parquet"
    subset.write_parquet(out, compression="zstd")

    report = distribution_report(full_keys, subset)
    manifest = {
        "source": "voilaj/swiss-caselaw data/*.parquet",
        "rows_full_deduped": full_keys.height,
        "rows_subset": subset.height,
        "strata": STRATA,
        "n_strata": full_keys.select(STRATA).n_unique(),
        "params": vars(args) | {"min_text_chars": MIN_TEXT_CHARS},
        "distribution": report,
    }
    out.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"wrote {out} ({subset.height:,} rows of {full_keys.height:,})")
    for col in STRATA:
        top = list(report[col].items())[:8]
        print(f"  {col:9s}", "  ".join(f"{k}:{v['full']:.3f}/{v['subset']:.3f}" for k, v in top))


if __name__ == "__main__":
    main()
