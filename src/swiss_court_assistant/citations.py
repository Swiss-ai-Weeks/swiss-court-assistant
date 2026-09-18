"""Who cites whom: a lookup index over the corpus citation graph.

    uv run python -m swiss_court_assistant.citations build
    uv run python -m swiss_court_assistant.citations show bge_BGE_122_V_157

`data/raw/graph/citations.parquet` holds 11.8M edges for the whole corpus. The index keeps the edges
that touch the development subset (either end), plus a label for every decision they mention — most
citing decisions are outside the subset, so their court and docket come from the full corpus files.
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from pathlib import Path

import polars as pl

CITATIONS = Path("data/raw/graph/citations.parquet")
CORPUS = Path("data/raw/data")
DEFAULT_SUBSET = Path("data/subset/decisions_50k_seed42.parquet")
CHUNK = 200_000


def index_path(subset: Path) -> Path:
    return Path("data/graph") / f"{subset.stem}.citations.sqlite"


def _schema(con: sqlite3.Connection) -> None:
    con.executescript("""
        DROP TABLE IF EXISTS edges;
        DROP TABLE IF EXISTS decisions;
        DROP TABLE IF EXISTS counts;
        CREATE TABLE edges (source TEXT, target TEXT, confidence REAL);
        CREATE TABLE decisions (decision_id TEXT PRIMARY KEY, court TEXT, docket TEXT, date TEXT);
        CREATE TABLE counts (decision_id TEXT PRIMARY KEY, cited_by INTEGER, cites INTEGER);
    """)


def _labels(ids: set[str]) -> pl.DataFrame:
    """court, docket and date for every decision in `ids`, from the full corpus."""
    frames = []
    for f in sorted(CORPUS.glob("*.parquet")):
        frame = (pl.scan_parquet(f)
                 .select("decision_id", "court", "docket_number", "decision_date")
                 .filter(pl.col("decision_id").is_in(ids))
                 .collect())
        if len(frame):
            frames.append(frame)
    out = pl.concat(frames) if frames else pl.DataFrame(
        schema={"decision_id": pl.Utf8, "court": pl.Utf8, "docket_number": pl.Utf8, "decision_date": pl.Utf8})
    return out.with_columns(pl.col("decision_date").cast(pl.Utf8)).unique(subset="decision_id")


def build(subset: Path = DEFAULT_SUBSET) -> Path:
    t0 = time.time()
    ids = set(pl.read_parquet(subset, columns=["decision_id"])["decision_id"])
    edges = (pl.read_parquet(CITATIONS, columns=["source_decision_id", "target_decision_id", "confidence_score"])
             .filter(pl.col("target_decision_id").is_in(ids) | pl.col("source_decision_id").is_in(ids))
             .rename({"source_decision_id": "source", "target_decision_id": "target",
                      "confidence_score": "confidence"}))
    print(f"{len(edges):,} edges touch the subset ({time.time() - t0:.0f}s)")

    mentioned = set(edges["source"]) | set(edges["target"])
    labels = _labels(mentioned)
    print(f"{len(labels):,} of {len(mentioned):,} decisions have metadata in the corpus "
          f"({time.time() - t0:.0f}s)")

    out = index_path(subset)
    out.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(out)
    _schema(con)
    for frame in edges.iter_slices(CHUNK):
        con.executemany("INSERT INTO edges VALUES (?, ?, ?)", frame.iter_rows())
    con.executemany("INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, ?)", labels.iter_rows())
    con.executescript("""
        CREATE INDEX edges_target ON edges(target);
        CREATE INDEX edges_source ON edges(source);
        INSERT INTO counts (decision_id, cited_by, cites)
          SELECT d, SUM(inc), SUM(outg) FROM (
            SELECT target AS d, COUNT(*) AS inc, 0 AS outg FROM edges GROUP BY target
            UNION ALL
            SELECT source AS d, 0 AS inc, COUNT(*) AS outg FROM edges GROUP BY source)
          GROUP BY d;
    """)
    con.commit()
    size = out.stat().st_size / 1e6
    print(f"{out} — {size:.0f} MB, {time.time() - t0:.0f}s")
    con.close()
    return out


def show(decision_id: str, subset: Path = DEFAULT_SUBSET, limit: int = 10) -> None:
    con = sqlite3.connect(f"file:{index_path(subset)}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    row = con.execute("SELECT * FROM counts WHERE decision_id = ?", [decision_id]).fetchone()
    print(f"{decision_id}: cited by {row['cited_by'] if row else 0}, cites {row['cites'] if row else 0}")
    for r in con.execute("""SELECT e.source, d.court, d.docket, d.date FROM edges e
                            LEFT JOIN decisions d ON d.decision_id = e.source
                            WHERE e.target = ? ORDER BY d.date DESC NULLS LAST LIMIT ?""",
                         [decision_id, limit]):
        print(f"  ← {r['date'] or '?':10} {r['court'] or '?':10} {r['docket'] or r['source']}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    s = sub.add_parser("show")
    s.add_argument("decision_id")
    s.add_argument("--subset", type=Path, default=DEFAULT_SUBSET)
    s.add_argument("--limit", type=int, default=10)
    args = p.parse_args()
    if args.command == "build":
        build(args.subset)
    else:
        show(args.decision_id, args.subset, args.limit)


if __name__ == "__main__":
    main()
