"""Keyword index over the passages: SQLite FTS5, in its own file next to the vector DB.

A separate file, so it can be built while the vector DB is being written to. Row ids
are `chunks.id` (the row in the chunks parquet), so hits join back to the vector DB's
`chunks` table. The table is contentless: it stores the index, not the text.

Query syntax: text in double quotes is an exact phrase, other words must all appear,
e.g.  Sperrfrist "Art. 271a OR"  or  "4A_705/2016".

Usage:
    uv run python -m swiss_court_assistant.fts build
    uv run python -m swiss_court_assistant.fts search '"Art. 271a OR" Sperrfrist'
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import time
from pathlib import Path

import polars as pl

DEFAULT_DECISIONS = Path("data/subset/decisions_50k_seed42.parquet")


def default_path(decisions: Path) -> Path:
    return Path("data/vectordb") / f"{decisions.stem}.fts.sqlite"


def build(chunks: Path, out: Path, batch: int = 50_000) -> None:
    tmp = out.with_name(out.name + ".tmp")
    tmp.unlink(missing_ok=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(tmp)
    con.execute("CREATE VIRTUAL TABLE passages USING fts5("
                "text, content='', tokenize='unicode61 remove_diacritics 2')")
    texts = pl.read_parquet(chunks, columns=["text"])["text"]
    t0 = time.time()
    with con:
        for s in range(0, len(texts), batch):
            con.executemany("INSERT INTO passages(rowid, text) VALUES (?, ?)",
                            enumerate(texts.slice(s, batch).to_list(), start=s))
            print(f"  {min(s + batch, len(texts)):,}/{len(texts):,} passages", flush=True)
        con.execute("INSERT INTO passages(passages) VALUES ('optimize')")
    con.close()
    tmp.replace(out)  # readers never see a half-built index
    print(f"done: {out} ({out.stat().st_size / 1e9:.1f} GB, {time.time() - t0:.0f} s)")


_TERM = re.compile(r'"([^"]*)"|(\S+)')


def match_expr(keyword: str) -> str:
    """FTS5 MATCH expression: quoted text stays a phrase, every part must match (AND)."""
    parts = []
    for phrase, word in _TERM.findall(keyword):
        t = (phrase or word).strip()
        if re.search(r"\w", t):
            parts.append('"' + t.replace('"', '""') + '"')
    return " ".join(parts)


class KeywordIndex:
    def __init__(self, path: Path):
        self.con = sqlite3.connect(f"file:{path}?mode=ro", uri=True, check_same_thread=False)

    def search(self, keyword: str, k: int = 50) -> list[tuple[int, float]]:
        """(chunks.id, bm25) best first; bm25 is negative, lower is better."""
        expr = match_expr(keyword)
        if not expr:
            return []
        return self.con.execute(
            "SELECT rowid, bm25(passages) FROM passages WHERE passages MATCH ? ORDER BY rank LIMIT ?",
            (expr, k)).fetchall()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    ap.add_argument("--out", type=Path, help="index file (default: data/vectordb/<subset>.fts.sqlite)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    s = sub.add_parser("search")
    s.add_argument("keyword")
    s.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    out = args.out or default_path(args.decisions)
    chunks = args.decisions.with_name(args.decisions.stem + ".chunks.parquet")
    if args.cmd == "build":
        build(chunks, out)
    else:
        hits = KeywordIndex(out).search(args.keyword, args.k)
        rows = pl.read_parquet(chunks, columns=["chunk_id", "text"])
        print(f"MATCH {match_expr(args.keyword)}")
        for rowid, score in hits:
            cid, text = rows.row(rowid)
            print(f"{score:7.2f}  {cid}\n         {text[:200].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
