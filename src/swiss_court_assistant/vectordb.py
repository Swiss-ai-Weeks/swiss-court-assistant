"""SQLite vector database: decisions, passages and embeddings in one file.

Built on the sqlite-vec extension. The file holds

  decisions         every metadata column of the subset, plus full_text (citations
                    point into it)
  chunks            citable passages: char offsets into full_text, Erwägung numbers
  vec_<model>       one vec0 KNN table per embedding model (cosine distance), with
                    filter columns copied from the metadata: language, canton,
                    jurisdiction, court, branch, period, year, section
  embedding_models  which models are in the file and how complete they are

`chunks.id` is the row position in the chunks parquet, which is also the row of
the .npy embedding files written by retrieval.py.

Building is idempotent and resumable: embeddings are committed batch by batch
and a rerun only embeds the chunks that are still missing, so an interrupted
build loses at most one batch. Existing .npy embeddings can be imported instead
of recomputed.

Usage:
    uv run python -m swiss_court_assistant.vectordb build --model bge-m3 \
        --from-npy data/index/decisions_50k_seed42/bge-m3.npy
    uv run python -m swiss_court_assistant.vectordb build --model nemotron-embed   # needs the nim-embed container
    uv run python -m swiss_court_assistant.vectordb search "Kündigung wegen Eigenbedarf" --model bge-m3 --language fr
    uv run python -m swiss_court_assistant.vectordb info
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import sqlite_vec

from swiss_court_assistant import retrieval as R
from swiss_court_assistant.sampling import period_expr

DEFAULT_DECISIONS = Path("data/subset/decisions_50k_seed42.parquet")
DB_DIR = Path("data/vectordb")
# Copied onto every vector row so KNN can filter without a join. sqlite-vec
# metadata columns can't hold NULL, so unknowns are "" / 0.
VEC_FILTERS = {
    "language": "text", "canton": "text", "jurisdiction": "text", "court": "text",
    "branch": "text", "period": "text", "year": "integer", "section": "text",
}


def chunks_path_for(decisions: Path) -> Path:
    return decisions.with_name(decisions.stem + ".chunks.parquet")


def default_db(decisions: Path) -> Path:
    return DB_DIR / f"{decisions.stem}.sqlite"


def vec_table(model: str) -> str:
    return "vec_" + re.sub(r"\W", "_", model)


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    con.execute("PRAGMA journal_mode=WAL")  # readers keep working during a build
    con.execute("PRAGMA synchronous=NORMAL")
    return con


def _exists(con: sqlite3.Connection, table: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE name = ?", [table]).fetchone() is not None


def _sql_type(dtype: pl.DataType) -> str:
    if dtype.is_integer() or dtype == pl.Boolean:
        return "INTEGER"
    return "REAL" if dtype.is_float() else "TEXT"


# ── tables ──────────────────────────────────────────────────────────────
def load_decisions(con: sqlite3.Connection, path: Path) -> None:
    if _exists(con, "decisions"):
        return
    df = pl.read_parquet(path)
    cols = ", ".join(
        f'"{c}" {_sql_type(t)}' + (" PRIMARY KEY" if c == "decision_id" else "")
        for c, t in df.schema.items()
    )
    with con:  # one transaction: the table is either complete or absent
        con.execute(f"CREATE TABLE decisions ({cols})")
        con.executemany(f"INSERT INTO decisions VALUES ({', '.join('?' * df.width)})", df.iter_rows())
        for c in ("court", "canton", "language", "decision_date"):
            if c in df.columns:
                con.execute(f'CREATE INDEX decisions_{c} ON decisions("{c}")')
    print(f"decisions: {df.height:,} rows, {df.width} columns")


def load_chunks(con: sqlite3.Connection, path: Path) -> None:
    if _exists(con, "chunks"):
        return
    df = pl.read_parquet(path).with_row_index("id").with_columns(
        # list cells arrive as Series
        erwaegungen=pl.col("erwaegungen").map_elements(lambda s: json.dumps(s.to_list()), return_dtype=pl.String)
    )
    cols = ["id", "chunk_id", "decision_id", "chunk_index", "section", "language",
            "char_start", "char_end", "erwaegungen", "text"]
    with con:
        con.execute("""
            CREATE TABLE chunks (
                id INTEGER PRIMARY KEY,            -- row in the chunks parquet / .npy files
                chunk_id TEXT UNIQUE NOT NULL,     -- "<decision_id>#<chunk_index>"
                decision_id TEXT NOT NULL REFERENCES decisions(decision_id),
                chunk_index INTEGER,               -- negative = Regeste (headnote)
                section TEXT,                      -- regeste | erwaegung | body
                language TEXT,
                char_start INTEGER,                -- offsets into decisions.full_text;
                char_end INTEGER,                  -- NULL for Regeste chunks
                erwaegungen TEXT,                  -- JSON list of Erwägung numbers, e.g. ["2.1"]
                text TEXT NOT NULL
            )""")
        con.executemany(f"INSERT INTO chunks VALUES ({', '.join('?' * len(cols))})",
                        df.select(cols).iter_rows())
        con.execute("CREATE INDEX chunks_decision ON chunks(decision_id)")
    print(f"chunks: {df.height:,} rows")


def chunk_frame(decisions: Path, chunks: Path) -> pl.DataFrame:
    """Chunk id, text and the vec filter columns, in chunk-id order."""
    d = pl.read_parquet(decisions, columns=["decision_id", "canton", "court", "branch", "decision_date"])
    d = d.with_columns(
        branch=pl.col("branch").fill_null("unknown"),
        jurisdiction=pl.when(pl.col("canton") == "CH").then(pl.col("court")).otherwise(pl.col("canton")),
        period=period_expr(),
        year=pl.col("decision_date").str.slice(0, 4).cast(pl.Int64, strict=False).fill_null(0),
    )
    c = pl.read_parquet(chunks, columns=["decision_id", "section", "language", "text"]).with_row_index("id")
    f = c.join(d, on="decision_id", how="left").sort("id")
    return f.with_columns(pl.col(n).fill_null("") for n, t in VEC_FILTERS.items() if t == "text")


# ── embeddings ──────────────────────────────────────────────────────────
def encode_passages(enc, texts: list[str]) -> np.ndarray:
    if isinstance(enc, R.NimEncoder):
        return enc.embed(texts, "passage")
    return enc.encode(texts, batch_size=128, normalize_embeddings=True, convert_to_numpy=True)


def encode_query(model: str, enc, text: str) -> np.ndarray:
    if isinstance(enc, R.NimEncoder):
        return enc.embed([text], "query")[0]
    prompt = R.DENSE_MODELS[model].get("query_prompt") or None
    return enc.encode([text], prompt=prompt, normalize_embeddings=True, convert_to_numpy=True)[0]


def _check_npy_order(npy: Path, chunks: Path) -> None:
    """The .npy rows must follow the chunks parquet; retrieval.py saves the ids next to it."""
    ids_file = npy.parent / "chunk_ids.parquet"
    if not ids_file.exists():
        print(f"warning: {ids_file} missing, cannot verify that {npy.name} matches the chunk order")
        return
    a = pl.read_parquet(ids_file)["chunk_id"]
    b = pl.read_parquet(chunks, columns=["chunk_id"])["chunk_id"]
    if not a.equals(b):
        raise SystemExit(f"{npy} was built from a different chunk table than {chunks}; rebuild without --from-npy")


def embed_missing(con: sqlite3.Connection, model: str, frame: pl.DataFrame, device: str,
                  batch: int, from_npy: Path | None) -> None:
    table = vec_table(model)
    emb, enc = None, None
    if from_npy:
        emb = np.load(from_npy, mmap_mode="r")
        if emb.shape[0] != frame.height:
            raise SystemExit(f"{from_npy} has {emb.shape[0]:,} rows, chunks table has {frame.height:,}")
        dim, source = emb.shape[1], f"npy:{from_npy}"
    else:
        enc = R.load_encoder(model, device)
        dim, source = encode_passages(enc, ["dimension probe"]).shape[1], "computed"

    cols = list(VEC_FILTERS)
    with con:
        con.execute(f"""CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(
            chunk_rowid INTEGER PRIMARY KEY,
            embedding float[{dim}] distance_metric=cosine,
            {", ".join(f"{c} {t}" for c, t in VEC_FILTERS.items())})""")
        con.execute("""CREATE TABLE IF NOT EXISTS embedding_models (
            model TEXT PRIMARY KEY, model_name TEXT, dim INTEGER, vec_table TEXT,
            n_embedded INTEGER, n_chunks INTEGER, source TEXT, updated_at TEXT)""")

    have = pl.DataFrame({"id": [r[0] for r in con.execute(f"SELECT chunk_rowid FROM {table}")]},
                        schema={"id": pl.UInt32})
    todo = frame.join(have, on="id", how="anti")
    done = have.height
    print(f"{model}: {done:,}/{frame.height:,} chunks already embedded, {todo.height:,} to go")
    insert = (f"INSERT INTO {table}(chunk_rowid, embedding, {', '.join(cols)}) "
              f"VALUES (?, ?, {', '.join('?' * len(cols))})")
    name = R.DENSE_MODELS[model].get("name") or R.DENSE_MODELS[model].get("model")
    t0 = time.time()
    for s in range(0, todo.height, batch):
        part = todo.slice(s, batch)
        ids = part["id"].to_numpy()
        vecs = (np.asarray(emb[ids], dtype=np.float32) if emb is not None
                else encode_passages(enc, part["text"].to_list()).astype(np.float32))
        rows = ((int(i), v.tobytes(), *meta) for i, v, meta in zip(ids, vecs, part.select(cols).iter_rows()))
        with con:  # commit per batch: an interrupted build resumes from here
            con.executemany(insert, rows)
            done += part.height
            con.execute(
                "INSERT OR REPLACE INTO embedding_models VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [model, name, dim, table, done, frame.height, source,
                 datetime.now(timezone.utc).isoformat(timespec="seconds")],
            )
        rate = (s + part.height) / (time.time() - t0)
        eta = (todo.height - s - part.height) / rate / 60
        print(f"  {done:,}/{frame.height:,} embedded  ({rate:,.0f} chunks/s, ~{eta:.0f} min left)", flush=True)


def build(decisions: Path, chunks: Path, db: Path, model: str, device: str, batch: int,
          from_npy: Path | None) -> None:
    if from_npy:
        _check_npy_order(from_npy, chunks)
    con = connect(db)
    load_decisions(con, decisions)
    load_chunks(con, chunks)
    embed_missing(con, model, chunk_frame(decisions, chunks), device, batch, from_npy)
    con.close()
    print(f"done: {db} ({db.stat().st_size / 1e9:.1f} GB)")


# ── search ──────────────────────────────────────────────────────────────
class VectorDB:
    """KNN search over the SQLite file, returning passages with their decision metadata.

    >>> db = VectorDB("data/vectordb/decisions_50k_seed42.sqlite")
    >>> db.search("Kündigung wegen Eigenbedarf", model="bge-m3", k=5, language="fr", year_from=2010)
    """

    def __init__(self, path: str | Path, device: str = "cpu"):
        self.con = connect(Path(path))
        self.con.row_factory = sqlite3.Row
        self.device = device
        self._encoders: dict = {}

    def models(self) -> list[dict]:
        return [dict(r) for r in self.con.execute("SELECT * FROM embedding_models")]

    def encode(self, query: str, model: str) -> np.ndarray:
        if model not in self._encoders:
            self._encoders[model] = R.load_encoder(model, self.device)
        return encode_query(model, self._encoders[model], query)

    def search(self, query: str | np.ndarray, model: str, k: int = 10, year_from: int | None = None,
               year_to: int | None = None, **filters: str | None) -> list[dict]:
        if isinstance(query, str):
            query = self.encode(query, model)
        where, params = ["embedding MATCH ?", "k = ?"], [np.asarray(query, dtype=np.float32).tobytes(), k]
        for col, val in filters.items():
            if col not in VEC_FILTERS:
                raise ValueError(f"unknown filter {col!r}; use one of {list(VEC_FILTERS)}")
            if val is not None:
                where.append(f"{col} = ?"), params.append(val)
        if year_from is not None:
            where.append("year >= ?"), params.append(year_from)
        if year_to is not None:
            where.append("year <= ?"), params.append(year_to)
        sql = f"""
            SELECT v.distance, c.chunk_id, c.decision_id, c.section, c.erwaegungen,
                   c.char_start, c.char_end, c.text,
                   d.court, d.canton, d.docket_number, d.decision_date, d.language,
                   d.title, d.source_url
            FROM (SELECT chunk_rowid, distance FROM {vec_table(model)} WHERE {" AND ".join(where)}) v
            JOIN chunks c ON c.id = v.chunk_rowid
            JOIN decisions d ON d.decision_id = c.decision_id
            ORDER BY v.distance"""
        out = []
        for r in self.con.execute(sql, params):
            row = dict(r)
            row["erwaegungen"] = json.loads(row["erwaegungen"] or "[]")
            row["similarity"] = 1 - row.pop("distance")
            out.append(row)
        return out


def info(db: Path) -> None:
    con = connect(db)
    print(f"{db}  ({db.stat().st_size / 1e9:.1f} GB)")
    for t in ("decisions", "chunks"):
        n = con.execute(f"SELECT count(*) FROM {t}").fetchone()[0] if _exists(con, t) else 0
        print(f"  {t:10s} {n:>10,} rows")
    if _exists(con, "embedding_models"):
        for m, name, dim, n, total, src, ts in con.execute(
                "SELECT model, model_name, dim, n_embedded, n_chunks, source, updated_at FROM embedding_models"):
            print(f"  vec_{m:20s} {n:>10,}/{total:,} chunks  dim {dim}  {name}  ({src}, {ts})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS, help="subset parquet (sampling.py)")
    ap.add_argument("--db", type=Path, help="SQLite file (default: data/vectordb/<subset>.sqlite)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="create/extend the database (resumable)")
    b.add_argument("--model", required=True, choices=list(R.DENSE_MODELS))
    b.add_argument("--from-npy", type=Path, help="import embeddings saved by retrieval.py instead of computing")
    b.add_argument("--device", default="cuda:0", help="for local sentence-transformers models")
    b.add_argument("--batch", type=int, default=8192, help="chunks per committed batch")

    s = sub.add_parser("search", help="KNN search with optional metadata filters")
    s.add_argument("query")
    s.add_argument("--model", required=True, choices=list(R.DENSE_MODELS))
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--device", default="cpu")
    for col in VEC_FILTERS:
        if col != "year":
            s.add_argument(f"--{col}")
    s.add_argument("--year-from", type=int)
    s.add_argument("--year-to", type=int)

    sub.add_parser("info", help="row counts and embedded models")
    args = ap.parse_args()
    db = args.db or default_db(args.decisions)

    if args.cmd == "build":
        build(args.decisions, chunks_path_for(args.decisions), db, args.model, args.device,
              args.batch, args.from_npy)
    elif args.cmd == "info":
        info(db)
    else:
        filters = {c: getattr(args, c) for c in VEC_FILTERS if c != "year"}
        hits = VectorDB(db, args.device).search(args.query, args.model, args.k, args.year_from,
                                                args.year_to, **filters)
        for i, h in enumerate(hits, 1):
            e = f" E. {', '.join(h['erwaegungen'])}" if h["erwaegungen"] else ""
            print(f"{i:2d}. {h['similarity']:.3f}  {h['court']} {h['docket_number']} "
                  f"({h['decision_date']}, {h['language']}){e}\n    {h['text'][:220].replace(chr(10), ' ')}")


if __name__ == "__main__":
    main()
