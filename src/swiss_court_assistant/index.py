"""Build and update the full-corpus index from the HuggingFace dataset.

One command builds everything the app searches - decisions, passages, embeddings, keyword index
and the citation graph - and the same command run again only does what is missing. Embeddings come
from our own NIM (nothing leaves this machine at query time); the dataset is fetched at build time.

    decisions          every column of the upstream shards, plus the derived jurisdiction/period
    chunks             citable passages, char offsets into decisions.full_text
    vec_<model>        one vec0 KNN table per embedding model
    pending_embeddings ids waiting for a vector, so an interrupted run resumes exactly
    index_sources      which upstream file (by sha256) is already in, for incremental updates
    index_state        the delta watermark

Incremental updates ride on the dataset's own `artifacts/manifest.json`: a dated snapshot plus one
parquet per day of changes, each with a sha256. `update` applies every delta newer than the
watermark - the delta schema is identical to the shard schema, so a delta row is just an upsert.

Chunk ids are append-only. They are the rowid of both the vector table and the keyword index, so a
decision that changes gets its old chunks (and their vectors and FTS rows) deleted and new chunks
appended at the end; ids are never renumbered.

Which decisions: `--since YEAR` keeps decisions from that year on and `--fraction 0.25` takes a
stratified sample of them (sampling.py: language x jurisdiction x branch x period), remembered in
`selected_decisions` so reruns and deltas index the same set. `--laws` adds every statute article
(federal and cantonal) from voilaj/swiss-legislation as passages next to the decisions'; the
article's `law_id` stands in for the decision_id and the `laws` table holds its metadata.

`build` and `update` finish by re-exporting the vectors as an in-memory matrix (vecmatrix.py), which
the app searches instead of scanning sqlite-vec.

Usage:
    uv run python -m swiss_court_assistant.index build --laws --since 1980 --fraction 0.25
    uv run python -m swiss_court_assistant.index build --courts bger,bge      # start small
    uv run python -m swiss_court_assistant.index grow --fraction 0.43         # more of the same sample
    uv run python -m swiss_court_assistant.index grow --fraction 1.0 --all-years   # every decision
    uv run python -m swiss_court_assistant.index update                       # apply new deltas
    uv run python -m swiss_court_assistant.index citations                    # rebuild the citation graph
    uv run python -m swiss_court_assistant.index status
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from swiss_court_assistant import chunking
from swiss_court_assistant import retrieval as R
from swiss_court_assistant import sampling as S
from swiss_court_assistant import vectordb as V
from swiss_court_assistant.sampling import period_expr

REPO = "voilaj/swiss-caselaw"
LAWS_REPO = "voilaj/swiss-legislation"  # article-level statute text, federal + cantonal
MANIFEST = "artifacts/manifest.json"
NAME = "corpus"  # the index is a corpus, not a sample: data/vectordb/corpus.sqlite etc.
DB = V.DB_DIR / f"{NAME}.sqlite"
FTS_DB = V.DB_DIR / f"{NAME}.fts.sqlite"
IDS_PARQUET = Path("data/subset") / f"{NAME}.parquet"  # citations.build keys its output off the stem
SCRATCH = Path("data/index-work")
# The shards under data/ are the current layout; bge_historical exists only at the repo root, so
# indexing the data/ prefix alone would silently drop the historical BGE volumes.
ROOT_ONLY = ["bge_historical.parquet"]


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# ── upstream ────────────────────────────────────────────────────────────
def manifest() -> dict:
    return json.load(open(hf_hub_download(REPO, MANIFEST, repo_type="dataset")))


def shard_files(courts: list[str] | None) -> list[str]:
    """Upstream decision shards to index, newest layout first."""
    files = [s.rfilename for s in HfApi().repo_info(REPO, repo_type="dataset").siblings]
    # a dated delta file sits in data/ alongside the court shards, in the upstream scraper's own
    # layout (id/content_text, no decision_id); deltas belong to `update`, which reads the manifest
    shards = [f for f in files if f.startswith("data/") and f.endswith(".parquet")
              and not Path(f).name.startswith("delta-")]
    shards += [f for f in ROOT_ONLY if f in files]
    if courts:
        want = {c.strip() for c in courts}
        shards = [f for f in shards if Path(f).stem.split(".")[0] in want]
    return sorted(shards)


def fetch(repo_path: str) -> Path:
    """Download (or reuse the cached copy of) one upstream file."""
    return Path(hf_hub_download(REPO, repo_path, repo_type="dataset"))


# ── schema ──────────────────────────────────────────────────────────────
def open_db() -> sqlite3.Connection:
    con = V.connect(DB)
    with con:
        con.execute("""CREATE TABLE IF NOT EXISTS index_sources (
            path TEXT PRIMARY KEY, sha256 TEXT, rows INTEGER, indexed_at TEXT)""")
        con.execute("CREATE TABLE IF NOT EXISTS index_state (key TEXT PRIMARY KEY, value TEXT)")
        con.execute("CREATE TABLE IF NOT EXISTS pending_embeddings (id INTEGER PRIMARY KEY)")
        con.execute("CREATE TABLE IF NOT EXISTS selected_decisions (decision_id TEXT PRIMARY KEY)")
        con.execute("""CREATE TABLE IF NOT EXISTS laws (
            law_id TEXT PRIMARY KEY, canton TEXT, sr_number TEXT, abbreviation TEXT, article_num TEXT,
            seq INTEGER, language TEXT, law_title TEXT, heading TEXT, category TEXT, text TEXT NOT NULL,
            original_url TEXT)""")
        con.execute("CREATE INDEX IF NOT EXISTS laws_article ON laws(sr_number, article_num, language)")
        con.execute("CREATE INDEX IF NOT EXISTS laws_abbrev ON laws(abbreviation, article_num, language)")
    _chunks_table(con)  # laws may be ingested before any decision, and their passages go here too
    return con


def open_fts() -> sqlite3.Connection:
    FTS_DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(FTS_DB)
    con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS passages USING fts5("
                "text, content='', tokenize='unicode61 remove_diacritics 2')")
    return con


def state(con: sqlite3.Connection, key: str, value: str | None = None) -> str | None:
    if value is None:
        row = con.execute("SELECT value FROM index_state WHERE key = ?", [key]).fetchone()
        return row[0] if row else None
    with con:
        con.execute("INSERT OR REPLACE INTO index_state VALUES (?, ?)", [key, value])
    return value


def _decisions_table(con: sqlite3.Connection, df: pl.DataFrame) -> None:
    """Create the table from the first shard, then widen it as shards with more columns arrive -
    the shards do not all share one layout, and the first one sorted is the narrowest."""
    if not V._exists(con, "decisions"):
        cols = ", ".join(f'"{c}" {V._sql_type(t)}' + (" PRIMARY KEY" if c == "decision_id" else "")
                         for c, t in df.schema.items())
        with con:
            con.execute(f"CREATE TABLE decisions ({cols})")
            for c in ("court", "canton", "language", "decision_date"):
                if c in df.columns:
                    con.execute(f'CREATE INDEX decisions_{c} ON decisions("{c}")')
        return
    have = {r[1] for r in con.execute("PRAGMA table_info(decisions)")}
    with con:
        for c, t in df.schema.items():
            if c not in have:
                con.execute(f'ALTER TABLE decisions ADD COLUMN "{c}" {V._sql_type(t)}')


def _chunks_table(con: sqlite3.Connection) -> None:
    with con:
        con.execute("""CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY,
            chunk_id TEXT UNIQUE NOT NULL,
            decision_id TEXT NOT NULL REFERENCES decisions(decision_id),
            chunk_index INTEGER,
            section TEXT,
            language TEXT,
            char_start INTEGER,
            char_end INTEGER,
            erwaegungen TEXT,
            text TEXT NOT NULL)""")
        con.execute("CREATE INDEX IF NOT EXISTS chunks_decision ON chunks(decision_id)")


# ── one shard ───────────────────────────────────────────────────────────
def prepare(df: pl.DataFrame) -> pl.DataFrame:
    """The two columns sampling.py derives; the rest ships with the shard."""
    if "branch" not in df.columns:  # the root-level bge_historical.parquet is an older, 33-column layout
        df = df.with_columns(branch=pl.lit(None, dtype=pl.String))
    return df.with_columns(
        branch=pl.col("branch").fill_null("unknown"),
        jurisdiction=pl.when(pl.col("canton") == "CH").then(pl.col("court")).otherwise(pl.col("canton")),
        period=period_expr(),
    )


def drop_decisions(con: sqlite3.Connection, fts: sqlite3.Connection, ids: list[str]) -> int:
    """Remove a decision's passages from all three indexes before re-adding them."""
    if not ids:
        return 0
    gone = 0
    for s in range(0, len(ids), 500):
        part = ids[s:s + 500]
        holes = ",".join("?" * len(part))
        rows = con.execute(f"SELECT id, text FROM chunks WHERE decision_id IN ({holes})", part).fetchall()
        if not rows:
            continue
        # a contentless FTS5 table needs the original text to delete a row
        with fts:
            fts.executemany("INSERT INTO passages(passages, rowid, text) VALUES('delete', ?, ?)", rows)
        with con:
            ids_ = [[r[0]] for r in rows]
            for table in (t for t in (V.vec_table(m) for m in models(con)) if V._exists(con, t)):
                con.executemany(f"DELETE FROM {table} WHERE chunk_rowid = ?", ids_)
            con.executemany("DELETE FROM pending_embeddings WHERE id = ?", ids_)
            con.executemany("DELETE FROM chunks WHERE id = ?", ids_)
        gone += len(rows)
    return gone


def models(con: sqlite3.Connection) -> list[str]:
    if not V._exists(con, "embedding_models"):
        return []
    return [r[0] for r in con.execute("SELECT model FROM embedding_models")]


def add_chunks(con: sqlite3.Connection, chunks: pl.DataFrame) -> tuple[int, int]:
    """Append passages with fresh ids and queue them for embedding. Returns (first id, count)."""
    if chunks.is_empty():
        return 0, 0
    start = con.execute("SELECT COALESCE(MAX(id), -1) FROM chunks").fetchone()[0] + 1
    chunks = chunks.with_row_index("id", offset=start).with_columns(
        erwaegungen=pl.col("erwaegungen").map_elements(
            lambda s: json.dumps(s.to_list() if hasattr(s, "to_list") else list(s)), return_dtype=pl.String))
    cols = ["id", "chunk_id", "decision_id", "chunk_index", "section", "language",
            "char_start", "char_end", "erwaegungen", "text"]
    with con:
        con.executemany(f"INSERT INTO chunks VALUES ({', '.join('?' * len(cols))})",
                        chunks.select(cols).iter_rows())
        con.executemany("INSERT OR IGNORE INTO pending_embeddings VALUES (?)",
                        [[i] for i in chunks["id"].to_list()])
    return start, chunks.height


def index_keywords(con: sqlite3.Connection, fts: sqlite3.Connection, ids: list[int]) -> None:
    """FTS rowid is chunks.id, so hits join straight back to the passage."""
    for s in range(0, len(ids), 50_000):
        part = ids[s:s + 50_000]
        holes = ",".join("?" * len(part))
        rows = con.execute(f"SELECT id, text FROM chunks WHERE id IN ({holes})", part).fetchall()
        with fts:
            fts.executemany("INSERT INTO passages(rowid, text) VALUES (?, ?)", rows)


def ingest(con: sqlite3.Connection, fts: sqlite3.Connection, path: Path, label: str,
           skip: set[str] | None = None) -> tuple[int, int]:
    """Index one upstream parquet (a shard or a delta), leaving out the decision ids in `skip`.
    Returns (decisions, chunks)."""
    df = prepare(pl.read_parquet(path))
    if "decision_id" not in df.columns or "full_text" not in df.columns:
        print(f"  skipping {label}: not a decision shard ({len(df.columns)} columns, no decision_id/full_text)")
        return 0, 0
    selected = [r[0] for r in con.execute("SELECT decision_id FROM selected_decisions")]
    if selected:  # only the chosen sample; a delta may bring decisions outside it, which are skipped
        df = df.filter(pl.col("decision_id").is_in(selected))
    if skip:
        df = df.filter(~pl.col("decision_id").is_in(list(skip)))
    if df.is_empty():
        return 0, 0
    _decisions_table(con, df)
    _chunks_table(con)
    # decisions already held under one of these ids are being replaced: drop their passages first,
    # otherwise the old vectors and FTS rows would survive alongside the new ones
    con.execute("CREATE TEMP TABLE IF NOT EXISTS _incoming (decision_id TEXT PRIMARY KEY)")
    con.execute("DELETE FROM _incoming")
    con.executemany("INSERT OR IGNORE INTO _incoming VALUES (?)", [[i] for i in df["decision_id"]])
    replaced = [r[0] for r in con.execute(
        "SELECT d.decision_id FROM decisions d JOIN _incoming i USING(decision_id)")]
    drop_decisions(con, fts, replaced)

    # insert by name: the table's column order is fixed by whichever shard created it
    cols = [c for c in (r[1] for r in con.execute("PRAGMA table_info(decisions)")) if c in df.columns]
    names = ", ".join(f'"{c}"' for c in cols)
    with con:
        con.executemany(f"INSERT OR REPLACE INTO decisions ({names}) VALUES ({', '.join('?' * len(cols))})",
                        df.select(cols).iter_rows())

    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp = SCRATCH / f"{label}.decisions.parquet"
    df.write_parquet(tmp)  # build_chunks reads a path and joins the upstream paragraph table
    try:
        chunks = chunking.build_chunks(tmp)
    finally:
        tmp.unlink(missing_ok=True)
    first, n = add_chunks(con, chunks)
    index_keywords(con, fts, list(range(first, first + n)))
    return df.height, n


# ── which decisions ─────────────────────────────────────────────────────
def select_decisions(con: sqlite3.Connection, since: int | None, fraction: float | None, seed: int) -> int:
    """Choose the decisions to index and remember them. A stratified sample (sampling.py) of the
    decisions from `since` on; the selection is stored so reruns and deltas index the same set.
    Returns the number selected; 0 means no restriction."""
    have = con.execute("SELECT COUNT(*) FROM selected_decisions").fetchone()[0]
    if have:
        print(f"selection: {have:,} decisions (chosen earlier: {state(con, 'selection')})")
        return have
    if since is None and fraction is None:
        return 0
    where = None
    if since is not None:
        where = pl.col("decision_date").str.slice(0, 4).cast(pl.Int32, strict=False) >= since
    subset, pool = S.sample(0, floor=3, seed=seed, where=where, fraction=fraction or 1.0)
    with con:
        con.executemany("INSERT OR IGNORE INTO selected_decisions VALUES (?)",
                        [[d] for d in subset["decision_id"]])
    params = json.dumps({"since": since, "fraction": fraction, "seed": seed, "pool": pool.height})
    state(con, "selection", params)
    print(f"selection: {subset.height:,} of {pool.height:,} decisions ({params})")
    return subset.height


# ── laws ────────────────────────────────────────────────────────────────
LAW_COLS = ["law_id", "canton", "sr_number", "abbreviation", "article_num", "seq", "language", "law_title",
            "heading", "category", "text", "original_url"]


def law_frame(df: pl.DataFrame) -> pl.DataFrame:
    """One schema for both exports. The federal file has no canton, seq or is_active and calls its
    source `work_uri`; the cantonal files carry is_active as 0/1 integers, not booleans."""
    defaults = {"canton": ("CH", pl.String), "seq": (0, pl.Int64), "abbreviation": (None, pl.String),
                "law_title": (None, pl.String), "heading": (None, pl.String), "category": (None, pl.String),
                "original_url": (None, pl.String)}
    df = df.with_columns([pl.lit(v, dtype=t).alias(c) for c, (v, t) in defaults.items() if c not in df.columns])
    if "work_uri" in df.columns:
        df = df.with_columns(original_url=pl.coalesce("original_url", "work_uri"))
    # cantonal article numbers carry markup - "38<sup>bis</sup>\xa0<strong>*</strong>" - which would
    # defeat a lookup by article number; keep the tag contents ("38bis"), drop the asterisk marker
    df = df.with_columns(article_num=pl.col("article_num").cast(pl.String).str.replace_all(r"<[^>]+>", "")
                         .str.replace_all(r"[* ]", " ").str.strip_chars())
    if "is_active" in df.columns:
        df = df.filter(pl.col("is_active").cast(pl.Boolean).fill_null(True))
    df = df.filter(pl.col("text").is_not_null() & (pl.col("text").str.strip_chars() != ""))
    df = df.with_columns(law_id=pl.format("law_{}_{}_{}_{}_{}", pl.col("canton").fill_null("CH"), "sr_number",
                                          pl.col("article_num").fill_null(""), "language",
                                          pl.col("seq").fill_null(0)))
    # keep file order: chunk ids are assigned in append order, so a rebuild gets the same ids
    return df.unique(subset="law_id", keep="first", maintain_order=True).select(LAW_COLS)


def law_chunks(rows: pl.DataFrame) -> pl.DataFrame:
    """Statute articles as passages with the same columns as decision chunks; offsets into laws.text."""
    out = []
    for r in rows.iter_rows(named=True):
        text = r["text"]
        for i, (s, e) in enumerate(chunking.split_text(text)):
            if piece := text[s:e].strip():
                out.append({"chunk_id": f"{r['law_id']}#{i}", "decision_id": r["law_id"], "chunk_index": i,
                            "section": "law", "language": r["language"], "char_start": s, "char_end": e,
                            "erwaegungen": [], "text": piece})
    if not out:
        return pl.DataFrame()
    return pl.DataFrame(out, schema_overrides={"char_start": pl.Int32, "char_end": pl.Int32,
                                              "erwaegungen": pl.List(pl.String)})


def ingest_laws(con: sqlite3.Connection, fts: sqlite3.Connection) -> tuple[int, int]:
    """Every article of the legislation dataset, one parquet per jurisdiction, skipped when unchanged."""
    root = Path(snapshot_download(LAWS_REPO, repo_type="dataset"))
    files = sorted(root.glob("**/*.parquet"))
    n_rows = n_chunks = 0
    for i, f in enumerate(files, 1):
        rel = f"{LAWS_REPO}/{f.relative_to(root)}"
        sha = _sha256(f)
        row = con.execute("SELECT sha256 FROM index_sources WHERE path = ?", [rel]).fetchone()
        if row and row[0] == sha:
            print(f"[laws {i}/{len(files)}] {f.relative_to(root)} unchanged")
            continue
        t0 = time.time()
        df = law_frame(pl.read_parquet(f))
        # replaced articles lose their old passages, as decisions do
        con.execute("CREATE TEMP TABLE IF NOT EXISTS _incoming (decision_id TEXT PRIMARY KEY)")
        con.execute("DELETE FROM _incoming")
        con.executemany("INSERT OR IGNORE INTO _incoming VALUES (?)", [[x] for x in df["law_id"]])
        replaced = [r[0] for r in con.execute(
            "SELECT l.law_id FROM laws l JOIN _incoming i ON i.decision_id = l.law_id")]
        drop_decisions(con, fts, replaced)
        with con:
            con.executemany(f"INSERT OR REPLACE INTO laws ({', '.join(LAW_COLS)}) "
                            f"VALUES ({', '.join('?' * len(LAW_COLS))})", df.iter_rows())
        first, n = add_chunks(con, law_chunks(df))
        index_keywords(con, fts, list(range(first, first + n)))
        with con:
            con.execute("INSERT OR REPLACE INTO index_sources VALUES (?, ?, ?, ?)", [rel, sha, df.height, _utc()])
        print(f"[laws {i}/{len(files)}] {f.relative_to(root)}: {df.height:,} articles, {n:,} passages "
              f"({time.time() - t0:.0f}s)")
        n_rows, n_chunks = n_rows + df.height, n_chunks + n
    return n_rows, n_chunks


# ── embeddings ──────────────────────────────────────────────────────────
def embed(con: sqlite3.Connection, model: str, device: str, batch: int) -> int:
    """Drain pending_embeddings. Committed batch by batch, so an interrupted run loses one batch."""
    todo = con.execute("SELECT COUNT(*) FROM pending_embeddings").fetchone()[0]
    if not todo:
        return 0
    enc = R.load_encoder(model, device)
    dim = V.encode_passages(enc, ["dimension probe"]).shape[1]
    table = V.vec_table(model)
    cols = list(V.VEC_FILTERS)
    with con:
        con.execute(f"""CREATE VIRTUAL TABLE IF NOT EXISTS {table} USING vec0(
            chunk_rowid INTEGER PRIMARY KEY,
            embedding float[{dim}] distance_metric=cosine,
            {", ".join(f"{c} {t}" for c, t in V.VEC_FILTERS.items())})""")
        con.execute("""CREATE TABLE IF NOT EXISTS embedding_models (
            model TEXT PRIMARY KEY, model_name TEXT, dim INTEGER, vec_table TEXT,
            n_embedded INTEGER, n_chunks INTEGER, source TEXT, updated_at TEXT)""")
    name = R.DENSE_MODELS[model].get("name") or R.DENSE_MODELS[model].get("model")
    insert = (f"INSERT INTO {table}(chunk_rowid, embedding, {', '.join(cols)}) "
              f"VALUES (?, ?, {', '.join('?' * len(cols))})")
    done, t0 = 0, time.time()
    while True:
        # filter columns come from the decision, or from the law when the passage is an article
        d = V._exists(con, "decisions")
        rows = con.execute(
            f"""SELECT c.id, c.text, c.language, c.section,
                       COALESCE({'d.canton' if d else 'NULL'}, l.canton), {'d.court' if d else 'NULL'},
                       COALESCE({'d.branch' if d else 'NULL'}, CASE WHEN l.law_id IS NOT NULL THEN 'law' END),
                       {'d.decision_date' if d else 'NULL'}
                FROM pending_embeddings p JOIN chunks c ON c.id = p.id
                {'LEFT JOIN decisions d ON d.decision_id = c.decision_id' if d else ''}
                LEFT JOIN laws l ON l.law_id = c.decision_id LIMIT ?""", [batch]).fetchall()
        if not rows:
            break
        # explicit dtypes: a batch of statute articles has decision_date NULL in every row, and
        # polars would infer dtype Null, which period_expr()'s .str.slice() cannot take
        frame = pl.DataFrame(rows, orient="row", schema={
            "id": pl.Int64, "text": pl.String, "language": pl.String, "section": pl.String,
            "canton": pl.String, "court": pl.String, "branch": pl.String, "decision_date": pl.String,
        }).with_columns(
            branch=pl.col("branch").fill_null("unknown"),
            jurisdiction=pl.when(pl.col("canton") == "CH").then(pl.col("court")).otherwise(pl.col("canton")),
            period=period_expr(),
            year=pl.col("decision_date").str.slice(0, 4).cast(pl.Int64, strict=False).fill_null(0),
        ).with_columns(pl.col(c).fill_null("") for c, t in V.VEC_FILTERS.items() if t == "text")
        vecs = V.encode_passages(enc, frame["text"].to_list()).astype("float32")
        with con:
            con.executemany(insert, ((int(i), v.tobytes(), *meta) for i, v, meta
                                     in zip(frame["id"], vecs, frame.select(cols).iter_rows())))
            con.executemany("DELETE FROM pending_embeddings WHERE id = ?",
                            [[int(i)] for i in frame["id"]])
            total = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            embedded = con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            con.execute("INSERT OR REPLACE INTO embedding_models VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        [model, name, dim, table, embedded, total, "computed", _utc()])
        done += frame.height
        rate = done / max(time.time() - t0, 1e-6)
        print(f"  embedded {done:,}/{todo:,}  ({rate:,.0f} chunks/s, ~{(todo - done) / rate / 3600:.1f} h left)",
              flush=True)
    return done


# ── commands ────────────────────────────────────────────────────────────
def build(courts: list[str] | None, model: str, device: str, batch: int, limit: int | None,
          skip_embed: bool, since: int | None = None, fraction: float | None = None, seed: int = 42,
          laws: bool = False, only_laws: bool = False) -> None:
    con, fts = open_db(), open_fts()
    if laws:
        n_rows, n_ch = ingest_laws(con, fts)
        print(f"laws: {n_rows:,} new or changed articles, {n_ch:,} passages")
    shards = [] if only_laws else shard_files(courts)[: limit or None]
    if shards:
        select_decisions(con, since, fraction, seed)
    print(f"{len(shards)} shards to consider")
    ingested = 0
    for i, repo_path in enumerate(shards, 1):
        local = fetch(repo_path)
        sha = _sha256(local)
        row = con.execute("SELECT sha256 FROM index_sources WHERE path = ?", [repo_path]).fetchone()
        if row and row[0] == sha:
            print(f"[{i}/{len(shards)}] {repo_path} unchanged")
            continue
        t0 = time.time()
        n_dec, n_chunks = ingest(con, fts, local, Path(repo_path).stem)
        with con:
            con.execute("INSERT OR REPLACE INTO index_sources VALUES (?, ?, ?, ?)",
                        [repo_path, sha, n_dec, _utc()])
        ingested += 1
        print(f"[{i}/{len(shards)}] {repo_path}: {n_dec:,} decisions, {n_chunks:,} passages "
              f"({time.time() - t0:.0f}s)")
    if shards:
        state(con, "snapshot_date", manifest()["snapshot"]["date"])
    if ingested:
        # A shard holds the snapshot, which predates every delta: decisions that a delta had already
        # updated just went back in time. Clear the watermark so `update` replays them.
        state(con, "delta_date", "")
    if not skip_embed:
        embed(con, model, device, batch)
    con.close()
    fts.close()
    print(f"done: {DB} ({DB.stat().st_size / 1e9:.1f} GB), {FTS_DB} ({FTS_DB.stat().st_size / 1e9:.1f} GB)")


def grow(fraction: float, model: str, device: str, batch: int, skip_embed: bool,
         since: int | None = None, all_years: bool = False) -> None:
    """Enlarge the sample to `fraction` of the pool and index only the decisions that are new.
    The sample ranks decisions inside each stratum by a seeded hash, so with the same `since` and seed
    a larger fraction contains the smaller one: nothing already indexed is chunked or embedded again.
    Lowering `since` (or `all_years`) widens the pool; at fraction 1.0 that is simply every decision.
    Resumable: a decision counts as done once its passages are in, and embedding drains the queue."""
    con, fts = open_db(), open_fts()
    params = json.loads(state(con, "selection") or "null") or {}
    old_since = params.get("since")
    new_since = None if all_years else (since if since is not None else old_since)
    wider = old_since is not None and (new_since is None or new_since < old_since)
    if not params or fraction < (params.get("fraction") or 1.0) or (
            fraction == params.get("fraction") and not wider):
        raise SystemExit(f"grow needs a larger fraction or an earlier year than the selection {params}")
    if wider and fraction < 1.0:
        raise SystemExit("widening the years keeps the old sample nested only at --fraction 1.0")
    where = None
    if new_since is not None:
        where = pl.col("decision_date").str.slice(0, 4).cast(pl.Int32, strict=False) >= new_since
    subset, pool = S.sample(0, floor=3, seed=params["seed"], where=where, fraction=fraction)
    before = con.execute("SELECT COUNT(*) FROM selected_decisions").fetchone()[0]
    with con:
        con.executemany("INSERT OR IGNORE INTO selected_decisions VALUES (?)",
                        [[d] for d in subset["decision_id"]])
    after = con.execute("SELECT COUNT(*) FROM selected_decisions").fetchone()[0]
    state(con, "selection", json.dumps({**params, "since": new_since, "fraction": fraction, "pool": pool.height}))
    print(f"selection: {before:,} -> {after:,} decisions ({fraction:.0%} of {pool.height:,})")

    done = {r[0] for r in con.execute("SELECT DISTINCT decision_id FROM chunks WHERE section != 'law'")}
    shards = shard_files(None)
    added = 0
    for i, repo_path in enumerate(shards, 1):
        t0 = time.time()
        n_dec, n_chunks = ingest(con, fts, fetch(repo_path), Path(repo_path).stem, skip=done)
        added += n_dec
        print(f"[{i}/{len(shards)}] {repo_path}: {n_dec:,} new decisions, {n_chunks:,} passages "
              f"({time.time() - t0:.0f}s)")
    if added:
        state(con, "delta_date", "")  # the new decisions come from the snapshot: let `update` replay deltas
    if not skip_embed:
        embed(con, model, device, batch)
    con.close()
    fts.close()
    print(f"done: {added:,} decisions added; {DB} ({DB.stat().st_size / 1e9:.1f} GB)")


def update(model: str, device: str, batch: int, skip_embed: bool) -> None:
    con, fts = open_db(), open_fts()
    m = manifest()
    watermark = state(con, "delta_date") or state(con, "snapshot_date") or m["snapshot"]["date"]
    todo = sorted((d for d in m["deltas"] if d["date"] > watermark), key=lambda d: d["date"])
    print(f"watermark {watermark}: {len(todo)} deltas to apply")
    for d in todo:
        local = fetch(d["parquet"]["path"])
        if (got := _sha256(local)) != d["parquet"]["sha256"]:
            raise SystemExit(f"{d['parquet']['path']} sha256 {got} != manifest {d['parquet']['sha256']}")
        n_dec, n_chunks = ingest(con, fts, local, f"delta-{d['date']}")
        with con:
            con.execute("INSERT OR REPLACE INTO index_sources VALUES (?, ?, ?, ?)",
                        [d["parquet"]["path"], got, n_dec, _utc()])
        state(con, "delta_date", d["date"])
        print(f"  {d['date']}: {n_dec:,} decisions, {n_chunks:,} passages")
    if not skip_embed:
        embed(con, model, device, batch)
    con.close()
    fts.close()


def citations() -> None:
    """The citation graph is keyed off a parquet of the indexed ids (see citations.index_path)."""
    from swiss_court_assistant import citations as C

    con = open_db()
    ids = pl.DataFrame({"decision_id": [r[0] for r in con.execute("SELECT decision_id FROM decisions")]})
    con.close()
    IDS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    ids.write_parquet(IDS_PARQUET)
    C.build(IDS_PARQUET)


def vectors(model: str) -> None:
    """Re-export the in-memory search matrix (vecmatrix.py) after the vectors changed. The app does not
    use a matrix older than the index: it falls back to sqlite-vec, whose scans take ~11 s. The search
    filters (facets.py) are rebuilt with it, and so is the GPU copy (gpuvec.py) where there is one."""
    import sqlite3

    from swiss_court_assistant import facets, gpuvec, vecmatrix

    if not vecmatrix.current(DB, model):
        vecmatrix.export(DB, model)
        con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
        facets.build(DB, model, vecmatrix.VectorMatrix.open(DB, model, con))
        con.close()
    else:
        print("vector matrix is current")
    if vecmatrix._file(vecmatrix.base(DB, model), ".f16.json").exists() and not gpuvec.current(DB, model):
        gpuvec.export(DB, model)


def status() -> None:
    if not DB.exists():
        print(f"no index at {DB}")
        return
    con = open_db()
    dec = con.execute("SELECT COUNT(*) FROM decisions").fetchone()[0] if V._exists(con, "decisions") else 0
    ch = con.execute("SELECT COUNT(*) FROM chunks").fetchone()[0] if V._exists(con, "chunks") else 0
    pend = con.execute("SELECT COUNT(*) FROM pending_embeddings").fetchone()[0]
    src = con.execute("SELECT COUNT(*) FROM index_sources").fetchone()[0]
    print(f"{DB} ({DB.stat().st_size / 1e9:.1f} GB)")
    print(f"  {dec:,} decisions, {ch:,} passages, {src} upstream files indexed")
    print(f"  snapshot {state(con, 'snapshot_date')}, deltas applied through {state(con, 'delta_date')}")
    print(f"  {pend:,} passages waiting for a vector")
    if V._exists(con, "embedding_models"):
        for m, n, total, dim, at in con.execute(
                "SELECT model, n_embedded, n_chunks, dim, updated_at FROM embedding_models"):
            print(f"  vec_{m}: {n:,}/{total:,} chunks, dim {dim} ({at})")
    if FTS_DB.exists():
        print(f"  keyword index {FTS_DB.stat().st_size / 1e9:.1f} GB")
    con.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", default="nemotron-embed", choices=list(R.DENSE_MODELS))
        p.add_argument("--device", default="cuda:1", help="only for local sentence-transformers models")
        p.add_argument("--batch", type=int, default=2048, help="passages per embedding batch and commit")
        p.add_argument("--skip-embed", action="store_true", help="index text now, embed in a later run")

    b = sub.add_parser("build", help="index the upstream shards (resumable, skips unchanged files)")
    b.add_argument("--courts", help="comma-separated shard names, e.g. bger,bge (default: all)")
    b.add_argument("--limit", type=int, help="only the first N shards, to try the pipeline out")
    b.add_argument("--since", type=int, help="only decisions from this year on, e.g. 1980")
    b.add_argument("--fraction", type=float, help="stratified sample of the decisions, e.g. 0.25")
    b.add_argument("--seed", type=int, default=42, help="sampling seed")
    b.add_argument("--laws", action="store_true", help="also index every statute article")
    b.add_argument("--only-laws", action="store_true", help="statute articles, no decisions")
    common(b)
    g = sub.add_parser("grow", help="enlarge the stratified sample and index only the new decisions")
    g.add_argument("--fraction", type=float, required=True, help="new share of the pool, e.g. 0.43")
    g.add_argument("--since", type=int, help="widen the pool to decisions from this year on (needs 1.0)")
    g.add_argument("--all-years", action="store_true", help="every decision, whatever its date (needs 1.0)")
    common(g)
    u = sub.add_parser("update", help="apply the dataset's dated deltas since the watermark")
    common(u)
    sub.add_parser("citations", help="rebuild the citation graph over the indexed decisions")
    sub.add_parser("status", help="what is indexed")

    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up in a redirected log as it happens
    if args.cmd == "build":
        build(args.courts.split(",") if args.courts else None, args.model, args.device, args.batch,
              args.limit, args.skip_embed, args.since, args.fraction, args.seed,
              args.laws or args.only_laws, args.only_laws)
        if not args.skip_embed:
            vectors(args.model)
    elif args.cmd == "grow":
        grow(args.fraction, args.model, args.device, args.batch, args.skip_embed, args.since, args.all_years)
        if not args.skip_embed:
            vectors(args.model)  # restart the app afterwards: it opens the matrix at startup
            citations()
    elif args.cmd == "update":
        update(args.model, args.device, args.batch, args.skip_embed)
        if not args.skip_embed:
            vectors(args.model)  # restart the app afterwards: it opens the matrix at startup
    elif args.cmd == "citations":
        citations()
    else:
        status()


if __name__ == "__main__":
    main()
