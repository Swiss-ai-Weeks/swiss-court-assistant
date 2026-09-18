"""The index's vectors as one matrix in memory, for exact search in a fraction of sqlite-vec's time.

sqlite-vec's vec0 search is a brute-force scan that reads every stored vector whatever the filter.
On the full-corpus index (4.9M passages x 2048 dims, 40 GB) one query takes ~11 s, and scans run in
parallel only compete for memory bandwidth (eight year-range shards: 25 s). The same brute force as a
BLAS matrix-vector product over a matrix held in memory reads only the rows it needs: the rows are
sorted by kind (decision passage or statute article) and language, so a search in German decisions
touches the German decision slice and nothing else.

The index stores unit-length vectors for cosine distance, so the dot product ranks exactly as
sqlite-vec does — same passages, same order.

The export reads vec0's own storage tables (1,024 vectors per block) instead of the virtual table,
which returns about 1,300 vectors a second and would take an hour.

    uv run python -m swiss_court_assistant.vecmatrix export     # after `index build` or `index update`
    uv run python -m swiss_court_assistant.vecmatrix status

Files next to the index, for DB = data/vectordb/corpus.sqlite and model nemotron-embed:
    corpus.nemotron-embed.f32    float32 rows, row-major
    corpus.nemotron-embed.ids    int64 chunk ids (chunks.id), one per row
    corpus.nemotron-embed.json   dim, segments {"decision/de": [start, end], ...}, and the index state it
                                 was exported from — a matrix older than the index is not used
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

from swiss_court_assistant import vectordb as V

log = logging.getLogger(__name__)

def base(db: Path, model: str) -> Path:
    return db.with_name(f"{db.stem}.{model}")


def _file(path: Path, ext: str) -> Path:
    """`path` plus an extension. Not `with_suffix`: the model name ("nemotron-embed") already looks
    like one, and would be replaced."""
    return path.with_name(path.name + ext)


def _state(con: sqlite3.Connection, model: str) -> dict:
    """What the index says about the model's vectors; a matrix exported from another state is stale."""
    row = con.execute("SELECT dim, n_embedded, n_chunks, updated_at FROM embedding_models WHERE model = ?",
                      [model]).fetchone()
    if row is None:
        raise SystemExit(f"no embeddings for {model!r} in this index")
    dim, n_embedded, n_chunks, updated_at = row
    return {"dim": dim, "n_embedded": n_embedded, "n_chunks": n_chunks, "updated_at": updated_at}


def export(db: Path, model: str) -> None:
    t0 = time.time()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    state = _state(con, model)
    dim, table = state["dim"], V.vec_table(model)

    # the row order: decision passages, then statute articles, each by language, then by id
    print("reading passage kinds and languages …")
    meta = np.array(con.execute("SELECT id, section = 'law', language FROM chunks ORDER BY id").fetchall(),
                    dtype=object)
    ids = meta[:, 0].astype(np.int64)
    kinds = np.where(meta[:, 1].astype(bool), "law", "decision")
    languages = meta[:, 2].astype(str)
    keys = np.char.add(np.char.add(kinds.astype(str), "/"), languages)
    order = np.lexsort((ids, keys))
    ids, keys = ids[order], keys[order]
    segments: dict[str, list[int]] = {}
    for key in dict.fromkeys(keys.tolist()):
        where = np.flatnonzero(keys == key)
        segments[key] = [int(where[0]), int(where[-1]) + 1]
    position = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
    position[ids] = np.arange(len(ids))

    out = base(db, model)
    vectors = np.memmap(_file(out, ".f32.tmp"), dtype=np.float32, mode="w+", shape=(len(ids), dim))
    filled = np.zeros(len(ids), dtype=bool)
    n_blocks = con.execute(f"SELECT COUNT(*) FROM {table}_chunks").fetchone()[0]
    blocks = con.execute(f"""SELECT c.rowids, c.validity, v.vectors FROM {table}_chunks c
                             JOIN {table}_vector_chunks00 v ON v.rowid = c.chunk_id ORDER BY c.chunk_id""")
    for i, (rowids, validity, blob) in enumerate(blocks, 1):
        slots = np.frombuffer(rowids, dtype=np.int64)
        valid = np.unpackbits(np.frombuffer(validity, dtype=np.uint8), bitorder="little")[:len(slots)].astype(bool)
        block = np.frombuffer(blob, dtype=np.float32).reshape(-1, dim)
        chunk_ids = slots[valid]
        known = chunk_ids < len(position)
        rows = position[chunk_ids[known]]
        keep = rows >= 0  # a vector whose passage is gone (should not happen) is skipped
        vectors[rows[keep]] = block[valid][known][keep]
        filled[rows[keep]] = True
        if i % 500 == 0 or i == n_blocks:
            print(f"  {i:,}/{n_blocks:,} blocks ({time.time() - t0:.0f} s)")
    if not filled.all():
        missing = int((~filled).sum())
        raise SystemExit(f"{missing:,} passages have no vector yet; finish `index build` first")
    vectors.flush()
    del vectors
    ids.tofile(_file(out, ".ids.tmp"))
    _file(out, ".f32.tmp").replace(_file(out, ".f32"))
    _file(out, ".ids.tmp").replace(_file(out, ".ids"))
    _file(out, ".json").write_text(json.dumps(
        {"model": model, "rows": len(ids), "segments": segments, "index": state}, indent=1))
    con.close()
    print(f"done: {len(ids):,} vectors in {time.time() - t0:.0f} s -> {out}.f32")
    for key, (a, b) in segments.items():
        print(f"  {key:12s} {b - a:>10,}")


class VectorMatrix:
    """Exact nearest-neighbour search over the exported matrix. Thread-safe: it only reads."""

    def __init__(self, path: Path, meta: dict):
        self.meta = meta
        dim = meta["index"]["dim"]
        # memory-mapped: the first searches page it in, after that it is served from memory
        self.vectors = np.memmap(_file(path, ".f32"), dtype=np.float32, mode="r",
                                 shape=(meta["rows"], dim))
        self.ids = np.fromfile(_file(path, ".ids"), dtype=np.int64)
        self.segments = {k: tuple(v) for k, v in meta["segments"].items()}

    @classmethod
    def open(cls, db: Path, model: str, con: sqlite3.Connection) -> VectorMatrix | None:
        """The matrix, or None when it is missing or older than the index (the caller falls back to
        sqlite-vec, which is slow but always current)."""
        path = base(db, model)
        if not _file(path, ".json").exists():
            return None
        meta = json.loads(_file(path, ".json").read_text())
        try:
            now = _state(con, model)
        except SystemExit:
            return None
        if meta["index"] != now:
            log.warning("vector matrix %s is stale (exported from %s, index is at %s); run "
                        "`python -m swiss_court_assistant.vecmatrix export`", path, meta["index"], now)
            return None
        return cls(path, meta)

    def warm(self) -> None:
        """Read the whole matrix once so the first question does not pay for paging it in."""
        step = 1 << 16
        for a in range(0, len(self.ids), step):
            self.vectors[a:a + step].sum()

    def search(self, query: np.ndarray, k: int, kind: str = "decision",
               language: str | None = None) -> list[tuple[int, float]]:
        """(chunk id, cosine similarity) of the k nearest rows, best first."""
        q = np.asarray(query, dtype=np.float32)
        spans = [span for key, span in self.segments.items()
                 if key.startswith(f"{kind}/") and (language is None or key == f"{kind}/{language}")]
        best: list[tuple[int, float]] = []
        for a, b in spans:
            scores = self.vectors[a:b] @ q
            top = np.argpartition(-scores, min(k, len(scores) - 1))[:k] if len(scores) > k else np.arange(len(scores))
            best += [(int(self.ids[a + i]), float(scores[i])) for i in top]
        return sorted(best, key=lambda x: -x[1])[:k]


def current(db: Path, model: str) -> bool:
    """Whether the exported matrix matches the index as it is now."""
    path = _file(base(db, model), ".json")
    if not path.exists():
        return False
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        return json.loads(path.read_text())["index"] == _state(con, model)
    finally:
        con.close()


def status(db: Path, model: str) -> None:
    path = base(db, model)
    if not _file(path, ".json").exists():
        print(f"no matrix at {path}.f32")
        return
    meta = json.loads(_file(path, ".json").read_text())
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    fresh = meta["index"] == _state(con, model)
    print(f"{path}.f32: {meta['rows']:,} x {meta['index']['dim']} ({'current' if fresh else 'STALE'})")
    for key, (a, b) in meta["segments"].items():
        print(f"  {key:12s} {b - a:>10,}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["export", "status"])
    ap.add_argument("--db", type=Path, default=V.DB_DIR / "corpus.sqlite")
    ap.add_argument("--model", default="nemotron-embed")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # progress shows up in a redirected log as it happens
    (export if args.command == "export" else status)(args.db, args.model)


if __name__ == "__main__":
    main()
