"""A matter's case file as its own collection in a vector store, for the research to search.

A case file of a dozen letters, contracts and transcripts does not fit in the prompt, and the head of
each document is rarely the part an issue turns on. So every asset is cut into passages the way the
decisions are (`chunking.split_text`, ~1,400 characters on sentence ends), embedded with the same
Nemotron embedder, and kept in one sqlite-vec file, one partition per matter. The research agent then
finds what the case file says about an issue the way it finds case law: by meaning and by exact words,
fused and reranked.

Passages keep their character offsets into the document's stored text, so a hit can be read on with
read_document and cited like any other part of the document.
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import sqlite_vec

from swiss_court_assistant import retrieval as R
from swiss_court_assistant.chunking import split_text

from .parsing import DocumentStore
from .schemas import DocumentInfo

log = logging.getLogger(__name__)

INDEX_PATH = Path(os.environ.get("SCA_CASE_INDEX", "data/app/case_index.sqlite"))
_EMBED = R.DENSE_MODELS["nemotron-embed"]
_RERANK = R.RERANKERS["nemotron"]
EMBED_BATCH = 16
CANDIDATES = 30  # from each of the two searches, before fusion and reranking


class IndexUnavailable(RuntimeError):
    """The embedder could not be reached: the case file is not searchable by meaning."""


@dataclass
class CaseHit:
    document_id: str
    name: str
    page: int | None
    char_start: int
    char_end: int
    text: str
    score: float


def collection_of(matter_id: str) -> str:
    return f"matter_{matter_id}"


def _page_lines(text: str) -> str:
    return re.sub(r"(?m)^\[Page \d+\]\n+", "", text).strip()


class CaseIndex:
    def __init__(self, documents: DocumentStore, path: Path = INDEX_PATH):
        self.documents = documents
        self.embed_url = os.environ.get(_EMBED["url_env"], _EMBED["url"]).split(",")[0].strip()
        self.rerank_url = os.environ.get(_RERANK["url_env"], _RERANK["url"])
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path, check_same_thread=False)
        self.con.enable_load_extension(True)
        sqlite_vec.load(self.con)
        self.con.enable_load_extension(False)
        self.lock = threading.Lock()  # one connection, used from worker threads
        with self.lock, self.con:
            self.con.execute("PRAGMA journal_mode=WAL")
            self.con.execute("""CREATE TABLE IF NOT EXISTS passages (
                id INTEGER PRIMARY KEY, collection TEXT NOT NULL, document_id TEXT NOT NULL,
                name TEXT, page INTEGER, char_start INTEGER, char_end INTEGER, text TEXT NOT NULL)""")
            self.con.execute("CREATE INDEX IF NOT EXISTS passages_doc ON passages(collection, document_id)")
            self.con.execute("CREATE VIRTUAL TABLE IF NOT EXISTS passages_fts USING fts5("
                             "text, content='passages', content_rowid='id', tokenize='unicode61 remove_diacritics 2')")
            self.con.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT)")
        self.dim = self._meta("dim")

    # ── storage ─────────────────────────────────────────────────────────
    def _meta(self, key: str) -> int | None:
        row = self.con.execute("SELECT value FROM meta WHERE key = ?", [key]).fetchone()
        return int(row[0]) if row else None

    def _vectors(self, dim: int) -> None:
        """The vec0 table, created on the first embedding (its width is the embedder's)."""
        if self.dim is None:
            self.con.execute(f"CREATE VIRTUAL TABLE IF NOT EXISTS passages_vec USING vec0(id INTEGER PRIMARY KEY, "
                             f"collection TEXT PARTITION KEY, embedding float[{dim}] distance_metric=cosine)")
            self.con.execute("INSERT OR REPLACE INTO meta VALUES ('dim', ?)", [str(dim)])
            self.dim = dim

    def count(self, collection: str) -> int:
        with self.lock:
            return self.con.execute("SELECT COUNT(*) FROM passages WHERE collection = ?", [collection]).fetchone()[0]

    def indexed(self, collection: str) -> set[str]:
        with self.lock:
            return {r[0] for r in self.con.execute(
                "SELECT DISTINCT document_id FROM passages WHERE collection = ?", [collection])}

    def drop(self, collection: str) -> None:
        with self.lock, self.con:
            ids = [r[0] for r in self.con.execute("SELECT id FROM passages WHERE collection = ?", [collection])]
            for i in range(0, len(ids), 500):
                part = ids[i:i + 500]
                marks = ",".join("?" * len(part))
                self.con.execute(f"DELETE FROM passages_fts WHERE rowid IN ({marks})", part)
                if self.dim is not None:
                    self.con.execute(f"DELETE FROM passages_vec WHERE id IN ({marks})", part)
            self.con.execute("DELETE FROM passages WHERE collection = ?", [collection])

    # ── models ──────────────────────────────────────────────────────────
    def _embed(self, texts: list[str], input_type: str) -> np.ndarray:
        out = []
        try:
            with httpx.Client(base_url=self.embed_url, timeout=120) as client:
                for i in range(0, len(texts), EMBED_BATCH):
                    r = client.post("/embeddings", json={
                        "model": _EMBED["model"], "input": texts[i:i + EMBED_BATCH],
                        "input_type": input_type, "truncate": "END"})
                    r.raise_for_status()
                    out.extend(d["embedding"] for d in sorted(r.json()["data"], key=lambda d: d["index"]))
        except httpx.HTTPError as e:
            raise IndexUnavailable(f"the embedder at {self.embed_url} is unavailable: {e}") from e
        emb = np.asarray(out, dtype=np.float32)
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)

    def _rerank(self, query: str, hits: list[CaseHit]) -> list[CaseHit]:
        if len(hits) < 2:
            return hits
        try:
            r = httpx.post(self.rerank_url, timeout=30, json={
                "model": _RERANK["model"], "query": {"text": query},
                "passages": [{"text": f"{h.name}\n{h.text}"} for h in hits], "truncate": "END"})
            r.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("reranker unavailable (%s), keeping the fused order", e)
            return hits
        order = sorted(r.json()["rankings"], key=lambda x: -x["logit"])
        return [CaseHit(**{**hits[x["index"]].__dict__, "score": x["logit"]}) for x in order]

    # ── building ────────────────────────────────────────────────────────
    def add(self, collection: str, assets: list[DocumentInfo]) -> int:
        """Index the assets not yet in the collection; returns how many passages were added. Blocking."""
        done = self.indexed(collection)
        rows: list[tuple[DocumentInfo, int | None, int, int, str]] = []
        for d in assets:
            if d.id in done or not (text := self.documents.text(d.id)):
                continue
            for start, end in split_text(text):
                passage = text[start:end]
                if _page_lines(passage):
                    rows.append((d, DocumentStore.page_at(text, start), start, end, passage))
        if not rows:
            return 0
        # the document's name tells the embedder what a bare paragraph is ("Lease agreement", "Notice")
        vectors = self._embed([f"{d.name}\n{_page_lines(p)}" for d, _, _, _, p in rows], "passage")
        with self.lock, self.con:
            self._vectors(vectors.shape[1])
            for (d, page, start, end, passage), v in zip(rows, vectors, strict=True):
                cur = self.con.execute(
                    "INSERT INTO passages (collection, document_id, name, page, char_start, char_end, text) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)", [collection, d.id, d.name, page, start, end, passage])
                self.con.execute("INSERT INTO passages_fts (rowid, text) VALUES (?, ?)", [cur.lastrowid, passage])
                self.con.execute("INSERT INTO passages_vec (id, collection, embedding) VALUES (?, ?, ?)",
                                 [cur.lastrowid, collection, v.tobytes()])
        log.info("case index %s: %d passages from %d documents", collection, len(rows), len({r[0].id for r in rows}))
        return len(rows)

    # ── searching ───────────────────────────────────────────────────────
    def _hits(self, ids: list[int]) -> dict[int, CaseHit]:
        if not ids:
            return {}
        rows = self.con.execute(f"SELECT id, document_id, name, page, char_start, char_end, text FROM passages "
                                f"WHERE id IN ({','.join('?' * len(ids))})", ids)
        return {r[0]: CaseHit(r[1], r[2], r[3], r[4], r[5], r[6], 0.0) for r in rows}

    def search(self, collection: str, query: str, k: int = 6) -> list[CaseHit]:
        """The passages of the case file that answer the query: by meaning and by exact words, fused by
        reciprocal rank and reranked. Blocking."""
        vector = self._embed([query], "query")[0] if self.dim is not None else None
        words = [w for w in re.findall(r"\w+", query) if len(w) >= 3]
        with self.lock:
            semantic = [r[0] for r in self.con.execute(
                "SELECT id FROM passages_vec WHERE embedding MATCH ? AND k = ? AND collection = ? ORDER BY distance",
                [vector.tobytes(), CANDIDATES, collection])] if vector is not None else []
            keyword = [r[0] for r in self.con.execute(
                "SELECT p.id FROM passages_fts f JOIN passages p ON p.id = f.rowid "
                "WHERE passages_fts MATCH ? AND p.collection = ? ORDER BY bm25(passages_fts) LIMIT ?",
                [" OR ".join(f'"{w}"' for w in words), collection, CANDIDATES])] if words else []
            fused: dict[int, float] = {}
            for ranking in (semantic, keyword):
                for rank, i in enumerate(ranking):
                    fused[i] = fused.get(i, 0.0) + 1 / (60 + rank)
            best = sorted(fused, key=lambda i: -fused[i])[:max(k * 3, 12)]
            hits = self._hits(best)
        return self._rerank(query, [hits[i] for i in best if i in hits])[:k]
