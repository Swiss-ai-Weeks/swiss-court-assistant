from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any

import httpx

from swiss_court_assistant import retrieval as R
from swiss_court_assistant.fts import KeywordIndex
from swiss_court_assistant.statutes import localize
from swiss_court_assistant.vecmatrix import VectorMatrix
from swiss_court_assistant.vectordb import VectorDB

from .citations import CitationIndex
from .decisions import DecisionStore, SqliteDecisionStore

log = logging.getLogger(__name__)

_CHUNK_COLS = "id, chunk_id, decision_id, section, erwaegungen, char_start, char_end, text"


@dataclass
class Passage:
    chunk_id: str
    decision_id: str
    section: str
    erwaegungen: list[str]
    char_start: int | None
    char_end: int | None
    text: str
    docket: str
    court_label: str
    date: str | None
    language: str
    score: float = 0.0
    cited_by: int = 0  # later decisions citing this one, from the citation graph
    kind: str = "decision"  # or "law": a statute article, whose law_id is in decision_id


class Corpus:
    """Search and lookup for the agent tools.

    The vector DB connection and the query encoder are not thread-safe, so both live
    on one worker thread and the async methods hop onto it.
    """

    def __init__(self, db: Path, fts: Path, decisions: DecisionStore | SqliteDecisionStore,
                 embed_model: str | None = None,
                 device: str = "cpu", rerank: bool = True, candidates: int = 40,
                 citations: "CitationIndex | None" = None):
        self.decisions = decisions
        self.candidates = candidates
        self.citations = citations  # how often each decision is cited: an authority signal on results
        cfg = R.RERANKERS["nemotron"]
        self.rerank_url = os.environ.get(cfg["url_env"], cfg["url"]) if rerank else None
        self.rerank_model = cfg["model"]
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="corpus")
        self._pool.submit(self._open, db, fts, embed_model, device).result()
        # A KNN query scans every vector (~2 s); with a connection per thread the languages scan in parallel
        self._local = threading.local()
        self._knn_pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="knn",
                                            initializer=self._open_knn, initargs=(db,))

    def _open(self, db: Path, fts: Path, embed_model: str | None, device: str) -> None:
        self.vdb = VectorDB(db, device)
        models = {m["model"]: m for m in self.vdb.models()}
        complete = [m for m, r in models.items() if r["n_embedded"] == r["n_chunks"]]
        if embed_model is None:
            embed_model = next((m for m in ("nemotron-embed", "bge-m3") if m in complete), None)
            embed_model = embed_model or (complete[0] if complete else next(iter(models), None))
        if embed_model not in models:
            raise RuntimeError(f"no embeddings for {embed_model!r} in {db}; have {list(models)}")
        m = models[embed_model]
        if m["n_embedded"] < m["n_chunks"]:
            log.warning("%s covers only %d/%d chunks", embed_model, m["n_embedded"], m["n_chunks"])
        self.embed_model = embed_model
        self.vdb.search("Aufwärmen", embed_model, k=1)  # load the encoder now, not on the first question
        # the vectors as a matrix in memory, when exported: sqlite-vec reads all of them on every query
        self.matrix = VectorMatrix.open(db, embed_model, self.vdb.con)
        if self.matrix is not None:
            threading.Thread(target=self.matrix.warm, name="warm-vectors", daemon=True).start()
        else:
            log.info("no current vector matrix for %s; searching with sqlite-vec", db)
        # statute articles: indexed with `index.py build --laws`, searchable by meaning through the matrix
        self.has_laws = (self.matrix is not None and any(k.startswith("law/") for k in self.matrix.segments)
                         and hasattr(self.decisions, "find_articles"))
        self.n_articles = (self.vdb.con.execute("SELECT COUNT(*) FROM laws").fetchone()[0]
                           if self.has_laws else 0)
        self.fts = KeywordIndex(fts) if fts.exists() else None
        if self.fts is None:
            log.warning("no keyword index at %s; run `python -m swiss_court_assistant.fts build`", fts)
        log.info("corpus: embeddings=%s (%s), vector matrix=%s, keyword index=%s", embed_model, device,
                 bool(self.matrix), bool(self.fts))

    def _open_knn(self, db: Path) -> None:
        self._local.vdb = VectorDB(db)  # no encoder: it gets query vectors

    def _knn(self, vector, language: str | None) -> list[dict]:
        """The nearest decision passages, optionally in one language, with a `similarity` each."""
        if self.matrix is None:
            return self._local.vdb.search(vector, self.embed_model, self.candidates, language=language)
        hits = self.matrix.search(vector, self.candidates, "decision", language)
        if not hits:
            return []
        rows = {r["id"]: dict(r) for r in self._local.vdb.con.execute(
            f"SELECT {_CHUNK_COLS} FROM chunks WHERE id IN ({','.join('?' * len(hits))})", [i for i, _ in hits])}
        return [rows[i] | {"similarity": score} for i, score in hits if i in rows]

    async def _call(self, fn, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self._pool, partial(fn, *args, **kwargs))

    def _passage(self, row: dict, score: float = 0.0) -> Passage | None:
        """A decision passage; None for a statute article, which shares the passage table in the
        full-corpus index but is not a decision."""
        d = self.decisions.summary(row["decision_id"])
        if d is None:
            return None
        erw = row["erwaegungen"]
        return Passage(
            chunk_id=row["chunk_id"], decision_id=row["decision_id"], section=row["section"],
            erwaegungen=json.loads(erw or "[]") if isinstance(erw, str) else erw,
            char_start=row["char_start"], char_end=row["char_end"], text=row["text"],
            docket=d.docket, court_label=d.court_label, date=d.date, language=d.language, score=score,
        )

    def _rows(self, where: str, params: list) -> list[dict]:
        return [dict(r) for r in self.vdb.con.execute(f"SELECT {_CHUNK_COLS} FROM chunks WHERE {where}", params)]

    async def _authority(self, hits: list[Passage]) -> list[Passage]:
        """Annotate each passage with how often its decision is cited by later ones."""
        if self.citations is None or not hits:
            return hits
        counts = await asyncio.to_thread(self.citations.cited_by_counts,
                                         sorted({h.decision_id for h in hits}))
        return [replace(h, cited_by=counts.get(h.decision_id, 0)) for h in hits]

    def _law_passage(self, row: dict, score: float = 0.0) -> Passage | None:
        law = self.decisions.law_summary(row["decision_id"])
        if law is None:
            return None
        return Passage(
            chunk_id=row["chunk_id"], decision_id=row["decision_id"], section="law", erwaegungen=[],
            char_start=row["char_start"], char_end=row["char_end"], text=row["text"], docket=law.docket,
            court_label=law.court_label, date=None, language=law.language, score=score, kind="law")

    def _knn_laws(self, vector, language: str | None, federal: bool) -> list[dict]:
        """The nearest statute passages. Federal law only by default: cantonal acts are nearly half the
        articles, and their procedural and tax rules crowd out the federal provision a question is about."""
        hits = self.matrix.search(vector, self.candidates * (6 if federal else 1), "law", language)
        if not hits:
            return []
        rows = {r["id"]: dict(r) for r in self._local.vdb.con.execute(
            f"""SELECT c.id, c.chunk_id, c.decision_id, c.section, c.erwaegungen, c.char_start, c.char_end,
                       c.text, l.canton FROM chunks c JOIN laws l ON l.law_id = c.decision_id
                WHERE c.id IN ({','.join('?' * len(hits))})""", [i for i, _ in hits])}
        out = [rows[i] | {"similarity": score} for i, score in hits
               if i in rows and (not federal or rows[i]["canton"] == "CH")]
        return out[:self.candidates]

    async def semantic_search_laws(self, queries: dict[str, str], k: int = 6,
                                   federal: bool = True) -> list[Passage]:
        """Statute articles by meaning, one query per language like `semantic_search_by_language`.
        An article exists in three languages; only its best-ranked version is kept."""
        if not self.has_laws:
            raise RuntimeError("this index has no statute articles, or their vectors are not exported")
        queries = {lang: localize(q.strip(), lang) for lang, q in queries.items() if q and q.strip()}
        vectors = [await self._call(self.vdb.encode, q, self.embed_model) for q in queries.values()]
        loop = asyncio.get_running_loop()
        found = await asyncio.gather(*(loop.run_in_executor(self._knn_pool, self._knn_laws, v, lang, federal)
                                       for v, lang in zip(vectors, queries)))
        ranked = await asyncio.gather(*(
            self._rerank(q, [p for r in rows if (p := self._law_passage(r, r["similarity"]))])
            for q, rows in zip(queries.values(), found)))
        best: dict[str, Passage] = {}
        for hit in sorted((h for hits in ranked for h in hits), key=lambda h: -h.score):
            best.setdefault(_article_of(hit.decision_id), hit)
        return list(best.values())[:k]

    async def law_chunk(self, chunk_id: str) -> Passage | None:
        rows = await self._call(self._rows, "chunk_id = ?", [chunk_id])
        return self._law_passage(rows[0]) if rows else None

    async def law_chunks_of(self, law_id: str) -> list[Passage]:
        rows = await self._call(self._rows, "decision_id = ? ORDER BY chunk_index", [law_id])
        return [p for r in rows if (p := self._law_passage(r))]

    def _passages(self, rows: list[dict], scores: list[float] | None = None) -> list[Passage]:
        found = (self._passage(r, r.get("similarity", 0.0) if scores is None else s)
                 for r, s in zip(rows, scores or [0.0] * len(rows)))
        return [p for p in found if p is not None]

    async def semantic_search(self, query: str, k: int = 8, language: str | None = None) -> list[Passage]:
        vector = await self._call(self.vdb.encode, query, self.embed_model)
        rows = await asyncio.get_running_loop().run_in_executor(self._knn_pool, self._knn, vector, language)
        hits = await self._rerank(query, self._passages(rows))
        return await self._authority(_diverse(hits, k))

    async def semantic_search_by_language(self, queries: dict[str, str], k: int = 8,
                                          per_language: int = 2) -> list[Passage]:
        """Search each language's decisions with the query written in that language ({"de": ..., "fr": ...})
        and rerank them against it. The merged result keeps the best ``per_language`` passages of each
        language, then fills up by reranker score."""
        # the agent sometimes writes "art. 336 OR" in the French query; courts write "art. 336 CO"
        queries = {lang: localize(q.strip(), lang) for lang, q in queries.items() if q and q.strip()}
        vectors = [await self._call(self.vdb.encode, q, self.embed_model) for q in queries.values()]
        loop = asyncio.get_running_loop()
        found = await asyncio.gather(*(loop.run_in_executor(self._knn_pool, self._knn, v, lang)
                                       for v, lang in zip(vectors, queries)))
        ranked = await asyncio.gather(*(self._rerank(q, self._passages(rows))
                                        for q, rows in zip(queries.values(), found)))
        picked = [h for hits in ranked for h in _diverse(hits, per_language)]
        rest = sorted((h for hits in ranked for h in hits), key=lambda h: -h.score)
        seen: set[str] = set()
        merged = [h for h in picked + rest if not (h.chunk_id in seen or seen.add(h.chunk_id))]
        return await self._authority(sorted(_diverse(merged, k), key=lambda h: -h.score))

    async def keyword_search(self, keyword: str, k: int = 8) -> list[Passage]:
        if self.fts is None:
            raise RuntimeError("the keyword index is not built")
        # more candidates than needed: in the full-corpus index statute articles match too, and are dropped
        ranked = await self._call(self.fts.search, keyword, k * 8)
        if not ranked:
            return []
        ids = [rowid for rowid, _ in ranked]
        rows = {r["id"]: r for r in await self._call(self._rows, f"id IN ({','.join('?' * len(ids))})", ids)}
        hits = self._passages([rows[i] for i, _ in ranked if i in rows], [-s for i, s in ranked if i in rows])
        return await self._authority(_diverse(hits, k))

    async def chunk(self, chunk_id: str) -> Passage | None:
        rows = await self._call(self._rows, "chunk_id = ?", [chunk_id])
        return self._passage(rows[0]) if rows else None

    async def chunks_of(self, decision_id: str) -> list[Passage]:
        rows = await self._call(self._rows, "decision_id = ? ORDER BY chunk_index", [decision_id])
        return self._passages(rows)

    async def _rerank(self, query: str, hits: list[Passage]) -> list[Passage]:
        if not self.rerank_url or len(hits) < 2:
            return hits
        body = {"model": self.rerank_model, "query": {"text": query},
                "passages": [{"text": h.text} for h in hits], "truncate": "END"}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                r = await client.post(self.rerank_url, json=body)
                r.raise_for_status()
        except httpx.HTTPError as e:
            log.warning("reranker unavailable (%s), keeping vector order", e)
            return hits
        order = sorted(r.json()["rankings"], key=lambda x: -x["logit"])
        return [replace(hits[x["index"]], score=x["logit"]) for x in order]


def _article_of(law_id: str) -> str:
    """The article behind a law_id, whatever its language: law_CH_220_271a_de_0 -> law_CH_220_271a_0."""
    return re.sub(r"_(de|fr|it|rm)_(\d+)$", r"_\2", law_id)


def _diverse(hits: list[Passage], k: int, per_decision: int = 2) -> list[Passage]:
    out, seen = [], {}
    for h in hits:
        if seen.get(h.decision_id, 0) < per_decision:
            out.append(h)
            seen[h.decision_id] = seen.get(h.decision_id, 0) + 1
            if len(out) == k:
                break
    return out
