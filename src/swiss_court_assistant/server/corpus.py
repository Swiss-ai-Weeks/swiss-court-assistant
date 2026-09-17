from __future__ import annotations

import asyncio
import json
import logging
import os
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
from swiss_court_assistant.vectordb import VectorDB

from .citations import CitationIndex
from .decisions import DecisionStore

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


class Corpus:
    """Search and lookup for the agent tools.

    The vector DB connection and the query encoder are not thread-safe, so both live
    on one worker thread and the async methods hop onto it.
    """

    def __init__(self, db: Path, fts: Path, decisions: DecisionStore, embed_model: str | None = None,
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
        self.fts = KeywordIndex(fts) if fts.exists() else None
        if self.fts is None:
            log.warning("no keyword index at %s; run `python -m swiss_court_assistant.fts build`", fts)
        log.info("corpus: embeddings=%s (%s), keyword index=%s", embed_model, device, bool(self.fts))

    def _open_knn(self, db: Path) -> None:
        self._local.vdb = VectorDB(db)  # no encoder: it gets query vectors

    def _knn(self, vector, language: str) -> list[dict]:
        return self._local.vdb.search(vector, self.embed_model, self.candidates, language=language)

    async def _call(self, fn, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self._pool, partial(fn, *args, **kwargs))

    def _passage(self, row: dict, score: float = 0.0) -> Passage:
        d = self.decisions.summary(row["decision_id"])
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

    async def semantic_search(self, query: str, k: int = 8, language: str | None = None) -> list[Passage]:
        rows = await self._call(self.vdb.search, query, self.embed_model, self.candidates, language=language)
        hits = await self._rerank(query, [self._passage(r, r["similarity"]) for r in rows])
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
        ranked = await asyncio.gather(*(self._rerank(q, [self._passage(r, r["similarity"]) for r in rows])
                                        for q, rows in zip(queries.values(), found)))
        picked = [h for hits in ranked for h in _diverse(hits, per_language)]
        rest = sorted((h for hits in ranked for h in hits), key=lambda h: -h.score)
        seen: set[str] = set()
        merged = [h for h in picked + rest if not (h.chunk_id in seen or seen.add(h.chunk_id))]
        return await self._authority(sorted(_diverse(merged, k), key=lambda h: -h.score))

    async def keyword_search(self, keyword: str, k: int = 8) -> list[Passage]:
        if self.fts is None:
            raise RuntimeError("the keyword index is not built")
        ranked = await self._call(self.fts.search, keyword, k * 4)
        if not ranked:
            return []
        ids = [rowid for rowid, _ in ranked]
        rows = {r["id"]: r for r in await self._call(self._rows, f"id IN ({','.join('?' * len(ids))})", ids)}
        return await self._authority(_diverse([self._passage(rows[i], -s) for i, s in ranked if i in rows], k))

    async def chunk(self, chunk_id: str) -> Passage | None:
        rows = await self._call(self._rows, "chunk_id = ?", [chunk_id])
        return self._passage(rows[0]) if rows else None

    async def chunks_of(self, decision_id: str) -> list[Passage]:
        rows = await self._call(self._rows, "decision_id = ? ORDER BY chunk_index", [decision_id])
        return [self._passage(r) for r in rows]

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


def _diverse(hits: list[Passage], k: int, per_decision: int = 2) -> list[Passage]:
    out, seen = [], {}
    for h in hits:
        if seen.get(h.decision_id, 0) < per_decision:
            out.append(h)
            seen[h.decision_id] = seen.get(h.decision_id, 0) + 1
            if len(out) == k:
                break
    return out
