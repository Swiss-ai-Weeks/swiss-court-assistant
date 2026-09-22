from __future__ import annotations

import logging
import sqlite3
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from swiss_court_assistant.identity import canonical_id

log = logging.getLogger(__name__)


@dataclass
class Citing:
    decision_id: str
    court: str | None
    docket: str | None
    date: str | None
    in_corpus: bool


class CitationIndex:
    """Read-only citation graph, including compatibility with pre-canonicalisation graph builds."""

    def __init__(self, path: Path, known: set[str] | None = None):
        self.path = path
        self.known = {canonical_id(d) for d in (known or set())}
        self._local = threading.local()
        con = self._connect()
        self._twins: dict[str, list[str]] = {}
        for (did,) in con.execute("SELECT decision_id FROM decisions WHERE court = 'bge'"):
            self._twins.setdefault(canonical_id(did), []).append(did)

    def _connect(self) -> sqlite3.Connection:
        con = getattr(self._local, 'con', None)
        if con is None:
            con = sqlite3.connect(f'file:{self.path}?mode=ro', uri=True, check_same_thread=False)
            con.row_factory = sqlite3.Row
            self._local.con = con
        return con

    @lru_cache(maxsize=50000)
    def counts(self, decision_id: str) -> tuple[int, int]:
        cid = canonical_id(decision_id)
        ids = self._twins.get(cid, [cid])
        con = self._connect()
        if len(ids) == 1:
            row = con.execute('SELECT cited_by, cites FROM counts WHERE decision_id = ?', ids).fetchone()
            return (row[0], row[1]) if row else (0, 0)
        # Summing precomputed twin counts double-counts sources that cite both exports. Count
        # distinct canonical neighbours instead, excluding artificial self-edges between twins.
        holes = ','.join('?' * len(ids))
        incoming = {canonical_id(r[0]) for r in con.execute(
            f'SELECT source FROM edges WHERE target IN ({holes})', ids)} - {cid}
        outgoing = {canonical_id(r[0]) for r in con.execute(
            f'SELECT target FROM edges WHERE source IN ({holes})', ids)} - {cid}
        return len(incoming), len(outgoing)

    def cited_by_counts(self, decision_ids: list[str]) -> dict[str, int]:
        return {d: self.counts(d)[0] for d in decision_ids}

    def _edges(self, decision_id: str, column: str, limit: int) -> list[Citing]:
        other = 'source' if column == 'target' else 'target'
        cid = canonical_id(decision_id)
        ids = self._twins.get(cid, [cid])
        holes = ','.join('?' * len(ids))
        rows = self._connect().execute(
            f'''SELECT e.{other} AS id, d.court, d.docket, d.date FROM edges e
                LEFT JOIN decisions d ON d.decision_id = e.{other}
                WHERE e.{column} IN ({holes}) ORDER BY d.date DESC NULLS LAST''', ids)
        out, seen = [], {cid}
        for row in rows:
            did = canonical_id(row['id'])
            if did in seen:
                continue
            seen.add(did)
            out.append(Citing(did, row['court'], row['docket'], row['date'], did in self.known))
            if len(out) >= limit:
                break
        return out[:max(0, limit)]

    def cited_by(self, decision_id: str, limit: int = 20) -> list[Citing]:
        return self._edges(decision_id, 'target', limit)

    def cites(self, decision_id: str, limit: int = 20) -> list[Citing]:
        return self._edges(decision_id, 'source', limit)


def open_index(path: Path, known: set[str] | None = None) -> CitationIndex | None:
    if not path.exists():
        log.warning('no citation index at %s; run `python -m swiss_court_assistant.citations build`', path)
        return None
    try:
        return CitationIndex(path, known)
    except sqlite3.Error:
        log.exception('could not open the citation index at %s', path)
        return None
