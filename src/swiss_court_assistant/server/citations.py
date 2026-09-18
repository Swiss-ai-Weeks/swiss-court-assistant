from __future__ import annotations

import logging
import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)


def _key(docket: str | None) -> str:
    """Docket number without the reporter prefix or separators: "BGE 122 V 157" == "122 V 157"."""
    if not docket:
        return ""
    plain = docket.upper()
    for prefix in ("BGE ", "ATF ", "DTF "):
        plain = plain.replace(prefix, "")
    return re.sub(r"[^A-Z0-9]", "", plain)


@dataclass
class Citing:
    """A decision at the other end of a citation edge."""

    decision_id: str
    court: str | None
    docket: str | None
    date: str | None
    in_corpus: bool  # in the subset this app serves, so it can be opened here


class CitationIndex:
    """How often a decision is cited by later ones, and by which — from the corpus citation graph
    (`python -m swiss_court_assistant.citations build`). Read-only, one connection per thread."""

    def __init__(self, path: Path, known: set[str] | None = None):
        self.path = path
        self.known = known or set()
        self._local = threading.local()
        self._connect()  # fail fast if the file is missing

    def _connect(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:
            con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
            con.row_factory = sqlite3.Row
            self._local.con = con
        return con

    def counts(self, decision_id: str) -> tuple[int, int]:
        """(cited by, cites)."""
        row = self._connect().execute("SELECT cited_by, cites FROM counts WHERE decision_id = ?",
                                      [decision_id]).fetchone()
        return (row["cited_by"], row["cites"]) if row else (0, 0)

    def cited_by_counts(self, decision_ids: list[str]) -> dict[str, int]:
        """How often each decision is cited, for a batch of search results."""
        if not decision_ids:
            return {}
        holes = ",".join("?" * len(decision_ids))
        rows = self._connect().execute(
            f"SELECT decision_id, cited_by FROM counts WHERE decision_id IN ({holes})", decision_ids)
        return {r["decision_id"]: r["cited_by"] for r in rows}

    def _docket(self, decision_id: str) -> str | None:
        row = self._connect().execute("SELECT docket FROM decisions WHERE decision_id = ?",
                                      [decision_id]).fetchone()
        return row["docket"] if row else None

    def _edges(self, decision_id: str, column: str, limit: int) -> list[Citing]:
        other = "source" if column == "target" else "target"
        # the corpus holds some decisions under two ids ("bge_122 V 157" and "bge_BGE_122_V_157"), which
        # makes a decision look as if it cites itself; drop the twin by comparing docket numbers
        own = _key(self._docket(decision_id))
        rows = self._connect().execute(
            f"""SELECT DISTINCT e.{other} AS id, d.court, d.docket, d.date FROM edges e
                LEFT JOIN decisions d ON d.decision_id = e.{other}
                WHERE e.{column} = ? AND e.{other} != ? ORDER BY d.date DESC NULLS LAST LIMIT ?""",
            [decision_id, decision_id, limit + 5])
        out: list[Citing] = []
        seen = {own} if own else set()
        for r in rows:
            key = _key(r["docket"]) or r["id"]  # the same decision can appear under two ids
            if key in seen:
                continue
            seen.add(key)
            out.append(Citing(decision_id=r["id"], court=r["court"], docket=r["docket"], date=r["date"],
                              in_corpus=r["id"] in self.known))
        return out[:limit]

    def cited_by(self, decision_id: str, limit: int = 20) -> list[Citing]:
        """Later decisions citing this one, most recent first."""
        return self._edges(decision_id, "target", limit)

    def cites(self, decision_id: str, limit: int = 20) -> list[Citing]:
        """Decisions this one relies on."""
        return self._edges(decision_id, "source", limit)


def open_index(path: Path, known: set[str] | None = None) -> CitationIndex | None:
    if not path.exists():
        log.warning("no citation index at %s; run `python -m swiss_court_assistant.citations build`", path)
        return None
    try:
        return CitationIndex(path, known)
    except sqlite3.Error:
        log.exception("could not open the citation index at %s", path)
        return None
