
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import polars as pl

from .schemas import Decision, DecisionSummary

COURT_LABELS = {
    "bger": "Swiss Federal Supreme Court",
    "bge": "Swiss Federal Supreme Court (BGE)",
    "bvger": "Federal Administrative Court",
    "bstger": "Federal Criminal Court",
    "bpatger": "Federal Patent Court",
    "mkg": "Military Court of Cassation",
}
_COLUMNS = ["decision_id", "court", "canton", "chamber", "docket_number", "decision_date", "language",
            "title", "regeste", "legal_area", "source_url", "pdf_url", "full_text"]
_BGER_DOCKET = re.compile(r"^(\d+[A-Z]) (\d+/\d{4})$")  # "4A 705/2016" -> "4A_705/2016"


def court_label(court: str, canton: str | None) -> str:
    if court in COURT_LABELS:
        return COURT_LABELS[court]
    return f"Cantonal court {canton}" if canton and canton != "CH" else court


def _summary_fields(r: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision_id": r["decision_id"], "court": r["court"],
        "court_label": court_label(r["court"], r["canton"]), "canton": r["canton"],
        "chamber": r["chamber"], "docket": _BGER_DOCKET.sub(r"\1_\2", r["docket_number"] or r["decision_id"]),
        "date": r["decision_date"], "language": r["language"], "title": r["title"],
        "regeste": r["regeste"], "legal_area": r["legal_area"],
        "source_url": r["source_url"], "pdf_url": r["pdf_url"],
    }


class DecisionStore:
    """~50k decisions, ~1.2 GB in memory, ~2 s to load."""

    def __init__(self, path: Path):
        self.df = pl.read_parquet(path, columns=_COLUMNS)
        self._row = {d: i for i, d in enumerate(self.df["decision_id"].to_list())}
        self._docket: dict[str, list[str]] = {}
        for did, docket in self.df.select("decision_id", "docket_number").iter_rows():
            if docket:
                self._docket.setdefault(_docket_key(docket), []).append(did)

    def __len__(self) -> int:
        return self.df.height

    def _record(self, decision_id: str) -> dict[str, Any] | None:
        i = self._row.get(decision_id)
        return None if i is None else self.df.row(i, named=True)

    def summary(self, decision_id: str) -> DecisionSummary | None:
        r = self._record(decision_id)
        return None if r is None else DecisionSummary(**_summary_fields(r))

    def get(self, decision_id: str) -> Decision | None:
        r = self._record(decision_id)
        return None if r is None else Decision(**_summary_fields(r), full_text=r["full_text"])

    def find(self, ref: str) -> list[str]:
        """Decision ids for a decision_id, a chunk_id or a docket number ("4A_705/2016", "4A 705/2016")."""
        ref = ref.strip().split("#")[0]
        if ref in self._row:
            return [ref]
        return self._docket.get(_docket_key(ref), [])


def _docket_key(docket: str) -> str:
    return re.sub(r"[\s_./-]", "", docket).lower()
