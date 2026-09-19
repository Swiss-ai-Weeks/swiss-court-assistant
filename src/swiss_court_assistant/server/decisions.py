
from __future__ import annotations

import re
import sqlite3
import threading
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
    "ch_vb": "Federal administrative practice (VPB)",
    "ch_bundesrat": "Federal Council",
    "emark": "Swiss Asylum Appeals Commission",
    "edoeb": "Federal Data Protection Commissioner (EDÖB)",
    "finma": "FINMA", "finma_versicherungsrecht": "FINMA (insurance supervision)",
    "weko": "Competition Commission (WEKO)", "elcom": "Electricity Commission (ElCom)",
    "comcom": "Communications Commission (ComCom)", "postcom": "Postal Services Commission (PostCom)",
    "ubi": "Independent Complaints Authority for Radio and Television (UBI)",
    "eschk": "Federal Arbitration Commission (ESchK)", "esbk": "Federal Gaming Board (ESBK)",
    "bazg": "Federal Office for Customs (BAZG)", "estv": "Federal Tax Administration (ESTV)",
    "rab": "Federal Audit Oversight Authority (RAB)", "preisueberwacher": "Price Supervisor",
}
# cantonal court codes are "<canton>_<court>"; these name a collection, not a court
_GENERIC_COURTS = {"gerichte", "findinfo", "omni", "publikationen", "weitere", "zivilstraf", "kantone"}
_CANTONAL_NAMES = {
    "bvd": "Bau- und Verkehrsdirektion", "jurisprudence_adm": "Tribunal administratif",
    "departement_bvu": "Departement Bau, Verkehr und Umwelt", "departement_vi": "Departement Volkswirtschaft und Inneres",
    "departement_bks": "Departement Bildung, Kultur und Sport", "departement_gs": "Departement Gesundheit und Soziales",
}

_COLUMNS = ["decision_id", "court", "canton", "chamber", "docket_number", "decision_date", "language",
            "title", "regeste", "legal_area", "source_url", "pdf_url", "full_text"]
_BGER_DOCKET = re.compile(r"^(\d+[A-Z]) (\d+/\d{4})$")  # "4A 705/2016" -> "4A_705/2016"


def court_label(court: str, canton: str | None) -> str:
    if court in COURT_LABELS:
        return COURT_LABELS[court]
    if not canton or canton == "CH":
        return court
    name = court.split("_", 1)[1] if "_" in court else ""
    if not name or name in _GENERIC_COURTS:
        return f"Cantonal court {canton}"
    # "zh_bezirksgericht_zuerich" -> "Bezirksgericht Zuerich (ZH)"
    name = _CANTONAL_NAMES.get(name) or " ".join(w.capitalize() for w in name.split("_"))
    return f"{name} ({canton})"


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

    def ids(self) -> set[str]:
        return set(self._row)

    def law(self, law_id: str) -> Decision | None:  # the subset holds no statutes
        return None

    def law_summary(self, law_id: str) -> DecisionSummary | None:
        return None


class SqliteDecisionStore:
    """The decisions of the full-corpus index (`index.py`), read from its `decisions` table on demand.

    Same interface as `DecisionStore`. Loading 254k decisions with their full text into memory would
    take ~6 GB and a slow start, and the index already holds them next to the passages; only the
    docket lookup is kept in memory, because the table has no index on the docket number."""

    def __init__(self, path: Path):
        self.path = path
        self._local = threading.local()
        self._docket: dict[str, list[str]] = {}
        self._known: set[str] = set()
        for did, docket in self._con().execute("SELECT decision_id, docket_number FROM decisions"):
            self._known.add(did)
            if docket:
                self._docket.setdefault(_docket_key(docket), []).append(did)

    def _con(self) -> sqlite3.Connection:
        con = getattr(self._local, "con", None)
        if con is None:  # one read-only connection per thread: the agent's tools run on several
            con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
            con.row_factory = sqlite3.Row
            self._local.con = con
        return con

    def __len__(self) -> int:
        return len(self._known)

    def _record(self, decision_id: str) -> dict[str, Any] | None:
        if decision_id not in self._known:  # statute articles share the passage table; they are not decisions
            return None
        row = self._con().execute(f"SELECT {', '.join(_COLUMNS)} FROM decisions WHERE decision_id = ?",
                                  [decision_id]).fetchone()
        return dict(row) if row else None

    def summary(self, decision_id: str) -> DecisionSummary | None:
        r = self._record(decision_id)
        return None if r is None else DecisionSummary(**_summary_fields(r))

    def get(self, decision_id: str) -> Decision | None:
        r = self._record(decision_id)
        return None if r is None else Decision(**_summary_fields(r), full_text=r["full_text"] or "")

    def find(self, ref: str) -> list[str]:
        """Decision ids for a decision_id, a chunk_id or a docket number ("4A_705/2016", "4A 705/2016")."""
        ref = ref.strip().split("#")[0]
        if ref in self._known:
            return [ref]
        return self._docket.get(_docket_key(ref), [])

    def ids(self) -> set[str]:
        return set(self._known)

    # ── statute articles: not decisions, but citable sources shaped like them ──
    def _law(self, law_id: str) -> dict[str, Any] | None:
        if not law_id.startswith("law_"):
            return None
        try:
            row = self._con().execute("SELECT * FROM laws WHERE law_id = ?", [law_id]).fetchone()
        except sqlite3.OperationalError:  # an index built without --laws
            return None
        return dict(row) if row else None

    def law(self, law_id: str) -> Decision | None:
        r = self._law(law_id)
        return None if r is None else Decision(**_law_fields(r), full_text=r["text"] or "")

    def law_summary(self, law_id: str) -> DecisionSummary | None:
        r = self._law(law_id)
        return None if r is None else DecisionSummary(**_law_fields(r))

    def find_acts(self, code: str, canton: str = "CH") -> list[str]:
        """The SR numbers of the act an abbreviation (any language: OR, CO) or SR number names."""
        code = re.sub(r"(?i)^SR\s*", "", code.strip())
        try:
            return [r[0] for r in self._con().execute(
                "SELECT DISTINCT sr_number FROM laws WHERE canton = ? AND (abbreviation = ? COLLATE NOCASE "
                "OR sr_number = ?) AND sr_number IS NOT NULL", [(canton or "CH").strip().upper(), code, code])]
        except sqlite3.OperationalError:  # an index built without --laws
            return []

    def find_articles(self, code: str, article: str, canton: str = "CH") -> list[dict[str, Any]]:
        """An article of an act in every language it is published in, with its label.

        `code` is an abbreviation or an SR number. Abbreviations differ by language (OR in German, CO
        in French and Italian), so the code is resolved to the act's SR number first and the article
        comes back in all languages - asking for "OR" finds the French text too."""
        canton = (canton or "CH").strip().upper()
        srs = self.find_acts(code, canton)
        if not srs:
            return []
        con = self._con()
        want = _article_key(article)
        holes = ",".join("?" * len(srs))
        rows = con.execute(f"SELECT * FROM laws WHERE canton = ? AND sr_number IN ({holes}) ORDER BY language, seq",
                           [canton, *srs])
        return [dict(r) | {"label": law_label(dict(r))} for r in rows if _article_key(r["article_num"]) == want]


# article numbers as stored: "271a", but also "ikel 9" (a truncated "Artikel 9"), ". 1", "§ 9",
# "Ziff. 9", "Règle 9", ranges and Roman numerals
_ARTICLE_JUNK = re.compile(r"^(?:ikel|icle|icolo|[.:])\s*", re.I)
_NOT_ARTICLE = re.compile(r"^(?:§|Ziff|Ch\.|chiffre|cifra|Règle|Regel|Regola)", re.I)


def article_number(raw: str | None) -> str:
    return _ARTICLE_JUNK.sub("", (raw or "").strip()).strip()


def _article_key(raw: str | None) -> str:
    """"271a", "271 a", "Art. 271a" and "ikel 271a" are the same article."""
    num = re.sub(r"(?i)^art(?:ikel|icle|icolo)?\.?\s*", "", article_number(raw))
    return re.sub(r"[\s.]", "", num).lower()


def law_label(r: dict[str, Any]) -> str:
    """How an article is cited: "Art. 271a OR". Many federal acts have no abbreviation in the data;
    their SR number stands in ("Art. 5 SR 832.10"), and a cantonal act says whose it is."""
    num = article_number(r.get("article_num"))
    art = (num if _NOT_ARTICLE.match(num) else f"Art. {num}") if num else (r.get("heading") or r["law_id"])
    canton = r.get("canton") or "CH"
    if r.get("abbreviation"):
        code = r["abbreviation"] if canton == "CH" else f"{r['abbreviation']} ({canton})"
    elif r.get("sr_number"):
        code = f"SR {r['sr_number']}" if canton == "CH" else f"{canton} {r['sr_number']}"
    else:
        code = ""
    return f"{art} {code}".strip()


def _law_fields(r: dict[str, Any]) -> dict[str, Any]:
    canton = r.get("canton") or "CH"
    return {
        "decision_id": r["law_id"], "court": "law",
        "court_label": "Federal law" if canton == "CH" else f"Cantonal law {canton}",
        "canton": canton, "chamber": None, "docket": law_label(r), "date": None,
        "language": r["language"], "title": r.get("law_title"), "regeste": None,
        "legal_area": r.get("heading") or None, "source_url": r.get("original_url"), "pdf_url": None,
    }


def _docket_key(docket: str) -> str:
    return re.sub(r"[\s_./-]", "", docket).lower()
