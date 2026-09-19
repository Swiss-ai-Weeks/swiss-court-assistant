"""What each decision and statute article is - canton, court, legal area, year, kind of proceeding -
as arrays aligned with the vector matrix, so the agent's searches can be restricted to a part of the
corpus ("Geneva decisions", "the Federal Supreme Court since 2020", "criminal law", "Art. … of the OR").

A filter is a mask over the matrix rows, applied inside the search: a search restricted to the 285
decisions of canton Uri ranks those decisions' passages, instead of hoping some survive in the top 40
of an unrestricted search.

The metadata as the dataset has it: court and canton are always there; the legal area ("branch") is
missing for 36 % of the decisions, nearly all cantonal ones; the kind of proceeding for 44 %; the
outcome for 89 %. A missing area is inferred from the statutes the decision cites (OR, ZGB, ZPO ->
civil; StGB, StPO -> criminal; IVG, UVG, ATSG -> social insurance; VwVG, AIG, RPG -> public): on
decisions whose area is recorded, the inference agrees 89 % of the time, and it names an area for
72 % of the unknown ones. Filtering by proceeding drops every decision that has none recorded.

    uv run python -m swiss_court_assistant.facets build    # after `vecmatrix export`; the app also builds it
    uv run python -m swiss_court_assistant.facets status

File next to the index: corpus.nemotron-embed.facets.npz, tied to the matrix it was built for.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

from swiss_court_assistant import vecmatrix as M
from swiss_court_assistant import vectordb as V

log = logging.getLogger(__name__)

AREAS = ["unknown", "civil", "criminal", "public", "social_insurance"]
_BRANCH = {"zivil": "civil", "straf": "criminal", "oeffentlich": "public", "sozialversicherung": "social_insurance"}
_AREA_OF_CODE = {code: area for area, codes in {
    "civil": "OR ZGB ZPO SchKG CO CC CPC LP LEF VVG LCA",
    "criminal": "StGB StPO CP CPP BetmG LStup MStG",
    "social_insurance": "ATSG LPGA IVG LAI UVG LAA AHVG LAVS KVG LAMal BVG LPP AVIG LACI ELG LPC FZG LFLP",
    "public": "VwVG PA AIG LEI LEtr AuG AsylG LAsi RPG LAT DBG LIFD StHG LHID USG LPE LTF",
}.items() for code in codes.split()}
# "Art. 271a OR", "art. 336c al. 1 let. c CO": the act named after an article number
_CITED_ACT = re.compile(
    r"\b(?:Art|art|Artikel|Article|article|Articolo|articolo|artt?)\.?\s*\d+[a-z]{0,2}"
    r"(?:\s*(?:Abs|al|cpv|para|lit|let|lett|Ziff|ch|n|Nr)\.?\s*\d*[a-z]?\)?)*\s+([A-Z][A-Za-z]{1,6})\b")

COURTS = {  # the `court` filter: which courts each value takes in
    "federal_supreme": {"bger", "bge"},
    "leading_cases": {"bge"},  # published in the official collection (BGE / ATF / DTF)
    "federal_administrative": {"bvger"},
    "federal_criminal": {"bstger"},
    "federal_patent": {"bpatger"},
}
FEDERAL_OTHER = "federal_other"  # the other federal courts, commissions and authorities (FINMA, WEKO, ...)
CANTONAL = "cantonal"

# proceeding types (as the dataset records them) grouped into what a question would ask for
PROCEEDINGS = {
    "appeal": {"zpo_berufung", "stpo_berufung", "og_berufung"},  # Berufung / appel: full review
    "objection": {"zpo_beschwerde", "stpo_beschwerde"},  # Beschwerde / recours under ZPO or StPO
    "debt_enforcement": {"schkg_aufsichtsbeschwerde", "schkg_rechtsoeffnung", "schkg_beschwerde"},
    "constitutional_complaint": {"bgg_verfassungsbeschwerde", "og_staatsrechtliche_beschwerde"},
    "revision": {"bgg_revision", "bgg_erlaeuterung"},
    "first_instance": {"zpo_ordentlich", "zpo_erstinstanz", "vwv_beschwerde_erstinstanz", "kartellverfahren"},
}
PROCEEDING_LABELS = {
    "zpo_berufung": "Berufung (ZPO)", "stpo_berufung": "Berufung (StPO)", "og_berufung": "Berufung (OG)",
    "zpo_beschwerde": "Beschwerde (ZPO)", "stpo_beschwerde": "Beschwerde (StPO)",
    "schkg_aufsichtsbeschwerde": "SchKG-Aufsichtsbeschwerde", "schkg_rechtsoeffnung": "Rechtsöffnung (SchKG)",
    "schkg_beschwerde": "Beschwerde (SchKG)", "bgg_verfassungsbeschwerde": "subsidiäre Verfassungsbeschwerde (BGG)",
    "og_staatsrechtliche_beschwerde": "staatsrechtliche Beschwerde (OG)", "bgg_revision": "Revision (BGG)",
    "bgg_erlaeuterung": "Erläuterung (BGG)", "zpo_ordentlich": "ordentliches Verfahren (ZPO)",
    "zpo_erstinstanz": "erste Instanz (ZPO)", "bgg_beschwerde_zivil": "Beschwerde in Zivilsachen (BGG)",
    "bgg_beschwerde_straf": "Beschwerde in Strafsachen (BGG)",
    "bgg_beschwerde_oeff": "Beschwerde in öffentlich-rechtlichen Angelegenheiten (BGG)",
    "bgg_beschwerde_soz": "Beschwerde, Sozialversicherung (BGG)", "vwvg_beschwerde": "Beschwerde (VwVG)",
    "vwv_beschwerde": "Verwaltungsbeschwerde", "vwv_rekurs": "Rekurs", "sozialversicherungsbeschwerde":
    "Beschwerde (ATSG)", "evg_verwaltungsgerichtsbeschwerde": "Verwaltungsgerichtsbeschwerde (EVG)",
    "og_verwaltungsgerichtsbeschwerde": "Verwaltungsgerichtsbeschwerde (OG)",
    "og_nichtigkeitsbeschwerde": "Nichtigkeitsbeschwerde (OG)", "kassationsbeschwerde": "Kassationsbeschwerde",
}
_OUTCOMES = {  # the dataset's outcome values, as one vocabulary
    "dismissed": "dismissed", "REJ - Rejeté": "dismissed", "inadmissible": "inadmissible",
    "IRR - Irrecevable": "inadmissible", "approved": "upheld", "ADV - Admis (annulée / renvoyée)": "upheld",
    "ADF - Admis (réformée)": "upheld", "partial_approval": "partly upheld",
    "PAD - Partiellement admis (réformée)": "partly upheld", "moot": "moot", "RSO - Sans objet": "moot",
    "RER - Retrait": "withdrawn",
}
CANTONS = ["AG", "AI", "AR", "BE", "BL", "BS", "FR", "GE", "GL", "GR", "JU", "LU", "NE", "NW", "OW", "SG",
           "SH", "SO", "SZ", "TG", "TI", "UR", "VD", "VS", "ZG", "ZH"]


def infer_area(text: str | None) -> str:
    """The legal area of a decision from the acts it cites most, or "unknown" when too few or mixed."""
    n = Counter(_AREA_OF_CODE[c] for c in _CITED_ACT.findall(text or "") if c in _AREA_OF_CODE)
    if not n:
        return "unknown"
    (area, k), = n.most_common(1)
    return area if k >= 2 and k >= 0.5 * sum(n.values()) else "unknown"


def _path(db: Path, model: str) -> Path:
    return M._file(M.base(db, model), ".facets.npz")


def _vocab(values: list[str | None]) -> tuple[list[str], np.ndarray]:
    names = sorted({v or "" for v in values})
    index = {v: i for i, v in enumerate(names)}
    return names, np.array([index[v or ""] for v in values], dtype=np.int32)


def build(db: Path, model: str, matrix: M.VectorMatrix) -> Path:
    t0 = time.time()
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = con.execute("SELECT decision_id, court, canton, branch, proceeding_type, decision_date, outcome "
                       "FROM decisions ORDER BY decision_id").fetchall()
    ids = [r[0] for r in rows]
    where = {d: i for i, d in enumerate(ids)}
    area = np.array([AREAS.index(_BRANCH.get(r[3], "unknown")) for r in rows], dtype=np.uint8)
    unknown = [ids[i] for i in np.flatnonzero(area == 0)]
    print(f"{len(ids):,} decisions; inferring the area of {len(unknown):,} from the statutes they cite …")
    for n, (did, text) in enumerate(con.execute(
            "SELECT decision_id, full_text FROM decisions WHERE branch IS NULL OR branch NOT IN "
            "('zivil', 'straf', 'oeffentlich', 'sozialversicherung')"), 1):
        area[where[did]] = AREAS.index(infer_area(text))
        if n % 20000 == 0:
            print(f"  {n:,} ({time.time() - t0:.0f} s)")
    courts, court = _vocab([r[1] for r in rows])
    cantons, canton = _vocab([r[2] for r in rows])
    proceedings, proceeding = _vocab([r[4] for r in rows])
    outcomes, outcome = _vocab([_OUTCOMES.get(r[6] or "", "") for r in rows])
    date = np.array([int(re.sub(r"\D", "", r[5] or "")[:8] or 0) for r in rows], dtype=np.int32)

    print("statute articles …")
    laws = con.execute("SELECT law_id, canton, sr_number FROM laws").fetchall() \
        if con.execute("SELECT 1 FROM sqlite_master WHERE name = 'laws'").fetchone() else []
    law_where = {r[0]: i for i, r in enumerate(laws)}
    law_cantons, law_canton = _vocab([r[1] or "CH" for r in laws])
    acts, law_act = _vocab([f"{r[1] or 'CH'}/{r[2] or ''}" for r in laws])

    print("passages …")
    position = np.full(int(matrix.ids.max()) + 1, -1, dtype=np.int64)
    position[matrix.ids] = np.arange(len(matrix.ids))
    row_decision = np.full(len(matrix.ids), -1, dtype=np.int32)
    row_law = np.full(len(matrix.ids), -1, dtype=np.int32)
    for cid, did in con.execute("SELECT id, decision_id FROM chunks"):
        if cid >= len(position) or (row := position[cid]) < 0:
            continue
        if (i := where.get(did)) is not None:
            row_decision[row] = i
        elif (i := law_where.get(did)) is not None:
            row_law[row] = i
    con.close()
    out = _path(db, model)
    tmp = out.with_name(out.name + ".tmp.npz")
    np.savez(tmp, decision_ids=np.array(ids), area=area, court=court, courts=np.array(courts),
             canton=canton, cantons=np.array(cantons), proceeding=proceeding, proceedings=np.array(proceedings),
             outcome=outcome, outcomes=np.array(outcomes), date=date, law_canton=law_canton,
             law_cantons=np.array(law_cantons), law_act=law_act, acts=np.array(acts),
             row_decision=row_decision, row_law=row_law, matrix=np.array(json.dumps(matrix.meta["index"])))
    tmp.replace(out)
    print(f"done in {time.time() - t0:.0f} s -> {out}")
    print("  areas:", {AREAS[a]: int(n) for a, n in zip(*np.unique(area, return_counts=True))})
    return out


class Facets:
    """Filters over the corpus, as masks over the rows of the vector matrix."""

    def __init__(self, data: dict[str, np.ndarray], matrix: M.VectorMatrix):
        self.matrix = matrix
        self.ids = data["decision_ids"].tolist()
        self._index = {d: i for i, d in enumerate(self.ids)}
        self.area, self.date = data["area"], data["date"]
        self.court, self.courts = data["court"], data["courts"].tolist()
        self.canton, self.cantons = data["canton"], data["cantons"].tolist()
        self.proceeding, self.proceedings = data["proceeding"], data["proceedings"].tolist()
        self.outcome, self.outcomes = data["outcome"], data["outcomes"].tolist()
        self.law_canton, self.law_cantons = data["law_canton"], data["law_cantons"].tolist()
        self.law_act, self.acts = data["law_act"], data["acts"].tolist()
        self.row_decision, self.row_law = data["row_decision"], data["row_law"]
        # chunk id -> matrix row, for filtering keyword search results
        self.position = np.full(int(matrix.ids.max()) + 1, -1, dtype=np.int64)
        self.position[matrix.ids] = np.arange(len(matrix.ids))

    @classmethod
    def open(cls, db: Path, model: str, matrix: M.VectorMatrix | None) -> Facets | None:
        """The facets for this matrix, built first when missing or built for another one (~2 min)."""
        if matrix is None:
            return None
        path = _path(db, model)
        if path.exists():
            data = dict(np.load(path))
            if json.loads(str(data["matrix"])) == matrix.meta["index"]:
                return cls(data, matrix)
            log.info("facets %s are for another export of the vectors; rebuilding", path)
        else:
            log.info("no facets at %s; building them (about two minutes, once)", path)
        build(db, model, matrix)
        return cls(dict(np.load(path)), matrix)

    # ── decisions ──
    def decision_mask(self, canton: list[str] | None = None, court: list[str] | None = None,
                      area: list[str] | None = None, proceeding: list[str] | None = None,
                      year_from: int | None = None, year_to: int | None = None) -> np.ndarray | None:
        """Which decisions pass the filters (None: no filter given). Values within one filter are
        alternatives, the filters combine: canton=["GE", "VD"], area=["civil"] is civil law in GE or VD."""
        mask = None

        def both(m: np.ndarray) -> None:
            nonlocal mask
            mask = m if mask is None else mask & m

        if canton:
            both(np.isin(self.canton, [self.cantons.index(c) for c in canton if c in self.cantons]))
        if court:
            codes: set[str] = set()
            for c in court:
                if c == CANTONAL:
                    codes |= {x for x, k in zip(self.courts, self._court_canton()) if k != "CH"}
                elif c == FEDERAL_OTHER:
                    named = set().union(*COURTS.values())
                    codes |= {x for x, k in zip(self.courts, self._court_canton()) if k == "CH" and x not in named}
                else:
                    codes |= COURTS.get(c, set())
            both(np.isin(self.court, [self.courts.index(x) for x in codes if x in self.courts]))
        if area:
            both(np.isin(self.area, [AREAS.index(a) for a in area if a in AREAS]))
        if proceeding:
            kinds = set().union(*(PROCEEDINGS.get(p, set()) for p in proceeding))
            both(np.isin(self.proceeding, [self.proceedings.index(x) for x in kinds if x in self.proceedings]))
        if year_from:
            both(self.date >= int(year_from) * 10000)
        if year_to:
            both((self.date > 0) & (self.date < (int(year_to) + 1) * 10000))
        return mask

    def _court_canton(self) -> list[str]:
        if not hasattr(self, "_cc"):
            first = {}
            for c, k in zip(self.court.tolist(), self.canton.tolist()):
                first.setdefault(c, self.cantons[k])
            self._cc = [first.get(i, "") for i in range(len(self.courts))]
        return self._cc

    def court_canton(self, court: str) -> str:
        return self._court_canton()[self.courts.index(court)] if court in self.courts else "CH"

    def rows(self, decisions: np.ndarray) -> np.ndarray:
        """The matrix rows of the passages of the decisions in the mask."""
        return (self.row_decision >= 0) & decisions[np.maximum(self.row_decision, 0)]

    def keep(self, chunk_ids: list[int], decisions: np.ndarray) -> list[bool]:
        """Which passages (by chunks.id) belong to a decision in the mask."""
        ids = np.asarray(chunk_ids, dtype=np.int64)
        rows = np.where(ids < len(self.position), self.position[np.minimum(ids, len(self.position) - 1)], -1)
        d = np.where(rows >= 0, self.row_decision[np.maximum(rows, 0)], -1)
        return ((d >= 0) & decisions[np.maximum(d, 0)]).tolist()

    def describe(self, decision_id: str) -> dict[str, str]:
        """Area, proceeding and outcome of one decision, where known."""
        i = self._index.get(decision_id)
        if i is None:
            return {}
        p, o = self.proceedings[self.proceeding[i]], self.outcomes[self.outcome[i]]
        return {k: v for k, v in {"area": AREAS[self.area[i]].replace("_", " ") if self.area[i] else "",
                                  "proceeding": PROCEEDING_LABELS.get(p, p.replace("_", " ")),
                                  "outcome": o}.items() if v}

    def newest(self, decisions: np.ndarray, n: int = 10, oldest: bool = False) -> list[str]:
        idx = np.flatnonzero(decisions)
        order = np.argsort(self.date[idx], kind="stable")
        pick = idx[order[:n]] if oldest else idx[order[::-1][:n]]
        return [self.ids[i] for i in pick]

    def breakdown(self, decisions: np.ndarray) -> dict[str, Counter]:
        idx = np.flatnonzero(decisions)
        years = self.date[idx] // 10000
        return {
            "court": Counter(self.courts[c] for c in self.court[idx].tolist()),
            "area": Counter(AREAS[a] for a in self.area[idx].tolist()),
            "period": Counter(f"{y // 10 * 10}s" for y in years.tolist() if y),
        }

    # ── statute articles ──
    def law_rows(self, canton: str | None = "CH", acts: list[str] | None = None) -> np.ndarray:
        """Matrix rows of statute articles of one canton ("CH": federal law, None: all) and, if given,
        of these acts only ("CH/220" = the OR)."""
        ok = self.row_law >= 0
        law = np.maximum(self.row_law, 0)
        if canton:
            k = self.law_cantons.index(canton) if canton in self.law_cantons else -1
            ok &= self.law_canton[law] == k
        if acts:
            ok &= np.isin(self.law_act[law], [self.acts.index(a) for a in acts if a in self.acts])
        return ok

    def status(self) -> str:
        return (f"{len(self.ids):,} decisions ({int((self.area > 0).sum()):,} with a legal area), "
                f"{len(self.law_act):,} statute articles")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["build", "status"])
    ap.add_argument("--db", type=Path, default=V.DB_DIR / "corpus.sqlite")
    ap.add_argument("--model", default="nemotron-embed")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    con = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    matrix = M.VectorMatrix.open(args.db, args.model, con)
    if matrix is None:
        raise SystemExit("no current vector matrix; run `python -m swiss_court_assistant.vecmatrix export`")
    if args.command == "build":
        build(args.db, args.model, matrix)
    else:
        path = _path(args.db, args.model)
        f = Facets.open(args.db, args.model, matrix) if path.exists() else None
        print(f"{path}: {f.status() if f else 'missing'}")


if __name__ == "__main__":
    main()
