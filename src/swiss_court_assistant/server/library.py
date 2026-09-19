"""Lookup and search over the full-corpus index (decisions and statute articles).

The index `swiss_court_assistant.index` builds holds both kinds of text in one `chunks` table:
a passage whose `decision_id` is found in `decisions` is court text, one found in `laws` is a
statute article. This module is the read side the agent's tools sit on.

The agent must never have to guess a filter value. Courts are codes ("bger", not "Bundesgericht"),
cantons are two letters, branches are German words ("oeffentlich"), and a law's abbreviation depends
on the language it is published in (OR in German, CO in French). So:

  * `vocabulary()` lists the values actually present, with counts - the agent can read them;
  * `resolve()` turns a loose value into the exact one ("Zurich" -> "ZH") and reports what it did,
    or, when nothing matches, names the closest values instead of silently returning nothing.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from swiss_court_assistant.fts import match_expr

log = logging.getLogger(__name__)

# Fields a caller may filter or group by, per kind of text.
DECISION_FIELDS = ("court", "canton", "branch", "language", "jurisdiction", "chamber")
LAW_FIELDS = ("canton", "language", "abbreviation", "category", "sr_number")

# Spellings people (and models) reach for, mapped to what the database stores. Everything else is
# matched against the live values, so this only has to cover what similarity would get wrong.
ALIASES = {
    "court": {
        "federal supreme court": "bger", "supreme court": "bger", "bundesgericht": "bger",
        "tribunal federal": "bger", "tribunale federale": "bger", "tf": "bger",
        "federal administrative court": "bvger", "bundesverwaltungsgericht": "bvger",
        "federal criminal court": "bstger", "bundesstrafgericht": "bstger",
        "federal patent court": "bpatger", "bundespatentgericht": "bpatger",
        "leading decisions": "bge", "atf": "bge", "dtf": "bge", "published": "bge",
    },
    "branch": {
        "public": "oeffentlich", "public law": "oeffentlich", "administrative": "oeffentlich",
        "öffentlich": "oeffentlich", "droit public": "oeffentlich",
        "civil": "zivil", "civil law": "zivil", "private": "zivil", "droit civil": "zivil",
        "criminal": "straf", "criminal law": "straf", "penal": "straf", "droit penal": "straf",
        "social insurance": "sozialversicherung", "social security": "sozialversicherung",
        "insurance": "sozialversicherung", "assurances sociales": "sozialversicherung",
    },
    "language": {"german": "de", "deutsch": "de", "allemand": "de",
                 "french": "fr", "français": "fr", "francais": "fr", "franzoesisch": "fr",
                 "italian": "it", "italiano": "it", "italien": "it"},
    "canton": {"zurich": "ZH", "zürich": "ZH", "bern": "BE", "berne": "BE", "geneva": "GE",
               "genève": "GE", "geneve": "GE", "vaud": "VD", "ticino": "TI", "tessin": "TI",
               "lucerne": "LU", "luzern": "LU", "basel": "BS", "basel-stadt": "BS",
               "basel-landschaft": "BL", "valais": "VS", "wallis": "VS", "aargau": "AG",
               "st. gallen": "SG", "sankt gallen": "SG", "federal": "CH", "switzerland": "CH",
               "confederation": "CH"},
}


def _ref_key(ref: str) -> str:
    """A decision reference with its separators removed: "4A_13/2019" == "4A 13 2019"."""
    return re.sub(r"[^a-z0-9]", "", str(ref).lower())


def _fold(s: str) -> str:
    """Lowercase, strip accents and punctuation: 'Genève' and 'geneve' compare equal."""
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


@dataclass
class Hit:
    """One passage, of a decision or of a statute article."""

    kind: str  # "decision" | "law"
    chunk_id: str
    source_id: str  # decision_id or law_id
    text: str
    score: float
    section: str
    erwaegungen: list[str]
    char_start: int | None
    char_end: int | None
    label: str  # how a lawyer would name it: "BGer 4A_1/2024" or "Art. 271 OR (SR 220)"
    date: str | None = None
    language: str = ""
    court: str | None = None
    canton: str | None = None
    branch: str | None = None


class Library:
    """Read-only access to the corpus index. One connection per instance; callers run it in a
    worker thread (see Corpus) because SQLite connections are not shared across threads."""

    def __init__(self, db: Path, fts: Path):
        self.db = sqlite3.connect(f"file:{db}?mode=ro", uri=True, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.fts = sqlite3.connect(f"file:{fts}?mode=ro", uri=True, check_same_thread=False)
        self._vocab: dict[tuple[str, str], list[tuple[str, int]]] = {}

    # ── the vocabulary the agent is allowed to filter on ────────────────
    def vocabulary(self, table: str, field: str, limit: int = 60) -> list[tuple[str, int]]:
        """The values actually stored, most common first. Cached: the index does not change here."""
        fields = DECISION_FIELDS if table == "decisions" else LAW_FIELDS
        if field not in fields:
            raise ValueError(f"{table} has no filter {field!r}; use one of {list(fields)}")
        key = (table, field)
        if key not in self._vocab:
            rows = self.db.execute(
                f'SELECT "{field}" v, COUNT(*) n FROM {table} WHERE "{field}" IS NOT NULL '
                f'AND "{field}" <> \'\' GROUP BY 1 ORDER BY n DESC').fetchall()
            self._vocab[key] = [(r["v"], r["n"]) for r in rows]
        return self._vocab[key][:limit]

    def resolve(self, table: str, field: str, value: str | None) -> tuple[str | None, str | None]:
        """(exact value, note). The note is meant to be shown to the agent: it says which value was
        used, or - when nothing matched - which values exist, so the next call can be right."""
        if value is None or str(value).strip() == "":
            return None, None
        raw = str(value).strip()
        values = [v for v, _ in self.vocabulary(table, field, limit=10_000)]
        if raw in values:
            return raw, None
        folded = {_fold(v): v for v in values}
        f = _fold(raw)
        if f in folded:
            return folded[f], f"{field}={raw!r} read as {folded[f]!r}"
        if (alias := ALIASES.get(field, {}).get(f)) and alias in values:
            return alias, f"{field}={raw!r} read as {alias!r}"
        near = [v for v in values if f and (_fold(v).startswith(f) or f in _fold(v))]
        if len(near) == 1:
            return near[0], f"{field}={raw!r} read as {near[0]!r}"
        choices = near[:12] or values[:12]
        return None, (f"no {field} matches {raw!r} in the index. "
                      f"{'Did you mean' if near else 'The most common are'}: {', '.join(map(str, choices))}")

    def _filters(self, table: str, wanted: dict, alias: str = "") -> tuple[list[str], list, list[str]]:
        """SQL conditions for the given filters, plus the notes explaining how each was read."""
        where, params, notes = [], [], []
        for field, value in wanted.items():
            if value is None:
                continue
            exact, note = self.resolve(table, field, value)
            if note:
                notes.append(note)
            if exact is None:
                where.append("0")  # an unmatched filter must return nothing, not everything
            else:
                where.append(f'{alias}"{field}" = ?'), params.append(exact)
        return where, params, notes

    # ── labels ──────────────────────────────────────────────────────────
    @staticmethod
    def _law_label(r) -> str:
        """"Art. 271 OR (SR 220)" - how the article would be cited."""
        r = dict(r)
        art = f"Art. {r['article_num']}" if r.get("article_num") else (r.get("heading") or "")
        code = r.get("abbreviation") or r.get("law_title") or ""
        canton, sr = r.get("canton") or "CH", r.get("sr_number") or ""
        where = f"SR {sr}" if canton == "CH" else f"{canton} {sr}"
        return " ".join(x for x in (art, code, f"({where})") if x.strip())

    @staticmethod
    def _decision_label(r) -> str:
        r = dict(r)
        docket = r.get("docket_number") or r.get("decision_id")
        date = r.get("decision_date")
        return f"{r.get('court')} {docket}" + (f" · {date}" if date else "")

    # ── reading one item ────────────────────────────────────────────────
    def read_law(self, abbreviation: str | None = None, article: str | None = None,
                 sr_number: str | None = None, language: str | None = None,
                 canton: str | None = None, law_id: str | None = None, limit: int = 8) -> dict:
        """One statute article, verbatim. Abbreviation is language-specific (OR/CO/CO)."""
        if law_id:
            rows = self.db.execute("SELECT * FROM laws WHERE law_id = ?", [law_id]).fetchall()
            return {"notes": [], "articles": [dict(r) | {"label": self._law_label(r)} for r in rows]}
        where, params, notes = [], [], []
        for field, value in (("abbreviation", abbreviation), ("sr_number", sr_number),
                             ("language", language), ("canton", canton)):
            if value is None:
                continue
            exact, note = self.resolve("laws", field, value)
            if note:
                notes.append(note)
            if exact is None:
                return {"notes": notes, "articles": []}
            where.append(f'"{field}" = ?'), params.append(exact)
        if article:
            # "271a", "271 a", "Art. 271a" all mean the same article
            where.append("REPLACE(LOWER(article_num), ' ', '') = ?")
            params.append(re.sub(r"(?i)^art\.?\s*", "", str(article)).replace(" ", "").lower())
        if not where:
            raise ValueError("give at least an abbreviation, an sr_number or a law_id")
        rows = self.db.execute(
            f"SELECT * FROM laws WHERE {' AND '.join(where)} ORDER BY language, seq LIMIT ?",
            [*params, limit]).fetchall()
        return {"notes": notes, "articles": [dict(r) | {"label": self._law_label(r)} for r in rows]}

    def find_decision(self, ref: str) -> list[str]:
        """Decision ids matching a reference. The agent passes back whatever it saw - an id, a bare
        docket, or the label a result was printed with ("bger 4A 13/2019") - so all three resolve."""
        ref = str(ref).strip()
        if (r := self.db.execute("SELECT decision_id FROM decisions WHERE decision_id = ?",
                                 [ref]).fetchone()):
            return [r["decision_id"]]
        key = _ref_key(ref)
        if not key:
            return []
        rows = self.db.execute(
            """SELECT decision_id FROM decisions
               WHERE REPLACE(REPLACE(REPLACE(LOWER(decision_id), '_', ''), ' ', ''), '/', '') = ?1
                  OR REPLACE(REPLACE(REPLACE(LOWER(COALESCE(docket_number, '')), '_', ''), ' ', ''),
                             '/', '') = ?1
               LIMIT 10""", [key]).fetchall()
        if rows:
            return [r["decision_id"] for r in rows]
        # the label carries the court in front of the docket: "bger 4A 13/2019"
        head, _, rest = ref.partition(" ")
        return self.find_decision(rest) if rest and _fold(head) else []

    def read_decision(self, decision_id: str, offset: int = 0, window: int = 8000) -> dict | None:
        ids = self.find_decision(decision_id)
        r = (self.db.execute("SELECT * FROM decisions WHERE decision_id = ?", [ids[0]]).fetchone()
             if ids else None)
        if r is None:
            return None
        text = r["full_text"] or ""
        start = max(0, min(offset, len(text)))
        return {"decision": dict(r), "label": self._decision_label(r), "length": len(text),
                "start": start, "end": min(len(text), start + window),
                "text": text[start:start + window]}

    # ── search ──────────────────────────────────────────────────────────
    def _hits(self, ids: list[int], scores: dict[int, float]) -> list[Hit]:
        """Turn chunk ids into hits, taking metadata from whichever table owns them."""
        if not ids:
            return []
        holes = ",".join("?" * len(ids))
        rows = self.db.execute(f"""
            SELECT c.id, c.chunk_id, c.decision_id, c.section, c.erwaegungen, c.char_start, c.char_end,
                   c.text, c.language,
                   d.court, d.canton AS d_canton, d.branch, d.decision_date, d.docket_number,
                   l.law_id, l.canton AS l_canton, l.sr_number, l.article_num, l.abbreviation,
                   l.law_title, l.heading
            FROM chunks c
            LEFT JOIN decisions d ON d.decision_id = c.decision_id
            LEFT JOIN laws l ON l.law_id = c.decision_id
            WHERE c.id IN ({holes})""", ids).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            is_law = d["law_id"] is not None
            d["canton"] = d["l_canton"] if is_law else d["d_canton"]
            out.append(Hit(
                kind="law" if is_law else "decision",
                chunk_id=r["chunk_id"], source_id=r["decision_id"], text=r["text"],
                score=scores.get(r["id"], 0.0), section=r["section"],
                erwaegungen=json.loads(r["erwaegungen"] or "[]"),
                char_start=r["char_start"], char_end=r["char_end"],
                label=self._law_label(d) if is_law else self._decision_label(d),
                date=None if is_law else r["decision_date"], language=r["language"] or "",
                court=None if is_law else r["court"],
                canton=r["l_canton"] if is_law else r["d_canton"],
                branch="law" if is_law else r["branch"]))
        return sorted(out, key=lambda h: -h.score)

    def keyword_search(self, keyword: str, kind: str = "any", k: int = 8,
                       **filters) -> tuple[list[Hit], list[str]]:
        """FTS5 over every passage. Quoted text is a phrase; other words must all appear."""
        table = "laws" if kind == "law" else "decisions"
        where, params, notes = self._filters(table, filters, alias="t.") if filters else ([], [], [])
        expr = match_expr(keyword)
        if not expr:
            return [], notes
        # over-fetch: the filters are applied after the keyword match
        rows = self.fts.execute(
            "SELECT rowid, bm25(passages) FROM passages WHERE passages MATCH ? ORDER BY rank LIMIT ?",
            (expr, max(k * 40, 400))).fetchall()
        if not rows:
            return [], notes
        scores = {int(i): -float(s) for i, s in rows}
        ids = list(scores)
        holes = ",".join("?" * len(ids))
        join = ("JOIN laws t ON t.law_id = c.decision_id" if kind == "law" else
                "JOIN decisions t ON t.decision_id = c.decision_id" if kind == "decision" else "")
        sql = f"SELECT c.id FROM chunks c {join} WHERE c.id IN ({holes})"
        if where:
            sql += " AND " + " AND ".join(where)
        keep = [r[0] for r in self.db.execute(sql, [*ids, *params])]
        keep.sort(key=lambda i: -scores[i])
        return self._hits(keep[:k], scores), notes

    def count(self, table: str, group_by: str, limit: int = 25, **filters) -> tuple[list[tuple], list[str]]:
        """How many decisions (or articles) there are per value - the metadata query."""
        fields = DECISION_FIELDS if table == "decisions" else LAW_FIELDS
        if group_by not in fields:
            raise ValueError(f"cannot group {table} by {group_by!r}; use one of {list(fields)}")
        where, params, notes = self._filters(table, filters)
        sql = f'SELECT "{group_by}" v, COUNT(*) n FROM {table}'
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += f" GROUP BY 1 ORDER BY n DESC LIMIT {int(limit)}"
        return [(r["v"], r["n"]) for r in self.db.execute(sql, params)], notes

    def browse(self, table: str, limit: int = 10, **filters) -> tuple[list[dict], list[str]]:
        """Items matching metadata only - no query text. Newest decisions first."""
        where, params, notes = self._filters(table, filters)
        order = "decision_date DESC" if table == "decisions" else "sr_number, seq"
        sql = f"SELECT * FROM {table}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        rows = self.db.execute(f"{sql} ORDER BY {order} LIMIT ?", [*params, limit]).fetchall()
        label = self._decision_label if table == "decisions" else self._law_label
        return [dict(r) | {"label": label(r)} for r in rows], notes

    def stats(self) -> dict:
        def one(sql, default=0):
            try:
                return self.db.execute(sql).fetchone()[0]
            except sqlite3.Error:
                return default
        return {"decisions": one("SELECT COUNT(*) FROM decisions"),
                "laws": one("SELECT COUNT(*) FROM laws"),
                "passages": one("SELECT COUNT(*) FROM chunks"),
                "embedded": one("SELECT SUM(n_embedded) FROM embedding_models")}
