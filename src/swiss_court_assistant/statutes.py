from __future__ import annotations

import re

# Swiss statutes have a different official abbreviation in each language; LLMs
# otherwise invent literal translations ("LTF" -> "RVG" instead of "BGG").
STATUTE_GLOSSARY = """de / fr / it
BV / Cst. / Cost. — ZGB / CC / CC — OR / CO / CO — StGB / CP / CP — StPO / CPP / CPP
ZPO / CPC / CPC — SchKG / LP / LEF — BGG / LTF / LTF — VwVG / PA / PA — ATSG / LPGA / LPGA
IVG / LAI / LAI — AHVG / LAVS / LAVS — UVG / LAA / LAINF — KVG / LAMal / LAMal — BVG / LPP / LPP
AVIG / LACI / LADI — AIG (AuG) / LEI (LEtr) / LStrI (LStr) — AsylG / LAsi / LAsi — DBG / LIFD / LIFD
MWSTG / LTVA / LIVA — RPG / LAT / LPT — USG / LPE / LPAmb — SVG / LCR / LCStr — BetmG / LStup / LStup"""

# Official reports of the Federal Supreme Court: BGE 138 III 59 = ATF 138 III 59 = DTF 138 III 59
REPORTER = {"de": "BGE", "fr": "ATF", "it": "DTF"}


def _parse_glossary() -> list[dict[str, list[str]]]:
    """Rows of {lang: [official form, *aliases]} from STATUTE_GLOSSARY."""
    rows = []
    for line in STATUTE_GLOSSARY.splitlines()[1:]:
        for entry in line.split(" — "):
            forms = [re.findall(r"[\w.]+", part) for part in entry.split(" / ")]
            rows.append(dict(zip(["de", "fr", "it"], forms)))
    return rows


GLOSSARY = _parse_glossary()


def _form_index() -> dict[str, dict[str, list[str]]]:
    """Abbreviation -> its glossary row, for abbreviations that belong to exactly one statute."""
    rows: dict[str, list[dict[str, list[str]]]] = {}
    for row in GLOSSARY:
        for form in {f for forms in row.values() for f in forms}:
            rows.setdefault(form, []).append(row)
    return {form: r[0] for form, r in rows.items() if len(r) == 1}


_FORMS = _form_index()
_TOKEN = re.compile(r"(?<![\w.])[A-Z]\w*\.?")
_REPORTER = re.compile(r"\b(?:BGE|ATF|DTF)\b")


def localize(text: str, lang: str) -> str:
    """Statute abbreviations and the reporter in the official form of ``lang``:
    "art. 271 OR, BGE 138 III 59" -> "art. 271 CO, ATF 138 III 59" for French."""
    if lang not in REPORTER:
        return text

    def sub(m: re.Match) -> str:
        tok = m.group(0)
        core = tok if tok in _FORMS else tok.rstrip(".")  # "Cst." keeps its dot, "ZGB." ends a sentence
        row = _FORMS.get(core)
        if row is None or core in row[lang]:
            return tok
        return row[lang][0] + tok[len(core):]

    return _REPORTER.sub(REPORTER[lang], _TOKEN.sub(sub, text))
