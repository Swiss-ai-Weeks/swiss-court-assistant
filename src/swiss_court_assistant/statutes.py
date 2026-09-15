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
