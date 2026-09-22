"""Stable Swiss decision identities. No benchmark-specific aliases or relevance judgements."""
from __future__ import annotations

import re

_BGE = re.compile(r"^(?:bge[_ ]*(?:historical[_ ]*)?)?(?:(?:BGE|ATF|DTF)[_ ]*)?(\d{1,3})[_ ]+([IVX]+[ab]?)[_ ]+(\d+)\s*$", re.I)
_DOCKET = re.compile(r"\b\d{1,2}[A-Z][_. ]\d{1,5}/\d{4}\b")
_EXACT = re.compile(r"(?:bge[_ ]*)?(?:BGE|ATF|DTF)?[_ ]*\d{1,3}[_ ]+[IVX]+[ab]?[_ ]+\d+|"
                    r"(?:BVGE|ATAF|DTAF)\s+\d{4}/\d+|\d{1,2}[A-Z][_. ]\d+/\d{4}", re.I)


def canonical_id(identifier: str) -> str:
    """Normalise only unambiguous BGE identities, preserving all other IDs unchanged."""
    if match := _BGE.fullmatch(identifier):
        volume, division, page = match.groups()
        division = division.upper()
        if division.endswith(('A', 'B')):
            division = division[:-1] + division[-1].lower()
        return f"bge_BGE_{volume}_{division}_{page}"
    return identifier


def docket_key(ref: str) -> str:
    ref = ref.strip().split('#')[0]
    canonical = canonical_id(ref)
    if canonical.startswith('bge_BGE_'):
        return canonical
    # ID forms used by exports, e.g. bger_1A.122_2005 and bvger_BVGE 2013_10.
    ref = re.sub(r"^(?:bger|bvger|bstger)_", "", ref, flags=re.I)
    return re.sub(r"[^a-z0-9]", "", ref.lower())


def own_dockets(text: str | None) -> set[str]:
    """Dockets from a BGE's *own heading*, never from reasoning or citations to other cases."""
    text = text or ''
    start = text.find('Urteilskopf')
    if start >= 0:
        heading = re.split(r"\n(?:Regeste|Regesto|Sachverhalt|Erwägungen)\b", text[start:], maxsplit=1)[0]
    else:
        # Upstream's three-line reporter header often already includes the own docket.
        heading = text.split('\n\n', 1)[0]
    return set(_DOCKET.findall(heading))


def exact_reference(query: str) -> bool:
    """A bare reporter/docket lookup must not silently become a loosely related semantic search."""
    return bool(_EXACT.fullmatch(query.strip()) or re.fullmatch(r"(?:bge|bger|bvger|bstger)_.+", query.strip()))
