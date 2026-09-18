"""Statute articles named in an answer, linked to their text.

The agent reads articles (`read_law`) and names them in its answers — "Art. 259d CO", "art. 336c al. 1
let. c CO" — but its numbered citations nearly always point at decisions: of the first answers after
the statute tools went live, none cited the article it had read, so there was nothing to open. So every
article an answer names that the index holds becomes a link to its text, found deterministically after
the answer is written. These are references, not evidence: they are kept apart from the numbered
citations (`Message.sources`), which keep meaning "this passage supports the sentence", and they are
not grounding-checked.
"""

from __future__ import annotations

import re
from typing import Any

from .schemas import Source, StatuteRef

# "Art. 271a OR", "art. 336c al. 1 let. c CO", "Art. 56 Abs. 1 OR", "Artikel 12 StGB", "Art. 8 Ziff. 2 EMRK"
_MENTION = re.compile(
    r"\b(?:Art|art|Artikel|Article|article|Articolo|articolo)\.?\s*"
    r"(?P<num>\d+[a-z]{0,2}(?:\s?(?:bis|ter|quater|quinquies|sexies|septies|octies))?)"
    r"(?:\s*(?:Abs|al|cpv|para|§)\.?\s*\d+[a-z]?)?"
    r"(?:\s*(?:lit|let|lett)\.?\s*[a-z]\)?)?"
    r"(?:\s*(?:Ziff|ch|n|Nr)\.?\s*\d+)?"
    r"\s+(?P<code>[A-Z][A-Za-z]{1,7})\b")

NOTE = {
    "de": "Im Text der Antwort genannt. Angezeigt aus dem Gesetzestext im Index; nicht als Beleg für einen Satz zitiert.",
    "fr": "Cité dans le texte de la réponse. Affiché d'après le texte légal de l'index ; il ne sert pas de source à une phrase.",
    "it": "Menzionato nel testo della risposta. Mostrato dal testo di legge nell'indice; non citato come fonte di una frase.",
    "en": "Named in the answer. Shown from the statute text in the index; not cited as evidence for a sentence.",
}


def _pick(rows: list[dict[str, Any]], code: str, language: str) -> dict[str, Any]:
    """The language version the answer means: the one whose abbreviation it used (OR is German, CO
    French or Italian), in the answer's own language when that is one of them."""
    same_code = [r for r in rows if (r.get("abbreviation") or "").lower() == code.lower()] or rows
    return next((r for r in same_code if r["language"] == language), same_code[0])


def statute_links(store: Any, text: str, language: str | None) -> list[StatuteRef]:
    """Every article named in `text` that the index holds, once per distinct mention."""
    if not hasattr(store, "find_articles"):
        return []  # an index without statutes
    lang = language or "de"
    found: dict[tuple[str, str], list[dict[str, Any]]] = {}
    out: list[StatuteRef] = []
    seen: set[str] = set()
    for m in _MENTION.finditer(text):
        mention = m.group(0)
        if mention in seen:
            continue
        seen.add(mention)
        key = (m.group("code"), m.group("num").replace(" ", ""))
        if key not in found:
            found[key] = store.find_articles(*key)
        if not found[key]:
            continue  # not an act this index holds (or not an abbreviation at all: "Art. 5 The ...")
        row = _pick(found[key], key[0], lang)
        law = store.law(row["law_id"])
        out.append(StatuteRef(text=mention, source=Source(
            # n=0: not a numbered citation, so the preview shows no "cited in this answer" line for it
            n=0, chunk_id=f"{row['law_id']}#0", decision_id=row["law_id"], text=law.full_text,
            section="law", erwaegungen=[], char_start=None, char_end=None, score=0.0,
            decision=store.law_summary(row["law_id"]), explanation=NOTE.get(lang, NOTE["en"]), verified=True)))
    return out
