from __future__ import annotations

import re

from bm25s import stopwords as sw

_STOP = {
    "de": set(sw.STOPWORDS_GERMAN), "fr": set(sw.STOPWORDS_FRENCH),
    "it": set(sw.STOPWORDS_ITALIAN), "en": set(sw.STOPWORDS_EN),
}
_WORD = re.compile(r"[^\W\d_]+")


def detect_language(text: str, default: str = "en") -> str:
    """Language of a question, by counting function words; `default` when there are none."""
    words = [w.lower() for w in _WORD.findall(text)]
    scores = {lang: sum(w in stop for w in words) for lang, stop in _STOP.items()}
    best = max(scores, key=scores.__getitem__)
    return best if scores[best] > 0 else default
