from __future__ import annotations

import re

from bm25s import stopwords as sw

# bm25s's English list is short; questions need these ("Can an employer … during …" tied with German on "an")
_EN_EXTRA = {"can", "could", "may", "might", "must", "should", "would", "does", "do", "did", "what", "when",
             "how", "which", "who", "whom", "whose", "why", "during", "after", "before", "has", "have", "had",
             "his", "her", "my", "our", "your", "its", "under", "about", "against", "without"}
_STOP = {
    "de": set(sw.STOPWORDS_GERMAN), "fr": set(sw.STOPWORDS_FRENCH),
    "it": set(sw.STOPWORDS_ITALIAN), "en": set(sw.STOPWORDS_EN) | _EN_EXTRA,
}
_WORD = re.compile(r"[^\W\d_]+")


def detect_language(text: str, default: str = "en") -> str:
    """Language of a question, by counting function words; `default` when there are none or two
    languages tie. Quoted lines ("> …", a passage the user replies to) are not the user's language."""
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(">"))
    words = [w.lower() for w in _WORD.findall(text)]
    scores = sorted(((sum(w in stop for w in words), lang) for lang, stop in _STOP.items()), reverse=True)
    (best, lang), (second, _) = scores[0], scores[1]
    return lang if best > second else default
