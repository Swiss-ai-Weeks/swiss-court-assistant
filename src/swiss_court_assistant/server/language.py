from __future__ import annotations

import re
from bm25s import stopwords as sw

_EN_EXTRA = {"can", "could", "may", "might", "must", "should", "would", "does", "do", "did", "what", "when",
             "how", "which", "who", "whom", "whose", "why", "during", "after", "before", "has", "have", "had",
             "his", "her", "my", "our", "your", "its", "under", "about", "against", "without"}
_STOP = {
    "de": set(sw.STOPWORDS_GERMAN), "fr": set(sw.STOPWORDS_FRENCH),
    "it": set(sw.STOPWORDS_ITALIAN), "en": set(sw.STOPWORDS_EN) | _EN_EXTRA,
    # Detect Spanish drift too, even though it is not a corpus search language.
    "es": {"el", "la", "los", "las", "del", "de", "un", "una", "que", "en", "por", "para", "con",
           "su", "sus", "es", "son", "se", "y", "al", "como", "pero", "este", "esta"},
}
_LEGAL = {
    "de": set("hundebiss notwehr angriff straflos strafrecht schadenersatz haftung zahlungsverzug entlassung kündigung mietrecht verjährung".split()),
    "fr": set("responsabilité dommage licenciement bail congé recours délai prescription arrêt préjudice droit visite enfant divorce permis construire voisin opposition entretien conjoint imposition impôt assurance rechute causalité".split()),
    "it": set("responsabilità danno morale risarcimento licenziamento abusivo contratto lavoro diritto visita minore affidamento ricorso tardivo restituzione termine locazione disdetta medica paziente".split()),
    "es": set("establece responsabilidad titular animal daño despido".split()),
}
_WORD = re.compile(r"[^\W\d_]+")


def _scores(text: str):
    # Uppercase OR is a German act, not English's conjunction "or".
    words = [w.lower() for w in _WORD.findall(text) if w not in {"OR", "CO", "CP", "CC"}]
    return sorted(((sum(w in stop for w in words), lang) for lang, stop in _STOP.items()), reverse=True)


def detect_language(text: str, default: str = "de") -> str:
    """Function words for prose, legal vocabulary/script for keyword queries, explicit fallback.

    The German default is used only without evidence, never to override a clear English question
    mentioning OR. Callers measuring unknown-language prose can still pass default="".
    """
    text = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith(">"))
    (best, lang), (second, _), *_ = _scores(text)
    if best >= 2 and best > second:
        return lang
    words = {w.lower() for w in _WORD.findall(text)}
    clues = sorted(((len(words & vocab), lang) for lang, vocab in _LEGAL.items()), reverse=True)
    if clues[0][0] > clues[1][0]:
        return clues[0][1]
    if best > second:
        return lang
    if re.search(r"[äöüß]|\b(?:OR|StGB|ZGB|ZPO|SchKG)\b", text):
        return "de"
    if re.search(r"[ìò]|\b(?:cpv|Cost)\b", text):
        return "it"
    if re.search(r"[éèçœâêëîïôûÿ]|\b(?:CO|CP|Cst)\b", text):
        return "fr"  # CO/CP alone are ambiguous between fr/it; vocabulary above takes precedence.
    return default


def language_matches(text: str, wanted: str) -> bool:
    """Whole-answer check plus confident sentence-level drift (not merely the dominant language)."""
    if detect_language(text, default="") != wanted:
        return False
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        if len(_WORD.findall(sentence)) < 8:
            continue
        (best, lang), (second, _), *_ = _scores(sentence)
        if lang != wanted and best >= 3 and best - second >= 2:
            return False
    return True
