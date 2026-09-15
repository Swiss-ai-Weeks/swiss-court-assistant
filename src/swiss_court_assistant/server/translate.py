from __future__ import annotations

import asyncio
import os
import re
from collections import OrderedDict

import riva.client

from ..statutes import GLOSSARY, REPORTER

RIVA_URI = os.environ.get("SCA_TRANSLATE_URI", "localhost:50051")
RIVA_MODEL = os.environ.get("SCA_TRANSLATE_MODEL", "")  # "" = the NIM's only model
MAX_PIECE = 800  # characters per request text; the NMT model truncates long inputs

_SENTENCE = re.compile(r"(?<=[.;:!?])\s+(?=[A-ZÀ-ÖØ-Þ(«\"„])")
_ABBR = {lang: {form: row for row in GLOSSARY for form in row[lang]} for lang in REPORTER}
_TOKEN = re.compile(r"(?<![\w.])[A-Z]\w*\.?")
_CITE = re.compile(r"\b(BGE|ATF|DTF)(?= \d+ [IVX]+[a-z]? \d+)")
_ANONYMIZED = re.compile(r"\b([A-Z]{1,3})\.?_{2,}")  # "C.______" -> "C."; the runs make Riva loop
# Riva sometimes leaks its dnt markup: "< dnt > StGB < / dnt >", "VwVG</dnt", "3 dnt AsylG dnt", "(DNT)"
_MARKUP = re.compile(r"\s*<\s*/?\s*dnt\s*>?|\s*\(dnt\)|\s+dnt(?=\s)", re.I)
# ... or the hash it put in place of a protected phrase: "Article 278 (EA931cf29efd5e8b)"
_PLACEHOLDER = re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{16}\b", re.I)
_DEGENERATE = re.compile(r"(\S)(?:\s?\1){15,}")  # "?????…", "_ _ _ _ …"


class UntranslatableError(ValueError):
    pass


def pieces(text: str, limit: int = MAX_PIECE) -> list[str]:
    """Sentence-aligned pieces of at most ``limit`` characters (a longer sentence stays whole)."""
    out: list[str] = []
    for sentence in _SENTENCE.split(text):
        if out and len(out[-1]) + 1 + len(sentence) <= limit:
            out[-1] += " " + sentence
        else:
            out.append(sentence)
    return out


def protected(text: str, source: str, target: str) -> dict[str, str]:
    """Do-not-translate phrases: statute abbreviations and reporter names in ``text``, mapped to the
    target language's official form (OR -> CO). Unprotected, Riva renders "OR" as "Code of Civil
    Procedure" and "BGE" as "FBI"."""
    out: dict[str, str] = {}
    for tok in _TOKEN.findall(text):
        for form in (tok, tok.rstrip(".")):  # "Cst." keeps its dot, "ZGB." ends a sentence
            row = _ABBR.get(source, {}).get(form)
            if row:
                out[form] = row[target][0] if target in row else form
                break
    for m in _CITE.finditer(text):
        out[m.group(1)] = REPORTER.get(target, m.group(1))
    return out


def _clean(out: str) -> str:
    return _MARKUP.sub("", out).strip()


def _broken(out: str) -> bool:
    return bool(_DEGENERATE.search(out) or _PLACEHOLDER.search(out))


class Translator:
    """Translates cited passages with the Riva Translate NIM (gRPC); results are cached."""

    def __init__(self, cache_size: int = 512):
        self.client = riva.client.NeuralMachineTranslationClient(riva.client.Auth(uri=RIVA_URI))
        self._cache: OrderedDict[tuple[str, str, str], str] = OrderedDict()
        self._size = cache_size

    def _riva(self, texts: list[str], source: str, target: str, dnt: dict[str, str] | None) -> list[str]:
        r = self.client.translate(texts, RIVA_MODEL, source, target, dnt_phrases_dict=dnt or None)
        return [_clean(t.text) for t in r.translations]

    def _translate(self, text: str, source: str, target: str) -> str:
        text = _ANONYMIZED.sub(r"\1.", text)
        parts = pieces(text)
        out = self._riva(parts, source, target, protected(text, source, target))
        for i, t in enumerate(out):
            if _broken(t):  # retry the piece without dnt phrases, then give up
                t = self._riva([parts[i]], source, target, None)[0]
                if _broken(t):
                    raise UntranslatableError("The translation model could not translate this passage.")
                out[i] = t
        return " ".join(out)

    async def translate(self, text: str, source: str, target: str) -> str:
        text = " ".join(text.split())  # the decision texts break lines around citations
        key = (text, source, target)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        out = await asyncio.to_thread(self._translate, text, source, target)
        self._cache[key] = out
        if len(self._cache) > self._size:
            self._cache.popitem(last=False)
        return out
