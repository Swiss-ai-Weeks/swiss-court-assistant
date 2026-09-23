"""A search query in German, French and Italian, for the search-only API.

The agent writes its own query per language; `/api/search` receives one query in one language, and
searched the French and Italian decisions with the German words (and vice versa). BM25 then finds
nothing, and the reranker compares languages. The local LLM (no external service) writes the three
versions; statute abbreviations are copied unchanged and mapped deterministically by `localize`
afterwards, since the model otherwise invents them ("Art. 41 DTF"). On any failure the caller keeps
the original query for every language, which is the previous behaviour.
"""
from __future__ import annotations

import logging
import re
from collections import OrderedDict

import httpx

from .llm import LLM_KEY, LLM_URL, served_model
from .mentions import _MENTION

log = logging.getLogger(__name__)

PROMPT = """Translate this Swiss legal search query into German, French and Italian, using the terms Swiss courts use (e.g. Genugtuung / tort moral / riparazione morale; fristlose Entlassung / licenciement immédiat / licenziamento immediato). Copy article numbers and statute abbreviations (OR, ZGB, LAA, ...) exactly as they are in the query. Keep it a search query: no explanations. Answer exactly three lines:
de: ...
fr: ...
it: ...

Query: {query}"""

_LINE = re.compile(r"^\s*(de|fr|it)\s*:\s*(.+?)\s*$", re.M)


class QueryTranslator:
    def __init__(self, url: str = LLM_URL, size: int = 4096, timeout: float = 10.0):
        self.url = url.rstrip("/") + "/chat/completions"
        self.timeout = timeout
        self.size = size
        self._model: str | None = None
        self._cache: OrderedDict[str, dict[str, str]] = OrderedDict()

    async def translate(self, query: str) -> dict[str, str] | None:
        if query in self._cache:
            self._cache.move_to_end(query)
            return self._cache[query]
        self._model = self._model or served_model()
        body = {"model": self._model, "messages": [{"role": "user", "content": PROMPT.format(query=query)}],
                "temperature": 0, "max_tokens": 200, "chat_template_kwargs": {"enable_thinking": False}}
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(self.url, json=body, headers={"Authorization": f"Bearer {LLM_KEY}"})
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"] or ""
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            log.warning("query translation unavailable (%s); searching with the original query", e)
            return None
        found = {lang: sanitize(query, q) for lang, q in _LINE.findall(text) if len(q) <= 4 * len(query) + 40}
        if set(found) != {"de", "fr", "it"}:
            log.warning("unusable query translation for %r: %r", query, text[:200])
            return None
        self._cache[query] = found
        if len(self._cache) > self.size:
            self._cache.popitem(last=False)
        return found


_ABBREVIATION = re.compile(r"\b[A-Z][A-Za-z]*[A-Z][a-z]*\b\.?")
_PARAGRAPH = re.compile(r"§+\s*\d+\w*")


def sanitize(original: str, translated: str) -> str:
    """Drop statute references the model added: articles, § signs and law abbreviations not in the query.

    Measured on the OpenCaseLaw development queries: "Raub Gewalt Drohung" came back as
    "Raub Gewalt Drohung OR § 143 ZGB", and invented articles pulled unrelated statute candidates.
    """
    from swiss_court_assistant.statute_links import canon, query_articles  # statute_links imports server code

    articles = query_articles(original)
    codes = {canon(t) for t in _ABBREVIATION.findall(original)}
    text = _MENTION.sub(lambda m: m.group(0) if (canon(m.group('code')), m.group('num').replace(' ', ''))
                        in articles else ' ', translated)
    if '§' not in original:
        text = _PARAGRAPH.sub(' ', text)
    text = _ABBREVIATION.sub(lambda m: m.group(0) if canon(m.group(0)) in codes else ' ', text)
    text = re.sub(r'\s*/\s*(?=/|$)|^\s*/\s*', ' ', text)
    # Keep the articles the user named even where the model garbled them ("Art. 41 DTF").
    kept = query_articles(text)
    for m in _MENTION.finditer(original):
        if (canon(m.group('code')), m.group('num').replace(' ', '')) not in kept:
            text = m.group(0) + ' ' + text
    return ' '.join(text.split()) or original
