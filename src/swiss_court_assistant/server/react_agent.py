from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from collections.abc import AsyncIterator
from typing import Annotated, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.utils.json import parse_partial_json
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, ValidationError

from .agent import AgentEvent, Cite, Delta, Status, Thought, ToolEnd, ToolStart, Verdict
from .corpus import Corpus, Passage
from .language import detect_language
from .llm import LLM_KEY, LLM_THINKING, LLM_URL, served_model
from .schemas import Message, Source

log = logging.getLogger(__name__)

READ_WINDOW = 8000
MAX_TOOL_CALLS = 8  # then the agent has to write the answer
LANGUAGE_NAMES = {"de": "German", "fr": "French", "it": "Italian", "rm": "Romansh", "en": "English"}
RECURSION_LIMIT = 2 * MAX_TOOL_CALLS + 6


# ── structured answer ───────────────────────────────────────────────────
class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class CitationPart(BaseModel):
    type: Literal["citation"]
    decision_id: str
    chunk_id: str | None = None
    quote: str = Field(description="verbatim span of the decision; highlighted in the UI")
    explanation: str = Field(description="why the passage supports the preceding text")


class AgentAnswer(BaseModel):
    """The final answer; the NIM decodes it against this model's JSON schema."""

    answer: list[Annotated[TextPart | CitationPart, Field(discriminator="type")]]


# vLLM structured outputs. A whitespace-free EBNF grammar would also rule out the rare padding loop that
# AnswerStream.stuck() catches, but on this NIM it decodes about 6x slower (18 vs 104 tokens/s).
ANSWER_FORMAT = {"type": "json_schema",
                 "json_schema": {"name": "agent_answer", "schema": AgentAnswer.model_json_schema()}}

SUPPORT_PROMPT = """You check a legal answer against its sources. Given one passage from a Swiss court decision and one statement from an answer that cites it, reply {"supported": true} if the passage states or directly implies the statement, and {"supported": false} if it does not (it is about something else, says less than the statement claims, or contradicts it). Judge only against this passage, not your own legal knowledge."""
SUPPORT_FORMAT = {"type": "json_schema", "json_schema": {"name": "support", "schema": {
    "type": "object", "properties": {"supported": {"type": "boolean"}}, "required": ["supported"]}}}
MAX_CHECKS = 8  # grounding checks per answer

RESEARCH_PROMPT = """You are a legal research assistant for Swiss case law, working on a corpus of {n_decisions:,} Swiss court decisions (Federal Supreme Court, other federal courts and cantonal courts), written in German, French or Italian.

Research the user's question with the tools, then call write_answer.
- semantic_search(query_de, query_fr, query_it): finds passages by meaning. Start here for most questions. Write the same search in German, French and Italian, each phrased like a Swiss court in that language, with that language's legal terms and statute abbreviations, e.g. query_de="Anfechtung der Kündigung wegen Verstoss gegen Treu und Glauben, Art. 271 OR", query_fr="annulation du congé contraire aux règles de la bonne foi, art. 271 CO", query_it="annullamento della disdetta contraria alla buona fede, art. 271 CO". Each query searches the decisions in its language. If results are thin, search again with other wording.
- keyword_search(keyword): exact words, e.g. a statute "Art. 271a OR", a docket number "4A_705/2016", a rare term. Put exact phrases in double quotes.
- read_decision(decision_id, offset): reads a decision's full text, 8,000 characters per call, to check the context or find the decisive reasoning.
- citing_decisions(decision_id): how many later decisions cite it, and the most recent ones. Search results already show "cited by N" — prefer decisions later courts still rely on, and check a leading case before resting the answer on it.
- write_answer(): call it as soon as the passages you found answer the question, or when more searching is unlikely to help. You have at most {max_calls} tool calls.
Prefer passages where a court states the rule and its reasoning over passages that only mention it.
Lines starting with ">" quote text the user selected (from a decision, named on the "> —" line, or from an earlier answer); the question below them is about that text."""

ANSWER_PROMPT = """You are a legal research assistant for Swiss case law. Using only the tool results in this conversation, answer the user's last question as a JSON object {"answer": [...]} whose parts alternate between text and citations:
- A "text" part holds one or two sentences of the answer (Markdown allowed). Write in the language of the user's question and name decisions by court and docket number.
- A "citation" part follows the text it supports. It gives the decision_id of a passage from the tool results, its chunk_id (only for search results; null for text read with read_decision), a "quote" copied character for character from that passage in the decision's own language (never translated or shortened; one to three consecutive sentences, at most about 300 characters), and an "explanation": one sentence in the user's language on why the passage supports the text.
- Every legal statement needs a citation. Do not add holdings, facts or statutes that are not in the tool results. If the results do not answer the question, say so and summarize what was found.

Shape:
{"answer": [
  {"type": "text", "text": "First statement."},
  {"type": "citation", "decision_id": "<from a tool result>", "chunk_id": "<from a tool result>", "quote": "<verbatim>", "explanation": "<why it supports the statement>"},
  {"type": "text", "text": "Second statement."},
  {"type": "citation", "decision_id": "...", "chunk_id": "...", "quote": "...", "explanation": "..."}
]}"""


# ── tools ───────────────────────────────────────────────────────────────
def _hit_block(i: int, p: Passage) -> str:
    erw = f" · E. {', '.join(p.erwaegungen)}" if p.erwaegungen else ""
    regeste = " · Regeste" if p.section == "regeste" else ""
    cited = f" · cited by {p.cited_by} later decisions" if p.cited_by else ""
    return (f"Result {i}: decision_id={p.decision_id} chunk_id={p.chunk_id}\n"
            f"{p.court_label} {p.docket} · {p.date or 'undated'} · {p.language}{erw}{regeste}{cited}\n{p.text}")


def _hits_result(hits: list[Passage]) -> tuple[str, dict]:
    if not hits:
        return "No passages found.", {"summary": "no passages"}
    decisions = sorted({h.decision_id for h in hits})
    return ("\n\n".join(_hit_block(i, h) for i, h in enumerate(hits, 1)),
            {"summary": f"{len(hits)} passages from {len(decisions)} decision{'s' * (len(decisions) > 1)}",
             "decisions": decisions})


def _docket_matches(corpus: Corpus, query: str) -> str:
    found: dict[str, str] = {}
    for tok in re.split(r'[\s"]+', query):
        tok = tok.strip(".,;:()")
        if len(tok) >= 5 and re.search(r"\d", tok) and re.search(r"\D", tok):
            for did in corpus.decisions.find(tok):
                found[did] = corpus.decisions.summary(did).docket
    return ", ".join(f"{did} ({docket})" for did, docket in list(found.items())[:5])


def _failed(e: Exception) -> tuple[str, dict]:
    log.exception("tool failed")
    return f"The tool failed: {e}", {"summary": "failed", "error": True}


def make_tools(corpus: Corpus) -> list:
    @tool(response_format="content_and_artifact")
    async def semantic_search(query_de: str, query_fr: str, query_it: str) -> tuple[str, dict]:
        """Find passages by meaning. Give the same search in German, French and Italian, each phrased the
        way a Swiss court writes in that language, with its legal terms and statute abbreviations
        (OR / CO / CO). Each query searches only the decisions in its language. Returns the best
        passages with decision_id and chunk_id."""
        try:
            hits = await corpus.semantic_search_by_language({"de": query_de, "fr": query_fr, "it": query_it})
        except Exception as e:
            return _failed(e)
        text, artifact = _hits_result(hits)
        if hits:
            per = Counter(h.language for h in hits)
            artifact["summary"] += " · " + ", ".join(f"{n} {lang}" for lang, n in sorted(per.items()))
        return text, artifact

    @tool(response_format="content_and_artifact")
    async def keyword_search(keyword: str) -> tuple[str, dict]:
        """Full-text search for exact words in the decisions. Text in double quotes is an exact
        phrase ("Art. 271a OR", "4A_705/2016"); other words must all appear. Returns passages."""
        try:
            text, artifact = _hits_result(await corpus.keyword_search(keyword))
        except Exception as e:
            return _failed(e)
        if dockets := _docket_matches(corpus, keyword):  # docket numbers are rarely in the passage text
            text = f"Docket numbers in the query match: {dockets}. Read them with read_decision.\n\n{text}"
            artifact["summary"] += " · docket match"
        return text, artifact

    @tool(response_format="content_and_artifact")
    async def read_decision(decision_id: str, offset: int = 0) -> tuple[str, dict]:
        """Read one decision: metadata, Regeste and full text, 8,000 characters per call.
        Takes a decision_id (or a docket number). Pass offset to continue reading."""
        ids = corpus.decisions.find(decision_id)
        if not ids:
            return f"No decision {decision_id!r} in the corpus.", {"summary": "not found", "error": True}
        if len(ids) > 1:
            return (f"Several decisions match: {', '.join(ids[:10])}. Call again with one decision_id.",
                    {"summary": f"{len(ids)} decisions match"})
        d = corpus.decisions.get(ids[0])
        start = max(0, min(offset, len(d.full_text)))
        end = min(len(d.full_text), start + READ_WINDOW)
        head = [f"decision_id: {d.decision_id}", f"{d.court_label} {d.docket} · {d.date or 'undated'} · {d.language}"]
        if d.title:
            head.append(f"Title: {d.title}")
        if d.regeste and start == 0:
            head.append(f"Regeste: {d.regeste}")
        more = f" Call read_decision(decision_id={d.decision_id!r}, offset={end}) to continue." \
            if end < len(d.full_text) else ""
        head.append(f"Full text, characters {start}-{end} of {len(d.full_text)}.{more}")
        return ("\n".join(head) + "\n\n" + d.full_text[start:end],
                {"summary": f"{d.docket}, characters {start:,}–{end:,} of {len(d.full_text):,}",
                 "decisions": [d.decision_id]})

    @tool(response_format="content_and_artifact")
    async def citing_decisions(decision_id: str) -> tuple[str, dict]:
        """How many later decisions cite this one, and the most recent of them. Use it to check whether
        a precedent is still followed before relying on it."""
        if corpus.citations is None:
            return "The citation index is not available.", {"summary": "unavailable", "error": True}
        ids = corpus.decisions.find(decision_id)
        did = ids[0] if ids else decision_id
        try:
            cited_by, cites = await asyncio.to_thread(corpus.citations.counts, did)
            recent = await asyncio.to_thread(corpus.citations.cited_by, did, 10)
        except Exception as e:
            return _failed(e)
        if not cited_by:
            return (f"No later decision in the corpus cites {did}.",
                    {"summary": "cited by 0", "decisions": [did]})
        # name the corpus id only for decisions that are actually here: a court+docket label looks like
        # an id, and the agent then wastes calls on read_decision("zh_gerichte 4A_82/2024")
        lines = [f"- {r.date or 'undated'} · {r.court or '?'} {r.docket or r.decision_id}"
                 f" ({'decision_id=' + r.decision_id if r.in_corpus else 'not in this corpus'})" for r in recent]
        return (f"{did} is cited by {cited_by} later decisions and itself cites {cites}. The most recent citing "
                f"decisions (only those with a decision_id can be opened with read_decision):\n" + "\n".join(lines),
                {"summary": f"cited by {cited_by}", "decisions": [did]})

    @tool
    async def write_answer() -> str:
        """Finish the research and write the answer. Call it once the passages found answer the
        question, or when more searching is unlikely to help."""
        return ""  # never runs: ResearchThenAnswer answers instead

    return [semantic_search, keyword_search, read_decision, citing_decisions, write_answer]


class ResearchThenAnswer(AgentMiddleware):
    """Every research step must be a tool call (constrained decoding), so the model cannot answer
    from memory or in free text. When it calls write_answer, the answer is generated instead,
    constrained to the AgentAnswer schema and streamed token by token."""

    def __init__(self, answer_llm: ChatOpenAI, max_tool_calls: int = MAX_TOOL_CALLS):
        super().__init__()
        self.answer_llm, self.max_tool_calls = answer_llm, max_tool_calls

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse | AIMessage:
        used = sum(isinstance(m, ToolMessage) for m in request.messages)
        choice = ({"type": "function", "function": {"name": "write_answer"}}
                  if used >= self.max_tool_calls else "required")
        response = await handler(request.override(tool_choice=choice))
        result = response.result if isinstance(response, ModelResponse) else [response]
        calls = [tc for m in result if isinstance(m, AIMessage) for tc in m.tool_calls]
        if not any(tc["name"] == "write_answer" for tc in calls):
            return response
        # Passages and searches in three languages pull the answer away from the question's language
        # (a French question got a German answer), so name it when the question makes it clear.
        question = next((m.content for m in reversed(request.messages) if isinstance(m, HumanMessage)), "")
        language = LANGUAGE_NAMES.get(detect_language(str(question), default=""))
        final = "Write the final answer now." + (
            f" Write its text parts and explanations in {language}." if language else "")
        return await self.answer_llm.ainvoke([SystemMessage(ANSWER_PROMPT), *request.messages, HumanMessage(final)])


# ── streaming the final JSON ────────────────────────────────────────────
class AnswerStream:
    """Parses the final JSON answer while it streams.

    feed() returns what is ready: text parts as they grow (each TextPart holds only the
    new text) and citation parts once they are complete.
    """

    def __init__(self) -> None:
        self.buf = ""
        self.done = 0        # parts fully emitted
        self.shown = ""      # text of the open part already emitted
        self.last = ""       # last character sent, to space consecutive parts
        self._parsed_at = 0
        self.prose = False   # not JSON: only possible if constrained decoding is off

    def feed(self, text: str) -> list[TextPart | CitationPart]:
        self.buf += text
        head = self.buf.lstrip()
        if self.prose or (head and head[0] not in "{`"):
            self.prose = True  # passed through by finish()
            return []
        # parse_partial_json is linear in the buffer: parse every ~48 chars, not every token
        if len(self.buf) - self._parsed_at < 48 and "}" not in text:
            return []
        self._parsed_at = len(self.buf)
        return self._advance(final=False)

    def stuck(self) -> bool:
        """Constrained decoding lets the model pad the JSON with whitespace, and it sometimes never stops."""
        return len(self.buf) - len(self.buf.rstrip()) >= 64

    def finish(self) -> list[TextPart | CitationPart]:
        if self.prose or "{" not in self.buf:
            if self.buf.strip():
                log.warning("the model answered in prose instead of the JSON format")
            return [TextPart(type="text", text=self.buf.strip())] if self.buf.strip() else []
        try:
            AgentAnswer.model_validate_json(self.buf[self.buf.find("{"):self.buf.rfind("}") + 1])
        except ValidationError as e:
            log.warning("final answer does not match the schema: %s", str(e)[:300])
        return self._advance(final=True)

    def _parts(self, final: bool) -> list | None:
        start = self.buf.find("{")
        if start < 0:
            return None
        body = self.buf[start:]
        obj = None
        if final:
            try:
                obj = json.loads(body[:body.rfind("}") + 1])
            except ValueError:
                pass
        if obj is None:
            try:
                obj = parse_partial_json(re.sub(r"\s*`*\s*$", "", body))
            except ValueError:
                return None
        parts = obj.get("answer") if isinstance(obj, dict) else None
        return parts if isinstance(parts, list) else None

    def _text(self, piece: str) -> TextPart:
        if not self.shown and self.last and not self.last.isspace() and piece[:1] not in " \n.,;:!?)":
            piece = " " + piece  # parts are written as separate sentences
        self.last = piece[-1]
        return TextPart(type="text", text=piece)

    def _advance(self, final: bool) -> list[TextPart | CitationPart]:
        parts = self._parts(final) or []
        out: list[TextPart | CitationPart] = []
        for i in range(self.done, len(parts)):
            p = parts[i]
            is_open = not final and i == len(parts) - 1
            if not isinstance(p, dict):
                if is_open:
                    break
                self.done += 1
                continue
            kind = p.get("type") or ("citation" if "quote" in p else "text")
            if kind == "text":
                t = p.get("text") or ""
                # the tail of a half-streamed string can still change (e.g. a cut \u escape): hold it back
                ready = t[:max(len(self.shown), len(t) - 12)] if is_open else t
                if ready.startswith(self.shown) and len(ready) > len(self.shown):
                    out.append(self._text(ready[len(self.shown):]))
                    self.shown = ready
                elif not self.shown.startswith(ready):  # a shorter re-parse is harmless
                    log.warning("streamed text diverged from what was already sent")
                if is_open:
                    break
                self.done, self.shown = self.done + 1, ""
            else:
                if is_open:  # wait until the citation is complete
                    break
                try:
                    out.append(CitationPart.model_validate(p | {"type": "citation"}))
                    self.last = "]"
                except ValidationError:
                    log.warning("dropping malformed citation %r", p)
                self.done += 1
        return out


# ── quotes → highlights ─────────────────────────────────────────────────
_FOLD = str.maketrans({"„": '"', "“": '"', "”": '"', "«": '"', "»": '"', "‹": "'", "›": "'",
                       "‘": "'", "’": "'", "–": "-", "—": "-"})
_ELLIPSIS = re.compile(r"\[?(?:\.\.\.|…)\]?")


def _fold(s: str) -> tuple[str, list[int]]:
    """Lowercase, unify quotes/dashes and drop all whitespace, keeping each char's original index."""
    chars, idx = [], []
    for i, ch in enumerate(s):
        if not ch.isspace():
            chars.append(ch.translate(_FOLD).lower())
            idx.append(i)
    return "".join(chars), idx


def locate_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Char span of `quote` in `text`, tolerant to whitespace, quote styles and '...' elisions."""
    hay, idx = _fold(text)
    segments = [s for s in (_fold(p)[0] for p in _ELLIPSIS.split(quote)) if len(s) >= 8]
    if not segments or not hay:
        return None
    start = hay.find(segments[0])
    if start < 0:  # anchor on the quote's first words
        q = "".join(segments)
        if len(q) < 40 or (start := hay.find(q[:40])) < 0:
            return None
        end = hay.find(q[-40:], start)
        if len(q) >= 60 and 0 <= end - start <= len(q) * 1.5:  # only the middle differs
            return idx[start], idx[end + 39] + 1
        n = 40  # else the longest verbatim prefix, if it is most of the quote
        while n < len(q) and start + n < len(hay) and hay[start + n] == q[n]:
            n += 1
        return (idx[start], idx[start + n - 1] + 1) if n >= 0.6 * len(q) else None
    end = start + len(segments[0])
    for seg in segments[1:]:
        nxt = hay.find(seg, end)
        if nxt < 0 or nxt - end > 3000:
            break
        end = nxt + len(seg)
    return idx[start], idx[end - 1] + 1


class _Citations:
    """Turns CitationParts into numbered Sources; the same span keeps its number."""

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.by_span: dict[tuple, Source] = {}
        self.seen: set[str] = set()  # decisions the tools returned this turn

    async def resolve(self, c: CitationPart) -> Source | None:
        store = self.corpus.decisions
        chunk = await self.corpus.chunk(c.chunk_id) if c.chunk_id else None
        ids = store.find(c.decision_id) or ([chunk.decision_id] if chunk else [])
        did = (chunk.decision_id if chunk and chunk.decision_id in ids else ids[0]) if ids else None
        span, section = None, "body"
        if did:
            d = store.get(did)
            span = locate_quote(d.full_text, c.quote)
            if not span and d.regeste and locate_quote(d.regeste, c.quote):
                section = "regeste"
        if not span and section != "regeste":
            # the model sometimes pins a quote on the wrong decision: look in the others it was shown
            for other in sorted(self.seen - {did}):
                if sp := locate_quote(store.get(other).full_text, c.quote):
                    log.warning("quote cited from %s found in %s instead", did, other)
                    did, span = other, sp
                    break
        if did is None:
            log.warning("citation of unknown decision %r", c.decision_id)
            return None
        d = store.get(did)
        erw, verified = [], True
        cid = c.chunk_id if chunk and chunk.decision_id == did else f"{did}#?"
        if span:
            inside = [p for p in await self.corpus.chunks_of(did)
                      if p.char_start is not None and p.char_start <= span[0] < p.char_end]
            if inside:
                section, erw, cid = inside[-1].section, inside[-1].erwaegungen, inside[-1].chunk_id
        elif section == "regeste":
            pass
        elif chunk and chunk.decision_id == did:  # quote not verbatim: fall back to the whole passage
            verified, section, erw = False, chunk.section, chunk.erwaegungen
            span = (chunk.char_start, chunk.char_end) if chunk.char_start is not None else None
        else:
            verified = False
        if not verified:
            log.warning("quote not found verbatim in %s: %r", did, c.quote[:200])
        key = (did, span or section)
        if key in self.by_span:
            return self.by_span[key]
        source = Source(
            n=len(self.by_span) + 1, chunk_id=cid, decision_id=did,
            text=d.full_text[span[0]:span[1]] if span else c.quote, section=section, erwaegungen=erw,
            char_start=span[0] if span else None, char_end=span[1] if span else None, score=0.0,
            decision=store.summary(did), explanation=c.explanation, verified=verified,
        )
        self.by_span[key] = source
        return source


# ── agent ───────────────────────────────────────────────────────────────
def _history(messages: list[Message], limit: int = 8) -> list[BaseMessage]:
    """Earlier turns as plain text, citations replaced by docket numbers."""
    out: list[BaseMessage] = []
    for m in messages[-limit:]:
        if m.role == "user":
            out.append(HumanMessage(m.content))
        else:
            dockets = {s.n: s.decision.docket for s in m.sources or []}
            out.append(AIMessage(re.sub(r"\[(\d+)\]", lambda x: f" ({dockets.get(int(x[1]), 'source')})", m.content)))
    return out


class ReasoningChatOpenAI(ChatOpenAI):
    """ChatOpenAI that keeps vLLM's streamed reasoning (``delta.reasoning_content``), which
    langchain-openai drops, as ``additional_kwargs["reasoning"]`` on each chunk."""

    def _convert_chunk_to_generation_chunk(self, chunk, default_chunk_class, base_generation_info):
        gen = super()._convert_chunk_to_generation_chunk(chunk, default_chunk_class, base_generation_info)
        choices = chunk.get("choices") or []
        delta = (choices[0].get("delta") or {}) if choices else {}
        if gen is not None and (text := delta.get("reasoning_content") or delta.get("reasoning")):
            gen.message.additional_kwargs["reasoning"] = text
        return gen


class ReactAgent:
    name = "react"

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.model = served_model()
        common = dict(base_url=LLM_URL, api_key=LLM_KEY, model=self.model,
                      temperature=0.2, streaming=True)
        # the research steps reason before each tool call (streamed to the UI); the answer is constrained
        # JSON and starts right away
        research_llm = ReasoningChatOpenAI(**common, max_tokens=4096,
                                           extra_body={"chat_template_kwargs": {"enable_thinking": LLM_THINKING}})
        # passed in the request body as is, bypassing LangChain's own response_format handling
        answer_llm = ChatOpenAI(**common, max_tokens=4096, extra_body={
            "chat_template_kwargs": {"enable_thinking": False}, "response_format": ANSWER_FORMAT})
        # a separate short call per citation: does the passage actually say what the sentence claims?
        self.check_llm = ChatOpenAI(**{**common, "temperature": 0}, max_tokens=32, extra_body={
            "chat_template_kwargs": {"enable_thinking": False}, "response_format": SUPPORT_FORMAT})
        prompt = RESEARCH_PROMPT.format(n_decisions=len(corpus.decisions), max_calls=MAX_TOOL_CALLS)
        self.graph = create_agent(research_llm, make_tools(corpus), system_prompt=prompt,
                                  middleware=[ResearchThenAnswer(answer_llm)])
        log.info("react agent: %s at %s (thinking=%s)", self.model, LLM_URL, LLM_THINKING)

    async def _check(self, n: int, claim: str, source: Source) -> tuple[int, bool | None]:
        """Does the cited passage state the sentence it is attached to?"""
        try:
            reply = await self.check_llm.ainvoke([
                SystemMessage(SUPPORT_PROMPT),
                HumanMessage(f"PASSAGE ({source.decision.docket}):\n{source.text[:2000]}\n\n"
                             f"STATEMENT:\n{claim}")])
            return n, bool(json.loads(str(reply.content))["supported"])
        except (ValidationError, ValueError, KeyError, TypeError):
            log.warning("grounding check returned no verdict for [%d]", n, exc_info=True)
        except Exception:
            log.warning("grounding check failed for [%d]", n, exc_info=True)
        return n, None

    async def answer(self, question: str, history: list[Message]) -> AsyncIterator[AgentEvent]:
        cites = _Citations(self.corpus)
        stream, writing = AnswerStream(), False
        claim: list[str] = []  # the sentences since the last citation
        previous = ""  # several citations in a row all support the same sentence
        checks: list[asyncio.Task[tuple[int, bool | None]]] = []
        yield Status("thinking", "Planning the research")

        async def emit(parts: list[TextPart | CitationPart]) -> AsyncIterator[AgentEvent]:
            nonlocal previous
            for p in parts:
                if isinstance(p, TextPart):
                    claim.append(p.text)
                    yield Delta(p.text)
                elif (src := await cites.resolve(p)) is not None:
                    statement = " ".join("".join(claim).split())[-400:] or previous
                    claim.clear()
                    previous = statement
                    if statement and len(checks) < MAX_CHECKS:
                        checks.append(asyncio.create_task(self._check(src.n, statement, src)))
                    yield Cite(src)

        events = self.graph.astream(
            {"messages": [*_history(history), HumanMessage(question)]},
            {"recursion_limit": RECURSION_LIMIT}, stream_mode=["messages", "updates"])
        thought: list[str] = []  # reasoning since the last tool call
        async for mode, data in events:
            if mode == "messages":
                chunk = data[0]
                if not isinstance(chunk, AIMessageChunk):
                    continue
                if (text := chunk.additional_kwargs.get("reasoning")) and not writing:
                    thought.append(text)
                    yield Thought(text)
                if not isinstance(chunk.content, str) or not chunk.content:
                    continue
                if not writing:  # research calls are tool calls only, so text means the answer has begun
                    writing = True
                    yield Status("answer", "Writing the answer")
                async for ev in emit(stream.feed(chunk.content)):
                    yield ev
                if stream.stuck():
                    log.warning("the answer degenerated into whitespace; keeping the parts written so far")
                    break
                continue
            for update in data.values():
                for m in (update or {}).get("messages", []):
                    if isinstance(m, AIMessage) and m.tool_calls:
                        stream = AnswerStream()
                        said = " ".join("".join(thought).split()) or None
                        thought.clear()
                        for tc in m.tool_calls:
                            yield ToolStart(tc["id"], tc["name"], tc["args"], said)
                    elif isinstance(m, ToolMessage):
                        art = m.artifact if isinstance(m.artifact, dict) else {}
                        cites.seen.update(art.get("decisions", []))
                        yield ToolEnd(m.tool_call_id, m.name or "", art.get("summary", ""),
                                      m.status == "error" or bool(art.get("error")))
                        yield Status("thinking", "Reading the results")
        await events.aclose()  # cancels the model call if we broke off
        async for ev in emit(stream.finish()):
            yield ev
        for finished in asyncio.as_completed(checks):  # the checks ran while the answer was streaming
            n, supported = await finished
            if supported is not None:
                yield Verdict(n, supported)
