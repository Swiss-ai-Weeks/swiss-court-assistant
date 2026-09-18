from __future__ import annotations

import asyncio
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from collections.abc import AsyncIterator, Callable
from contextvars import ContextVar
from typing import Annotated, Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_core.utils.json import parse_partial_json
from langchain_openai import ChatOpenAI
from langgraph.config import get_stream_writer
from pydantic import BaseModel, Field, ValidationError

from swiss_court_assistant.facets import AREAS, CANTONS, COURTS, PROCEEDINGS

from .agent import (AgentEvent, Cite, Clarify, Delta, Status, Thought, ToolEnd, ToolStart, Verdict,
                    original_question, with_clarification)
from .corpus import Corpus, FiltersUnavailable, Passage
from .decisions import court_label
from .language import detect_language
from .llm import LLM_KEY, LLM_THINKING, LLM_URL, served_model
from .schemas import Message, Source

log = logging.getLogger(__name__)

READ_WINDOW = 8000
MAX_TOOL_CALLS = 8  # then the agent has to write the answer
# Offered write_answer from the first step, the agent took it after one search on 19 of 33 eval
# questions (agent-eval/RESULTS.md) and answered from whatever came back — on exam questions those
# turns covered 59 % of the reference points against 72 % for turns that looked again. So
# write_answer is only offered once it has made this many research calls.
MIN_TOOL_CALLS = 2
# Left alone, the agent searches until it runs out of calls, rewording the same query (traced: 8
# near-identical semantic_search calls for a question the corpus cannot answer). Remind it earlier.
NUDGE_AFTER = 4
NUDGE = ("You have used {used} of {max_calls} tool calls. If the passages so far do not answer the "
         "question, call write_answer now and say that this corpus does not answer it: searching "
         "again with similar wording returns the same passages.")
LANGUAGE_NAMES = {"de": "German", "fr": "French", "it": "Italian", "rm": "Romansh", "en": "English"}
NO_ANSWER = {
    "de": "Die Recherche ist abgeschlossen, aber es konnte keine Antwort formuliert werden. Bitte stellen Sie die Frage noch einmal.",
    "fr": "La recherche est terminée, mais aucune réponse n'a pu être rédigée. Veuillez reposer la question.",
    "it": "La ricerca è conclusa, ma non è stato possibile formulare una risposta. La preghiamo di riformulare la domanda.",
    "en": "The research finished, but no answer could be written. Please ask the question again.",
}
# Every statement rested on citations that did not hold up, so nothing is left to show. Said without
# naming the machinery: what the user needs to know is that nothing here is sourced, and what to try.
NO_SUPPORT = {
    "de": "Keine der gefundenen Passagen belegt eine Antwort auf diese Frage — die Stellen, auf die sich die Antwort stützen sollte, sagen etwas anderes. Bitte formulieren Sie die Frage enger oder mit anderen Begriffen.",
    "fr": "Aucun des passages trouvés n'étaye une réponse à cette question : les extraits sur lesquels la réponse devait s'appuyer disent autre chose. Veuillez préciser la question ou la reformuler avec d'autres termes.",
    "it": "Nessuno dei passaggi trovati sostiene una risposta a questa domanda: i brani su cui la risposta avrebbe dovuto fondarsi dicono altro. La preghiamo di precisare la domanda o di riformularla con altri termini.",
    "en": "None of the passages found supports an answer to this question: the ones the answer would have rested on say something else. Please narrow the question or put it in different terms.",
}
RECURSION_LIMIT = 2 * MAX_TOOL_CALLS + 6
# The answer is drafted, checked against the tool results and, while the check finds problems, revised
# this many times before what passed is shown. Two revisions fix most drafts; a third mostly repeats.
MAX_REVISIONS = 2
# A statement whose citation failed and which comes back without one is the same statement when its
# words overlap this much (Jaccard on words of three letters or more).
SAME_STATEMENT = 0.6
# Constrained decoding lets the model pad the JSON with whitespace, and it sometimes never stops.
STUCK_AFTER = 64
# When the agent notes early on that something was not found, it is sent back once to look for it.
PUSHBACK = ("Before writing, you noted that the passages do not answer this: {gap}\n"
            "You have {left} tool calls left. Look for it once more — with clearly different wording, or with "
            "another tool (search_laws for a provision, keyword_search for an exact term or docket number, "
            "read_decision for the reasoning behind a passage) — then call write_answer again. If what is "
            "missing is something the case law may simply not contain (a fixed threshold, a rule of foreign "
            "law), call write_answer again right away. Either way the answer reports that point as not "
            "found; it is not filled in.")
_NOTHING = re.compile(r"(?i)^\W*(nothing|none|n/a|nichts|kein|rien|aucun|niente|nessun|all\b|everything|covered)")

# Asking the agent not to repeat a search only works if it notices that it is, so each turn
# remembers what it has searched for. Queries are compared as token sets, because the repeats are
# rewordings, not copies ("DSGVO Wettbewerbsrecht Sanktionen SVKG" then "DSGVO Wettbewerbsrecht").
SAME_SEARCH = 0.8  # Jaccard overlap at which two searches count as the same


@dataclass
class VerifiedAnswer:
    """What is left of a draft after the checks: parts, one Source per citation part, in order, and the
    grounding verdict for each (None when the check itself failed)."""

    parts: list[TextPart | CitationPart]
    sources: list[Source]
    supported: list[bool | None]
    dropped: int = 0  # statements removed because none of their citations held
    rounds: int = 0   # revisions the draft went through


@dataclass
class Turn:
    """What one turn's tools, middleware and answer() share.

    `make_tools` runs once and its tools are shared between turns, so per-turn state lives here, behind a
    ContextVar that answer() sets before the graph starts: the graph's tasks inherit the reference and
    mutate the same object, which is how the middleware hands the checked answer back."""

    searches: list[tuple[str, frozenset[str]]] = field(default_factory=list)
    # whether this turn may end with a question to the user (not in matters, evals, or right after one)
    may_ask: bool = False
    # the user's language: the research runs in German, French and Italian and pulls the model away
    # from it, and a reply to a question asked back mixes languages (see `original_question`)
    language: str | None = None
    pushback: str | None = None  # what write_answer returns when the agent is sent back to research
    pushed_back: bool = False    # only once per turn
    result: VerifiedAnswer | None = None


_turn: ContextVar[Turn | None] = ContextVar("turn", default=None)


def _tokens(text: str) -> frozenset[str]:
    """Words of three letters or more: "de", "la", "der" are in every query and distinguish nothing."""
    return frozenset(w for w in re.findall(r"\w+", text.lower()) if len(w) >= 3)


def _already_searched(text: str, kind: str = "decisions") -> str | None:
    """Remembers this search and, when it repeats one from this turn, says which.

    Two measures, because the traced repeats were of both kinds: rewording keeps most of the words
    (Jaccard), and dropping words searches for nothing the earlier query did not already cover
    (how much of this query it contained) — "DSGVO Wettbewerbsrecht Sanktionen SVKG" then "DSGVO
    Wettbewerbsrecht" only reaches a Jaccard of 0.73, but is fully contained.
    """
    turn = _turn.get()
    if turn is None:  # outside a turn (a tool called directly, e.g. from a test)
        return None
    searches = turn.searches
    tokens = _tokens(text)
    # the same words searched in the statutes are not a repeat of a search in the decisions
    for n, (was, seen) in enumerate(searches, 1):
        if was != kind or not tokens or not seen:
            continue
        both = len(tokens & seen)
        if max(both / len(tokens | seen), both / len(tokens)) >= SAME_SEARCH:
            return (f"This is search {n} again, so it returns the same passages. Search for something "
                    f"different, use another tool, or call write_answer and report what the passages "
                    f"found so far do and do not say.")
    searches.append((kind, tokens))
    return None


# ── structured answer ───────────────────────────────────────────────────
class TextPart(BaseModel):
    type: Literal["text"]
    text: str


class CitationPart(BaseModel):
    type: Literal["citation"]
    decision_id: str = Field(description="a decision_id, or the law_id of a statute article")
    chunk_id: str | None = None
    quote: str = Field(description="verbatim span of the decision; highlighted in the UI")
    explanation: str = Field(description="why the passage supports the preceding text")


class AgentAnswer(BaseModel):
    """The final answer; the NIM decodes it against this model's JSON schema."""

    answer: list[Annotated[TextPart | CitationPart, Field(discriminator="type")]]


# vLLM structured outputs. A whitespace-free EBNF grammar would also rule out the rare padding loop that
# ResearchThenAnswer._generate() cuts off, but on this NIM it decodes about 6x slower (18 vs 104 tokens/s).
ANSWER_FORMAT = {"type": "json_schema",
                 "json_schema": {"name": "agent_answer", "schema": AgentAnswer.model_json_schema()}}

# The checks are yes/no questions on purpose: asked to quote or describe what is wrong, this model always
# finds something (see agent-eval/); asked to confirm one thing, it is a usable judge.
SUPPORT_PROMPT = """You check a legal answer against its sources. You are given a passage from a Swiss court decision or a statute, in which the sentences the answer quotes are marked between ⟦ and ⟧, and one statement from the answer that cites them. Reply {"supported": true} if the quoted sentences, read in their context, state or directly imply the statement, and {"supported": false} if they do not: they are about something else, say less than the statement claims, or contradict it. Judge only against this passage, not your own legal knowledge. The statement and the passage may be in different languages."""
SUPPORT_FORMAT = {"type": "json_schema", "json_schema": {"name": "support", "schema": {
    "type": "object", "properties": {"supported": {"type": "boolean"}}, "required": ["supported"]}}}
COVERED_PROMPT = """You review a legal answer. You are given the sentences of the answer that are backed by a cited source, and one further sentence of the same answer that has no source. Reply {"covered": true} if that sentence only restates, summarises, introduces or frames what the backed sentences say, or reports what was searched for and not found, or what the passages found are about instead. Reply {"covered": false} if it adds something the backed sentences do not state — a rule, a holding, what a statute or a court says, a decision or article it names, a fact of a case, a deadline, an amount. When there are no backed sentences, only a sentence that reports what was searched for and not found is covered."""
COVERED_FORMAT = {"type": "json_schema", "json_schema": {"name": "covered", "schema": {
    "type": "object", "properties": {"covered": {"type": "boolean"}}, "required": ["covered"]}}}
ANSWERS_PROMPT = """You review whether a legal answer responds to the question it was written for. Reply {"answers": true} if the answer addresses what the question asks — its subject and the point it turns on — including when it says that the sources do not cover the question, or corrects a premise of the question. Reply {"answers": false} if it answers a neighbouring question instead, states a general rule without reaching the point asked, or talks past the question."""
ANSWERS_FORMAT = {"type": "json_schema", "json_schema": {"name": "answers", "schema": {
    "type": "object", "properties": {"answers": {"type": "boolean"}}, "required": ["answers"]}}}
MAX_CHECKS = 10  # grounding checks per draft
REVISE_PROMPT = """Your draft was checked against the tool results, and these problems were found:
{problems}

Write the corrected answer as the same JSON object. Fix every problem: a quote that was not found is replaced by text copied character for character from the tool result, or the citation is dropped; a passage that does not state the statement is replaced by a passage that does, or the statement is removed; a statement that needs a source gets a citation from the tool results, or is removed; a decision the tools did not return is not cited. Keeping a statement and only dropping its citation does not fix it: statements without support are removed from the final answer. Leave the parts that had no problem as they are, and add nothing that is not in the tool results."""
ASK_FORMAT = {"type": "json_schema", "json_schema": {"name": "question", "schema": {
    "type": "object", "properties": {"question": {"type": "string"}, "options": {"type": "array", "items": {
        "type": "string"}}}, "required": ["question", "options"]}}}

LAW_TOOLS = """- read_law(code, article, canton="CH"): the text of a statute article in German, French and Italian, e.g. code="OR", article="271a". Read the provision a question or a decision turns on, so the answer can quote what it says.
- search_laws(query_de, query_fr, query_it, canton="CH", code=""): finds statute articles by meaning, when you do not know which provision applies. canton="CH" is federal law; a canton's code ("ZH", "GE") searches that canton's law instead. code limits the search to one act ("OR", "StGB", "ZPO"), when you know the act but not the article. It searches the {n_articles:,} statute articles, not the decisions. When the question asks which provision, article or rule governs something, search the statutes before concluding that there is none: court passages rarely say that no provision exists.
"""

FILTER_TOOLS = """- Filters, for semantic_search, keyword_search and list_decisions: canton ("ZH", or several: "GE,VD"; "CH" is the federal courts), court (federal_supreme, leading_cases = the published BGE, federal_administrative, federal_criminal, federal_patent, federal_other, cantonal), area (civil, criminal, public, social_insurance), proceeding (appeal, objection, debt_enforcement, constitutional_complaint, revision, first_instance), year_from, year_to. Use a filter when the question names a canton, a court, a period or an area of law ("in Geneva", "the Federal Supreme Court", "since 2020"), and otherwise leave them out: every filter hides decisions. "Since 2020" is year_from=2020 with no year_to; the corpus runs to {this_year}. The proceeding is only recorded for about half of the decisions. A search that finds nothing with a filter may find something without it.
- list_decisions(filters, oldest=false): how many decisions match the filters, how they split by court, area and decade, and the newest ten (oldest=true: the oldest). For questions about the corpus itself or the latest decisions of a court — not to find what the law is.
"""

ASK_TOOL = """- ask_user(question, options, found_so_far): ask the user one short question instead of answering. Two things call for it.
  (a) The answer turns on a fact the question leaves open and the passages found go different ways on it — a residential or a commercial lease, which canton, whether a deadline has passed, employee or self-employed.
  (b) You found the article or the decisions that would govern, but they apply only if something the question does not say is true — Art. 337 OR only if the contract was ended immediately rather than with notice, Art. 271a OR only for a residential lease, a cantonal rule only for a case in that canton. Do not answer on the assumption that it holds: name what you found and let the user confirm it against their own facts, e.g. \"The passages point to Art. 337 OR, termination for good cause with immediate effect. Was the contract ended from one day to the next, or with the ordinary notice period?\" A confirmed premise is what makes the answer the user's answer rather than a plausible one, so it is worth the question whenever the premise is doing real work.
Ask about one fact only, the one the answer turns on most, in one sentence, in the language of the user's question (not the language you searched in). options are two to four short answers to that question, in the user's language: name the alternatives (\"Residential lease\" / \"Commercial lease\", \"Immediately\" / \"With notice\") rather than a bare yes and no whenever the question has a natural alternative; no \"other\" option, the user can always type their own. found_so_far says in two or three sentences what the research found and which decisions (decision_id) and statute articles matter, and for (b) which of them stand or fall on the premise you are asking about; the next turn reads it back. Never settle the open fact yourself: if answering means assuming one case — that the contract is an employment contract, that the user is the tenant — ask instead. Ask only after searching, when you can say why the answer differs; never for something the tools can look up, and not when a short answer can cover every case.
"""

RESEARCH_PROMPT = """You are a legal research assistant for Swiss case law, working on a corpus of {n_decisions:,} Swiss court decisions (Federal Supreme Court, other federal courts and cantonal courts), written in German, French or Italian.

Research the user's question with the tools, then call write_answer.
- semantic_search(query_de, query_fr, query_it): finds passages by meaning. Start here for most questions. Write the same search in German, French and Italian, each phrased like a Swiss court in that language, with that language's legal terms and statute abbreviations, e.g. query_de="Anfechtung der Kündigung wegen Verstoss gegen Treu und Glauben, Art. 271 OR", query_fr="annulation du congé contraire aux règles de la bonne foi, art. 271 CO", query_it="annullamento della disdetta contraria alla buona fede, art. 271 CO". Each query searches the decisions in its language. If results are thin, search again with other wording.
- keyword_search(keyword): exact words, e.g. a statute "Art. 271a OR", a docket number "4A_705/2016", a rare term. Put exact phrases in double quotes.
- read_decision(decision_id, offset): reads a decision's full text, 8,000 characters per call, to check the context or find the decisive reasoning.
- citing_decisions(decision_id): how many later decisions cite it, and the most recent ones. Search results already show "cited by N" — prefer decisions later courts still rely on, and check a leading case before resting the answer on it.
{law_tools}{filter_tools}{ask_tool}- write_answer(established, not_found): call it as soon as the passages you found answer the question, or when more searching is unlikely to help. It becomes available after your first {min_calls} research calls: use the second to read the most relevant decision, or to search for what the first results left open. You have at most {max_calls} tool calls. Its two arguments are your own stock-taking before the answer is written. established: what the passages actually say that answers the question, point by point, each with the decision_id or law_id it comes from — what they say, not what you know. not_found: what the question asks that no passage answers, or "nothing". Be exact there: the answer reports what was not found instead of filling it in, and early in the research you are sent back once to look for it. Before you call it, read the answer you are about to write: if it only holds under a fact the user never stated, and you are about to cover that by hedging — "that depends on the circumstances", "if the termination was immediate", "provided the lease is residential", "generally" — then the hedge is the question, and ask_user is the call to make instead. Hedging is not a way to stay safe about a fact you could simply have asked for.
Prefer passages where a court states the rule and its reasoning over passages that only mention it.
Never repeat a search you have already made: near-identical wording returns the same passages. If two searches with clearly different wording bring back nothing on point, stop and call write_answer. This corpus is a subset of Swiss case law, so many questions — foreign law such as the EU GDPR, statutes no court here applied, recent events — have no answer in it at all. Reporting that is a correct answer; assembling one out of loosely related passages is not.
Lines starting with ">" quote text the user selected (from a decision, named on the "> —" line, or from an earlier answer); the question below them is about that text."""

ANSWER_PROMPT = """You are a legal research assistant for Swiss case law. Using only the tool results in this conversation, answer the user's last question as a JSON object {"answer": [...]} whose parts alternate between text and citations:
- A "text" part holds one or two sentences of the answer (Markdown allowed). Write in the language of the user's question and name decisions by court and docket number.
- A "citation" part follows the text it supports. It gives the decision_id of a passage from the tool results, its chunk_id (only for search results; null for text read with read_decision), a "quote" copied character for character from that passage in the decision's own language (never translated or shortened; one to three consecutive sentences, at most about 300 characters), and an "explanation": one sentence in the user's language on why the passage supports the text.
- Every legal statement needs a citation. Do not add holdings, facts or statutes that are not in the tool results — neither from your own legal knowledge nor from an earlier answer in this conversation. Earlier turns tell you what is being asked; they are never a source, and an earlier answer is never repeated as the new one.
- A statute article from search_laws or read_law is cited the same way: its law_id goes in decision_id, with its chunk_id and a quote copied from the article's text. Name it as it is cited ("Art. 271a OR"). Cite the article for what the statute says and a decision for how courts apply it.
- What list_decisions reports about the corpus itself — how many decisions match, which ones, their dates and subjects — needs no citation: name each decision by court, docket number and date. What a decision holds still does.
- If the tool results do not answer the question, say so plainly in one or two text parts with no citations at all: name what was searched for and what those passages are actually about. Do not stretch a loosely related passage into an answer.
- The answer is checked before it is shown: every quote is looked up character for character in the cited text, and every cited passage is checked for whether it states the sentence in front of it. What fails is removed from the answer, sentence and citation together. So rest each sentence on a passage that says it, and quote the sentences that say it.

Shape:
{"answer": [
  {"type": "text", "text": "First statement."},
  {"type": "citation", "decision_id": "<from a tool result>", "chunk_id": "<from a tool result>", "quote": "<verbatim>", "explanation": "<why it supports the statement>"},
  {"type": "text", "text": "Second statement."},
  {"type": "citation", "decision_id": "...", "chunk_id": "...", "quote": "...", "explanation": "..."}
]}

When the results do not answer the question, the whole answer is text:
{"answer": [
  {"type": "text", "text": "The decisions found do not answer this. Searches for X returned only passages about Y."}
]}"""


# ── tools ───────────────────────────────────────────────────────────────
def _hit_block(i: int, p: Passage) -> str:
    erw = f" · E. {', '.join(p.erwaegungen)}" if p.erwaegungen else ""
    regeste = " · Regeste" if p.section == "regeste" else ""
    cited = f" · cited by {p.cited_by} later decisions" if p.cited_by else ""
    about = "".join(f" · {v}" for v in (p.about or {}).values())
    return (f"Result {i}: decision_id={p.decision_id} chunk_id={p.chunk_id}\n"
            f"{p.court_label} {p.docket} · {p.date or 'undated'} · {p.language}{about}{erw}{regeste}{cited}\n{p.text}")


def _law_block(i: int, p: Passage) -> str:
    # "Text:" on a line of its own: without a boundary the model copied the label into its quotes
    return (f"Result {i}: law_id={p.decision_id} chunk_id={p.chunk_id}\n"
            f"{p.docket} · {p.court_label} · {p.language}\nText:\n{p.text}")


def _hits_result(hits: list[Passage], filters: str = "") -> tuple[str, dict]:
    within = f" in decisions matching {filters}" if filters else ""
    if not hits:
        return (f"No passages found{within}." + (" Try fewer filters." if filters else ""),
                {"summary": "no passages"})
    decisions = sorted({h.decision_id for h in hits})
    return ((f"Only decisions matching {filters}.\n\n" if filters else "")
            + "\n\n".join(_hit_block(i, h) for i, h in enumerate(hits, 1)),
            {"summary": f"{len(hits)} passages from {len(decisions)} decision{'s' * (len(decisions) > 1)}",
             "decisions": decisions})


def _court_name(corpus: Corpus, court: str) -> str:
    return court_label(court, corpus.facets.court_canton(court))


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


Court = Literal[tuple(COURTS) + ("federal_other", "cantonal")]  # type: ignore[valid-type]
Area = Literal[tuple(a for a in AREAS if a != "unknown")]  # type: ignore[valid-type]
Proceeding = Literal[tuple(PROCEEDINGS)]  # type: ignore[valid-type]

# what the model writes for a canton, besides its code
_CANTON_NAMES = {
    "zurich": "ZH", "zürich": "ZH", "bern": "BE", "berne": "BE", "luzern": "LU", "lucerne": "LU", "uri": "UR",
    "schwyz": "SZ", "obwalden": "OW", "nidwalden": "NW", "glarus": "GL", "zug": "ZG", "fribourg": "FR",
    "freiburg": "FR", "solothurn": "SO", "basel-stadt": "BS", "basel-landschaft": "BL", "schaffhausen": "SH",
    "appenzell ausserrhoden": "AR", "appenzell innerrhoden": "AI", "st. gallen": "SG", "st gallen": "SG",
    "graubünden": "GR", "grisons": "GR", "aargau": "AG", "thurgau": "TG", "ticino": "TI", "tessin": "TI",
    "vaud": "VD", "waadt": "VD", "valais": "VS", "wallis": "VS", "neuchâtel": "NE", "neuenburg": "NE",
    "genève": "GE", "geneva": "GE", "genf": "GE", "ginevra": "GE", "jura": "JU", "federal": "CH", "bund": "CH",
}


def _cantons(canton: str | None) -> tuple[list[str], list[str]]:
    """Canton codes from "GE", "GE,VD", "Geneva"; and what was not understood."""
    codes, unknown = [], []
    for part in re.split(r"[,;/]| and | und | et ", canton or ""):
        part = part.strip()
        if not part or part.lower() in ("none", "null", "all", "any"):  # the model writes "None" for no filter
            continue
        code = part.upper() if part.upper() in (*CANTONS, "CH") else _CANTON_NAMES.get(part.lower())
        (codes if code else unknown).append(code or part)
    return codes, unknown


def _decision_filter(corpus: Corpus, canton: str | None, court: str | None, area: str | None,
                     proceeding: str | None, year_from: int | None, year_to: int | None):
    """(mask over decisions or None, "GE · civil · 2020–" for the result, or an error message)."""
    codes, unknown = _cantons(canton)
    if unknown:
        return None, "", f"Unknown canton {', '.join(unknown)!s}: use two-letter codes (ZH, GE, TI ...) or CH."
    filters = {"canton": codes, "court": [court] if court else [], "area": [area] if area else [],
               "proceeding": [proceeding] if proceeding else [], "year_from": year_from, "year_to": year_to}
    years = f"{year_from or ''}–{year_to or ''}" if (year_from or year_to) else ""
    label = " · ".join(x for x in (",".join(codes), court, area, proceeding, years) if x)
    try:
        where = corpus.where(**filters)
    except FiltersUnavailable as e:
        return None, label, f"{e}; search again without filters."
    if where is not None and not where.any():
        # say which filter is too narrow: each alone matches decisions, but not together
        alone = [f"{name}={value if isinstance(value, int) else ','.join(value)}: "
                 f"{int(m.sum()):,}" for name, value in filters.items() if value
                 and (m := corpus.where(**{name: value})) is not None]
        return None, label, (f"No decision in the corpus matches {label} (each filter alone: {'; '.join(alone)}). "
                             f"The kind of proceeding is missing for many cantonal decisions. Drop a filter.")
    return where, label, None


def make_tools(corpus: Corpus) -> list:
    @tool(response_format="content_and_artifact")
    async def semantic_search(query_de: str, query_fr: str, query_it: str, canton: str | None = None,
                              court: Court | None = None, area: Area | None = None,
                              proceeding: Proceeding | None = None, year_from: int | None = None,
                              year_to: int | None = None) -> tuple[str, dict]:
        """Find passages by meaning. Give the same search in German, French and Italian, each phrased the
        way a Swiss court writes in that language, with its legal terms and statute abbreviations
        (OR / CO / CO). Each query searches only the decisions in its language. Returns the best
        passages with decision_id and chunk_id. Optional filters, only when the question asks for them:
        canton ("ZH", "GE,VD", "CH" = federal courts), court, area of law, proceeding, year_from, year_to."""
        where, label, error = _decision_filter(corpus, canton, court, area, proceeding, year_from, year_to)
        if error:
            return error, {"summary": error, "error": True}
        if repeat := _already_searched(f"{query_de} {query_fr} {query_it} {label}"):
            return repeat, {"summary": "repeats an earlier search"}
        try:
            hits = await corpus.semantic_search_by_language({"de": query_de, "fr": query_fr, "it": query_it},
                                                            where=where)
        except Exception as e:
            return _failed(e)
        text, artifact = _hits_result(hits, label)
        if hits:
            per = Counter(h.language for h in hits)
            artifact["summary"] += " · " + ", ".join(f"{n} {lang}" for lang, n in sorted(per.items()))
        return text, artifact

    @tool(response_format="content_and_artifact")
    async def keyword_search(keyword: str, canton: str | None = None, court: Court | None = None,
                             area: Area | None = None, proceeding: Proceeding | None = None,
                             year_from: int | None = None, year_to: int | None = None) -> tuple[str, dict]:
        """Full-text search for exact words in the decisions. Text in double quotes is an exact
        phrase ("Art. 271a OR", "4A_705/2016"); other words must all appear. Returns passages.
        Takes the same optional filters as semantic_search."""
        where, label, error = _decision_filter(corpus, canton, court, area, proceeding, year_from, year_to)
        if error:
            return error, {"summary": error, "error": True}
        if repeat := _already_searched(f"{keyword} {label}"):
            return repeat, {"summary": "repeats an earlier search"}
        try:
            text, artifact = _hits_result(await corpus.keyword_search(keyword, where=where), label)
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
        about = corpus.facets.describe(d.decision_id) if corpus.facets else {}
        if details := " · ".join(x for x in (d.chamber, *about.values()) if x):
            head.append(details)
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

    @tool(response_format="content_and_artifact")
    async def search_laws(query_de: str, query_fr: str, query_it: str, canton: str = "CH",
                          code: str = "") -> tuple[str, dict]:
        """Find statute articles by meaning: the provisions themselves, not the decisions applying them.
        Give the same search in German, French and Italian, worded the way the statute would put it.
        canton: "CH" for federal law (the default), a canton's code ("ZH", "GE") for its law.
        code: only this act, by abbreviation or SR number ("OR", "CO", "StGB", "220"), when the act is
        known but not the article. Returns articles with law_id and chunk_id, cited like decision passages."""
        codes, unknown = _cantons(canton or "CH")
        if unknown or len(codes) != 1:
            return ("canton takes one code: CH for federal law, or ZH, GE, TI ...",
                    {"summary": "bad canton", "error": True})
        where = codes[0]
        acts = None
        if code.strip():
            srs = await asyncio.to_thread(corpus.decisions.find_acts, code, where)
            if not srs:
                return (f"No act {code!r} in the {'federal' if where == 'CH' else where} law of this index. Check "
                        f"the abbreviation, or search without code.", {"summary": "act not found", "error": True})
            acts = [f"{where}/{sr}" for sr in srs]
        label = " · ".join(x for x in (where if where != "CH" else "", code.strip()) if x)
        if repeat := _already_searched(f"{query_de} {query_fr} {query_it} {label}", "laws"):
            return repeat, {"summary": "repeats an earlier search"}
        try:
            hits = await corpus.semantic_search_laws({"de": query_de, "fr": query_fr, "it": query_it},
                                                     canton=where, acts=acts)
        except Exception as e:
            return _failed(e)
        if not hits:
            return "No statute articles found.", {"summary": "no articles"}
        return ("\n\n".join(_law_block(i, h) for i, h in enumerate(hits, 1)),
                {"summary": f"{len(hits)} articles: "
                 + ", ".join(h.docket for h in hits[:4]) + (" …" if len(hits) > 4 else "")})

    @tool(response_format="content_and_artifact")
    async def list_decisions(canton: str | None = None, court: Court | None = None, area: Area | None = None,
                             proceeding: Proceeding | None = None, year_from: int | None = None,
                             year_to: int | None = None, oldest: bool = False) -> tuple[str, dict]:
        """Which decisions the corpus holds for some filters: how many, how they split by court, area
        and decade, and the ten newest (oldest=true: the ten oldest) with their Regeste or title. For
        questions about the corpus or a court's latest decisions; semantic_search finds what the law is.
        At least one filter: canton, court, area, proceeding, year_from, year_to."""
        where, label, error = _decision_filter(corpus, canton, court, area, proceeding, year_from, year_to)
        if error:
            return error, {"summary": error, "error": True}
        if where is None:
            return ("Give at least one filter.", {"summary": "no filter", "error": True})
        # one token: a listing repeats another only when every filter is the same ("GE" and "ZH" are
        # too short to count as words on their own)
        key = re.sub(r"\W+", "_", f"{label} {'oldest' if oldest else 'newest'}")
        if repeat := _already_searched(key, "list"):
            return repeat, {"summary": "repeats an earlier listing"}
        total, split, found = await asyncio.to_thread(corpus.list_decisions, where, 10, oldest)

        def top(c: Counter, n: int = 6, name=lambda x: x) -> str:
            return ", ".join(f"{name(k)} {v:,}" for k, v in c.most_common(n))

        lines = [f"{total:,} decisions match {label}.",
                 f"By court: {top(split['court'], name=lambda c: _court_name(corpus, c))}",
                 f"By area: {top(split['area'])} (area inferred from cited statutes where not recorded)",
                 f"By decade: {', '.join(f'{k} {v:,}' for k, v in sorted(split['period'].items()))}",
                 f"The {'oldest' if oldest else 'newest'} {len(found)}:"]
        for d, about in found:
            gist = (d.regeste or d.title or "").replace("\n", " ").strip()
            lines.append(f"- {d.date or 'undated'} · {d.court_label} {d.docket} · decision_id={d.decision_id}"
                         + "".join(f" · {v}" for v in about.values())
                         + (f"\n  {gist[:240]}{'…' if len(gist) > 240 else ''}" if gist else ""))
        return "\n".join(lines), {"summary": f"{total:,} decisions",
                                  "decisions": [d.decision_id for d, _ in found]}

    @tool(response_format="content_and_artifact")
    async def read_law(code: str, article: str, canton: str = "CH") -> tuple[str, dict]:
        """The verbatim text of one statute article, in German, French and Italian. code is the act's
        abbreviation in any of its languages (OR or CO, ZGB or CC, StGB or CP) or its SR number
        ("220"); article is the article number ("271a"). canton is "CH" for federal law, otherwise two
        letters (ZH, GE, TI ...)."""
        try:
            found = await asyncio.to_thread(corpus.decisions.find_articles, code, article, canton)
        except Exception as e:
            return _failed(e)
        if not found:
            return (f"No article {article!r} of {code!r} ({canton}) in the index. Check the abbreviation, or "
                    f"find the provision with search_laws.", {"summary": "not found", "error": True})
        blocks = [f"law_id={a['law_id']} chunk_id={a['law_id']}#0\n{a['label']} · {a['language']}"
                  + (f" · {a['law_title']}" if a.get("law_title") else "") + f"\nText:\n{a['text']}"
                  for a in found]
        return "\n\n".join(blocks), {"summary": f"{found[0]['label']}, {len(found)} language"
                                                  f"{'s' * (len(found) > 1)}"}

    @tool
    async def ask_user(question: str, options: list[str], found_so_far: str) -> str:
        """Ask the user one short question instead of answering, when the answer depends on a fact the
        question leaves open (which canton, federal or cantonal law, a residential or a commercial
        lease), or when the article or the decisions you found would govern only if something the
        question does not say is true — then name what you found and ask the user to confirm it,
        instead of answering on the assumption that it holds.
        options: two to four short likely answers, naming the alternatives rather than yes and no.
        found_so_far: what the research found and which decision_ids and statute articles matter, and
        which of them depend on the premise being confirmed, for the next turn."""
        return ""  # never runs: ResearchThenAnswer ends the turn with the question instead

    @tool
    async def write_answer(established: str, not_found: str) -> str:
        """Finish the research and write the answer. Call it once the passages found answer the
        question, or when more searching is unlikely to help. First take stock: established lists what
        the passages actually say that answers the question, point by point, each with the decision_id
        or law_id it comes from — what they say, not what you know. not_found names what the question
        asks that no passage answers ("nothing" when they cover it); the answer reports it as not found
        rather than filling it in."""
        turn = _turn.get()
        if turn is not None and turn.pushback:  # sent back to research once: see ResearchThenAnswer
            text, turn.pushback = turn.pushback, None
            return text
        return ""  # otherwise never runs: ResearchThenAnswer writes the answer instead

    laws = [read_law, search_laws] if corpus.has_laws else []
    listing = [list_decisions] if corpus.facets else []
    return [semantic_search, keyword_search, read_decision, citing_decisions, *listing, *laws, ask_user,
            write_answer]


_OTHER = re.compile(r"(?i)^(andere[sr]?|sonstiges|other|autre|altro|else|etwas anderes)\b")


def _gap(not_found: Any) -> str | None:
    """What the agent said it did not find, or None when it said it found everything."""
    text = " ".join(str(not_found or "").split())
    return None if len(text) < 15 or _NOTHING.match(text) else text


def _writer() -> Callable[[Any], None]:
    """The graph's custom stream, for Status events from inside the middleware; a no-op outside a run."""
    try:
        return get_stream_writer()
    except Exception:  # noqa: BLE001 — called directly, e.g. from a test
        return lambda _: None


class ResearchThenAnswer(AgentMiddleware):
    """Every research step must be a tool call (constrained decoding), so the model cannot answer
    from memory or in free text. When it calls write_answer, the answer is drafted instead,
    constrained to the AgentAnswer schema, then checked against the tool results and revised until
    the check passes or the revisions run out; only what passed is handed to answer()."""

    def __init__(self, answer_llm: ChatOpenAI, verifier: Verifier, translate_llm: ChatOpenAI | None = None,
                 max_tool_calls: int = MAX_TOOL_CALLS, max_revisions: int = MAX_REVISIONS):
        super().__init__()
        self.answer_llm, self.verifier, self.translate_llm = answer_llm, verifier, translate_llm
        self.max_tool_calls, self.max_revisions = max_tool_calls, max_revisions

    async def _in_language(self, question: str, options: list[str]) -> tuple[str, list[str]]:
        """The question asked back and its options in the user's language. The research model writes
        them in whatever language it last searched in (an English question was asked back in German),
        so a question in another language is translated, options with it."""
        turn = _turn.get()
        want = turn.language if turn else None
        if not want or want not in LANGUAGE_NAMES or self.translate_llm is None \
                or detect_language(question, default=want) == want:
            return question, options
        prompt = (f"Translate this question and its answer options into {LANGUAGE_NAMES[want]}. Keep legal "
                  f"terms and statute abbreviations as they are cited in Switzerland. Reply with JSON "
                  f'{{"question": ..., "options": [...]}}, the options in the same order.\n\n'
                  + json.dumps({"question": question, "options": options}, ensure_ascii=False))
        try:
            reply = await self.translate_llm.ainvoke([HumanMessage(prompt)])
            out = json.loads(str(reply.content))
            translated = [str(o).strip() for o in out.get("options") or []]
            return (str(out["question"]).strip() or question,
                    translated if len(translated) == len(options) else options)
        except Exception:  # noqa: BLE001 — a question in the wrong language beats no question
            log.warning("could not translate the question asked back", exc_info=True)
            return question, options

    async def awrap_model_call(self, request: ModelRequest, handler) -> ModelResponse | AIMessage:
        turn = _turn.get()
        may_ask = turn.may_ask if turn else False
        used = sum(isinstance(m, ToolMessage) for m in request.messages)
        choice = ({"type": "function", "function": {"name": "write_answer"}}
                  if used >= self.max_tool_calls else "required")
        messages = request.messages
        if NUDGE_AFTER <= used < self.max_tool_calls:
            messages = [*messages, HumanMessage(NUDGE.format(used=used, max_calls=self.max_tool_calls))]
        tools = request.tools
        if used < MIN_TOOL_CALLS:  # not yet: see MIN_TOOL_CALLS
            tools = [t for t in tools if _tool_name(t) != "write_answer"]
        if used < 1 or not may_ask:  # asking back needs a search to say why it matters
            tools = [t for t in tools if _tool_name(t) != "ask_user"]
        response = await handler(request.override(tool_choice=choice, messages=messages, tools=tools))
        result = response.result if isinstance(response, ModelResponse) else [response]
        replies = [m for m in result if isinstance(m, AIMessage)]
        calls = [tc for m in replies for tc in m.tool_calls]
        if (ask := next((tc for tc in calls if tc["name"] == "ask_user"), None)) and may_ask:
            # the turn ends here: a message with no tool call, carrying the question for answer()
            args = ask["args"]
            options = [o for o in (str(x).strip() for x in (args.get("options") or []))
                       if o and not _OTHER.match(o)][:4]  # "Other": the user can always type
            if str(args.get("question") or "").strip():
                asked, options = await self._in_language(str(args["question"]).strip(), options)
                return AIMessage(content="", additional_kwargs={"ask_user": {
                    "question": asked, "options": options,
                    "notes": str(args.get("found_so_far") or "").strip()}})
            calls = [tc for tc in calls if tc["name"] != "ask_user"]
        # A research step that returns neither a tool call nor any text would end the turn with no
        # answer and nothing logged (seen once in 33 eval turns); answer from what was found instead.
        stalled = not calls and not any(str(m.content).strip() for m in replies)
        write = next((tc for tc in calls if tc["name"] == "write_answer"), None)
        if stalled:
            log.warning("a research step returned neither a tool call nor text; writing the answer")
        elif write is None:
            return response
        args = dict(write["args"]) if write else {}
        # The agent's own stock-taking says something was not found. Early in the research that is
        # worth one more look, so write_answer runs as a tool this once and sends it back; what it
        # returns is set here, where the count of calls is known.
        if (write and turn is not None and not turn.pushed_back and used < NUDGE_AFTER
                and (gap := _gap(args.get("not_found")))):
            turn.pushed_back = True
            turn.pushback = PUSHBACK.format(gap=gap, left=self.max_tool_calls - used - 1)
            log.info("sent back to research once for: %s", gap)
            return response
        return await self._write(request, args)

    async def _generate(self, prompt: list[BaseMessage]) -> str:
        """One constrained answer call, streamed so that the whitespace padding loop can be cut off."""
        pieces: list[str] = []
        trailing = 0
        stream = self.answer_llm.astream(prompt)
        try:
            async for chunk in stream:
                piece = chunk.content if isinstance(chunk.content, str) else ""
                if not piece:
                    continue
                pieces.append(piece)
                trailing = trailing + len(piece) if piece.isspace() else len(piece) - len(piece.rstrip())
                if trailing >= STUCK_AFTER:
                    log.warning("the answer degenerated into whitespace; keeping the parts written so far")
                    break
        finally:
            await stream.aclose()
        return "".join(pieces)

    async def _write(self, request: ModelRequest, args: dict[str, Any]) -> AIMessage:
        """Draft the answer, check it against the tool results, revise while the check finds problems,
        and hand answer() what survived."""
        turn = _turn.get()
        status = _writer()
        # Passages and searches in three languages pull the answer away from the question's language
        # (a French question got a German answer), so name it when the question makes it clear.
        question = str(next((m.content for m in reversed(request.messages) if isinstance(m, HumanMessage)), ""))
        code = (turn.language if turn else None) or detect_language(question, default="")
        language = LANGUAGE_NAMES.get(code)
        notes = ""
        if established := " ".join(str(args.get("established") or "").split()):
            notes += f"\n- What the passages establish: {established}"
        if gap := " ".join(str(args.get("not_found") or "").split()):
            notes += f"\n- Not found in the passages: {gap}"
        # Name the question. This instruction is the last message the answer model sees, and when it
        # only said "write the final answer now", a follow-up got the previous turn's answer again.
        final = (f"The user's last question is:\n{question}\n\n"
                 + (f"Your research notes before writing:{notes}\n\n" if notes else "")
                 + "Answer that question now, from the tool results above. What the notes say was not "
                   "found is reported as not found, never filled in from your own knowledge. An earlier "
                   "answer in this conversation is not a source for it and is never repeated as the answer.")
        if language:
            final += f" Write its text parts and explanations in {language}."
        base = [SystemMessage(ANSWER_PROMPT), *request.messages, HumanMessage(final)]
        # "checking", not "thinking": the research is over and no more reasoning will stream, so the UI
        # drops the thinking line and follows these instead through the drafting and the checks.
        status(Status("checking", "Drafting the answer"))
        text = await self._generate(base)
        parts = _parse_answer(text)
        if not parts:
            # {"answer": []} is valid against the schema, so nothing downstream would complain
            log.warning("the answer call returned no answer; asking once more")
            text = await self._generate(base)
            parts = _parse_answer(text)
        if not parts:
            answer = VerifiedAnswer([TextPart(type="text", text=NO_ANSWER.get(code, NO_ANSWER["en"]))], [], [])
            return self._finish(turn, answer, [])
        seen = _seen(request.messages)
        failed: list[str] = []  # statements whose citations did not hold, across the rounds
        rounds = 0
        while True:
            n = sum(isinstance(p, CitationPart) for p in parts)
            status(Status("checking", f"Checking {n} citation{'s' * (n != 1)} against the passages" if n
                          else "Checking the draft against the results"))
            report = await self.verifier.verify(parts, question, code, seen,
                                                lenient=rounds >= self.max_revisions, check_answers=rounds == 0)
            if not report.problems or rounds >= self.max_revisions:
                break
            rounds += 1
            failed += report.failed_statements
            log.info("answer draft %d, %d problem(s): %s", rounds, len(report.problems), " | ".join(report.problems))
            status(Status("checking", f"Revising the draft: {len(report.problems)} problem"
                                      f"{'s' * (len(report.problems) != 1)} found"))
            listed = "\n".join(f"{i}. {p}" for i, p in enumerate(report.problems, 1))
            revised = await self._generate([*base, AIMessage(text), HumanMessage(REVISE_PROMPT.format(problems=listed))])
            if not (new_parts := _parse_answer(revised)):
                log.warning("the revision returned no answer; keeping the previous draft")
                break
            text, parts = revised, new_parts
        answer = report.prune(parts, failed)
        answer.rounds = rounds
        if not any(isinstance(p, TextPart) and p.text.strip() for p in answer.parts):
            log.warning("nothing of the draft held up against the passages")
            answer = VerifiedAnswer([TextPart(type="text", text=NO_SUPPORT.get(code, NO_SUPPORT["en"]))], [], [],
                                    dropped=answer.dropped, rounds=rounds)
        elif answer.dropped:
            log.warning("removed %d statement(s) whose citations did not hold", answer.dropped)
        return self._finish(turn, answer, report.problems)

    @staticmethod
    def _finish(turn: Turn | None, answer: VerifiedAnswer, problems: list[str]) -> AIMessage:
        """The checked answer goes to answer() through the turn; the message carries its text for the
        graph's history, and what was still wrong at the end for the trace."""
        if turn is not None:
            turn.result = answer
        return AIMessage(content=AgentAnswer(answer=answer.parts).model_dump_json(),
                         additional_kwargs={"checked": {"dropped": answer.dropped, "rounds": answer.rounds,
                                                        "unresolved": problems}})


def _tool_name(tool) -> str | None:
    return getattr(tool, "name", None) or (tool.get("name") if isinstance(tool, dict) else None)


_ID_IN_RESULT = re.compile(r"\b(?:decision_id|law_id)[=:]\s*'?([\w./-]+)")


def _seen(messages: list[BaseMessage]) -> set[str]:
    """The decisions and statute articles the tools returned this turn: all an answer may cite."""
    seen: set[str] = set()
    for m in messages:
        if isinstance(m, ToolMessage):
            art = m.artifact if isinstance(m.artifact, dict) else {}
            seen.update(art.get("decisions", []))
            seen.update(_ID_IN_RESULT.findall(str(m.content)))
    return seen


def _parse_answer(text: str) -> list[TextPart | CitationPart]:
    """The parts of an answer call's JSON; what is complete of a cut-off one; prose as one text part."""
    text = text.strip()
    if not text:
        return []
    start = text.find("{")
    if start < 0 or text[0] not in "{`":  # only possible if constrained decoding is off
        log.warning("the model answered in prose instead of the JSON format")
        return [TextPart(type="text", text=text)]
    body = text[start:]
    obj = None
    try:
        obj = json.loads(body[:body.rfind("}") + 1])
    except ValueError:
        try:
            obj = parse_partial_json(re.sub(r"\s*`*\s*$", "", body))
        except ValueError:
            pass
    raw = obj.get("answer") if isinstance(obj, dict) else None
    if not isinstance(raw, list):
        log.warning("the answer is not in the expected shape: %r", text[:200])
        return []
    parts: list[TextPart | CitationPart] = []
    for p in raw:
        if not isinstance(p, dict):
            continue
        kind = p.get("type") or ("citation" if "quote" in p else "text")
        if kind == "text":
            if (t := str(p.get("text") or "")).strip():
                parts.append(TextPart(type="text", text=t))
            continue
        try:
            parts.append(CitationPart.model_validate(p | {"type": "citation"}))
        except ValidationError:
            log.warning("dropping malformed citation %r", p)
    return parts


# ── quotes → highlights ─────────────────────────────────────────────────
_FOLD = str.maketrans({"„": '"', "“": '"', "”": '"', "«": '"', "»": '"', "‹": "'", "›": "'",
                       "‘": "'", "’": "'", "–": "-", "—": "-"})
_ELLIPSIS = re.compile(r"\[?(?:\.\.\.|…)\]?")


def _fold(s: str) -> tuple[str, list[int]]:
    """Lowercase, unify quotes/dashes and drop all whitespace, keeping each char's original index.

    A word broken across lines in a decision's text ("wer-\\nden") is layout, not text: the hyphen goes
    too, so that a quote written as "werden" is still found character for character."""
    chars, idx = [], []
    n = len(s)
    for i, ch in enumerate(s):
        if ch.isspace() or ch == "­":
            continue
        if ch == "-":
            j = i + 1
            while j < n and s[j].isspace():
                j += 1
            if j > i + 1 and "\n" in s[i + 1:j] and j < n and s[j].islower():
                continue
        chars.append(ch.translate(_FOLD).lower())
        idx.append(i)
    return "".join(chars), idx


_BROKEN_WORD = re.compile(r"-\s+(?=[a-zäöüàâéèêîôûç])")


def locate_quote(text: str, quote: str) -> tuple[int, int] | None:
    """Char span of `quote` in `text`, tolerant to whitespace, quote styles, '...' elisions and words
    broken across lines."""
    span = _locate(text, quote)
    if span is None and _BROKEN_WORD.search(quote):  # the model kept "wer- den" from the passage
        span = _locate(text, _BROKEN_WORD.sub("", quote))
    return span


def _locate(text: str, quote: str) -> tuple[int, int] | None:
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


def _locate_trimmed(text: str, quote: str) -> tuple[int, int] | None:
    """`locate_quote`, and failing that the quote without its first words.

    A statute result starts with a label line ("Art. 271a OR · de · Bundesgesetz …"), and the model has
    copied it into the quote in front of the article's words. Dropping leading words until the rest is
    found keeps the check verbatim: what remains must still appear in the article, character for
    character, and must be long enough to mean something."""
    if span := locate_quote(text, quote):
        return span
    words = quote.split()
    for start in range(1, min(len(words), 40)):
        rest = " ".join(words[start:])
        if len(rest) < 40:
            break
        if span := locate_quote(text, rest):
            return span
    return None


class _Citations:
    """Turns CitationParts into numbered Sources; the same span keeps its number."""

    def __init__(self, corpus: Corpus):
        self.corpus = corpus
        self.by_span: dict[tuple, Source] = {}
        self.seen: set[str] = set()  # decisions the tools returned this turn

    async def _law(self, c: CitationPart) -> Source | None:
        """A citation of a statute article (its law_id in decision_id), or None if it is not one."""
        store = self.corpus.decisions
        chunk = await self.corpus.law_chunk(c.chunk_id) if c.chunk_id and c.chunk_id.startswith("law_") else None
        law_id = c.decision_id if c.decision_id.startswith("law_") else (chunk.decision_id if chunk else None)
        if not law_id or store.law(law_id) is None:
            return None
        # the quote may come from another language version of the same article than the one named
        versions = [law_id] + [re.sub(r"_(de|fr|it)_(\d+)$", rf"_{lang}_\2", law_id) for lang in ("de", "fr", "it")]
        span = None
        for version in dict.fromkeys(versions):
            law = store.law(version)
            if law and (span := _locate_trimmed(law.full_text, c.quote)):
                if version != law_id:
                    log.warning("quote cited from %s found in %s instead", law_id, version)
                law_id = version
                break
        law = store.law(law_id)
        verified = span is not None
        if not span and chunk and chunk.decision_id == law_id and chunk.char_start is not None:
            span = (chunk.char_start, chunk.char_end)  # not verbatim: the whole passage
        if not verified:
            log.warning("quote not found verbatim in %s: %r", law_id, c.quote[:200])
        key = (law_id, span or "law")
        if key in self.by_span:
            return self.by_span[key]
        cid = f"{law_id}#0"
        if span:
            inside = [p for p in await self.corpus.law_chunks_of(law_id)
                      if p.char_start is not None and p.char_start <= span[0] < p.char_end]
            cid = inside[-1].chunk_id if inside else cid
        source = Source(
            n=len(self.by_span) + 1, chunk_id=cid, decision_id=law_id,
            text=law.full_text[span[0]:span[1]] if span else c.quote, section="law", erwaegungen=[],
            char_start=span[0] if span else None, char_end=span[1] if span else None, score=0.0,
            decision=store.law_summary(law_id), explanation=c.explanation, verified=verified,
        )
        self.by_span[key] = source
        return source

    async def resolve(self, c: CitationPart) -> Source | None:
        if (c.decision_id.startswith("law_") or (c.chunk_id or "").startswith("law_")) \
                and hasattr(self.corpus, "law_chunk"):
            return await self._law(c)
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
                # `seen` may also hold statute ids and ids the corpus does not have: skip them
                if (doc := store.get(other)) and (sp := locate_quote(doc.full_text, c.quote)):
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


# ── checking a draft ────────────────────────────────────────────────────
@dataclass
class CheckedCitation:
    index: int          # position in the parts list
    n: int              # its number in the draft, for the problem list
    part: CitationPart
    statement: str      # the text it is attached to
    source: Source | None
    in_results: bool = False        # the cited decision or article was returned by a tool this turn
    supported: bool | None = None   # None: not checked, or the check failed
    # After the revisions, a quote the model would not copy verbatim (typically translated into the
    # answer's language) is accepted when the passage it named still states the sentence: the Source
    # then covers the whole passage and is marked as not verbatim, which the UI shows.
    lenient: bool = False

    @property
    def ok(self) -> bool:
        if self.source is None or not self.in_results or self.supported is False:
            return False
        return self.source.verified or (self.lenient and self.supported is True)


@dataclass
class Report:
    citations: list[CheckedCitation]
    problems: list[str]
    # text parts with no source that add something new: index → what is left of them once the
    # sentences that add something are removed ("" when nothing is)
    rewritten: dict[int, str] = field(default_factory=dict)

    @property
    def failed_statements(self) -> list[str]:
        """Statements left with no support at all: every citation attached to them failed.

        Per statement, not per citation. A statement that also carries a citation that held is not
        unsupported, and the next round must not drop it because one of its other citations was bad."""
        held = {c.statement for c in self.citations if c.ok}
        return [c.statement for c in self.citations if not c.ok and c.statement and c.statement not in held]

    def prune(self, parts: list[TextPart | CitationPart], failed_before: list[str]) -> VerifiedAnswer:
        """What is left once citations that did not hold are removed.

        Citations that failed are dropped. Their text is not dropped with them out of hand: a sentence
        stays if a citation attached to it held, or if the checks found that it only restates what the
        surviving citations say or reports what the search did and did not find (`rewritten` holds what
        is left of each part whose sentences did not all pass). This matters for the honest no-answer:
        "the decisions found say nothing about a 180-day deadline" is right even when the model staples
        an unrelated passage to it. A sentence is dropped anyway when it is one whose citation failed in
        an earlier round and which came back without one: a revision may not keep a claim by dropping
        its source. Sources are renumbered in order of appearance."""
        checked = {c.index: c for c in self.citations}
        kept: list[TextPart | CitationPart] = []
        sources: list[Source] = []
        supported: list[bool | None] = []
        numbered: dict[tuple, Source] = {}
        said: set[str] = set()  # text already kept, to catch a paragraph repeated word for word
        dropped = i = 0
        while i < len(parts):
            texts: list[TextPart] = []
            while i < len(parts) and isinstance(parts[i], TextPart):
                if i in self.rewritten:
                    dropped += 1
                    if self.rewritten[i]:
                        texts.append(TextPart(type="text", text=self.rewritten[i]))
                else:
                    texts.append(parts[i])  # type: ignore[arg-type]
                i += 1
            cites: list[CheckedCitation] = []
            while i < len(parts) and isinstance(parts[i], CitationPart):
                if (c := checked.get(i)) is not None:
                    cites.append(c)
                i += 1
            good = [c for c in cites if c.ok]
            # Sentence by sentence, because a revision may slip the claim in next to a supported one.
            # `failed_before` holds only statements that had no citation left at all (see
            # Report.failed_statements), so a statement that kept a citation that holds is not in it.
            for t in texts:
                if _same_statement(t.text, failed_before):
                    dropped += 1
                    continue
                # A revision sometimes restates a whole passage of the draft under the next citation,
                # and the answer then says the same thing twice, word for word. Keep the first.
                mark = " ".join(t.text.lower().split())
                if len(mark) >= 40 and mark in said:
                    dropped += 1
                    continue
                said.add(mark)
                kept.append(t)
            for c in good:
                src = c.source
                key = (src.decision_id, src.section, src.char_start, src.char_end)
                if key not in numbered:
                    numbered[key] = src.model_copy(update={"n": len(numbered) + 1})
                kept.append(c.part)
                sources.append(numbered[key])
                supported.append(c.supported)
        return VerifiedAnswer(kept, sources, supported, dropped=dropped)


def _same_statement(text: str, others: list[str]) -> bool:
    """Whether a sentence is one of the statements in `others`, by word overlap or by being contained in
    one: a statement is the text between two citations, which may be several sentences."""
    tokens = _tokens(text)
    if len(tokens) < 3:
        return False
    for other in others:
        seen = _tokens(other)
        both = len(tokens & seen)
        if seen and max(both / len(tokens | seen), both / len(tokens)) >= SAME_STATEMENT:
            return True
    return False


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜÉÈÀ«\"(\[*])")
_ABBREVIATION = re.compile(r"(?:\b(?:Art|Abs|lit|Ziff|vgl|bzw|al|let|consid|art|cpv|Nr|Rz|resp|etc|Bd|ca|Hrsg|"
                           r"insb|ev|evtl|ggf|inkl|sog|usw|z\.B|u\.a|i\.S|i\.V|d\.h|c\.-à-d|p\.ex|n|p|E|S|N)|\b[A-Z])\.$")


def _sentences(text: str) -> list[str]:
    """The sentences of a text part, keeping "Art. 336c" and "E. 4.1" together."""
    out: list[str] = []
    for piece in _SENTENCE_END.split(" ".join(text.split())):
        if out and _ABBREVIATION.search(out[-1]):
            out[-1] += " " + piece
        else:
            out.append(piece)
    return [s for s in out if s.strip()]


def _any_language(law_id: str) -> str:
    """A statute id without its language: law_CH_220_de_271a and law_CH_220_fr_271a are one article."""
    return re.sub(r"_(de|fr|it)_(\w+)$", r"_\2", law_id)


def _in_context(corpus: Corpus, source: Source, margin: int = 300) -> str:
    """The quoted span marked inside its surroundings, so the check can resolve what it refers to."""
    if source.char_start is None or source.char_end is None or not source.verified:
        return f"⟦{source.text[:2000]}⟧"
    store = corpus.decisions
    doc = store.law(source.decision_id) if source.section == "law" else store.get(source.decision_id)
    full = doc.full_text if doc else source.text
    s, e = source.char_start, source.char_end
    return full[max(0, s - margin):s] + "⟦" + full[s:e] + "⟧" + full[e:e + margin]


class Verifier:
    """Checks a drafted answer against the tool results: in code where it can (the cited decision was
    returned by a tool, the quote is in it character for character, the answer's language) and with one
    short yes/no model call where it cannot (the passage states the sentence, an unsourced sentence only
    restates what is sourced, the answer responds to the question)."""

    def __init__(self, corpus: Corpus, support_llm: ChatOpenAI, covered_llm: ChatOpenAI | None = None,
                 answers_llm: ChatOpenAI | None = None):
        self.corpus = corpus
        self.support_llm, self.covered_llm, self.answers_llm = support_llm, covered_llm, answers_llm

    async def _yes_no(self, llm: ChatOpenAI | None, system: str, user: str, key: str, what: str) -> bool | None:
        if llm is None:
            return None
        try:
            reply = await llm.ainvoke([SystemMessage(system), HumanMessage(user)])
            return bool(json.loads(str(reply.content))[key])
        except (ValidationError, ValueError, KeyError, TypeError):
            log.warning("%s check returned no verdict", what, exc_info=True)
        except Exception:  # noqa: BLE001 — a check that fails is not a verdict either way
            log.warning("%s check failed", what, exc_info=True)
        return None

    async def supported(self, statement: str, source: Source) -> bool | None:
        """Does the cited passage state the sentence it is attached to?"""
        passage = _in_context(self.corpus, source)
        return await self._yes_no(
            self.support_llm, SUPPORT_PROMPT,
            f"PASSAGE ({source.decision.docket}; the quoted sentences are between ⟦ and ⟧):\n{passage}\n\n"
            f"STATEMENT:\n{statement}", "supported", "grounding")

    async def covered(self, backed: str, statement: str) -> bool | None:
        """Does an unsourced sentence only restate what the sourced sentences say?"""
        return await self._yes_no(self.covered_llm, COVERED_PROMPT,
                                  f"BACKED SENTENCES:\n{backed[:4000] or '(none)'}\n\nSENTENCE WITHOUT A SOURCE:\n{statement}",
                                  "covered", "coverage")

    async def answers(self, question: str, answer: str) -> bool | None:
        return await self._yes_no(self.answers_llm, ANSWERS_PROMPT,
                                  f"QUESTION:\n{question[:1500]}\n\nANSWER:\n{answer[:4000]}", "answers", "answers")

    async def verify(self, parts: list[TextPart | CitationPart], question: str, language: str | None,
                     seen: set[str], lenient: bool = False, check_answers: bool = True) -> Report:
        """The problems with a draft. `lenient`: the last round, see CheckedCitation.lenient."""
        cites = _Citations(self.corpus)
        cites.seen = set(seen)
        free = {_any_language(s) for s in seen}
        checked: list[CheckedCitation] = []
        # the draft as statements: the text parts up to a run of citations, and those citations
        groups: list[tuple[list[int], str, list[CheckedCitation]]] = []
        previous = ""  # citations right after citations support the same statement
        i = 0
        while i < len(parts):
            indices: list[int] = []
            texts: list[str] = []
            while i < len(parts) and isinstance(parts[i], TextPart):
                indices.append(i)
                texts.append(parts[i].text)  # type: ignore[union-attr]
                i += 1
            whole = " ".join(" ".join(texts).split())
            group: list[tuple[int, CitationPart]] = []
            while i < len(parts) and isinstance(parts[i], CitationPart):
                group.append((i, parts[i]))  # type: ignore[arg-type]
                i += 1
            if not group:
                if whole:
                    groups.append((indices, whole, []))
                continue
            statement = whole[-400:] or previous  # the sentences right before the citation
            previous = statement
            these: list[CheckedCitation] = []
            for index, part in group:
                src = await cites.resolve(part)
                # a statute quoted from another language version than the one the tool showed still counts
                in_results = src is not None and (src.decision_id in seen or _any_language(src.decision_id) in free)
                these.append(CheckedCitation(index, len(checked) + len(these) + 1, part, statement, src,
                                             in_results=in_results, lenient=lenient))
            checked.extend(these)
            groups.append((indices, whole, these))

        async def support(c: CheckedCitation) -> None:
            # an unverified quote is checked too: in the last round the passage may stand in for it
            if c.source is not None and c.in_results and c.statement and (c.source.verified or c.lenient):
                c.supported = await self.supported(c.statement, c.source)

        text = " ".join(p.text for p in parts if isinstance(p, TextPart))
        answers = None
        for outcome in await asyncio.gather(
                *(support(c) for c in checked[:MAX_CHECKS]),
                self.answers(question, text) if check_answers and text.strip() else asyncio.sleep(0, None),
                return_exceptions=True):
            if isinstance(outcome, BaseException):
                log.warning("a check failed: %r", outcome)
            elif isinstance(outcome, bool):
                answers = outcome
        # Then, knowing which citations held, whether every sentence that none of them backs only restates
        # what they say. Sentence by sentence, because an honest "no decision says this" and a rule from
        # the model's own memory sit in the same text part; and for statements whose citations all failed
        # too, because the model staples a passage to that honest sentence and the passage does not say it.
        backed = " ".join(whole for _, whole, these in groups if any(c.ok for c in these))
        loose = {index for indices, _, these in groups if not any(c.ok for c in these) for index in indices}
        sentences = [(index, s) for index in sorted(loose)
                     for s in _sentences(parts[index].text)]  # type: ignore[union-attr]

        async def coverage(index: int, sentence: str) -> tuple[int, str, bool | None]:
            if len(sentence) < 25:  # "Zusammenfassend:" — framing, not a claim
                return index, sentence, True
            return index, sentence, await self.covered(backed, sentence)

        verdicts: dict[tuple[int, str], bool | None] = {}
        for outcome in await asyncio.gather(*(coverage(i, s) for i, s in sentences[:2 * MAX_CHECKS]),
                                            return_exceptions=True):
            if isinstance(outcome, BaseException):
                log.warning("a coverage check failed: %r", outcome)
            else:
                verdicts[(outcome[0], outcome[1])] = outcome[2]
        # An unchecked sentence (past the cap, or a check that failed) is kept where nothing was wrong with
        # its statement, and dropped where its citations did not hold: there it has no support either way.
        failed = {index for indices, _, these in groups if these and not any(c.ok for c in these)
                  for index in indices}
        rewritten: dict[int, str] = {}
        needed: list[str] = []  # uncovered sentences of statements that never had a citation
        for index in sorted(loose):
            was = _sentences(parts[index].text)  # type: ignore[union-attr]
            keep = []
            for sentence in was:
                verdict = verdicts.get((index, sentence))
                if verdict is None:  # not checked, or the check failed
                    verdict = index not in failed
                if verdict:
                    keep.append(sentence)
                elif verdicts.get((index, sentence)) is False and index not in failed:
                    needed.append(sentence)  # where the citation failed, that problem already says it
            if len(keep) < len(was):
                rewritten[index] = " ".join(keep)
        # Statements and quotes are given in full: shortened with "…", the model copied the shortened
        # form back into its revision, ellipsis included.
        problems: list[str] = []
        for c in checked:
            where = f"Citation [{c.n}], attached to the statement «{' '.join(c.statement.split())}»"
            if c.source is None:
                problems.append(f"{where}: there is no decision or article {c.part.decision_id!r} in the corpus. "
                                f"Cite only passages the tools returned, by their decision_id or law_id.")
            elif not c.in_results:
                problems.append(f"{where}: {c.source.decision.docket} ({c.source.decision_id}) was not returned by "
                                f"any tool in this turn. Cite only passages from the tool results.")
            elif not c.source.verified and not c.ok:
                # the usual reason: the quote was translated into the answer's language
                spoken = c.source.decision.language
                wrote = detect_language(c.part.quote, default="")
                why = (f" The decision is written in {LANGUAGE_NAMES.get(spoken, spoken)} and the quote is in "
                       f"{LANGUAGE_NAMES.get(wrote, wrote)}: a quote is never translated."
                       if wrote and spoken in LANGUAGE_NAMES and wrote != spoken else "")
                problems.append(f"{where}: its quote «{' '.join(c.part.quote.split())}» was not found in "
                                f"{c.source.decision.docket}.{why} Copy the exact characters of the tool result, "
                                f"in its language, wording and spelling.")
            elif c.supported is False:
                problems.append(f"{where}: the quoted passage from {c.source.decision.docket} does not state that. "
                                f"Cite a passage that says it, or remove the statement.")
        for statement in needed:
            problems.append(f"The sentence «{statement}» has no citation and says more than the cited sentences "
                            f"do. Cite a passage from the tool results that says it, or remove it.")
        if language in LANGUAGE_NAMES and (got := detect_language(text, default="")) and got != language:
            problems.append(f"The answer is written in {LANGUAGE_NAMES.get(got, got)}; the question is in "
                            f"{LANGUAGE_NAMES[language]}. Write the text parts and explanations in "
                            f"{LANGUAGE_NAMES[language]}.")
        # Only when some citation held: this model marks an answer down for what it leaves out (see
        # agent-eval/), so on a draft that reports having found nothing it always says "does not answer",
        # and no revision can satisfy it. There the objection would only burn the revisions.
        if answers is False and any(c.ok for c in checked):
            problems.append(f"The answer does not respond to the question asked: \"{question[:300]}\". "
                            f"Answer that question directly, from the tool results; if they do not cover it, "
                            f"say so.")
        return Report(checked, problems, rewritten)


# ── agent ───────────────────────────────────────────────────────────────
def _history(messages: list[Message], limit: int = 8) -> list[BaseMessage]:
    """Earlier turns as plain text, with their citation markers removed.

    The markers used to be replaced by the docket number of the decision behind them, which made an
    earlier answer look sourced — invented sentences included — to the model writing the next one.
    """
    out: list[BaseMessage] = []
    for m in messages[-limit:]:
        if m.role == "user":
            out.append(HumanMessage(m.content))
        elif m.clarification:
            # the question asked back, with what the research had found: the user's reply comes next
            notes = f"\n\n(Research before asking: {m.clarification.notes})" if m.clarification.notes else ""
            out.append(AIMessage(m.content + notes, additional_kwargs={"asked_user": True}))
        else:
            out.append(AIMessage(re.sub(r"\s*\[\d+\]", "", m.content)))
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
        # Passed in the request body as is, bypassing LangChain's own response_format handling.
        # "nostream": drafts are checked before anything is shown, so no call inside the graph streams to
        # the client — the checked answer arrives as one message.
        answer_llm = ChatOpenAI(**common, max_tokens=4096, tags=["nostream"], extra_body={
            "chat_template_kwargs": {"enable_thinking": False}, "response_format": ANSWER_FORMAT})
        translate_llm = self._judge(common, ASK_FORMAT, max_tokens=512)
        # one short yes/no call per check: does the passage say what the sentence claims, does an unsourced
        # sentence only restate the sourced ones, does the answer respond to the question
        verifier = Verifier(corpus, self._judge(common, SUPPORT_FORMAT), self._judge(common, COVERED_FORMAT),
                            self._judge(common, ANSWERS_FORMAT))
        law_tools = LAW_TOOLS.format(n_articles=corpus.n_articles) if corpus.has_laws else ""
        prompt = RESEARCH_PROMPT.format(n_decisions=len(corpus.decisions), max_calls=MAX_TOOL_CALLS,
                                        min_calls=MIN_TOOL_CALLS, law_tools=law_tools,
                                        filter_tools=FILTER_TOOLS.format(this_year=date.today().year)
                                        if corpus.facets else "", ask_tool=ASK_TOOL)
        self.graph = create_agent(research_llm, make_tools(corpus), system_prompt=prompt,
                                  middleware=[ResearchThenAnswer(answer_llm, verifier, translate_llm)])
        log.info("react agent: %s at %s (thinking=%s, up to %d revisions)", self.model, LLM_URL, LLM_THINKING,
                 MAX_REVISIONS)

    @staticmethod
    def _judge(common: dict, response_format: dict, max_tokens: int = 32) -> ChatOpenAI:
        """A deterministic model for one constrained yes/no or short JSON reply."""
        return ChatOpenAI(**{**common, "temperature": 0, "streaming": False}, max_tokens=max_tokens,
                          tags=["nostream"], extra_body={"chat_template_kwargs": {"enable_thinking": False},
                                                         "response_format": response_format})

    async def answer(self, question: str, history: list[Message], ask: bool = True) -> AsyncIterator[AgentEvent]:
        cites = _Citations(self.corpus)
        # one question back per question: after the user has answered one, the agent answers
        last = next((m for m in reversed(history) if m.role == "assistant"), None)
        turn = Turn(may_ask=ask and not (last and last.clarification),
                    language=detect_language(original_question(question, history), default="") or None)
        _turn.set(turn)  # before the graph starts, so its tasks share this turn's state
        question = with_clarification(question, history)
        yield Status("thinking", "Planning the research")

        events = self.graph.astream(
            {"messages": [*_history(history), HumanMessage(question)]},
            {"recursion_limit": RECURSION_LIMIT}, stream_mode=["messages", "updates", "custom"])
        thought: list[str] = []  # reasoning since the last tool call
        shown = False  # whether any part of an answer has been sent
        try:
            async for mode, data in events:
                if mode == "custom":  # progress of the drafting and checking, from the middleware
                    if isinstance(data, Status):
                        yield data
                    continue
                if mode == "messages":
                    chunk = data[0]
                    if isinstance(chunk, AIMessageChunk) and (text := chunk.additional_kwargs.get("reasoning")):
                        thought.append(text)
                        yield Thought(text)
                    continue  # research steps are tool calls; the answer arrives as an update, once checked
                for update in data.values():
                    for m in (update or {}).get("messages", []):
                        if isinstance(m, AIMessage) and (asked := m.additional_kwargs.get("ask_user")):
                            shown = True
                            seen = ", ".join(sorted(cites.seen)[:12])
                            notes = asked["notes"] + (f" Decisions found: {seen}." if seen else "")
                            yield Delta(asked["question"])
                            yield Clarify(asked["question"], asked["options"], notes)
                        elif isinstance(m, AIMessage) and "checked" in m.additional_kwargs:
                            shown = True
                            yield Status("answer", "Writing the answer")
                            async for ev in self._emit(turn, m, cites):
                                yield ev
                        elif isinstance(m, AIMessage) and m.tool_calls:
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
        finally:
            await events.aclose()  # cancels the model call if the client went away
        if not shown:  # never end a turn in silence: the client cannot tell it from an answer
            log.warning("the turn ended without an answer")
            yield Delta(NO_ANSWER.get(detect_language(question, default="en"), NO_ANSWER["en"]))

    async def _emit(self, turn: Turn, message: AIMessage, cites: _Citations) -> AsyncIterator[AgentEvent]:
        """The checked answer as events: its text, a Cite after each statement with the verified Source,
        and the grounding verdict of every citation (all passed, or they would not be here)."""
        result = turn.result
        if result is None:  # cannot happen in a turn; the message text is the fallback
            log.warning("the checked answer did not reach answer(); resolving its citations again")
            parts = _parse_answer(str(message.content))
            sources = [await cites.resolve(p) for p in parts if isinstance(p, CitationPart)]
            result = VerifiedAnswer(parts, [s for s in sources if s], [None] * len(sources))
        last = ""  # last character sent, to space consecutive parts
        verdicts: dict[int, bool] = {}
        k = 0
        for p in result.parts:
            if isinstance(p, TextPart):
                text = p.text
                if last and not last.isspace() and text[:1] not in " \n.,;:!?)":
                    text = " " + text  # parts are written as separate sentences
                last = text[-1]
                yield Delta(text)
                continue
            if k >= len(result.sources):
                break
            src, supported = result.sources[k], result.supported[k] if k < len(result.supported) else None
            k += 1
            last = "]"
            yield Cite(src)
            if supported is not None:
                verdicts[src.n] = supported
        for n, supported in verdicts.items():
            yield Verdict(n, supported)
