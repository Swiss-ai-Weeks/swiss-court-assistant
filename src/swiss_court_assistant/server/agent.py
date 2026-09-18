from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import polars as pl

from .decisions import DecisionStore
from .schemas import Message, Source, Stage
from .store import new_id


@dataclass
class Status:
    stage: Stage
    detail: str


@dataclass
class Thought:
    """Streamed reasoning while the agent decides on its next tool call."""
    text: str


@dataclass
class ToolStart:
    id: str
    name: str
    args: dict[str, Any]
    thought: str | None = None  # the reasoning that led to this call


@dataclass
class ToolEnd:
    id: str
    name: str
    summary: str
    error: bool = False


@dataclass
class Delta:
    text: str


@dataclass
class Cite:
    source: Source


@dataclass
class Verdict:
    """Whether the cited passage really states the sentence it was attached to."""

    n: int
    supported: bool


@dataclass
class Clarify:
    """The agent needs a fact from the user before it can answer: the turn ends with this question
    (also sent as Delta text), a few likely answers to pick from, and a note of what the research found
    so far, which the next turn reads back."""

    question: str
    options: list[str]
    notes: str


AgentEvent = Status | Thought | ToolStart | ToolEnd | Delta | Cite | Verdict | Clarify


def with_clarification(question: str, history: list[Message]) -> str:
    """A reply to a question the assistant asked back ("Wohnmietvertrag") is not a question on its own:
    research and answer the original question with the reply added, in the original's language."""
    if len(history) < 2 or history[-1].role != "assistant" or not history[-1].clarification:
        return question
    asked = history[-1].clarification
    original = next((m.content for m in reversed(history[:-1]) if m.role == "user"), "")
    return f"{original}\n\n{asked.question}\n→ {question}" if original else question


def original_question(question: str, history: list[Message]) -> str:
    """The question whose language the turn is in: for a reply to a question asked back ("Arbeitsvertrag"),
    the user's original question — the reply is often one of the suggested answers, and those can be
    in another language than the user writes."""
    if len(history) < 2 or history[-1].role != "assistant" or not history[-1].clarification:
        return question
    return next((m.content for m in reversed(history[:-1]) if m.role == "user"), question)


class Agent(Protocol):
    name: str

    def answer(self, question: str, history: list[Message], ask: bool = True) -> AsyncIterator[AgentEvent]:
        """Stream one turn (with `ask`, the agent may end it with a Clarify question instead): Status, Thought and ToolStart/ToolEnd while researching, then the answer as
        Delta text with a Cite right after each statement a source supports."""
        ...


# ── stub: canned answers on real passages, no LLM (SCA_AGENT=stub) ──────
@dataclass
class Scenario:
    match: re.Pattern[str]
    query: str
    chunks: list[str]
    answer: str


TENANCY = Scenario(
    match=re.compile(r"miet|bail|locat|tenan|lease|rent|landlord|vermiet|loyer", re.I),
    query="Anfechtung Kündigung Mietverhältnis Treu und Glauben Art. 271 271a OR",
    chunks=["bger_4A_705_2016#12", "bger_4A_482_2014#4", "bger_4A_482_2014#8"],
    answer="""An ordinary termination of a residential or commercial lease needs no particular reason: the parties may end an open-ended lease by observing the statutory notice periods (Art. 266a OR). The only limit is good faith — a termination that violates it can be challenged (Art. 271 para. 1 OR) [1].

- **Contrary to good faith (Art. 271 OR).** The Federal Supreme Court treats a termination as abusive when it is given without an objective, serious and legitimate interest — i.e. purely as harassment — or when the parties' interests are grossly disproportionate. Hardship for the tenant is not enough; it only matters for an extension of the lease under Art. 272 OR [1].
- **Relevant moment.** Whether a termination breaches good faith is assessed as of the time it is given [1].
- **Protected periods (Art. 271a para. 1 lit. d OR).** A landlord's termination given during conciliation or court proceedings connected with the lease can be challenged unless the tenant started the proceedings abusively — regardless of whether the termination is actually abusive [2].
- **Landlord's knowledge.** The retaliation motive is presumed by law; the Court reasoned that whether the landlord actually had that motive, or could have had it given his knowledge of the proceedings, cannot be decisive for fixing the protected period [3].""",
)

DISMISSAL = Scenario(
    match=re.compile(r"licenci|congé|travail|employ|dismiss|arbeit|kündigungsschutz|grossesse|pregnan"
                     r"|schwanger|licenzi", re.I),
    query="licenciement abusif grossesse congé nul art. 336 336c CO",
    chunks=["bger_4C.414_2005#12", "bger_4C.414_2005#11", "bger_4A_328_2014#8"],
    answer="""Selon les passages retrouvés, un congé donné pendant la grossesse n'est pas abusif mais **nul** : dans l'arrêt 4C.414/2005, la travailleuse était enceinte lors du licenciement du 30 janvier 2002, de sorte que ce congé était nul (art. 336c al. 1 let. c et al. 2 CO) [1].

- **Seul le congé valable compte.** Le Tribunal fédéral a jugé qu'en qualifiant d'abusif un congé par ailleurs nul, la cour cantonale avait procédé à une fausse appréciation juridique ; seul le second licenciement, donné le 10 janvier 2003 au terme de la période de protection, était déterminant sous l'angle de l'art. 336 CO [1].
- **Circonstances retenues par l'instance cantonale.** Les juges cantonaux avaient qualifié de « brutale » la décision de remettre la lettre de licenciement sous les yeux de l'époux, de collègues et des forces de l'ordre, et retenu un licenciement « par convenance personnelle » [2].
- **Une résiliation claire est nécessaire.** L'art. 336c CO présuppose une résiliation communiquée par l'employeur, qui doit reposer sur une manifestation de volonté claire et dépourvue d'incertitudes, interprétée au besoin selon le principe de la confiance [3].""",
)

SCENARIOS = [DISMISSAL, TENANCY]
FALLBACK_NOTE = ("*Stub agent: it only knows two topics (tenancy termination and abusive dismissal), "
                 "so this is the closest canned answer.*\n\n")


class StubAgent:
    name = "stub"

    def __init__(self, chunks_path: Path, decisions: DecisionStore, delay: float = 1.0):
        ids = [c for s in SCENARIOS for c in s.chunks]
        rows = pl.scan_parquet(chunks_path).filter(pl.col("chunk_id").is_in(ids)).collect()
        self.passages = {r["chunk_id"]: r for r in rows.iter_rows(named=True)}
        missing = [c for c in ids if c not in self.passages or not decisions.find(self.passages[c]["decision_id"])]
        if missing:
            raise RuntimeError(f"stub passages missing from the corpus: {missing}")
        self.decisions, self.delay = decisions, delay

    def _sources(self, s: Scenario) -> list[Source]:
        out = []
        for i, cid in enumerate(s.chunks):
            p = self.passages[cid]
            out.append(Source(
                n=i + 1, chunk_id=cid, decision_id=p["decision_id"], text=p["text"], section=p["section"],
                erwaegungen=p["erwaegungen"], char_start=p["char_start"], char_end=p["char_end"],
                score=0.0, decision=self.decisions.summary(p["decision_id"]),
                explanation="Canned citation from the stub agent.",
            ))
        return out

    async def answer(self, question: str, history: list[Message], ask: bool = True) -> AsyncIterator[AgentEvent]:
        matched = next((s for s in SCENARIOS if s.match.search(question)), None)
        s = matched or TENANCY
        yield Status("thinking", "Planning the research (stub)")
        call = new_id()
        yield ToolStart(call, "semantic_search", {"query": s.query})
        await asyncio.sleep(1.0 * self.delay)
        yield ToolEnd(call, "semantic_search", f"{len(s.chunks)} passages (stub, canned)")
        yield Status("answer", "Writing the answer (stub)")
        sources = self._sources(s)
        for piece in re.split(r"(\[\d+\])", ("" if matched else FALLBACK_NOTE) + s.answer):
            if m := re.fullmatch(r"\[(\d+)\]", piece):
                yield Cite(sources[int(m[1]) - 1])
                continue
            for i in range(0, len(piece), 6):
                yield Delta(piece[i:i + 6])
                await asyncio.sleep(0.012 * self.delay)
