
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

Stage = Literal["thinking", "answer"]


class Model(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class DecisionSummary(Model):
    decision_id: str
    court: str
    court_label: str
    canton: str | None
    chamber: str | None
    docket: str
    date: str | None
    language: str
    title: str | None
    regeste: str | None
    legal_area: str | None
    source_url: str | None
    pdf_url: str | None


class Decision(DecisionSummary):
    full_text: str


class Source(Model):
    n: int
    chunk_id: str
    decision_id: str
    text: str
    section: Literal["regeste", "erwaegung", "body"]
    erwaegungen: list[str]
    char_start: int | None 
    char_end: int | None
    score: float
    decision: DecisionSummary
    explanation: str | None = None
    verified: bool = True
    # whether the cited passage really states the sentence it is attached to; null = not checked
    supported: bool | None = None


class ToolCall(Model):
    id: str
    name: str
    args: dict[str, Any]
    summary: str | None = None
    error: bool = False
    thought: str | None = None  # the agent's reasoning before the call


class Message(Model):
    id: str
    role: Literal["user", "assistant"]
    content: str
    search_query: str | None = None
    sources: list[Source] | None = None
    tool_calls: list[ToolCall] | None = None
    language: str | None = None  # of the question, on assistant messages
    created_at: str


class ConversationSummary(Model):
    id: str
    title: str
    updated_at: str


class Conversation(ConversationSummary):
    messages: list[Message]


class ChatRequest(Model):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=4000)


Language = Literal["de", "fr", "it", "rm", "en"]


class TranslateRequest(Model):
    text: str = Field(min_length=1, max_length=8000)
    source: Language
    target: Language


class TranslateResponse(Model):
    translation: str
    source: Language
    target: Language


class CitingDecision(Model):
    """A decision at the other end of a citation edge."""

    decision_id: str
    court: str | None = None
    docket: str | None = None
    date: str | None = None
    in_corpus: bool = False  # part of this app's corpus, so it can be opened here


class Citations(Model):
    decision_id: str
    cited_by_count: int
    cites_count: int
    cited_by: list[CitingDecision] = []
    cites: list[CitingDecision] = []


class SpeechRequest(Model):
    text: str = Field(min_length=1, max_length=20000)
    language: Language


class Health(Model):
    status: str
    agent: str
    decisions: int
    speech_languages: list[str] = []  # what the ASR NIM understands; empty when it is not reachable


# ── matters: one client case, worked through intake, research, assessment and drafting ──
MatterStage = Literal["new", "intake", "research", "assessment", "drafting", "done"]


class Issue(Model):
    """One legal question hidden in the client's story, and what the research found on it."""

    n: int
    question: str
    why: str  # why it decides this case
    area: str | None = None
    answer: str | None = None
    sources: list[Source] | None = None


class Intake(Model):
    summary: str
    parties: list[str] = []
    timeline: list[str] = []  # dated facts, in order


class MatterSummary(Model):
    id: str
    title: str
    stage: MatterStage = "new"
    source_name: str | None = None
    source_kind: Literal["document", "recording", "text"] = "text"
    created_at: str
    updated_at: str


class Matter(MatterSummary):
    language: str
    facts: str  # the client's story as text, however it arrived
    intake: Intake | None = None
    issues: list[Issue] = []
    assessment: str | None = None
    memo: str | None = None


class MatterRequest(Model):
    title: str | None = None
    text: str = Field(min_length=20, max_length=40_000)
