
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


class ToolCall(Model):
    id: str
    name: str
    args: dict[str, Any]
    summary: str | None = None
    error: bool = False


class Message(Model):
    id: str
    role: Literal["user", "assistant"]
    content: str
    search_query: str | None = None
    sources: list[Source] | None = None
    tool_calls: list[ToolCall] | None = None
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


class Health(Model):
    status: str
    agent: str
    decisions: int
