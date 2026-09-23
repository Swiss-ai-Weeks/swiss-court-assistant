
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

# What the agent is doing this moment: researching, checking the drafted answer against the passages
# it found (drafting, checking citations, revising), or writing out what passed.
Stage = Literal["thinking", "checking", "answer"]


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
    # "law": a statute article, law_id in decision_id; "document": a document the user attached, its id there
    section: Literal["regeste", "erwaegung", "body", "law", "document"]
    erwaegungen: list[str]
    char_start: int | None 
    char_end: int | None
    score: float
    decision: DecisionSummary
    explanation: str | None = None
    verified: bool = True
    # whether the cited passage really states the sentence it is attached to; null = not checked
    supported: bool | None = None


class StatuteRef(Model):
    """An article named in an answer's text ("Art. 259d CO") and its text - a link, not a citation."""

    text: str  # the mention exactly as it appears in the answer
    source: Source


class ToolCall(Model):
    id: str
    name: str
    args: dict[str, Any]
    summary: str | None = None
    error: bool = False
    thought: str | None = None  # the agent's reasoning before the call


class ClarifyQuestion(Model):
    question: str
    options: list[str] = []


class Clarification(Model):
    """A question the assistant asked back instead of answering."""

    question: str  # the whole text asked, all questions included
    options: list[str] = []  # likely answers, offered as buttons
    notes: str = ""  # what the research found before asking; read back by the next turn
    # when it asked several questions at once, each with its own likely answers
    questions: list[ClarifyQuestion] = []


class DocumentInfo(Model):
    """A document the user attached, as parsed and stored (see parsing.py)."""

    id: str
    name: str
    # a file the user attached, a recording of the client (kept as WAV, its text the transcript), or typed notes
    kind: Literal["document", "recording", "notes", "generated"] = "document"  # generated: a matter's case prep
    pages: int
    chars: int
    parser: str  # "nemotron-parse", "python-docx", "text", "nemotron-asr" or "typed"
    language: str | None = None
    seconds: float | None = None  # length of a recording
    created_at: str


class UploadStatus(Model):
    """A file being read in the background (POST /api/documents/jobs), polled until it is done: reading a
    forty-page scan with Nemotron Parse takes minutes, longer than a proxy holds a request open."""

    id: str
    name: str
    state: Literal["reading", "ready", "failed"]
    seconds: float  # how long it has been read
    document: DocumentInfo | None = None  # set once it is ready
    error: str | None = None  # why it could not be read
    status: int | None = None  # the HTTP status the same failure would have had on POST /api/documents


class Message(Model):
    id: str
    role: Literal["user", "assistant"]
    content: str
    search_query: str | None = None
    sources: list[Source] | None = None
    tool_calls: list[ToolCall] | None = None
    language: str | None = None  # of the question, on assistant messages
    statutes: list[StatuteRef] | None = None  # articles the answer names, linked to their text
    clarification: Clarification | None = None  # set when the assistant asked back instead of answering
    attachments: list[DocumentInfo] | None = None  # documents attached to a user message
    created_at: str


class ConversationSummary(Model):
    id: str
    title: str
    updated_at: str
    matter_id: str | None = None  # asked from Case Prep about this matter; None for the general assistant


class Conversation(ConversationSummary):
    messages: list[Message]


class ChatRequest(Model):
    conversation_id: str | None = None
    message: str = Field(min_length=1, max_length=8000)  # a pasted fact pattern or clause reaches this
    allow_questions: bool = True  # whether the assistant may ask back instead of answering (off for evals)
    document_ids: list[str] = Field(default=[], max_length=5)  # uploaded with POST /api/documents
    # a new conversation about this matter (Case Prep's "Ask the assistant"); a conversation keeps its matter
    matter_id: str | None = None


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
    # which engine runs the vector search ("cuVS brute-force float16, GPU 1", "numpy float32, CPU",
    # "sqlite-vec"), with the GPU's search count and average time since startup
    vector_search: dict[str, Any] | None = None


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
    statutes: list[StatuteRef] | None = None


class Intake(Model):
    summary: str
    parties: list[str] = []
    timeline: list[str] = []  # dated facts, in order
    # where each dated fact was found in the case file, entry by entry (null: nowhere clear enough);
    # null for the whole list on matters taken in before the passages were located
    timeline_sources: list[Source | None] | None = None


class MatterSummary(Model):
    id: str
    title: str
    stage: MatterStage = "new"
    source_name: str | None = None
    # "bundle": several files, recordings and notes together
    source_kind: Literal["document", "recording", "text", "bundle"] = "text"
    document_id: str | None = None  # the first asset (matters opened before assets kept only this)
    created_at: str
    updated_at: str


class Matter(MatterSummary):
    language: str
    facts: str  # the client's story as text, however it arrived: every asset's text, one after the other
    assets: list[DocumentInfo] = []  # the case file: documents, recordings and notes, kept in the document store
    indexed: int = 0  # passages of the case file in its collection of the case index (0: not indexed)
    added_since_run: int = 0  # assets added to the case file since the last run, which has not seen them
    intake: Intake | None = None
    issues: list[Issue] = []
    assessment: str | None = None
    memo: str | None = None


class MatterRequest(Model):
    title: str | None = None
    text: str = Field(min_length=20, max_length=40_000)
