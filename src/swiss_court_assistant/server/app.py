
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request, Response, UploadFile, WebSocket,
                     WebSocketDisconnect)
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import tracing
from .agent import (Agent, Cite, Clarify, Delta, Mention, Status, StubAgent, Thought, ToolEnd, ToolStart, Verdict,
                    original_question)
from .decisions import DecisionStore, SqliteDecisionStore
from .documents import MAX_CHARS, UnreadableError, transcribe
from .matters import (MatterStore, Pipeline, case_prep_document, create_matter, docx_memo, in_order,
                      locate_timeline, memo, prep_document_id)
from .language import detect_language
from .mentions import statute_links
from .parsing import ACCEPTED, DocumentStore, ParserUnavailable
from .case_index import CaseIndex, IndexUnavailable, collection_of
from .citations import CitationIndex, open_index
from .schemas import (ChatRequest, Citations, CitingDecision, Clarification, Conversation, ConversationSummary,
                      Decision, DocumentInfo, Health, Matter, MatterRequest, MatterSummary, Message, Source, SpeechRequest, StatuteRef, ToolCall,
                      TranslateRequest, TranslateResponse, UploadStatus)
from .speech import SAMPLE_RATE, Speaker, UnspeakableError
from .store import ConversationStore, new_id, now
from .translate import Translator, UntranslatableError
from .voice import SAMPLE_RATE as ASR_RATE, Listener, narrate

# Which index the app serves: "corpus", the full-corpus index that `index.py` builds and updates
# (254k decisions since 1980 plus statute articles), or "subset", the 50k-decision evaluation
# subset the app started on (its decisions come from a parquet, and the stub agent needs it).
INDEX = os.environ.get("SCA_INDEX", "corpus")
DECISIONS = Path(os.environ.get("SCA_DECISIONS", "data/subset/decisions_50k_seed42.parquet"))
CHUNKS = DECISIONS.with_name(DECISIONS.stem + ".chunks.parquet")
_NAME = "corpus" if INDEX == "corpus" else DECISIONS.stem
VECTOR_DB = Path(os.environ.get("SCA_VECTOR_DB", f"data/vectordb/{_NAME}.sqlite"))
KEYWORD_INDEX = VECTOR_DB.with_name(VECTOR_DB.stem + ".fts.sqlite")
CITATION_INDEX = Path(os.environ.get("SCA_CITATIONS", f"data/graph/{_NAME}.citations.sqlite"))
DB = Path(os.environ.get("SCA_DB", "data/app/conversations.sqlite"))
MATTERS_DB = Path(os.environ.get("SCA_MATTERS_DB", "data/app/matters.sqlite"))
MAX_UPLOAD = 25 * 1024 * 1024  # a long recording or a scanned brief; anything larger is a mistake
AGENT = os.environ.get("SCA_AGENT", "react")  # react | stub
WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"

log = logging.getLogger(__name__)
# uvicorn configures only its own loggers, so this package's INFO lines - what the agent checked in a
# draft and why it revised it - would otherwise be dropped (warnings reached stderr via the last-resort
# handler). One handler for the package, at INFO.
_package = logging.getLogger("swiss_court_assistant")
if not _package.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    _package.addHandler(_handler)
    _package.setLevel(logging.INFO)
    _package.propagate = False


@dataclass
class Services:
    decisions: DecisionStore | SqliteDecisionStore
    store: ConversationStore
    agent: Agent
    translator: Translator
    speaker: Speaker
    listener: Listener
    citations: CitationIndex | None
    matters: MatterStore
    pipeline: Pipeline
    documents: DocumentStore
    case_index: CaseIndex | None
    uploads: dict[str, "UploadJob"] = field(default_factory=dict)  # files being read in the background


@asynccontextmanager
async def lifespan(app: FastAPI):
    tracing.setup()  # before the agent is built, so LangChain autologging patches it
    decisions = (SqliteDecisionStore(VECTOR_DB) if INDEX == "corpus" and AGENT != "stub"
                 else DecisionStore(DECISIONS))
    citations = open_index(CITATION_INDEX, decisions.ids())
    documents = DocumentStore()
    case_index = CaseIndex(documents) if AGENT != "stub" else None
    agent = _make_agent(decisions, citations, documents, case_index)
    matters = MatterStore(MATTERS_DB)
    app.state.services = Services(decisions, ConversationStore(DB), agent, Translator(),
                                  Speaker(), Listener(), citations, matters,
                                  Pipeline(agent, matters, decisions, case_index=case_index), documents, case_index)
    log.info("index=%s (%s): %d decisions; agent=%s", INDEX, VECTOR_DB, len(decisions),
             app.state.services.agent.name)
    yield


def _make_agent(decisions: DecisionStore | SqliteDecisionStore, citations: CitationIndex | None = None,
                documents: DocumentStore | None = None, case_index: CaseIndex | None = None) -> Agent:
    if AGENT == "stub":
        return StubAgent(CHUNKS, decisions)
    import torch

    from .corpus import Corpus
    from .react_agent import ReactAgent

    device = os.environ.get("SCA_EMBED_DEVICE", "cuda:1" if torch.cuda.device_count() > 1 else "cpu")
    corpus = Corpus(VECTOR_DB, KEYWORD_INDEX, decisions, os.environ.get("SCA_EMBED_MODEL"), device,
                    rerank=os.environ.get("SCA_RERANK", "1") == "1", citations=citations)
    return ReactAgent(corpus, documents, case_index)


def services(request: Request) -> Services:
    return request.app.state.services


Svc = Annotated[Services, Depends(services)]
app = FastAPI(title="Swiss Court Assistant", lifespan=lifespan)


@app.get("/api/health", response_model=Health)
async def health(s: Svc) -> Health:
    try:
        languages = sorted(await asyncio.to_thread(s.listener.languages))
    except Exception:  # voice mode is optional: the rest of the app works without the ASR NIM
        languages = []
    return Health(status="ok", agent=s.agent.name, decisions=len(s.decisions), speech_languages=languages,
                  vector_search=_vector_search(s.agent))


def _vector_search(agent: Agent) -> dict | None:
    corpus = getattr(agent, "corpus", None)
    if corpus is None:
        return None
    matrix = corpus.matrix
    if matrix is None:
        return {"backend": "sqlite-vec"}
    return {"backend": matrix.backend, "vectors": len(matrix.ids),
            **(matrix.stats() if hasattr(matrix, "stats") else {})}


@app.get("/api/conversations", response_model=list[ConversationSummary])
async def list_conversations(s: Svc) -> list[ConversationSummary]:
    return s.store.list()


@app.get("/api/conversations/{conversation_id}", response_model=Conversation)
async def get_conversation(conversation_id: str, s: Svc) -> Conversation:
    conv = s.store.get(conversation_id)
    if conv is None:
        raise HTTPException(404, "Conversation not found")
    question = None
    for m in conv.messages:  # an answer's language is its question's, recomputed rather than stored
        if m.role == "user":
            question = m.content
        elif m.language is None and question:
            m.language = detect_language(question)
    for m in conv.messages:  # answers saved before articles were linked get their links on the way out
        if m.role == "assistant" and m.statutes is None:
            m.statutes = await asyncio.to_thread(statute_links, s.decisions, m.content, m.language) or None
    return conv


@app.delete("/api/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, s: Svc) -> Response:
    if not s.store.delete(conversation_id):
        raise HTTPException(404, "Conversation not found")
    return Response(status_code=204)


@app.get("/api/decisions/{decision_id}", response_model=Decision)
async def get_decision(decision_id: str, s: Svc) -> Decision:
    """A decision, a statute article (a law_id) or an attached document (a doc_ id), which answers cite
    the same way."""
    if decision_id.startswith("doc_"):
        d = await asyncio.to_thread(s.documents.as_decision, decision_id)
        if d is None:
            raise HTTPException(404, "Document not found")
        return d
    d = s.decisions.get(decision_id) or s.decisions.law(decision_id)
    if d is None:
        raise HTTPException(404, "Decision not found")
    return d


@app.get("/api/decisions/{decision_id}/citations", response_model=Citations)
async def decision_citations(decision_id: str, s: Svc, limit: int = 20) -> Citations:
    """Which later decisions cite this one - the corpus citation graph, not the agent."""
    if s.citations is None:
        raise HTTPException(503, "The citation index is not built.")
    cited_by_count, cites_count = await asyncio.to_thread(s.citations.counts, decision_id)
    cited_by = await asyncio.to_thread(s.citations.cited_by, decision_id, limit)
    cites = await asyncio.to_thread(s.citations.cites, decision_id, limit)
    return Citations(decision_id=decision_id, cited_by_count=cited_by_count, cites_count=cites_count,
                     cited_by=[CitingDecision(**vars(c)) for c in cited_by],
                     cites=[CitingDecision(**vars(c)) for c in cites])


@app.post("/api/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest, s: Svc) -> TranslateResponse:
    if req.source == req.target:
        return TranslateResponse(translation=req.text, source=req.source, target=req.target)
    try:
        out = await s.translator.translate(req.text, req.source, req.target)
    except UntranslatableError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:
        log.exception("translation failed")
        raise HTTPException(502, "The translation model is not available right now.") from e
    return TranslateResponse(translation=out, source=req.source, target=req.target)


@app.post("/api/speech")
async def speech(req: SpeechRequest, s: Svc) -> StreamingResponse:
    try:
        audio = await s.speaker.stream(req.text, req.language)
    except UnspeakableError as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:
        log.exception("speech synthesis failed")
        raise HTTPException(502, "The speech model is not available right now.") from e
    return StreamingResponse(audio, media_type="application/octet-stream", headers={
        "X-Sample-Rate": str(SAMPLE_RATE), "Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


def _title(text: str) -> str:
    own = [line for line in text.splitlines() if not line.lstrip().startswith(">")]  # not the quoted selection
    text = " ".join(" ".join(own).split()) or " ".join(text.lstrip("> ").split())
    return text if len(text) <= 60 else text[:57] + "…"


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# ── attached documents: parsed by Nemotron Parse and kept on disk ─────────
async def _read_upload(file: UploadFile) -> tuple[str, bytes]:
    data = await file.read()
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "That file is larger than 25 MB.")
    return file.filename or "document", data


async def _ingest(s: Services, name: str, data: bytes) -> DocumentInfo:
    """Keep an upload in the document store: a document parsed to text, or a recording (16 kHz PCM, which
    the browser decoded for us, named *.pcm) transcribed by the ASR NIM."""
    try:
        if name.endswith(".pcm"):
            transcript = await transcribe(s.listener, data)
            return await asyncio.to_thread(s.documents.keep_recording, name, data, transcript, ASR_RATE)
        return await s.documents.ingest(name, data)
    except UnreadableError as e:
        raise HTTPException(422, str(e)) from e
    except ParserUnavailable as e:
        raise HTTPException(503, str(e)) from e
    except Exception as e:
        log.exception("could not read %s", name)
        raise HTTPException(502, "That file could not be read right now.") from e


@app.post("/api/documents", response_model=DocumentInfo)
async def upload_document(s: Svc, file: UploadFile = File(...)) -> DocumentInfo:
    """Parse and store a document the user attaches to a question or a matter; they are sent with its id.
    A recording (*.pcm) is transcribed and kept as audio. The answer waits for the parser, so a long scan
    can outlast a proxy's patience: the app itself uploads through the jobs below."""
    return await _ingest(s, *await _read_upload(file))


# ── the same upload as a background job, polled by the browser ───────────
@dataclass
class UploadJob:
    """A file being read while the request that brought it is already answered."""

    id: str
    name: str
    started: float
    task: asyncio.Task[DocumentInfo]
    finished: float | None = None  # so the reported time stops when the reading does


JOB_TTL = 30 * 60  # a finished job is kept this long, in case the tab asks again after a sleep


def _job_status(job: UploadJob) -> UploadStatus:
    if job.task.done() and job.finished is None:
        job.finished = time.monotonic()
    seconds = round((job.finished or time.monotonic()) - job.started, 1)
    if not job.task.done():
        return UploadStatus(id=job.id, name=job.name, state="reading", seconds=seconds)
    if job.task.cancelled():
        return UploadStatus(id=job.id, name=job.name, state="failed", seconds=seconds,
                            error="Reading that file was stopped.", status=499)
    error = job.task.exception()
    if error is None:
        return UploadStatus(id=job.id, name=job.name, state="ready", seconds=seconds,
                            document=job.task.result())
    status, detail = ((error.status_code, str(error.detail)) if isinstance(error, HTTPException)
                      else (502, "That file could not be read right now."))
    return UploadStatus(id=job.id, name=job.name, state="failed", seconds=seconds, error=detail, status=status)


@app.post("/api/documents/jobs", response_model=UploadStatus, status_code=202)
async def start_upload(s: Svc, file: UploadFile = File(...)) -> UploadStatus:
    """Take the file and answer at once; it is read in the background. The browser polls
    GET /api/documents/jobs/{id} until the document is ready, and DELETEs the job to give up."""
    name, data = await _read_upload(file)
    for old in [j for j in s.uploads.values() if j.task.done() and time.monotonic() - j.started > JOB_TTL]:
        s.uploads.pop(old.id, None)
    job = UploadJob(f"job_{new_id()}", name, time.monotonic(), asyncio.create_task(_ingest(s, name, data)))
    s.uploads[job.id] = job
    return _job_status(job)


@app.get("/api/documents/jobs/{job_id}", response_model=UploadStatus)
async def upload_job(job_id: str, s: Svc) -> UploadStatus:
    job = s.uploads.get(job_id)
    if job is None:
        raise HTTPException(404, "That upload is no longer known - add the file again.")
    return _job_status(job)


@app.delete("/api/documents/jobs/{job_id}", status_code=204)
async def cancel_upload(job_id: str, s: Svc) -> Response:
    """Give up on an upload: stop reading it, and throw away what was already stored."""
    job = s.uploads.pop(job_id, None)
    if job is None:
        raise HTTPException(404, "That upload is no longer known.")
    if not job.task.done():
        job.task.cancel()
    elif not job.task.cancelled() and job.task.exception() is None:
        await asyncio.to_thread(s.documents.delete, job.task.result().id)
    return Response(status_code=204)


@app.get("/api/documents/accepted")
async def accepted_documents() -> list[str]:
    return ACCEPTED


@app.get("/api/documents/{document_id}", response_model=DocumentInfo)
async def get_document(document_id: str, s: Svc) -> DocumentInfo:
    info = s.documents.info(document_id)
    if info is None:
        raise HTTPException(404, "Document not found")
    return info


@app.delete("/api/documents/{document_id}", status_code=204)
async def delete_document(document_id: str, s: Svc) -> Response:
    """Remove an attached document (the evaluation cleans up after itself with this)."""
    if not await asyncio.to_thread(s.documents.delete, document_id):
        raise HTTPException(404, "Document not found")
    return Response(status_code=204)


@app.get("/api/documents/{document_id}/file")
async def document_file(document_id: str, s: Svc) -> FileResponse:
    """The document as it was uploaded (a recording as WAV)."""
    info, path = s.documents.info(document_id), s.documents.file(document_id)
    if info is None or path is None:
        raise HTTPException(404, "Document not found")
    name = info.name if info.name.lower().endswith(path.suffix) else f"{Path(info.name).stem}{path.suffix}"
    return FileResponse(path, filename=name, content_disposition_type="inline")


# ── matters: a client case through intake, research, assessment and drafting ──
_LABEL = {"document": "Document", "recording": "Recording of the client", "notes": "Notes"}


def _facts(s: Services, assets: list[DocumentInfo]) -> str:
    """The client's story as one text: every asset's text, headed by what it is when there are several."""
    texts = [(a, _without_page_lines(s.documents.text(a.id) or "")) for a in assets]
    if len(texts) == 1:
        return texts[0][1][:MAX_CHARS]
    each = MAX_CHARS // len(texts)  # a long file must not crowd the others out of the intake prompt
    return "\n\n".join(f"## {_LABEL[a.kind]}: {a.name}\n\n{text[:each]}" for a, text in texts)


def _without_page_lines(text: str) -> str:
    """The client's story without the "[Page n]" lines the stored text marks pages with."""
    return re.sub(r"(?m)^\[Page \d+\]\n+", "", text).strip()


def _describe(assets: list[DocumentInfo]) -> tuple[str | None, str]:
    """What the case file is, for the matter's list entry: one thing by name, or so many items."""
    if len(assets) == 1:
        one = assets[0]
        return (None if one.kind == "notes" else one.name,
                {"document": "document", "recording": "recording", "notes": "text"}[one.kind])
    return f"{len(assets)} items", "bundle"


def _open_matter(s: Services, assets: list[DocumentInfo], title: str | None) -> Matter:
    if not assets:
        raise HTTPException(422, "Add a document, a recording, or the facts as text.")
    name, kind = _describe(assets)
    return create_matter(s.matters, _facts(s, assets), name, kind, title, assets)


@app.get("/api/matters", response_model=list[MatterSummary])
async def list_matters(s: Svc) -> list[MatterSummary]:
    return s.matters.list()


@app.post("/api/matters", response_model=Matter)
async def new_matter(s: Svc, document_ids: list[str] = Form([]), file: UploadFile | None = File(None),
                     text: str | None = Form(None), title: str | None = Form(None)) -> Matter:
    """Open a matter on its case file: documents and recordings already uploaded with POST /api/documents
    (`document_ids`, in order), a file sent along (`file`), and typed notes (`text`)."""
    assets = []
    for document_id in dict.fromkeys(document_ids):
        if (info := s.documents.info(document_id)) is None:
            raise HTTPException(422, f"Document {document_id} not found - add it again.")
        assets.append(info)
    if file is not None and file.filename:
        assets.append(await _ingest(s, *await _read_upload(file)))
    if text and text.strip():
        assets.append(await asyncio.to_thread(s.documents.keep_notes, text.strip()))
    return _open_matter(s, assets, title)


@app.post("/api/matters/text", response_model=Matter)
async def new_matter_from_text(req: MatterRequest, s: Svc) -> Matter:
    return _open_matter(s, [await asyncio.to_thread(s.documents.keep_notes, req.text.strip())], req.title)


async def _located(s: Services, matter: Matter) -> Matter:
    """Matters taken in before the timeline was put in date order and traced back to the case file are
    caught up here, once, the first time the page is opened."""
    if not (matter.intake and matter.intake.timeline):
        return matter
    changed = in_order(matter.intake)
    if matter.intake.timeline_sources is None and matter.assets:
        matter.intake.timeline_sources = await asyncio.to_thread(locate_timeline, s.documents, matter)
        changed = True
    if changed:
        s.matters.save(matter)
    return matter


def _with_assets(s: Services, matter: Matter) -> Matter:
    """Matters opened before the case file was kept have only their one document's id."""
    if not matter.assets and matter.document_id and (info := s.documents.info(matter.document_id)):
        matter.assets = [info]
    return matter


@app.get("/api/matters/{matter_id}", response_model=Matter)
async def get_matter(matter_id: str, s: Svc) -> Matter:
    matter = s.matters.get(matter_id)
    if matter is None:
        raise HTTPException(404, "Matter not found")
    return await _located(s, _with_assets(s, matter))


@app.delete("/api/matters/{matter_id}", status_code=204)
async def delete_matter(matter_id: str, s: Svc) -> Response:
    matter = s.matters.get(matter_id)
    if matter is None or not s.matters.delete(matter_id):
        raise HTTPException(404, "Matter not found")
    for asset in _with_assets(s, matter).assets:  # the case file goes with the matter
        s.documents.delete(asset.id)
    s.documents.delete(prep_document_id(matter_id))  # and what was generated from it
    if s.case_index is not None:
        await asyncio.to_thread(s.case_index.drop, collection_of(matter_id))
    return Response(status_code=204)


DOCX_TYPE = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@app.get("/api/matters/{matter_id}/memo")
async def matter_memo(matter_id: str, s: Svc, format: str = "docx") -> Response:
    """The memo as a Word file for the client file (`?format=md` for the Markdown behind it)."""
    matter = s.matters.get(matter_id)
    if matter is None:
        raise HTTPException(404, "Matter not found")
    name = re.sub(r"[^\w\-]+", "-", matter.title).strip("-").lower() or "memo"
    if format == "md":
        return Response(matter.memo or memo(matter), media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="{name}.md"'})
    body = await asyncio.to_thread(docx_memo, matter)
    return Response(body, media_type=DOCX_TYPE,
                    headers={"Content-Disposition": f'attachment; filename="{name}.docx"'})


@app.post("/api/matters/{matter_id}/assets", response_model=Matter)
async def add_matter_assets(matter_id: str, s: Svc, document_ids: list[str] = Form([]),
                            file: UploadFile | None = File(None), text: str | None = Form(None)) -> Matter:
    """Add to the case file of a matter that already exists: documents and recordings already uploaded
    (`document_ids`), a file sent along (`file`), or notes (`text`). The facts are rebuilt from the whole
    case file and the new pieces are indexed at once, so the assistant can use them before the research
    is run again; the matter remembers how much its last run has not seen."""
    matter = s.matters.get(matter_id)
    if matter is None:
        raise HTTPException(404, "Matter not found")
    matter = _with_assets(s, matter)
    have = {a.id for a in matter.assets}
    added: list[DocumentInfo] = []
    for document_id in dict.fromkeys(document_ids):
        if document_id in have:
            continue  # added twice from the same page: keep the case file as it is
        if (info := s.documents.info(document_id)) is None:
            raise HTTPException(422, f"Document {document_id} not found - add it again.")
        added.append(info)
    if file is not None and file.filename:
        added.append(await _ingest(s, *await _read_upload(file)))
    if text and text.strip():
        added.append(await asyncio.to_thread(s.documents.keep_notes, text.strip()))
    if not added:
        if document_ids:  # every one of them is already in the case file: a double click, or a retry
            return matter
        raise HTTPException(422, "Add a document, a recording, or notes.")

    matter.assets = [*matter.assets, *added]
    matter.facts = _facts(s, matter.assets)
    matter.source_name, matter.source_kind = _describe(matter.assets)  # type: ignore[assignment]
    matter.document_id = matter.assets[0].id
    if matter.stage == "done":  # a run in progress or still to come reads the whole case file anyway
        matter.added_since_run += len(added)
    if s.case_index is not None:
        collection = collection_of(matter.id)
        try:
            await asyncio.to_thread(s.case_index.add, collection, matter.assets)
            matter.indexed = await asyncio.to_thread(s.case_index.count, collection)
        except IndexUnavailable as e:  # the run indexes what is missing when the embedder is back
            log.warning("matter %s: %s not indexed (%s)", matter.id, ", ".join(a.name for a in added), e)
    log.info("matter %s: %d item(s) added to the case file", matter.id, len(added))
    return s.matters.save(matter)


@app.post("/api/matters/{matter_id}/run")
async def run_matter(matter_id: str, s: Svc) -> StreamingResponse:
    matter = s.matters.get(matter_id)
    if matter is None:
        raise HTTPException(404, "Matter not found")
    matter = _with_assets(s, matter)  # so an older matter's document is indexed too

    async def stream() -> AsyncIterator[str]:
        async for event in s.pipeline.run(matter):
            yield _sse(event)

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"})


@app.post("/api/chat")
async def chat(req: ChatRequest, s: Svc) -> StreamingResponse:
    attachments = []
    for document_id in dict.fromkeys(req.document_ids):
        if (info := s.documents.info(document_id)) is None:
            raise HTTPException(422, f"Document {document_id} not found - attach it again.")
        attachments.append(info)
    if req.conversation_id:
        conv = s.store.get(req.conversation_id)
        if conv is None:
            raise HTTPException(404, "Conversation not found")
        cid, title, history = conv.id, conv.title, conv.messages
        matter_id = conv.matter_id
    else:
        if req.matter_id and s.matters.get(req.matter_id) is None:
            raise HTTPException(404, "Matter not found")
        created = s.store.create(_title(req.message), req.matter_id)
        cid, title, history, matter_id = created.id, created.title, [], created.matter_id
    user = Message(id=new_id(), role="user", content=req.message, attachments=attachments or None,
                   created_at=now())
    s.store.add_message(cid, user)
    summary = ConversationSummary(id=cid, title=title, updated_at=user.created_at, matter_id=matter_id)
    return StreamingResponse(
        # a reply to a question asked back is short ("Wohnmietvertrag"); its language is the question's
        _stream(s, summary, req.message, history, detect_language(original_question(req.message, history)),
                req.allow_questions, attachments),
        media_type="text/event-stream",
        # no-transform: compressing proxies (code-server's port proxy, CDNs) would otherwise buffer the stream
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


async def _events(s: Services, conv: ConversationSummary, question: str, history: list[Message],
                  language: str, ask: bool = True,
                  attachments: list[DocumentInfo] | None = None) -> AsyncIterator[dict[str, Any]]:
    """One assistant turn as events (the SSE route and the voice socket both send these), saved when done.
    If the caller stops early (voice barge-in), what was written so far is still saved."""
    yield {"type": "conversation", "conversation": conv.model_dump(by_alias=True)}
    yield {"type": "meta", "language": language}
    parts: list[str] = []
    sources: list[Source] = []
    calls: dict[str, ToolCall] = {}
    clarification: Clarification | None = None
    mentions: list[StatuteRef] = []  # documents and decisions named in the text, linked like statutes

    def save(statutes: list | None = None) -> Message:
        msg = Message(id=new_id(), role="assistant", content="".join(parts), sources=sources,
                      tool_calls=list(calls.values()) or None, language=language, statutes=statutes or None,
                      clarification=clarification, created_at=now())
        s.store.add_message(conv.id, msg)
        return msg

    # a conversation asked from Case Prep answers against its matter: the case file, and the prep as background
    case_file, collection, prep = attachments or [], None, None
    if conv.matter_id and (matter := s.matters.get(conv.matter_id)) is not None:
        matter = _with_assets(s, matter)
        kept = {d.id for d in matter.assets}
        case_file = [*matter.assets, *(d for d in attachments or [] if d.id not in kept)]
        collection = collection_of(matter.id) if matter.indexed else None
        prep = await asyncio.to_thread(case_prep_document, s.documents, matter)

    with tracing.turn(question, conv.id, language) as trace:
        try:
            async for ev in s.agent.answer(question, history, ask=ask, attachments=case_file,
                                           collection=collection, case_prep=prep):
                trace.event(ev)
                match ev:
                    case Status():
                        yield {"type": "status", "stage": ev.stage, "detail": ev.detail}
                    case Thought():
                        yield {"type": "thinking", "text": ev.text}
                    case ToolStart():
                        calls[ev.id] = ToolCall(id=ev.id, name=ev.name, args=ev.args, thought=ev.thought)
                        yield {"type": "tool_start", "call": calls[ev.id].model_dump(by_alias=True)}
                    case ToolEnd():
                        if ev.id in calls:
                            calls[ev.id].summary, calls[ev.id].error = ev.summary, ev.error
                        yield {"type": "tool_end", "id": ev.id, "summary": ev.summary, "error": ev.error}
                    case Delta():
                        parts.append(ev.text)
                        yield {"type": "delta", "text": ev.text}
                    case Cite():
                        if all(x.n != ev.source.n for x in sources):
                            sources.append(ev.source)
                        parts.append(f"[{ev.source.n}]")
                        yield {"type": "citation", "source": ev.source.model_dump(by_alias=True)}
                    case Clarify():
                        clarification = Clarification(question=ev.question, options=ev.options, notes=ev.notes)
                        yield {"type": "clarify", "clarification": clarification.model_dump(by_alias=True)}
                    case Mention():
                        if all(m.text != ev.text for m in mentions):
                            mentions.append(StatuteRef(text=ev.text, source=ev.source))
                    case Verdict():
                        for source in sources:  # saved with the message, so the check survives a reload
                            if source.n == ev.n:
                                source.supported = ev.supported
                        yield {"type": "verdict", "n": ev.n, "supported": ev.supported}
        except asyncio.CancelledError:
            # keep the abandoned turn in the history, so the next one answers the new question instead of
            # carrying on with the old one
            parts.append(" … [cut off when the user started speaking]" if parts
                         else "[the user asked something else before this was answered]")
            trace.finish("".join(parts), sources)
            save()
            raise
        except Exception:
            log.exception("agent failed on %r", question)
            trace.finish("".join(parts), sources)
            yield {"type": "error", "message": "The assistant failed to answer. Please try again."}
            return
        # the articles the answer names, linked to their text (see mentions.py), and the documents and
        # decisions it names
        msg = save([*mentions, *await asyncio.to_thread(statute_links, s.decisions, "".join(parts), language)])
        trace.finish(msg.content, sources)
        yield {"type": "done", "message": msg.model_dump(by_alias=True)}


async def _stream(s: Services, conv: ConversationSummary, question: str, history: list[Message], language: str,
                  ask: bool = True, attachments: list[DocumentInfo] | None = None) -> AsyncIterator[str]:
    async for event in _events(s, conv, question, history, language, ask, attachments):
        yield _sse(event)


# ── voice mode ──────────────────────────────────────────────────────────
_BOUNDARY = re.compile(r"[.!?;:]\s+(?=[\"'«(]?[A-ZÀ-ÖØ-Þ0-9])")
# a dot after these is not the end of a sentence ("Art. 257d", "BGE 138 III 59 E. 2.1")
_ABBREVIATIONS = {"art", "abs", "al", "lit", "let", "lett", "cpv", "ziff", "ch", "nr", "no", "consid", "e", "bge",
                  "atf", "dtf", "vgl", "bzw", "ca", "etc", "s", "p", "z", "b", "lic", "iur", "dr"}
_WORD_BEFORE = re.compile(r"([^\W\d_]+)\W*$")
SPEAK_MIN = 25  # characters; shorter fragments are held back for the next sentence
# silence after the last recognised speech before answering; with the ASR's own ~0.9 s endpointing
# that is about two seconds after the user stops talking
VOICE_PAUSE = float(os.environ.get("SCA_VOICE_PAUSE", "1.2"))


def _next_sentence(buffer: str) -> tuple[str, str] | None:
    """The first complete sentence in `buffer` and what is left, or None while it is still growing."""
    for m in _BOUNDARY.finditer(buffer):
        head = buffer[:m.end()]
        word = _WORD_BEFORE.search(buffer[:m.start() + 1])
        if word and word.group(1).lower() in _ABBREVIATIONS:
            continue
        if len(head.strip()) >= SPEAK_MIN:
            return head.strip(), buffer[m.end():]
    return None


class _Voice:
    """Speaks what the assistant says, one line at a time, over the socket. Synthesis runs behind a
    queue so the text keeps streaming while the previous line is still playing."""

    def __init__(self, ws: WebSocket, speaker: Speaker, language: str):
        self.ws, self.speaker, self.language = ws, speaker, language
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.task = asyncio.create_task(self._run())
        self.buffer = ""
        self.last = ""

    async def _run(self) -> None:
        while (text := await self.queue.get()) is not None:
            try:
                audio = await self.speaker.stream(text, self.language)
                await self.ws.send_json({"type": "speech_start", "text": text, "sampleRate": SAMPLE_RATE})
                async for chunk in audio:
                    await self.ws.send_bytes(chunk)
                await self.ws.send_json({"type": "speech_end"})
            except asyncio.CancelledError:
                raise
            except Exception:
                log.warning("could not speak %r", text[:60], exc_info=True)

    async def say(self, text: str) -> None:
        text = text.strip()
        if text and text != self.last:  # two reads in a row would repeat the same line
            self.last = text
            await self.queue.put(text)

    async def handle(self, event: dict[str, Any]) -> None:
        """Narrate the research, and speak the answer sentence by sentence as it streams."""
        match event.get("type"):
            case "tool_start":
                call = event["call"]
                if line := narrate(call["name"], call["args"], self.language):
                    await self.say(line)
            case "delta":
                self.buffer += event["text"]
                while (split := _next_sentence(self.buffer)) is not None:
                    sentence, self.buffer = split
                    await self.say(sentence)
            case "done" | "error":
                await self.flush()

    async def flush(self) -> None:
        await self.say(self.buffer)
        self.buffer = ""

    async def close(self) -> None:
        await self.queue.put(None)
        await self.task


async def _voice_turn(ws: WebSocket, s: Services, state: dict[str, Any], question: str) -> None:
    """One spoken turn: save the question, then stream the answer as events and as speech."""
    conv = s.store.get(state["cid"]) if state["cid"] else None
    if conv is None:
        created = s.store.create(_title(question), state.get("matter"))
        state["cid"], title, history, matter_id = created.id, created.title, [], created.matter_id
    else:
        title, history, matter_id = conv.title, conv.messages, conv.matter_id
    user = Message(id=new_id(), role="user", content=question, created_at=now())
    s.store.add_message(state["cid"], user)
    await ws.send_json({"type": "user", "message": user.model_dump(by_alias=True)})
    summary = ConversationSummary(id=state["cid"], title=title, updated_at=user.created_at, matter_id=matter_id)
    language = detect_language(original_question(question, history), default=state["language"])
    voice = _Voice(ws, s.speaker, language)
    try:
        async for event in _events(s, summary, question, history, language):
            await ws.send_json(event)
            await voice.handle(event)
        await voice.close()
    except asyncio.CancelledError:
        voice.task.cancel()
        raise


@app.websocket("/api/voice")
async def voice(ws: WebSocket) -> None:
    """Voice mode: microphone audio in (16 kHz PCM), transcripts, agent events and speech out.
    Speaking while the assistant talks interrupts it and starts a new turn."""
    await ws.accept()
    s: Services = ws.app.state.services
    start = await ws.receive_json()
    state = {"cid": start.get("conversationId"), "language": start.get("language") or "en",
             "matter": start.get("matterId") if s.matters.get(start.get("matterId") or "") else None}
    audio: asyncio.Queue[bytes | None] = asyncio.Queue()
    turn: asyncio.Task | None = None
    answering: asyncio.Task | None = None
    utterance: list[str] = []

    async def answer_after_pause() -> None:
        """The ASR ends an utterance after a short silence, which cuts sentences in half. Wait for a
        real pause and answer everything heard since the last one."""
        nonlocal turn
        await asyncio.sleep(VOICE_PAUSE)
        question = " ".join(utterance).strip()
        utterance.clear()
        if not question:
            return
        if turn:  # let an abandoned turn save its "cut off" marker before this one reads the history
            await asyncio.wait({turn}, timeout=5)
        turn = asyncio.create_task(_voice_turn(ws, s, state, question))

    async def read_audio() -> None:
        try:
            while True:
                message = await ws.receive()
                if message["type"] == "websocket.disconnect":
                    break
                if (chunk := message.get("bytes")) is not None:
                    await audio.put(chunk)
                elif json.loads(message.get("text") or "{}").get("type") == "stop":
                    break
        finally:
            await audio.put(None)

    reader = asyncio.create_task(read_audio())
    try:
        async for text, final in s.listener.transcribe(audio, state["language"]):
            if not text:
                continue
            # The user is talking. Always stop the voice: the browser is still playing audio the server
            # sent long ago, so a finished turn must be silenced too, not only a running one.
            if turn is not None:  # nothing has been spoken yet before the first turn
                await ws.send_json({"type": "cancel_speech"})
            if turn and not turn.done():
                turn.cancel()
            if answering:
                answering.cancel()  # still speaking: the pause has not happened yet
            await ws.send_json({"type": "partial", "text": text, "final": final})
            if final:
                utterance.append(text)
                answering = asyncio.create_task(answer_after_pause())
    except WebSocketDisconnect:
        pass
    except Exception:
        log.exception("voice session failed")
        with contextlib.suppress(Exception):
            await ws.send_json({"type": "error", "message": "Voice mode stopped: the speech model failed."})
    finally:
        if turn and not turn.done():
            turn.cancel()
        if answering:
            answering.cancel()
        reader.cancel()
        with contextlib.suppress(Exception):
            await ws.close()


if WEB_DIST.is_dir():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
