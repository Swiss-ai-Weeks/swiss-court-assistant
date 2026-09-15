
from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import Agent, Cite, Delta, Status, StubAgent, ToolEnd, ToolStart
from .decisions import DecisionStore
from .language import detect_language
from .schemas import (ChatRequest, Conversation, ConversationSummary, Decision, Health, Message, Source, ToolCall,
                      TranslateRequest, TranslateResponse)
from .store import ConversationStore, new_id, now
from .translate import Translator, UntranslatableError

DECISIONS = Path(os.environ.get("SCA_DECISIONS", "data/subset/decisions_50k_seed42.parquet"))
CHUNKS = DECISIONS.with_name(DECISIONS.stem + ".chunks.parquet")
VECTOR_DB = Path(os.environ.get("SCA_VECTOR_DB", f"data/vectordb/{DECISIONS.stem}.sqlite"))
KEYWORD_INDEX = VECTOR_DB.with_name(VECTOR_DB.stem + ".fts.sqlite")
DB = Path(os.environ.get("SCA_DB", "data/app/conversations.sqlite"))
AGENT = os.environ.get("SCA_AGENT", "react")  # react | stub
WEB_DIST = Path(__file__).resolve().parents[3] / "web" / "dist"

log = logging.getLogger(__name__)


@dataclass
class Services:
    decisions: DecisionStore
    store: ConversationStore
    agent: Agent
    translator: Translator


@asynccontextmanager
async def lifespan(app: FastAPI):
    decisions = DecisionStore(DECISIONS)
    app.state.services = Services(decisions, ConversationStore(DB), _make_agent(decisions), Translator())
    log.info("loaded %d decisions; agent=%s", len(decisions), app.state.services.agent.name)
    yield


def _make_agent(decisions: DecisionStore) -> Agent:
    if AGENT == "stub":
        return StubAgent(CHUNKS, decisions)
    import torch

    from .corpus import Corpus
    from .react_agent import ReactAgent

    device = os.environ.get("SCA_EMBED_DEVICE", "cuda:1" if torch.cuda.device_count() > 1 else "cpu")
    corpus = Corpus(VECTOR_DB, KEYWORD_INDEX, decisions, os.environ.get("SCA_EMBED_MODEL"), device,
                    rerank=os.environ.get("SCA_RERANK", "1") == "1")
    return ReactAgent(corpus)


def services(request: Request) -> Services:
    return request.app.state.services


Svc = Annotated[Services, Depends(services)]
app = FastAPI(title="Swiss Court Assistant", lifespan=lifespan)


@app.get("/api/health", response_model=Health)
async def health(s: Svc) -> Health:
    return Health(status="ok", agent=s.agent.name, decisions=len(s.decisions))


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
    return conv


@app.delete("/api/conversations/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, s: Svc) -> Response:
    if not s.store.delete(conversation_id):
        raise HTTPException(404, "Conversation not found")
    return Response(status_code=204)


@app.get("/api/decisions/{decision_id}", response_model=Decision)
async def get_decision(decision_id: str, s: Svc) -> Decision:
    d = s.decisions.get(decision_id)
    if d is None:
        raise HTTPException(404, "Decision not found")
    return d


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


def _title(text: str) -> str:
    text = " ".join(text.split())
    return text if len(text) <= 60 else text[:57] + "…"


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest, s: Svc) -> StreamingResponse:
    if req.conversation_id:
        conv = s.store.get(req.conversation_id)
        if conv is None:
            raise HTTPException(404, "Conversation not found")
        cid, title, history = conv.id, conv.title, conv.messages
    else:
        created = s.store.create(_title(req.message))
        cid, title, history = created.id, created.title, []
    user = Message(id=new_id(), role="user", content=req.message, created_at=now())
    s.store.add_message(cid, user)
    summary = ConversationSummary(id=cid, title=title, updated_at=user.created_at)
    return StreamingResponse(
        _stream(s, summary, req.message, history, detect_language(req.message)),
        media_type="text/event-stream",
        # no-transform: compressing proxies (code-server's port proxy, CDNs) would otherwise buffer the stream
        headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
    )


async def _stream(s: Services, conv: ConversationSummary, question: str,
                  history: list[Message], language: str) -> AsyncIterator[str]:
    yield _sse({"type": "conversation", "conversation": conv.model_dump(by_alias=True)})
    yield _sse({"type": "meta", "language": language})
    parts: list[str] = []
    sources: list[Source] = []
    calls: dict[str, ToolCall] = {}
    try:
        async for ev in s.agent.answer(question, history):
            match ev:
                case Status():
                    yield _sse({"type": "status", "stage": ev.stage, "detail": ev.detail})
                case ToolStart():
                    calls[ev.id] = ToolCall(id=ev.id, name=ev.name, args=ev.args)
                    yield _sse({"type": "tool_start", "call": calls[ev.id].model_dump(by_alias=True)})
                case ToolEnd():
                    if ev.id in calls:
                        calls[ev.id].summary, calls[ev.id].error = ev.summary, ev.error
                    yield _sse({"type": "tool_end", "id": ev.id, "summary": ev.summary, "error": ev.error})
                case Delta():
                    parts.append(ev.text)
                    yield _sse({"type": "delta", "text": ev.text})
                case Cite():
                    if all(x.n != ev.source.n for x in sources):
                        sources.append(ev.source)
                    parts.append(f"[{ev.source.n}]")
                    yield _sse({"type": "citation", "source": ev.source.model_dump(by_alias=True)})
    except Exception: 
        log.exception("agent failed on %r", question)
        yield _sse({"type": "error", "message": "The assistant failed to answer. Please try again."})
        return
    msg = Message(id=new_id(), role="assistant", content="".join(parts), sources=sources,
                  tool_calls=list(calls.values()) or None, language=language, created_at=now())
    s.store.add_message(conv.id, msg)
    yield _sse({"type": "done", "message": msg.model_dump(by_alias=True)})


if WEB_DIST.is_dir(): 
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
