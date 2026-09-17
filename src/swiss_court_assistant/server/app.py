
from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import re
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import Agent, Cite, Delta, Status, StubAgent, Thought, ToolEnd, ToolStart
from .decisions import DecisionStore
from .language import detect_language
from .schemas import (ChatRequest, Conversation, ConversationSummary, Decision, Health, Message, Source,
                      SpeechRequest, ToolCall, TranslateRequest, TranslateResponse)
from .speech import SAMPLE_RATE, Speaker, UnspeakableError
from .store import ConversationStore, new_id, now
from .translate import Translator, UntranslatableError
from .voice import Listener, narrate

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
    speaker: Speaker
    listener: Listener


@asynccontextmanager
async def lifespan(app: FastAPI):
    decisions = DecisionStore(DECISIONS)
    app.state.services = Services(decisions, ConversationStore(DB), _make_agent(decisions), Translator(), Speaker(),
                                  Listener())
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
    try:
        languages = sorted(await asyncio.to_thread(s.listener.languages))
    except Exception:  # voice mode is optional: the rest of the app works without the ASR NIM
        languages = []
    return Health(status="ok", agent=s.agent.name, decisions=len(s.decisions), speech_languages=languages)


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


async def _events(s: Services, conv: ConversationSummary, question: str, history: list[Message],
                  language: str) -> AsyncIterator[dict[str, Any]]:
    """One assistant turn as events (the SSE route and the voice socket both send these), saved when done.
    If the caller stops early (voice barge-in), what was written so far is still saved."""
    yield {"type": "conversation", "conversation": conv.model_dump(by_alias=True)}
    yield {"type": "meta", "language": language}
    parts: list[str] = []
    sources: list[Source] = []
    calls: dict[str, ToolCall] = {}

    def save() -> Message:
        msg = Message(id=new_id(), role="assistant", content="".join(parts), sources=sources,
                      tool_calls=list(calls.values()) or None, language=language, created_at=now())
        s.store.add_message(conv.id, msg)
        return msg

    try:
        async for ev in s.agent.answer(question, history):
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
    except asyncio.CancelledError:
        # keep the abandoned turn in the history, so the next one answers the new question instead of
        # carrying on with the old one
        parts.append(" … [cut off when the user started speaking]" if parts
                     else "[the user asked something else before this was answered]")
        save()
        raise
    except Exception:
        log.exception("agent failed on %r", question)
        yield {"type": "error", "message": "The assistant failed to answer. Please try again."}
        return
    yield {"type": "done", "message": save().model_dump(by_alias=True)}


async def _stream(s: Services, conv: ConversationSummary, question: str,
                  history: list[Message], language: str) -> AsyncIterator[str]:
    async for event in _events(s, conv, question, history, language):
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
        created = s.store.create(_title(question))
        state["cid"], title, history = created.id, created.title, []
    else:
        title, history = conv.title, conv.messages
    user = Message(id=new_id(), role="user", content=question, created_at=now())
    s.store.add_message(state["cid"], user)
    await ws.send_json({"type": "user", "message": user.model_dump(by_alias=True)})
    summary = ConversationSummary(id=state["cid"], title=title, updated_at=user.created_at)
    language = detect_language(question, default=state["language"])
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
    state = {"cid": start.get("conversationId"), "language": start.get("language") or "en"}
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
