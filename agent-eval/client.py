"""Ask the running assistant a question and collect the whole turn.

The evaluation drives the agent through its own HTTP API rather than importing it, so what is
measured is the assistant as it is served: the same corpus, the same retrieval, the same grounding
check. `POST /api/chat` streams the turn as SSE; the closing `done` event carries the finished
message with its sources (including `verified` and `supported`), so the turn is read off that and
the deltas are only a fallback.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class Turn:
    question: str
    answer: str = ""                          # with [n] citation markers, as the UI shows it
    sources: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    conversation_id: str | None = None
    seconds: float = 0.0
    thinking_chars: int = 0
    error: str | None = None

    @property
    def tools(self) -> list[str]:
        return [c.get("name", "") for c in self.tool_calls]


async def ask(client: httpx.AsyncClient, question: str, conversation_id: str | None = None) -> Turn:
    """One turn. Never raises: a failed turn comes back with `error` set, so one broken case does
    not take the run down with it."""
    turn = Turn(question=question, conversation_id=conversation_id)
    # the assistant may ask back instead of answering; a graded case needs the answer
    body: dict[str, Any] = {"message": question, "allowQuestions": False}
    if conversation_id:
        body["conversationId"] = conversation_id
    started = time.monotonic()
    deltas: list[str] = []
    try:
        async with client.stream("POST", "/api/chat", json=body) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except ValueError:
                    continue
                match event.get("type"):
                    case "conversation":
                        turn.conversation_id = event["conversation"]["id"]
                    case "thinking":
                        turn.thinking_chars += len(event.get("text", ""))
                    case "tool_start":
                        turn.tool_calls.append(dict(event["call"]))
                    case "tool_end":
                        for call in turn.tool_calls:
                            if call.get("id") == event.get("id"):
                                call["summary"], call["error"] = event.get("summary"), event.get("error")
                    case "delta":
                        deltas.append(event.get("text", ""))
                    case "citation":
                        deltas.append(f"[{event['source']['n']}]")
                    case "done":
                        message = event.get("message") or {}
                        turn.answer = message.get("content") or "".join(deltas)
                        turn.sources = message.get("sources") or []
                    case "clarify":  # should not happen with allowQuestions off
                        asked = (event.get("clarification") or {}).get("question", "")
                        turn.error = f"asked back instead of answering: {asked[:120]}"
                    case "error":
                        turn.error = str(event.get("detail") or event.get("message") or "stream error")
    except httpx.HTTPStatusError as e:
        turn.error = f"HTTP {e.response.status_code}"
    except Exception as e:  # a timeout, a dropped connection: the case fails, the run goes on
        turn.error = f"{type(e).__name__}: {e}"
    turn.seconds = time.monotonic() - started
    if not turn.answer:
        turn.answer = "".join(deltas)
        if not turn.answer and turn.error is None:
            turn.error = "the stream ended without an answer"
    return turn


async def delete_conversation(client: httpx.AsyncClient, conversation_id: str) -> None:
    """Remove a conversation this run created, so evaluating does not fill the user's sidebar."""
    try:
        await client.delete(f"/api/conversations/{conversation_id}")
    except httpx.HTTPError:
        pass


async def health(client: httpx.AsyncClient) -> dict[str, Any]:
    response = await client.get("/api/health")
    response.raise_for_status()
    return response.json()
