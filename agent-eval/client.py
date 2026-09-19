"""Ask the running assistant a question and collect the whole turn."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
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
    indexed: int | None = None                # Case Prep: passages of the case file in its search index

    @property
    def tools(self) -> list[str]:
        return [c.get("name", "") for c in self.tool_calls]


async def upload(client: httpx.AsyncClient, path: Path) -> str:
    """Attach a file the way the UI does (parsed, or transcribed, on the server); returns its id."""
    response = await client.post("/api/documents", files={"file": (path.name, path.read_bytes())})
    response.raise_for_status()
    return response.json()["id"]


async def delete_document(client: httpx.AsyncClient, document_id: str) -> None:
    try:
        await client.delete(f"/api/documents/{document_id}")
    except httpx.HTTPError:
        pass


async def ask(client: httpx.AsyncClient, question: str, conversation_id: str | None = None,
              document_ids: list[str] | None = None) -> Turn:
    """One turn. Never raises: a failed turn comes back with `error` set, so one broken case does
    not take the run down with it."""
    turn = Turn(question=question, conversation_id=conversation_id)
    # the assistant may ask back instead of answering; a graded case needs the answer
    body: dict[str, Any] = {"message": question, "allowQuestions": False}
    if conversation_id:
        body["conversationId"] = conversation_id
    if document_ids:
        body["documentIds"] = document_ids
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


async def prepare_matter(client: httpx.AsyncClient, document_ids: list[str]) -> tuple[Turn, str | None]:
    """Case Prep on a case file: open a matter on the uploaded documents and run it through intake,
    research, assessment and drafting. The researched issues become one answer - each issue as a
    heading with its answer below, the [n] markers renumbered across issues so they point into one
    list of sources - which is what gets graded. Returns the turn and the matter's id, to delete."""
    turn = Turn(question="(Case Prep)")
    started = time.monotonic()
    matter_id = None
    try:
        response = await client.post("/api/matters", data={"document_ids": document_ids})
        response.raise_for_status()
        matter_id = response.json()["id"]
        async with client.stream("POST", f"/api/matters/{matter_id}/run") as stream:
            stream.raise_for_status()
            async for line in stream.aiter_lines():
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except ValueError:
                    continue
                match event.get("type"):
                    case "indexed":
                        turn.indexed = event.get("passages")
                    case "issue_tool":
                        turn.tool_calls.append({"name": event["name"], "args": {"arg": event.get("arg")},
                                                "issue": event["n"]})
                    case "error":
                        turn.error = f"{event.get('stage')}: {event.get('message')}"
                    case "done":
                        matter = event["matter"]
                        parts, offset = [], 0
                        for issue in matter.get("issues") or []:
                            answer = re.sub(r"\[(\d+)\]", lambda m: f"[{int(m.group(1)) + offset}]",
                                            issue.get("answer") or "")
                            for source in issue.get("sources") or []:
                                turn.sources.append(source | {"n": source["n"] + offset})
                            offset += max((s["n"] for s in issue.get("sources") or []), default=0)
                            parts.append(f"## {issue['question']}\n\n{answer.strip()}")
                        turn.answer = "\n\n".join(parts)
                        turn.indexed = matter.get("indexed", turn.indexed)
    except httpx.HTTPStatusError as e:
        turn.error = f"HTTP {e.response.status_code}"
    except Exception as e:
        turn.error = f"{type(e).__name__}: {e}"
    turn.seconds = time.monotonic() - started
    if not turn.answer and turn.error is None:
        turn.error = "the matter finished without researched issues"
    return turn, matter_id


async def delete_matter(client: httpx.AsyncClient, matter_id: str) -> None:
    """Remove a matter this run opened - with its case file and its collection in the case index."""
    try:
        await client.delete(f"/api/matters/{matter_id}")
    except httpx.HTTPError:
        pass


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
