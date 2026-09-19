"""Conversation history in SQLite (one file, created on first start)."""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import TypeAdapter

from .schemas import (Clarification, Conversation, ConversationSummary, DocumentInfo, Message, Source, StatuteRef,
                      ToolCall)

_SOURCES = TypeAdapter(list[Source])
_TOOL_CALLS = TypeAdapter(list[ToolCall])
_STATUTES = TypeAdapter(list[StatuteRef])
_CLARIFICATION = TypeAdapter(Clarification)
_ATTACHMENTS = TypeAdapter(list[DocumentInfo])
_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    seq INTEGER NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    search_query TEXT,
    sources TEXT,  -- JSON list[Source]
    created_at TEXT NOT NULL,
    tool_calls TEXT  -- JSON list[ToolCall]
);
CREATE INDEX IF NOT EXISTS messages_by_conversation ON messages(conversation_id, seq);
"""


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class ConversationStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        self._db.executescript(_SCHEMA)
        columns = {r["name"] for r in self._db.execute("PRAGMA table_info(messages)")}
        if "tool_calls" not in columns:
            self._db.execute("ALTER TABLE messages ADD COLUMN tool_calls TEXT")
        if "statutes" not in columns:
            self._db.execute("ALTER TABLE messages ADD COLUMN statutes TEXT")  # JSON list[StatuteRef]
        if "clarification" not in columns:
            self._db.execute("ALTER TABLE messages ADD COLUMN clarification TEXT")  # JSON Clarification
        if "attachments" not in columns:
            self._db.execute("ALTER TABLE messages ADD COLUMN attachments TEXT")  # JSON list[DocumentInfo]
        if "matter_id" not in {r["name"] for r in self._db.execute("PRAGMA table_info(conversations)")}:
            # a conversation asked from Case Prep about one matter; NULL for the general assistant
            self._db.execute("ALTER TABLE conversations ADD COLUMN matter_id TEXT")
        self._lock = threading.Lock()

    def list(self) -> list[ConversationSummary]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, title, updated_at, matter_id FROM conversations ORDER BY updated_at DESC").fetchall()
        return [ConversationSummary(**dict(r)) for r in rows]

    def get(self, conversation_id: str) -> Conversation | None:
        with self._lock:
            c = self._db.execute("SELECT id, title, updated_at, matter_id FROM conversations WHERE id = ?",
                                 (conversation_id,)).fetchone()
            rows = self._db.execute(
                "SELECT id, role, content, search_query, sources, tool_calls, statutes, clarification, attachments, "
                "created_at "
                "FROM messages "
                "WHERE conversation_id = ? ORDER BY seq", (conversation_id,)).fetchall()
        if c is None:
            return None
        messages = [
            Message(**(dict(r) | {
                "sources": _SOURCES.validate_json(r["sources"]) if r["sources"] else None,
                "tool_calls": _TOOL_CALLS.validate_json(r["tool_calls"]) if r["tool_calls"] else None,
                "statutes": _STATUTES.validate_json(r["statutes"]) if r["statutes"] else None,
                "clarification": _CLARIFICATION.validate_json(r["clarification"]) if r["clarification"] else None,
                "attachments": _ATTACHMENTS.validate_json(r["attachments"]) if r["attachments"] else None,
            }))
            for r in rows
        ]
        return Conversation(**dict(c), messages=messages)

    def create(self, title: str, matter_id: str | None = None) -> ConversationSummary:
        cid, t = new_id(), now()
        with self._lock, self._db:
            self._db.execute("INSERT INTO conversations (id, title, created_at, updated_at, matter_id) "
                             "VALUES (?, ?, ?, ?, ?)", (cid, title, t, t, matter_id))
        return ConversationSummary(id=cid, title=title, updated_at=t, matter_id=matter_id)

    def add_message(self, conversation_id: str, msg: Message) -> None:
        sources = _SOURCES.dump_json(msg.sources).decode() if msg.sources is not None else None
        calls = _TOOL_CALLS.dump_json(msg.tool_calls).decode() if msg.tool_calls is not None else None
        statutes = _STATUTES.dump_json(msg.statutes).decode() if msg.statutes else None
        clarification = _CLARIFICATION.dump_json(msg.clarification).decode() if msg.clarification else None
        attachments = _ATTACHMENTS.dump_json(msg.attachments).decode() if msg.attachments else None
        with self._lock, self._db:
            seq = self._db.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM messages WHERE conversation_id = ?",
                                   (conversation_id,)).fetchone()[0]
            self._db.execute(
                "INSERT INTO messages (id, conversation_id, seq, role, content, search_query, sources, "
                "tool_calls, statutes, clarification, attachments, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (msg.id, conversation_id, seq, msg.role, msg.content, msg.search_query, sources, calls,
                 statutes, clarification, attachments, msg.created_at))
            self._db.execute("UPDATE conversations SET updated_at = ? WHERE id = ?",
                             (msg.created_at, conversation_id))

    def delete(self, conversation_id: str) -> bool:
        with self._lock, self._db:
            return self._db.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,)).rowcount > 0
