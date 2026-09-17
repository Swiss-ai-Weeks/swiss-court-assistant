"""What a client hands over, turned into text the assistant can work on.

A matter starts either from a document (PDF, Word, plain text) or from a recording of the client
telling their story. Recordings arrive as 16 kHz mono PCM because the browser decodes and resamples
them: every format the browser can play then works without ffmpeg on the server, and the bytes are
already exactly what the ASR NIM expects.
"""

from __future__ import annotations

import asyncio
import io
import logging
import re
from pathlib import PurePosixPath

from .voice import SAMPLE_RATE, Listener

log = logging.getLogger(__name__)

# About twenty pages. Intake only needs the story, and the whole text goes into one prompt.
MAX_CHARS = 40_000
CHUNK = SAMPLE_RATE * 2 // 5  # 200 ms of 16-bit mono audio
_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING = re.compile(r"[ \t]+$", re.M)


class UnreadableError(Exception):
    """The file holds no text we can use — an empty file, or a scan with no text layer."""


def _pdf(data: bytes) -> str:
    from pypdf import PdfReader

    return "\n\n".join(page.extract_text() or "" for page in PdfReader(io.BytesIO(data)).pages)


def _docx(data: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(data))
    lines = [p.text for p in document.paragraphs]
    for table in document.tables:  # termination letters and settlement offers often are tables
        lines += [" | ".join(c.text.strip() for c in row.cells) for row in table.rows]
    return "\n".join(lines)


def _tidy(text: str) -> str:
    return _BLANK_LINES.sub("\n\n", _TRAILING.sub("", text.replace("\f", "\n"))).strip()


def extract(filename: str, data: bytes) -> str:
    """The text of an uploaded document, trimmed to `MAX_CHARS`."""
    suffix = PurePosixPath(filename).suffix.lower()
    if suffix == ".doc":
        raise UnreadableError("Old .doc files are not supported — save it as .docx or PDF first.")
    try:
        if suffix == ".pdf":
            text = _pdf(data)
        elif suffix in (".docx", ".dotx"):
            text = _docx(data)
        else:
            text = data.decode("utf-8", errors="replace")
    except Exception as e:
        log.exception("could not read %s", filename)
        raise UnreadableError(f"{filename} could not be read as a document.") from e
    text = _tidy(text)
    if len(text) < 20:
        raise UnreadableError(
            f"No text found in {filename}. A scanned PDF has to be run through OCR first — "
            "this assistant does not read images.")
    return text[:MAX_CHARS]


async def transcribe(listener: Listener, pcm: bytes, language: str = "en") -> str:
    """A recording (16 kHz mono PCM) as text, through the same ASR NIM as voice mode."""
    audio: asyncio.Queue[bytes | None] = asyncio.Queue()
    for i in range(0, len(pcm), CHUNK):
        audio.put_nowait(pcm[i:i + CHUNK])
    audio.put_nowait(None)
    said = [text async for text, final in listener.transcribe(audio, language) if final and text]
    out = _tidy(" ".join(said))
    if not out:
        raise UnreadableError("Nothing was recognised in that recording.")
    return out[:MAX_CHARS]
