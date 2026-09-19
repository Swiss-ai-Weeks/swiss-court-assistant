"""Documents the user attaches, read with the Nemotron Parse NIM and kept on disk.

Nemotron Parse reads page images, not files: a PDF is rendered page by page (pypdfium2, locally) and
each page goes to the NIM, which returns the page's elements in reading order - titles, text, tables as
Markdown, captions - with their type. Page headers and footers are dropped (they repeat on every page
and would land in the middle of sentences), section headers become Markdown headings. Scans and photos
of letters work the same way, which the old text-layer extraction could not do. Word and plain-text
files have their text already and are read directly.

Every document lives in its own folder under `SCA_DOCUMENTS` (data/app/documents/<id>/): the file as
uploaded, the extracted text (`text.md`, with a "[Page n]" line before each page) and `meta.json`.
The agent reads the text with read_document and cites it by its id, the same way it cites a decision.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import os
import re
import shutil
import threading
import wave
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import httpx

from .documents import UnreadableError, _docx, _tidy
from .language import detect_language
from .schemas import Decision, DecisionSummary, DocumentInfo
from .store import new_id, now

log = logging.getLogger(__name__)

PARSE_URL = os.environ.get("SCA_PARSE_URL", "http://localhost:8002/v1")
PARSE_MODEL = os.environ.get("SCA_PARSE_MODEL", "nvidia/nemotron-parse")
DOCUMENTS = Path(os.environ.get("SCA_DOCUMENTS", "data/app/documents"))
MAX_PAGES = int(os.environ.get("SCA_PARSE_MAX_PAGES", "80"))
PARALLEL = 4  # pages in flight at once; the NIM batches them
PAGE_PX = 2048  # longer side of a rendered page, the resolution the model was trained on
IMAGES = {".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".gif"}
TEXTS = {".txt", ".md", ".markdown", ".csv", ".rtf", ".html", ".htm", ".xml", ".json", ""}
ACCEPTED = sorted({".pdf", ".docx", ".dotx", *IMAGES, *TEXTS} - {""})
# what a page's elements are, as the model labels them; these carry no content of their own
_SKIP = {"page-header", "page-footer", "picture"}
_PAGE = re.compile(r"^\[Page (\d+)\]$", re.M)


# ── Nemotron Parse ──────────────────────────────────────────────────────
# model markup inside an element's text: <tbc> marks text that continues in the next element
_MARKUP = re.compile(r"</?tbc>|<br\s*/?>\n?", re.I)


def _element_text(kind: str, text: str) -> str:
    text = _MARKUP.sub(lambda m: "" if "tbc" in m.group().lower() else "\n", text).strip()
    if kind in ("title", "section-header"):
        level, _, title = text.partition(" ") if text.startswith("#") else ("#" if kind == "title" else "##", "", text)
        return f"{level} {title.strip().strip('*').strip()}"  # headings come back as "## **Mietvertrag**"
    if kind == "list-item" and not re.match(r"^([-*•]|\d+[.)])\s", text):
        return "- " + text
    return text


def _page_text(arguments: str) -> str:
    """One page's elements (the markdown_bbox tool's arguments) as Markdown, in reading order."""
    try:
        data = json.loads(arguments)
    except ValueError:
        return arguments  # the plain-text tools return text, not JSON
    elements = data[0] if data and isinstance(data[0], list) else data
    out = []
    for el in elements if isinstance(elements, list) else []:
        if not isinstance(el, dict):
            continue
        kind = str(el.get("type") or "text").lower().replace("_", "-")
        if kind in _SKIP or not str(el.get("text") or "").strip():
            continue
        out.append(_element_text(kind, str(el["text"])))
    return "\n\n".join(out)


class NemotronParse:
    """The Nemotron Parse NIM's OpenAI-style API: one page image in, the page's elements out."""

    def __init__(self, url: str = PARSE_URL, model: str = PARSE_MODEL):
        self.url, self.model = url.rstrip("/"), model
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(180, connect=5))

    async def page(self, png: bytes) -> str:
        image = "data:image/png;base64," + base64.b64encode(png).decode()
        tool = {"type": "function", "function": {"name": "markdown_bbox"}}
        r = await self.client.post(f"{self.url}/chat/completions", json={
            "model": self.model,
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image}}]}],
            "tools": [tool], "tool_choice": tool, "max_tokens": 8192})
        r.raise_for_status()
        message = r.json()["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        if calls:
            return _page_text(calls[0]["function"]["arguments"])
        return str(message.get("content") or "")

    async def pages(self, images: list[bytes]) -> list[str]:
        gate = asyncio.Semaphore(PARALLEL)

        async def one(png: bytes) -> str:
            async with gate:
                return await self.page(png)

        return list(await asyncio.gather(*(one(p) for p in images)))


# ── files to page images ────────────────────────────────────────────────
def _png(image) -> bytes:
    from PIL import Image

    image = image.convert("RGB")
    scale = PAGE_PX / max(image.size)
    if scale < 1:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    image.save(buf, "PNG")
    return buf.getvalue()


def pdf_pages(data: bytes) -> list[bytes]:
    import pypdfium2 as pdfium

    try:
        pdf = pdfium.PdfDocument(data)
    except pdfium.PdfiumError as e:
        raise UnreadableError("That PDF could not be opened - it may be damaged or password-protected.") from e
    if len(pdf) > MAX_PAGES:
        raise UnreadableError(f"That PDF has {len(pdf)} pages; up to {MAX_PAGES} can be read.")
    out = []
    for page in pdf:
        width, height = page.get_size()  # points
        out.append(_png(page.render(scale=PAGE_PX / max(width, height)).to_pil()))
    return out


def image_pages(data: bytes) -> list[bytes]:
    from PIL import Image, ImageSequence, UnidentifiedImageError

    try:
        image = Image.open(io.BytesIO(data))
    except UnidentifiedImageError as e:
        raise UnreadableError("That image could not be opened.") from e
    frames = [_png(f) for f in ImageSequence.Iterator(image)]  # a multi-page TIFF is several pages
    if len(frames) > MAX_PAGES:
        raise UnreadableError(f"That image has {len(frames)} pages; up to {MAX_PAGES} can be read.")
    return frames


def _with_pages(pages: list[str]) -> str:
    return "\n\n".join(f"[Page {i}]\n\n{_tidy(text)}" for i, text in enumerate(pages, 1))


# ── the store ───────────────────────────────────────────────────────────
@dataclass
class Parsed:
    text: str
    pages: int
    parser: str


class DocumentStore:
    """Uploaded documents on disk: parse once, then read windows of the text or quote from it."""

    def __init__(self, root: Path = DOCUMENTS, parser: NemotronParse | None = None):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.parser = parser or NemotronParse()
        self._texts: dict[str, str] = {}
        self._lock = threading.Lock()

    async def parse(self, filename: str, data: bytes) -> Parsed:
        suffix = PurePosixPath(filename).suffix.lower()
        if suffix == ".doc":
            raise UnreadableError("Old .doc files are not supported - save it as .docx or PDF first.")
        if suffix in (".docx", ".dotx"):
            try:
                return Parsed(_tidy(await asyncio.to_thread(_docx, data)), 1, "python-docx")
            except Exception as e:
                raise UnreadableError(f"{filename} could not be read as a Word document.") from e
        if suffix == ".pdf":
            images = await asyncio.to_thread(pdf_pages, data)
        elif suffix in IMAGES:
            images = await asyncio.to_thread(image_pages, data)
        elif suffix in TEXTS:
            return Parsed(_tidy(data.decode("utf-8", errors="replace")), 1, "text")
        else:
            raise UnreadableError(f"{suffix} files are not supported. Attach a PDF, a Word file, an image "
                                  f"or a text file.")
        try:
            pages = await self.parser.pages(images)
        except httpx.HTTPError as e:
            log.exception("nemotron-parse failed on %s", filename)
            raise ParserUnavailable("The document parser (Nemotron Parse) is not available right now.") from e
        return Parsed(_with_pages(pages), len(pages), "nemotron-parse")

    async def ingest(self, filename: str, data: bytes) -> DocumentInfo:
        """Parse an upload and keep it: the file, its text and what it is."""
        name = _name(filename)
        parsed = await self.parse(name, data)
        if len(_PAGE.sub("", parsed.text).strip()) < 20:
            raise UnreadableError(f"No text was found in {name}.")
        return self._keep(name, PurePosixPath(name).suffix.lower(), data, parsed.text, parsed.pages, parsed.parser)

    def keep_recording(self, filename: str, pcm: bytes, transcript: str, sample_rate: int) -> DocumentInfo:
        """A recording of the client: the audio as a WAV anyone can play back, the transcript as its text."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm)
        name = _name(filename.removesuffix(".pcm"))
        return self._keep(name, ".wav", buf.getvalue(), transcript, 1, "nemotron-asr", kind="recording",
                          seconds=round(len(pcm) / 2 / sample_rate, 1))

    def keep_notes(self, text: str, name: str = "Notes.txt") -> DocumentInfo:
        """Facts typed in by the lawyer, kept like any other part of the case file."""
        return self._keep(name, ".txt", text.encode(), _tidy(text), 1, "typed", kind="notes")

    def keep_generated(self, doc_id: str, name: str, text: str) -> DocumentInfo:
        """Text the app wrote itself (a matter's case prep), under an id of the caller's choosing so it is
        found again; rewritten only when the text has changed."""
        if (info := self.info(doc_id)) is not None and self.text(doc_id) == text and info.name == name:
            return info
        return self._keep(name, ".md", text.encode(), text, 1, "case-prep", kind="generated", doc_id=doc_id)

    def _keep(self, name: str, suffix: str, data: bytes, text: str, pages: int, parser: str,
              kind: str = "document", seconds: float | None = None, doc_id: str | None = None) -> DocumentInfo:
        info = DocumentInfo(id=doc_id or f"doc_{new_id()}", name=name, kind=kind, pages=pages, chars=len(text),  # type: ignore[arg-type]
                            parser=parser, language=detect_language(text[:5000], default="") or None,
                            seconds=seconds, created_at=now())
        folder = self.root / info.id
        folder.mkdir(parents=True, exist_ok=doc_id is not None)
        (folder / f"source{suffix}").write_bytes(data)
        (folder / "text.md").write_text(text, encoding="utf-8")
        (folder / "meta.json").write_text(info.model_dump_json(indent=1), encoding="utf-8")
        with self._lock:
            self._texts[info.id] = text
        log.info("%s %s: %s, %d pages, %d chars (%s)", kind, info.id, name, info.pages, info.chars, info.parser)
        return info

    def _folder(self, doc_id: str) -> Path | None:
        if not re.fullmatch(r"doc_[0-9a-f]{12}", doc_id):
            return None
        folder = self.root / doc_id
        return folder if (folder / "meta.json").is_file() else None

    def info(self, doc_id: str) -> DocumentInfo | None:
        folder = self._folder(doc_id)
        return DocumentInfo.model_validate_json((folder / "meta.json").read_text()) if folder else None

    def text(self, doc_id: str) -> str | None:
        with self._lock:
            if doc_id in self._texts:
                return self._texts[doc_id]
        folder = self._folder(doc_id)
        if folder is None:
            return None
        text = (folder / "text.md").read_text(encoding="utf-8")
        with self._lock:
            self._texts[doc_id] = text
        return text

    def file(self, doc_id: str) -> Path | None:
        folder = self._folder(doc_id)
        return next(folder.glob("source*"), None) if folder else None

    def delete(self, doc_id: str) -> bool:
        folder = self._folder(doc_id)
        if folder is None:
            return False
        shutil.rmtree(folder)
        with self._lock:
            self._texts.pop(doc_id, None)
        return True

    @staticmethod
    def page_at(text: str, offset: int) -> int | None:
        """The page a character of the text is on."""
        pages = [m for m in _PAGE.finditer(text) if m.start() <= offset]
        return int(pages[-1][1]) if pages else None

    def summary(self, doc_id: str) -> DecisionSummary | None:
        """The document described in a decision's fields, so a citation of it opens in the preview."""
        info = self.info(doc_id)
        if info is None:
            return None
        if info.kind == "recording":
            label, what = "Client recording", f"{_duration(info.seconds or 0)}"
        elif info.kind == "notes":
            label, what = "Notes", "typed in"
        elif info.kind == "generated":
            label, what = "Case prep", "generated from the case file and the research"
        else:
            label, what = "Attached document", f"{info.pages} page{'s' * (info.pages != 1)}"
        return DecisionSummary(
            decision_id=info.id, court="document", court_label=label, canton=None, chamber=None,
            docket=info.name, date=info.created_at[:10], language=info.language or "en",
            title=what, regeste=None,
            legal_area=None, source_url=None, pdf_url=f"api/documents/{info.id}/file")

    def as_decision(self, doc_id: str) -> Decision | None:
        summary, text = self.summary(doc_id), self.text(doc_id)
        return Decision(**summary.model_dump(), full_text=text) if summary and text is not None else None


def _duration(seconds: float) -> str:
    return f"{int(seconds // 60)}:{int(seconds % 60):02d} min"


def _name(filename: str) -> str:
    return PurePosixPath(filename.replace("\\", "/")).name or "document"


class ParserUnavailable(Exception):
    """The Nemotron Parse NIM did not answer."""
