from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import queue
import threading
from collections.abc import AsyncIterator
from typing import Any

import riva.client
from riva.client.proto import riva_asr_pb2 as rasr

from .translate import GRPC_OPTIONS

ASR_URI = os.environ.get("SCA_ASR_URI", "localhost:50053")
ASR_LANGUAGE = os.environ.get("SCA_ASR_LANGUAGE")  # unset: "auto" when the NIM detects the language itself
SAMPLE_RATE = 16000  # what the browser sends
CODES = {"de": "de-DE", "fr": "fr-FR", "it": "it-IT", "en": "en-US"}

log = logging.getLogger(__name__)


class Listener:
    """Streams microphone audio to the Nemotron ASR NIM and yields (transcript, is_final)."""

    def __init__(self):
        self.service = riva.client.ASRService(riva.client.Auth(uri=ASR_URI, options=GRPC_OPTIONS))
        self._supported: set[str] | None = None

    def languages(self) -> set[str]:
        """The codes this NIM serves. The multilingual model lists all of them, plus "auto", in one
        comma-joined parameter; the English-only model lists a single code."""
        if self._supported is None:
            cfg = self.service.stub.GetRivaSpeechRecognitionConfig(rasr.RivaSpeechRecognitionConfigRequest())
            self._supported = {code.strip() for m in cfg.model_config
                               for code in m.parameters.get("language_code", "").split(",") if code.strip()}
            log.info("ASR languages: %s", ", ".join(sorted(self._supported)))
        return self._supported

    def _code(self, language: str) -> str:
        supported = self.languages()
        if ASR_LANGUAGE and ASR_LANGUAGE in supported:  # forced with SCA_ASR_LANGUAGE
            return ASR_LANGUAGE
        if not ASR_LANGUAGE and "auto" in supported:
            return "auto"  # the multilingual model recognises whichever language is spoken
        code = CODES.get(language, language)
        if code in supported:
            return code
        fallback = "en-US" if "en-US" in supported else sorted(supported)[0]
        log.info("ASR has no %s; listening in %s", code, fallback)
        return fallback

    def _config(self, language: str) -> rasr.StreamingRecognitionConfig:
        return rasr.StreamingRecognitionConfig(
            config=rasr.RecognitionConfig(
                encoding=riva.client.AudioEncoding.LINEAR_PCM, sample_rate_hertz=SAMPLE_RATE,
                language_code=self._code(language), max_alternatives=1, audio_channel_count=1,
                enable_automatic_punctuation=True),
            interim_results=True)

    async def transcribe(self, audio: asyncio.Queue[bytes | None],
                         language: str = "en") -> AsyncIterator[tuple[str, bool]]:
        """Transcribe audio chunks from the queue (None ends it). The gRPC stream blocks, so it runs on
        a daemon thread - never the event loop's executor, where a stuck read would hold up shutdown -
        and hands results back through the loop."""
        loop = asyncio.get_running_loop()
        inbox: queue.Queue[bytes | None] = queue.Queue()
        results: asyncio.Queue[tuple[str, bool] | BaseException | None] = asyncio.Queue()
        stop = threading.Event()

        def emit(item: tuple[str, bool] | BaseException | None) -> None:
            with contextlib.suppress(RuntimeError):  # the loop is gone: the session is over anyway
                loop.call_soon_threadsafe(results.put_nowait, item)

        def chunks():
            while not stop.is_set() and (chunk := inbox.get()) is not None:
                yield chunk

        def run():
            try:
                for response in self.service.streaming_response_generator(chunks(), self._config(language)):
                    if stop.is_set():
                        break
                    for result in response.results:
                        if result.alternatives:
                            emit((result.alternatives[0].transcript.strip(), result.is_final))
            except BaseException as e:  # noqa: BLE001 - reported to the caller
                emit(e)
            finally:
                emit(None)

        threading.Thread(target=run, name="asr-stream", daemon=True).start()

        async def pump() -> None:
            while (chunk := await audio.get()) is not None:
                inbox.put(chunk)
            inbox.put(None)

        pump = asyncio.create_task(pump())
        try:
            while (item := await results.get()) is not None:
                if isinstance(item, BaseException):
                    raise item
                yield item
        finally:
            stop.set()
            inbox.put(None)
            pump.cancel()


# ── what the assistant says while it works ──────────────────────────────
NARRATION = {
    "en": {"search": "Searching for {q}.", "keyword": "Looking up {q}.", "read": "Reading the decision.",
           "write": "Let me sum that up.", "generic": "Searching Swiss case law.",
           "statute": "Looking at the statute."},
    "de": {"search": "Ich suche nach {q}.", "keyword": "Ich schlage {q} nach.", "read": "Ich lese den Entscheid.",
           "write": "Ich fasse es zusammen.", "generic": "Ich durchsuche die Rechtsprechung.",
           "statute": "Ich schaue im Gesetz nach."},
    "fr": {"search": "Je cherche {q}.", "keyword": "Je recherche {q}.", "read": "Je lis la décision.",
           "write": "Je résume.", "generic": "Je consulte la jurisprudence.",
           "statute": "Je consulte la loi."},
    "it": {"search": "Cerco {q}.", "keyword": "Cerco il termine {q}.", "read": "Leggo la decisione.",
           "write": "Riassumo.", "generic": "Consulto la giurisprudenza.",
           "statute": "Consulto la legge."},
}
_TOOLS = {"semantic_search": "search", "keyword_search": "keyword", "read_decision": "read", "write_answer": "write",
          "read_law": "statute", "search_laws": "statute", "list_decisions": "generic"}


def narrate(tool: str, args: dict[str, Any], language: str) -> str | None:
    """One short spoken line for a tool call, in the user's language. The search queries are written in
    German, French and Italian, so an English turn hears the generic line rather than a German query."""
    phrases = NARRATION.get(language) or NARRATION["en"]
    kind = _TOOLS.get(tool)
    if kind is None:
        return None
    if kind in ("write", "read", "statute", "generic"):  # decision ids ("zh_arbeitsgericht_AH250019") are unspeakable
        return phrases[kind]
    query = args.get(f"query_{language}") if kind == "search" else args.get("keyword")
    query = " ".join(str(query or "").split())
    # a keyword query is full of quotes and statute abbreviations; only read short, plain ones aloud
    if kind == "keyword" and (len(query) > 40 or '"' in query):
        query = ""
    return phrases[kind].format(q=query[:120]) if query else phrases["generic"]
