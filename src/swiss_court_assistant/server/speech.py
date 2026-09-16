from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncIterator, Callable
from typing import TypeVar

import grpc
import riva.client
from riva.client.proto import riva_tts_pb2

from .translate import GRPC_OPTIONS, pieces

TTS_URI = os.environ.get("SCA_TTS_URI", "localhost:50052")
SAMPLE_RATE = 22050
MAX_PIECE = 400  # characters per synthesis request; audio starts after the first piece
PREFERRED = {"en": "EN-US.Aria"}  # otherwise the first voice the NIM lists for the language

_CITATION = re.compile(r"\s*\[\d+(?:\s*[-–,;]\s*\d+)*\](?!\()")
_HEADING = re.compile(r"^#{1,6}\s+(.*?)[.:]?\s*$", re.M)
_MARKDOWN = re.compile(r"(\*\*|__|`+|^\s*>\s?|^\s*[-*+]\s+|^\s*\|[-:| ]+\|\s*$)", re.M)
_ANONYMIZED = re.compile(r"\b([A-Z]{1,3})\.?_{2,}")

T = TypeVar("T")


class UnspeakableError(ValueError):
    pass


def speakable(text: str) -> str:
    """Answer markdown or a decision passage as plain sentences: no [n] markers, markup or "C.____"."""
    text = _HEADING.sub(r"\1.", _CITATION.sub("", text))  # a heading is read as its own sentence
    text = _MARKDOWN.sub("", text).replace("|", ", ")
    text = _ANONYMIZED.sub(r"\1.", text)
    return " ".join(text.split())


class Speaker:
    """Reads text aloud with the Magpie TTS NIM (Riva gRPC), one voice per language."""

    def __init__(self):
        self._connect()
        self._voices: dict[str, tuple[str, str]] | None = None  # "de" -> ("de-DE", voice name)

    def _connect(self):
        self.client = riva.client.SpeechSynthesisService(riva.client.Auth(uri=TTS_URI, options=GRPC_OPTIONS))

    async def _call(self, fn: Callable[..., T], *args) -> T:
        try:
            return await asyncio.to_thread(fn, *args)
        except grpc.RpcError as e:
            if e.code() != grpc.StatusCode.UNAVAILABLE:
                raise
            self._connect()  # a fresh channel instead of one still backing off from a failed connect
            return await asyncio.to_thread(fn, *args)

    def voices(self) -> dict[str, tuple[str, str]]:
        """One voice per language (PREFERRED, else the first listed); override with SCA_TTS_VOICE_<LANG>."""
        if self._voices is None:
            cfg = self.client.stub.GetRivaSynthesisConfig(riva_tts_pb2.RivaSynthesisConfigRequest())
            found: dict[str, tuple[str, str]] = {}
            for m in cfg.model_config:
                p = m.parameters
                for sub in filter(None, p.get("subvoices", "").split(",")):
                    sub = sub.split(":")[0]
                    code = sub.split(".")[0] if "." in sub else p.get("language_code", "")
                    lang, _, region = code.partition("-")
                    lang = lang.lower()
                    name = f"{p['voice_name']}.{sub}"
                    if lang not in found or PREFERRED.get(lang) == sub:
                        found[lang] = (f"{lang}-{region.upper()}" if region else lang, name)
            for lang in list(found):
                if voice := os.environ.get(f"SCA_TTS_VOICE_{lang.upper()}"):
                    found[lang] = (found[lang][0], voice)
            self._voices = found
        return self._voices

    def _synth(self, text: str, code: str, voice: str) -> bytes:
        return self.client.synthesize(text, voice, code, sample_rate_hz=SAMPLE_RATE).audio

    async def stream(self, text: str, language: str) -> AsyncIterator[bytes]:
        """16-bit mono PCM, sentence by sentence. The first piece is synthesized before returning, so a
        missing voice or an unreachable NIM fails the request instead of the stream."""
        voice = (await self._call(self.voices)).get(language)
        if voice is None:
            raise UnspeakableError(f"Read aloud is not available for language '{language}'.")
        parts = pieces(speakable(text), MAX_PIECE)
        if not parts or not parts[0]:
            raise UnspeakableError("There is nothing to read.")
        first = await self._call(self._synth, parts[0], *voice)

        async def audio() -> AsyncIterator[bytes]:
            chunk = first
            for part in parts[1:]:
                task = asyncio.ensure_future(asyncio.to_thread(self._synth, part, *voice))  # synthesize ahead
                yield chunk
                chunk = await task
            yield chunk

        return audio()
