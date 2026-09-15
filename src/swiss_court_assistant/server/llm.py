from __future__ import annotations

import logging
import os

import httpx

log = logging.getLogger(__name__)

LLM_URL = os.environ.get("SCA_LLM_URL", "http://localhost:9100/v1")
LLM_MODEL = os.environ.get("SCA_LLM_MODEL")  # default: the first model the server lists
LLM_KEY = os.environ.get("SCA_LLM_KEY", "nim")
LLM_THINKING = os.environ.get("SCA_LLM_THINKING", "0") == "1"


def served_model() -> str:
    if LLM_MODEL:
        return LLM_MODEL
    try:
        return httpx.get(f"{LLM_URL}/models", timeout=5).json()["data"][0]["id"]
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        log.warning("could not list models at %s; assuming nvidia/nemotron-3.5-lightning", LLM_URL)
        return "nvidia/nemotron-3.5-lightning"
