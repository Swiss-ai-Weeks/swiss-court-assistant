from __future__ import annotations

import logging
import os
from datetime import date

import httpx

log = logging.getLogger(__name__)

LLM_URL = os.environ.get("SCA_LLM_URL", "http://localhost:9100/v1")
LLM_MODEL = os.environ.get("SCA_LLM_MODEL")  # default: the first model the server lists
LLM_KEY = os.environ.get("SCA_LLM_KEY", "nim")
LLM_THINKING = os.environ.get("SCA_LLM_THINKING", "1") == "1"  # reasoning before each research step


def served_model() -> str:
    if LLM_MODEL:
        return LLM_MODEL
    try:
        return httpx.get(f"{LLM_URL}/models", timeout=5).json()["data"][0]["id"]
    except (httpx.HTTPError, KeyError, IndexError, ValueError):
        log.warning("could not list models at %s; assuming nvidia/nemotron-3.5-lightning", LLM_URL)
        return "nvidia/nemotron-3.5-lightning"


def today_note() -> str:
    """Today's date, for the end of a system prompt. Without it the model guessed whether a deadline had
    passed: a matter memo concluded the client had lost her right to contest a notice whose 30 days were
    still running. Called per request, not once at startup, so a server left running keeps the right date."""
    today = date.today()
    return (f"Today's date is {today.day} {today:%B %Y} ({today.isoformat()}). Where a deadline or time limit "
            "matters, work out its last day from the dates in the sources and say whether it has passed or is "
            "still running as of today. Do not assume that a step was or was not taken unless the question or "
            "the documents say so.")
