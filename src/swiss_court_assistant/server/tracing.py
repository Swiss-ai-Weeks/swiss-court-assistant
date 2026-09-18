"""MLflow tracing for the agent loop.

Every assistant turn becomes one trace: the exact prompt sent to the model at each research step,
the tool calls and their results, the constrained final-answer call and the grounding checks.
LangChain autologging records the model calls; `turn()` is the root span that groups them and holds
what the user actually saw (the answer, its citations and their verdicts).

Off unless `SCA_MLFLOW_URI` is set, and never fatal: a tracing failure must not cost an answer.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Any

from .agent import Cite, Delta, Thought, ToolEnd, ToolStart, Verdict
from .schemas import Source

log = logging.getLogger(__name__)

MLFLOW_URI = os.environ.get("SCA_MLFLOW_URI", "")  # e.g. http://localhost:5000
EXPERIMENT = os.environ.get("SCA_MLFLOW_EXPERIMENT", "swiss-court-assistant")

_on = False


def setup() -> bool:
    """Point the client at the tracking server and trace every LangChain model and tool call."""
    global _on
    if not MLFLOW_URI:
        log.info("MLflow tracing is off; set SCA_MLFLOW_URI to record the agent loop")
        return False
    try:
        import mlflow
        import mlflow.langchain

        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT)
        mlflow.langchain.autolog()
        _on = True
        log.info("MLflow tracing on: %s (experiment %r)", MLFLOW_URI, EXPERIMENT)
    except Exception:
        log.exception("could not start MLflow tracing; continuing without it")
    return _on


def enabled() -> bool:
    return _on


class Recorder:
    """Collects one turn's events for the root span. Cheap enough to run when tracing is off."""

    def __init__(self) -> None:
        self.tools: list[dict[str, Any]] = []
        self.by_id: dict[str, dict[str, Any]] = {}
        self.citations: list[dict[str, Any]] = []
        self.thinking: list[str] = []
        self.answer = ""
        self.sources: list[Source] = []

    def event(self, ev: Any) -> None:
        match ev:
            case ToolStart():
                call = {"name": ev.name, "args": ev.args, "thought": ev.thought}
                self.by_id[ev.id] = call
                self.tools.append(call)
            case ToolEnd():
                self.by_id.get(ev.id, {}).update(result=ev.summary, error=ev.error)
            case Thought():
                self.thinking.append(ev.text)
            case Cite():
                self.citations.append({"n": ev.source.n, "decision_id": ev.source.decision_id,
                                       "docket": ev.source.decision.docket, "verified": ev.source.verified})
            case Verdict():
                for c in self.citations:
                    if c["n"] == ev.n:
                        c["supported"] = ev.supported
            case Delta():
                pass  # the answer is recorded once, in finish()

    def finish(self, answer: str, sources: list[Source]) -> None:
        self.answer, self.sources = answer, list(sources)

    def outputs(self) -> dict[str, Any]:
        return {"answer": self.answer,
                "tool_calls": self.tools,
                "citations": self.citations,
                "unsupported_citations": [s.n for s in self.sources if s.supported is False]}


@contextmanager
def turn(question: str, conversation_id: str, language: str):
    """Root span for one assistant turn; yields a Recorder to feed that turn's events into."""
    rec = Recorder()
    if not _on:
        yield rec
        return
    import mlflow

    try:
        span_cm = mlflow.start_span(name="turn")
    except Exception:  # a broken tracing server must never cost the user an answer
        log.warning("could not open an MLflow span for this turn", exc_info=True)
        yield rec
        return
    with span_cm as span:
        span.set_inputs({"question": question, "language": language})
        try:
            mlflow.update_current_trace(tags={"conversation_id": conversation_id, "language": language})
        except Exception:
            log.debug("could not tag the trace", exc_info=True)
        try:
            yield rec
        finally:
            try:
                span.set_outputs(rec.outputs())
            except Exception:
                log.warning("could not record the turn's outputs", exc_info=True)
