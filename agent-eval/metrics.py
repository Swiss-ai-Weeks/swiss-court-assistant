"""What can be measured without asking a model, and how a case's scores are combined.

The judge grades the prose; these checks read the turn itself - the language the answer came back
in, whether it cited anything, whether the quotes were found verbatim in the decision (`verified`),
whether the assistant's own grounding check backed each citation (`supported`), which tools ran, and
what the turn cost in time and calls. They are cheap, deterministic, and they are what makes a
judged score falsifiable.
"""

from __future__ import annotations

import re
from typing import Any

from swiss_court_assistant.server.language import detect_language

from client import Turn

WEIGHTS = {"rubric": 0.40, "accuracy": 0.20, "grounding": 0.20, "usefulness": 0.10, "behaviour": 0.10}
RUBRIC_PASS = 0.6      # a correct answer makes at least this share of the rubric points
ACCURACY_PASS = 4      # ... and carries no legal objection the judge could make stick (5 = none)
_MARKER = re.compile(r"\[\d+\]")


def mechanical(case: dict[str, Any], turn: Turn) -> dict[str, Any]:
    expect = case.get("expect") or {}
    sources, tools = turn.sources, turn.tools
    checked = [s for s in sources if s.get("supported") is not None]
    prose = _MARKER.sub("", turn.answer)
    language = detect_language(prose, default="") if prose.strip() else ""
    wanted = expect.get("cites", True)
    out = {
        "answer_language": language,
        "language_ok": language == (expect.get("language") or case["language"]),
        "answer_chars": len(prose),
        "thinking_chars": turn.thinking_chars,
        "n_citations": len(sources),
        "n_unverified": sum(not s.get("verified", True) for s in sources),
        "verified_rate": _rate(sum(s.get("verified", True) for s in sources), len(sources)),
        "n_checked": len(checked),
        "support_rate": _rate(sum(s.get("supported") is True for s in checked), len(checked)),
        "cites_ok": True if wanted == "optional" else (bool(sources) if wanted else not sources),
        "n_tool_calls": len(turn.tool_calls),
        "tools": tools,
        "tool_errors": sum(bool(c.get("error")) for c in turn.tool_calls),
        "repeated_searches": sum("repeats an earlier search" in (c.get("summary") or "")
                                 for c in turn.tool_calls),
        "seconds": round(turn.seconds, 1),
        "decision_ok": None if "decision" not in expect else
        any(s.get("decisionId") == expect["decision"] for s in sources),
        "tool_ok": None if "tool" not in expect else expect["tool"] in tools,
        # attached files the answer must cite (by file name): the facts that are only in them
        "documents_ok": None if "documents" not in expect else all(
            any(s.get("section") == "document" and (s.get("decision") or {}).get("docket") == name
                for s in sources) for name in expect["documents"]),
        "indexed": turn.indexed,
        "indexed_ok": None if "indexed" not in expect else bool(turn.indexed) == bool(expect["indexed"]),
    }
    return out


def _rate(hits: int, total: int) -> float | None:
    return round(hits / total, 3) if total else None


def score(case: dict[str, Any], mech: dict[str, Any], judgment: dict[str, Any]) -> dict[str, Any]:
    """Per-case scores. `correct` is the strict bar the headline pass rate counts; `score` is the
    partial-credit blend, so an answer that is right but thin and one that is wrong do not land in
    the same place."""
    expect = case.get("expect") or {}
    verdicts = [item.get("verdict") for item in judgment.get("rubric", [])]
    n = len(case["rubric"])
    coverage = _rate(sum({"covered": 2, "partial": 1}.get(v, 0) for v in verdicts), 2 * n) or 0.0
    # only the judge can tell a refusal from an answer; with --no-judge the check is not made at all
    abstained = judgment.get("abstained")
    abstain_ok = abstained is None or abstained == bool(expect.get("abstain", False))
    behaviour = [mech["language_ok"], mech["cites_ok"], abstain_ok,
                 *(x for x in (mech["decision_ok"], mech["tool_ok"], mech.get("documents_ok"),
                               mech.get("indexed_ok")) if x is not None)]
    behaviour_ok = all(behaviour)
    accuracy, grounding, usefulness = (judgment.get(k) for k in ("legal_accuracy", "grounding", "usefulness"))
    blend = None
    if None not in (accuracy, grounding, usefulness):
        blend = round(100 * (WEIGHTS["rubric"] * coverage
                             + WEIGHTS["accuracy"] * (accuracy - 1) / 4
                             + WEIGHTS["grounding"] * (grounding - 1) / 4
                             + WEIGHTS["usefulness"] * (usefulness - 1) / 4
                             + WEIGHTS["behaviour"] * behaviour_ok), 1)
    return {
        "rubric_coverage": coverage,
        "legal_accuracy": accuracy,
        "legal_error": judgment.get("legal_error") if (accuracy or 5) < 5 else None,
        "grounding": grounding,
        "usefulness": usefulness,
        "abstain_ok": abstain_ok,
        "behaviour_ok": behaviour_ok,
        "score": blend,
        "correct": bool(blend is not None and behaviour_ok
                        and coverage >= RUBRIC_PASS and (accuracy or 0) >= ACCURACY_PASS),
    }


def failure(case: dict[str, Any], mech: dict[str, Any], scored: dict[str, Any]) -> str:
    """Why a case did not count as correct, in a few words - the column that makes the table useful."""
    expect = case.get("expect") or {}
    reasons = []
    if not mech["language_ok"]:
        reasons.append(f"answered in {mech['answer_language'] or '?'}, "
                       f"not {expect.get('language') or case['language']}")
    if not scored["abstain_ok"]:
        reasons.append("answered what it should have refused" if expect.get("abstain")
                       else "refused a question the corpus answers")
    if not mech["cites_ok"]:
        reasons.append("no citations" if expect.get("cites", True) else "cited passages while abstaining")
    if mech["decision_ok"] is False:
        reasons.append(f"did not cite {expect['decision']}")
    if mech["tool_ok"] is False:
        reasons.append(f"never called {expect['tool']}")
    if mech.get("documents_ok") is False:
        reasons.append(f"did not cite all of {', '.join(expect['documents'])}")
    if mech.get("indexed_ok") is False:
        reasons.append("the case file was not indexed" if expect["indexed"] else "the case file was indexed")
    if (scored["legal_accuracy"] or 5) < ACCURACY_PASS:
        reasons.append("a central legal error" if scored["legal_accuracy"] == 2 else "a legal error")
    if scored["rubric_coverage"] < RUBRIC_PASS:
        reasons.append(f"rubric {scored['rubric_coverage']:.0%}")
    return "; ".join(reasons)
