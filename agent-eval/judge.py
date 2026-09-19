"""LLM as a judge: grade one answer against the reference answer and the rubric."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from openai import AsyncOpenAI

SYSTEM = """You are a Swiss law professor grading a legal research assistant that answers questions from a corpus of Swiss court decisions. You grade strictly and you answer the one question you are asked, as a single JSON object, nothing else.

The corpus is a subset of Swiss case law, so the assistant cannot always support every point of the reference answer. Never reward an answer for being long."""

LANGUAGES = {"de": "German", "fr": "French", "it": "Italian", "en": "English"}
PASSAGE_CHARS = 1200  # per citation: long enough to judge support, short enough to keep the prompt sharp
MAX_PASSAGES = 10

RUBRIC_TASK = """TASK - one rubric point:
{point}

Does the answer under review state this point? Judge the substance, not the wording: a point made in other words, in another language, or without the article number still counts. Reply {{"verdict": "covered"}} if the answer states it, {{"verdict": "partial"}} if it gestures at it without the substance, {{"verdict": "missing"}} if it is not there."""

ERROR_TASK = """TASK - find one legal error.

Copy into "wrong" the one statement in the ANSWER UNDER REVIEW that is wrong about Swiss law - word for word from the answer, one sentence. If every statement in the answer is correct, reply {"wrong": ""}.

What the answer does not say is not an error. Omissions, missing rubric points, a one-sided treatment and a short answer are measured separately: an answer of two correct sentences has no error. Copy only from the answer; do not paraphrase it and do not write a statement of your own."""

# Asked to *copy* the contradicting sentence, this judge finds one every time - the objection is
# already on the table and it justifies it. Only the first objection is asked for by quotation; this
# second step stays a yes/no, which it answers conservatively.
CONTRADICTED_TASK = """TASK - is this statement from the answer contradicted by the reference answer?

STATEMENT:
{statement}

Reply {{"contradicted": true}} only if the reference answer above says something that cannot both be true with this statement - it states the opposite rule, the opposite legal consequence, or a different condition where this one gives a condition. Reply {{"contradicted": false}} if the reference is merely silent on the point, says less, says more, or puts the same thing differently."""

SEVERITY_TASK = """TASK - how bad is this error?

STATEMENT (wrong, from the answer):
{statement}

Reply {{"severity": "central"}} if this statement is one of the answer's main propositions - a reader relying on the answer for the question asked would be misled about the rule or its consequence. Reply {{"severity": "secondary"}} if it is an aside: the answer to the question stands without it."""

SCORES_TASK = """TASK - two scores, each from 1 to 5.

"grounding": do the cited passages support the statements they follow? Judge only against the passages printed above, never against your own knowledge of the law. 5 = every citation supports its statement. 3 = a citation is only loosely related, or is attached to the wrong statement. 1 = the citations do not support the statements. An answer that correctly reports that the corpus has nothing on the question, and cites nothing, gets 5.

"usefulness": would this help a Swiss lawyer working on the question - is it direct, ordered, and free of padding? 5 = yes. 1 = unusable.

Reply {"grounding": n, "usefulness": n}."""

ABSTAIN_TASK = """TASK - did the answer refuse?

Reply {"abstained": true} if the answer's substance is that this corpus does not answer the question - for example that the search found nothing on point, that the passages found are about something else, or that the decision asked about is not in the corpus. It counts as a refusal however politely it is put, and in whatever language.

Reply {"abstained": false} if the answer actually answers the question, even partially. An answer that answers and merely adds a caveat about incomplete coverage has not refused."""

COMMENT_TASK = """TASK - write two or three sentences, in English, on what this answer gets right and what it gets wrong, for a reader who will not see the answer itself. No preamble, no score, no bullet points."""


def _schema(name: str, properties: dict[str, Any]) -> dict[str, Any]:
    return {"type": "json_schema", "json_schema": {"name": name, "schema": {
        "type": "object", "properties": properties, "required": list(properties)}}}


SCORE = {"type": "integer", "enum": [1, 2, 3, 4, 5]}
FORMATS = {
    "rubric": _schema("verdict", {"verdict": {"type": "string", "enum": ["covered", "partial", "missing"]}}),
    "error": _schema("error", {"wrong": {"type": "string"}}),
    "contradicted": _schema("contradicted", {"contradicted": {"type": "boolean"}}),
    "severity": _schema("severity", {"severity": {"type": "string",
                                                  "enum": ["central", "secondary"]}}),
    "scores": _schema("scores", {"grounding": SCORE, "usefulness": SCORE}),
    "abstain": _schema("abstain", {"abstained": {"type": "boolean"}}),
}


def _expectation(case: dict[str, Any]) -> str:
    expect = case.get("expect") or {}
    lines = ["The corpus cannot answer this question. The correct behaviour is to say so plainly and "
             "cite nothing; a substantive answer is wrong, however plausible."] if expect.get("abstain") \
        else ["The corpus can answer this question; refusing to answer it is wrong."]
    if decision := expect.get("decision"):
        lines.append(f"The answer must be based on the decision {decision}.")
    if documents := expect.get("documents"):
        lines.append("The user's own files are attached (they appear among the passages as documents). "
                     "The facts of the case must be taken from them and cited to them - in particular "
                     f"{', '.join(documents)} - and the law cited to decisions and statutes.")
    if expect.get("cites") == "optional":
        lines.append("This question is about the corpus itself, so citations are not required.")
    lines.append("The answer must be written in "
                 f"{LANGUAGES.get(expect.get('language') or case['language'])}.")
    return "\n".join(f"- {line}" for line in lines)


def _passages(sources: list[dict[str, Any]]) -> str:
    if not sources:
        return "(the answer cites no passage)"
    blocks = []
    for source in sources[:MAX_PASSAGES]:
        decision = source.get("decision") or {}
        label = (f"{decision.get('courtLabel', '?')} {decision.get('docket', '?')} · "
                 f"{decision.get('date') or 'undated'}")
        flag = "" if source.get("verified", True) else \
            " · WARNING: this quote was not found verbatim in the decision"
        blocks.append(f"[{source['n']}] {label} · decision_id={source.get('decisionId')}{flag}\n"
                      f"{(source.get('text') or '')[:PASSAGE_CHARS]}")
    return "\n\n".join(blocks)


def context(case: dict[str, Any], answer: str, sources: list[dict[str, Any]]) -> str:
    """The shared prefix of every call about this case."""
    setup = ""
    if turns := case.get("setup"):
        earlier = "\n".join(f"- {t.strip()}" for t in turns)
        setup = (f"EARLIER IN THE SAME CONVERSATION, the user asked:\n{earlier}\n"
                 "The question below is a follow-up; the answer has to resolve it from that context.\n\n")
    return (f"{setup}QUESTION ({LANGUAGES.get(case['language'], case['language'])}, {case['area']}):\n"
            f"{case['question'].strip()}\n\n"
            f"REFERENCE ANSWER (by a Swiss lawyer, for comparison - the answer under review does not "
            f"have to match its wording):\n{case['reference'].strip()}\n\n"
            f"EXPECTED BEHAVIOUR:\n{_expectation(case)}\n\n"
            f"ANSWER UNDER REVIEW (its [n] markers point to the passages below):\n"
            f"{answer.strip() or '(the assistant produced no answer)'}\n\n"
            f"PASSAGES CITED BY THAT ANSWER:\n{_passages(sources)}")


def _parse(text: str) -> dict[str, Any]:
    """Constrained decoding on this NIM pads its JSON with whitespace and sometimes a code fence."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"no JSON object in the judge's reply: {text[:200]!r}")
    return json.loads(re.sub(r"\s+", " ", text[start:end + 1]))


def _contains(answer: str, statement: str) -> bool:
    """Is this really a sentence of the answer? Whitespace and the citation markers are ignored, and
    a long quote counts if most of its opening is there, because the judge trims and re-wraps."""
    fold = lambda s: re.sub(r"\s+|\[\d+\]", "", s).lower()
    hay, needle = fold(answer), fold(statement)
    return bool(needle) and (needle in hay or (len(needle) >= 60 and needle[:60] in hay))


class Judge:
    def __init__(self, client: AsyncOpenAI, model: str, attempts: int = 3):
        self.client, self.model, self.attempts = client, model, attempts

    async def _ask(self, prefix: str, task: str, fmt: dict | None, max_tokens: int) -> Any:
        last: Exception | None = None
        for _ in range(self.attempts):
            try:
                reply = await self.client.chat.completions.create(
                    model=self.model, temperature=0.0, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": f"{prefix}\n\n{task}"}],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False},
                                **({"response_format": fmt} if fmt else {})})
                text = reply.choices[0].message.content or ""
                return _parse(text) if fmt else text.strip()
            except Exception as e:
                last = e
        raise RuntimeError(f"{type(last).__name__}: {last}")

    async def _accuracy(self, prefix: str, answer: str) -> dict[str, Any]:
        """Legal accuracy, from an objection the judge has to be able to quote.

        Asked for a score on a scale, this judge marks an answer down for what it leaves out however
        often it is told not to, and twice invented a rule to mark a correct answer down with. So it
        is asked instead to copy out the one sentence it says is wrong, and the objection is kept only
        if that sentence is really in the answer and the reference answer contradicts it. The quote
        requirement is what rules out the commonest bad objection - that the answer left something
        out - because an omission has no sentence to copy.
        """
        objection = (await self._ask(prefix, ERROR_TASK, FORMATS["error"], 120)).get("wrong") or ""
        note, statement = "", objection.strip()
        if not statement:
            return {"legal_accuracy": 5, "legal_error": None, "accuracy_note": "nothing wrong found"}
        if not _contains(answer, statement):
            return {"legal_accuracy": 5, "legal_error": statement,
                    "accuracy_note": "objection dropped: not a sentence of the answer"}
        contradicted = await self._ask(prefix, CONTRADICTED_TASK.format(statement=statement),
                                       FORMATS["contradicted"], 16)
        if not contradicted.get("contradicted"):
            return {"legal_accuracy": 5, "legal_error": statement,
                    "accuracy_note": "objection dropped: the reference answer does not contradict it"}
        severity = (await self._ask(prefix, SEVERITY_TASK.format(statement=statement),
                                    FORMATS["severity"], 16)).get("severity")
        return {"legal_accuracy": 2 if severity == "central" else 3, "legal_error": statement,
                "accuracy_note": f"{severity} error, contradicted by the reference answer"}

    async def judge(self, case: dict[str, Any], answer: str,
                    sources: list[dict[str, Any]]) -> dict[str, Any]:
        """Grade one answer. Returns the judgment, or one carrying `error` if the judge failed."""
        prefix = context(case, answer, sources)
        try:
            verdicts, accuracy, scores, abstain, comment = await asyncio.gather(
                asyncio.gather(*(self._ask(prefix, RUBRIC_TASK.format(point=point),
                                           FORMATS["rubric"], 24) for point in case["rubric"])),
                self._accuracy(prefix, answer),
                self._ask(prefix, SCORES_TASK, FORMATS["scores"], 32),
                self._ask(prefix, ABSTAIN_TASK, FORMATS["abstain"], 16),
                self._ask(prefix, COMMENT_TASK, None, 200),
            )
        except Exception as e:
            return {"error": str(e), "rubric": [], "legal_accuracy": None, "grounding": None,
                    "usefulness": None, "abstained": None, "comment": ""}
        return {
            "rubric": [{"point": i, "verdict": v.get("verdict", "missing"), "text": point}
                       for i, (point, v) in enumerate(zip(case["rubric"], verdicts), 1)],
            **accuracy,
            **{k: scores.get(k) for k in ("grounding", "usefulness")},
            "abstained": abstain.get("abstained"),
            "comment": comment,
        }
