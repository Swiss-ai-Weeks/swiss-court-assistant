"""Score a LEXam multiple-choice run on its reasoning instead of the letter it picked.

    uv run python agent-eval/reasoning.py agent-eval/runs/latest

A LEXam multiple-choice question is a list of numbered statements ("i. ... ii. ... iii. ...") and
options that are subsets of them ("B) i and iii"). Scoring the letter asks two things at once: does
the assistant judge each statement correctly, and does it then combine its judgements into the
option that matches. This agent answers the first and mostly refuses the second - on 100 questions
it named no option on 59 and analysed the statements instead - so the letter measures the refusal,
not the law.

So the statements are scored one by one. The key gives the truth: a statement is true exactly when
the correct option lists it (for a `negative_question`, which asks which statements are *wrong*, the
other way round). The judge is asked only what the answer under review concludes about that
statement - correct, incorrect, or not addressed - which is reading, not a judgement about Swiss
law, and is the one thing this judge does reliably.

Two numbers come out of it:
  * **coverage** - the share of statements the answer takes a position on at all;
  * **accuracy** - of those, the share where its position matches the key.
Accuracy is over addressed statements only, so an answer that covers little is not rewarded for it;
coverage is reported beside it. `--strict` scores unaddressed statements as wrong instead.

Statute articles are also compared: the ones the question's statements turn on against the ones the
answer cites, which says whether it is reasoning from the right provisions.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml
from openai import AsyncOpenAI

from judge import FORMATS, Judge, _schema

HERE = Path(__file__).resolve().parent
JUDGE_URL = os.environ.get("SCA_JUDGE_URL", "http://localhost:9100/v1")
JUDGE_KEY = os.environ.get("SCA_JUDGE_KEY", "nim")

_ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6}
# "i. ", "ii) " at the start of a line, or after the sentence before it on the same line
_ITEM = re.compile(r"(?:^|\n)\s*(i{1,3}|iv|vi?)[.)]\s+", re.M)
# "Art. 271a OR", "art. 429 al. 1 CPP", "Art. 16 Abs. 2 StGB" - the article number and its code
_ARTICLE = re.compile(r"\b[Aa]rt(?:icolo|icle)?\.?\s*(\d+[a-z]?)\s*"
                      r"(?:(?:Abs|al|cpv|para)\.?\s*\d+\s*)?"
                      r"(?:(?:lit|let|bst)\.?\s*[a-z]\s*)?"
                      r"([A-ZÄÖÜ][A-Za-zÄÖÜäöü]{1,6})\b")

VERDICT = _schema("statement", {"verdict": {"type": "string",
                                            "enum": ["correct", "incorrect", "not addressed"]}})
STATEMENT_TASK = """TASK - what does the answer under review conclude about this one statement?

STATEMENT:
{statement}

Read the ANSWER UNDER REVIEW and report its position on this statement. This is a question about what the answer says, not about Swiss law: do not decide the statement yourself.

Reply {{"verdict": "correct"}} if the answer treats this statement as right - it says so, or it states the same rule as the statement does.
Reply {{"verdict": "incorrect"}} if the answer treats it as wrong - it says so, or it states a rule that contradicts it.
Reply {{"verdict": "not addressed"}} if the answer does not take a position on this statement: it is silent on the point, says only that it found nothing, or discusses the area without saying whether this statement holds."""


def statements(question: str) -> dict[int, str]:
    """The numbered statements of a multiple-choice question, by their number."""
    body = question.split("\n\nA)")[0]  # the options follow; they repeat the numerals
    marks = list(_ITEM.finditer(body))
    found: dict[int, str] = {}
    for mark, nxt in zip(marks, [*marks[1:], None]):
        if (n := _ROMAN.get(mark.group(1))) is None:
            continue
        text = body[mark.end():nxt.start() if nxt else len(body)].strip()
        if text and n not in found:  # the first run is the statement list
            found[n] = " ".join(text.split())
    return found


def option_set(option: str) -> set[int]:
    """The statements an option names. "none of the statements" names none."""
    return {_ROMAN[m] for m in re.findall(r"\b(i{1,3}|iv|vi?)\b", option.lower()) if m in _ROMAN}


def key(case: dict[str, Any]) -> dict[int, bool]:
    """Which statements are true, from the correct option. A `negative_question` asks which ones are
    wrong, so there the option lists the false ones."""
    letters = "ABCDEFGH"
    gold = case["choices"][letters.index(case["gold"])]
    listed, negative = option_set(gold), bool((case.get("source") or {}).get("negative_question"))
    return {n: (n in listed) != negative for n in sorted(statements(case["question"]))}


def articles(text: str) -> set[str]:
    """The statute articles a text cites, as "271a OR". The code is kept as written: the same article
    is "337 OR" in German and "337 CO" in French, and an answer in the other language is not wrong."""
    return {f"{number} {code}" for number, code in _ARTICLE.findall(text)}


async def one(judge: Judge, case: dict[str, Any], answer: str) -> dict[str, Any]:
    truth, items = key(case), statements(case["question"])
    prefix = (f"QUESTION:\n{case['question'].split(chr(10) + chr(10) + 'A)')[0].strip()}\n\n"
              f"ANSWER UNDER REVIEW:\n{answer.strip() or '(the assistant produced no answer)'}")

    async def verdict(n: int) -> tuple[int, str | None]:
        try:
            out = await judge._ask(prefix, STATEMENT_TASK.format(statement=items[n]), VERDICT, 16)
            return n, out.get("verdict")
        except Exception:  # noqa: BLE001 - an unscored statement is dropped, not counted wrong
            return n, None

    said = dict(await asyncio.gather(*(verdict(n) for n in items)))
    scored = {n: {"key": truth[n], "said": said.get(n),
                  "right": (said.get(n) == "correct") == truth[n] if said.get(n) in ("correct", "incorrect") else None}
              for n in items}
    wanted = articles(" ".join(items.values()))
    return {"id": case["id"], "language": case["language"], "area": case["area"],
            "gold": case["gold"], "negative": bool((case.get("source") or {}).get("negative_question")),
            "statements": scored,
            "articles_expected": sorted(wanted), "articles_cited": sorted(articles(answer)),
            "articles_hit": sorted(wanted & articles(answer))}


def report(rows: list[dict[str, Any]], strict: bool) -> str:
    total = [s for r in rows for s in r["statements"].values()]
    addressed = [s for s in total if s["right"] is not None]
    right = [s for s in addressed if s["right"]]
    unscored = sum(1 for s in total if s["said"] is None)
    denominator = total if strict else addressed
    out = ["# LEXam multiple choice, scored on the reasoning", "",
           f"{len(rows)} questions · {len(total)} statements", "",
           "| measure | value |", "|---|---|",
           f"| statements the answer takes a position on | {len(addressed)}/{len(total)} = "
           f"{100 * len(addressed) / max(len(total), 1):.0f}% |",
           f"| of those, judged the way the key has it | {len(right)}/{len(addressed)} = "
           f"{100 * len(right) / max(len(addressed), 1):.0f}% |",
           f"| over every statement ({'strict' if strict else 'for comparison'}) | {len(right)}/{len(total)} = "
           f"{100 * len(right) / max(len(total), 1):.0f}% |"]
    if unscored:
        out.append(f"| statements the judge could not read | {unscored} |")
    whole = [r for r in rows if all(s["right"] for s in r["statements"].values())]
    out += [f"| questions right on every statement | {len(whole)}/{len(rows)} = "
            f"{100 * len(whole) / max(len(rows), 1):.0f}% |", ""]
    wanted = sum(len(r["articles_expected"]) for r in rows)
    hit = sum(len(r["articles_hit"]) for r in rows)
    with_arts = [r for r in rows if r["articles_expected"]]
    out += ["## The provisions",  "",
            f"{len(with_arts)} of {len(rows)} questions name a statute article. Of the "
            f"{wanted} articles they name, the answers cite {hit} ({100 * hit / max(wanted, 1):.0f}%).", ""]
    by: dict[str, list[dict]] = {}
    for r in rows:
        for s in r["statements"].values():
            by.setdefault(r["language"], []).append(s)
    out += ["## By language", "", "| language | addressed | of those, right |", "|---|---|---|"]
    for lang, ss in sorted(by.items()):
        a = [s for s in ss if s["right"] is not None]
        out.append(f"| {lang} | {len(a)}/{len(ss)} = {100 * len(a) / max(len(ss), 1):.0f}% | "
                   f"{sum(1 for s in a if s['right'])}/{len(a)} = "
                   f"{100 * sum(1 for s in a if s['right']) / max(len(a), 1):.0f}% |")
    _ = denominator
    return "\n".join(out) + "\n"


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="a finished run directory with transcripts/")
    ap.add_argument("--cases", type=Path, default=HERE / "cases" / "lexam" / "mcq.yaml")
    ap.add_argument("--judge-url", default=JUDGE_URL)
    ap.add_argument("--judge-model")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--strict", action="store_true", help="count an unaddressed statement as wrong")
    args = ap.parse_args()

    cases = {c["id"]: c for c in yaml.safe_load(args.cases.read_text())}
    answers = {}
    for path in sorted((args.run / "transcripts").glob("*.json")):
        t = json.loads(path.read_text())
        if t["id"] in cases and not t.get("error"):
            answers[t["id"]] = t["answer"] or ""
    if not answers:
        raise SystemExit(f"no multiple-choice transcripts in {args.run}")

    llm = AsyncOpenAI(base_url=args.judge_url, api_key=JUDGE_KEY, timeout=300, max_retries=0)
    model = args.judge_model or (await llm.models.list()).data[0].id
    judge = Judge(llm, model)
    gate = asyncio.Semaphore(args.concurrency)

    async def run(case_id: str) -> dict[str, Any]:
        async with gate:
            return await one(judge, cases[case_id], answers[case_id])

    rows = await asyncio.gather(*(run(i) for i in sorted(answers)))
    text = report(list(rows), args.strict)
    (args.run / "reasoning.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    (args.run / "reasoning.md").write_text(text)
    print(text)
    print(f"wrote {args.run / 'reasoning.json'} and {args.run / 'reasoning.md'}")


if __name__ == "__main__":
    asyncio.run(main())
