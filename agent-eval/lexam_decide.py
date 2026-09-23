"""Replay the agent's multiple-choice decision on saved research, with other prompts.

An agent run takes hours; the decision at its end is one model call. Serve the agent with
SCA_DECIDE_DUMP=<file.jsonl> during a LEXam run and every decision's input - the question, the whole
research conversation, the checked answer - is saved. This script asks the decision again on those
inputs, once per variant and repeat, and scores it against the key:

    uv run python agent-eval/lexam_decide.py decide.jsonl --variants current no-answer knowledge-first --repeats 2

Variants differ only in what the decision sees and how it is told to weigh it; the research is the same.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

import yaml
from langchain_core.messages import HumanMessage, SystemMessage, messages_from_dict
from langchain_openai import ChatOpenAI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from swiss_court_assistant.server.react_agent import (DECIDE_PROMPT, EXAM_ANSWER_WORD, LANGUAGE_NAMES,  # noqa: E402
                                                      _CHOSEN)

URL = os.environ.get("SCA_LLM_URL", "http://localhost:9100/v1")

CAREFUL_ARTICLES = DECIDE_PROMPT.replace(
    "decide it from your own knowledge of Swiss law and name the statute article it rests on;",
    "decide it from your own knowledge of Swiss law and name the rule and the statute it comes from; give an "
    "article number only when the research shows it or you are certain of it, since a wrong article number is "
    "worse than none;")
assert CAREFUL_ARTICLES != DECIDE_PROMPT

KNOWLEDGE_FIRST = DECIDE_PROMPT.replace(
    "Where a passage in the research states the rule, rely on it and name that decision (by court and docket number) or article.",
    "Decide each statement first from your knowledge of Swiss law, then check it against the research: where a "
    "passage states the rule, follow it and name that decision (by court and docket number) or article; where the "
    "passages are about something else, they neither confirm nor refute the statement.")


ARTICLE = re.compile(r"\bArt\.?\s*(\d+[a-z]?)(?:\s*(?:Abs|al|para|cpv|lit|Ziff)\.?\s*\S+)*\s+([A-Z][A-Za-z]{1,6}\b)")


def research_text(dump: dict) -> str:
    return " ".join(str(m["data"].get("content", "")) for m in dump["messages"] if m.get("type") == "tool")


def in_research(article: tuple[str, str], text: str) -> bool:
    """Whether a tool result mentions this article of this statute ("Art. 271a OR", "art. 271a CO" ...)."""
    number, code = article
    return re.search(rf"\b(?:Art|art)\.?\s*{re.escape(number)}\b[^.;]{{0,40}}\b{re.escape(code)}\b", text) is not None


def variant(name: str, dump: dict) -> list:
    """The messages of one variant of the decision, for one saved question."""
    code, letters = dump["code"], dump["letters"]
    word = EXAM_ANSWER_WORD.get(code, EXAM_ANSWER_WORD["en"])
    fmt = dict(word=word, letters=", ".join(letters), language=LANGUAGE_NAMES.get(code, "English"))
    research = messages_from_dict(dump["messages"])
    written = dump["written"] or "(nothing held up against the passages)"
    checked = HumanMessage(f"The checked answer:\n{written}\n\nDecide the question now. End with the line \"{word}: X\".")
    plain = HumanMessage(f"Decide the question now. End with the line \"{word}: X\".")
    if name == "current":
        return [SystemMessage(DECIDE_PROMPT.format(**fmt)), *research, checked]
    if name == "careful-articles":  # no article numbers from memory unless certain
        return [SystemMessage(CAREFUL_ARTICLES.format(**fmt)), *research, checked]
    if name == "no-answer":        # the research, without the checked answer's framing
        return [SystemMessage(DECIDE_PROMPT.format(**fmt)), *research, plain]
    if name == "knowledge-first":  # own knowledge first, the passages as a check
        return [SystemMessage(KNOWLEDGE_FIRST.format(**fmt)), *research, checked]
    if name == "no-research":      # the same prompt with the question alone: what the research adds
        return [SystemMessage(DECIDE_PROMPT.format(**fmt)), research[0], plain]
    raise SystemExit(f"unknown variant {name}")


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump", type=Path)
    ap.add_argument("--variants", nargs="+", default=["current"])
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--cases", type=Path, default=HERE / "cases" / "lexam" / "mcq.yaml")
    ap.add_argument("--concurrency", type=int, default=48)
    args = ap.parse_args()

    gold = {c["question"].strip(): c["gold"] for c in yaml.safe_load(args.cases.read_text())}
    dumps = {}
    for line in args.dump.read_text().splitlines():  # the last decision per question (after any retry)
        d = json.loads(line)
        if d["question"].strip() in gold:
            dumps[d["question"].strip()] = d
    print(f"{len(dumps)} saved decisions with a key")
    model = ChatOpenAI(base_url=URL, api_key="nim", model="nvidia/nemotron-3.5-lightning", temperature=0.2,
                       max_tokens=16384, timeout=900, extra_body={"chat_template_kwargs": {"enable_thinking": True}})
    semaphore = asyncio.Semaphore(args.concurrency)
    articles = {name: [0, 0] for name in args.variants}  # articles named, of those not in the research

    quick = model.bind(extra_body={"chat_template_kwargs": {"enable_thinking": False}})

    async def ask(name: str, q: str) -> str:
        """The letter chosen, "none" (no option named) or "error", with the agent's own retry: a second
        call without thinking when the first names no option."""
        outcome = "error"
        for llm in (model, quick):
            async with semaphore:
                try:
                    text = str((await llm.ainvoke(variant(name, dumps[q]))).content)
                except Exception as e:  # noqa: BLE001
                    print(f"  {name}: call failed: {str(e)[:120]}", flush=True)
                    continue
            chosen = [x for x in _CHOSEN.findall(text) if x in dumps[q]["letters"]]
            if chosen:
                named = {a for a in ARTICLE.findall(text)}
                seen = research_text(dumps[q])
                articles[name][0] += len(named)
                articles[name][1] += sum(not in_research(a, seen) for a in named)
                return chosen[-1]
            outcome = "none"
        return outcome

    for name in args.variants:
        scores, lost, picks = [], Counter(), {q: [] for q in dumps}
        for _ in range(args.repeats):
            outcomes = dict(zip(dumps, await asyncio.gather(*(ask(name, q) for q in dumps))))
            scores.append(sum(o == gold[q] for q, o in outcomes.items()) / len(outcomes))
            lost.update(o for o in outcomes.values() if o in ("none", "error"))
            for q, o in outcomes.items():
                picks[q].append(o)
        # the majority of the repeats, ties to the first: is one decision worth asking several times
        vote = sum(max(p, key=lambda x: (p.count(x), -p.index(x))) == gold[q] for q, p in picks.items()) / len(picks)
        stable = sum(len(set(p)) == 1 for p in picks.values())
        print(f"{name:<16} " + "  ".join(f"{s:.1%}" for s in scores)
              + (f"   mean {sum(scores) / len(scores):.1%}   majority {vote:.1%}   same letter every time "
                 f"{stable}/{len(picks)}" if len(scores) > 1 else "")
              + (f"   (no option {lost['none']}, failed {lost['error']})" if lost else "")
              + f"   articles named {articles[name][0]}, not in the research {articles[name][1]}", flush=True)


if __name__ == "__main__":
    random.seed(0)
    asyncio.run(main())
