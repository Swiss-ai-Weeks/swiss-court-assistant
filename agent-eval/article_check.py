"""Are the statute articles the multiple-choice assessment names the right ones?

The assessment ("Beurteilung der Optionen") is reasoned out, not checked against passages, and the
model names articles from memory there - "Art. 469 ZGB" for the parentelic order, which is Art. 457 ff.
This looks every "Art. N CODE" of an assessment up in the statute index and asks the model whether that
article's text is about what the sentence uses it for.

    uv run python agent-eval/article_check.py agent-eval/runs/lexam-ab/F/latest
    uv run python agent-eval/article_check.py replay-texts.jsonl       # lexam_decide.py --save

Prints, per source: articles named; those of an act the index cannot resolve (not checked); those
that do not exist in their act; and of those found, the share whose text fits the sentence.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from openai import AsyncOpenAI

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from swiss_court_assistant.server.decisions import SqliteDecisionStore  # noqa: E402
from swiss_court_assistant.server.react_agent import EXAM_HEADING  # noqa: E402

URL = os.environ.get("SCA_LLM_URL", "http://localhost:9100/v1")
INDEX = HERE.parent / "data" / "vectordb" / "corpus.sqlite"
# the agent's own reference pattern and check, so that this measures what the agent checks
from swiss_court_assistant.server.react_agent import ARTICLE_FITS_PROMPT, _ARTICLE_REF as ARTICLE, fits_input  # noqa: E402
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ(])|\n+")



def assessments(source: Path) -> list[tuple[str, str]]:
    """(label, text of the assessment) for every answer in a run directory or a replay save file."""
    if source.is_dir():
        out = []
        for t in sorted((source / "transcripts").glob("lexam-mcq-*.json")):
            answer = json.loads(t.read_text())["answer"]
            for heading in EXAM_HEADING.values():
                if f"**{heading}**" in answer:
                    out.append(("agent", answer.split(f"**{heading}**", 1)[1]))
                    break
        return out
    return [(r["variant"], r["text"]) for r in map(json.loads, source.read_text().splitlines())]


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", type=Path)
    ap.add_argument("--concurrency", type=int, default=32)
    args = ap.parse_args()

    store = SqliteDecisionStore(INDEX)
    llm = AsyncOpenAI(base_url=URL, api_key="nim", timeout=300)
    model = (await llm.models.list()).data[0].id
    semaphore = asyncio.Semaphore(args.concurrency)
    tally: dict[str, Counter] = defaultdict(Counter)
    wrong: list[str] = []

    async def check(name: str, sentence: str, ref: str, number: str, code: str) -> None:
        if not await asyncio.to_thread(store.find_acts, code):
            tally[name]["act not in the index"] += 1  # nothing to check it by
            return
        found = await asyncio.to_thread(store.find_articles, code, number)
        if not found:
            tally[name]["no such article"] += 1
            return
        article = next((a for a in found if a["language"] == "de"), found[0])
        messages = [{"role": "system", "content": ARTICLE_FITS_PROMPT},
                    {"role": "user", "content": fits_input(sentence, ref, article)}]
        async with semaphore:
            try:
                chat = await llm.chat.completions.create(
                    model=model, messages=messages, temperature=0, max_tokens=20,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False},
                                "response_format": {"type": "json_object"}})
                fits = json.loads(chat.choices[0].message.content or "{}").get("fits")
            except Exception:  # noqa: BLE001
                fits = None
        tally[name][{True: "fits", False: "does not fit", None: "unchecked"}[fits]] += 1
        if fits is False and len(wrong) < 12:
            wrong.append(f"{article['label']}: {sentence[:160]}")

    jobs = []
    for name, text in assessments(args.source):
        for sentence in SENTENCE.split(text):
            for ref, number, code in dict.fromkeys((m.group(0), m.group(1), m.group(2)) for m in ARTICLE.finditer(sentence)):
                tally[name]["named"] += 1
                jobs.append(check(name, sentence.strip(), ref, number, code))
    await asyncio.gather(*jobs)
    for name, t in tally.items():
        checked = t["fits"] + t["does not fit"]
        bad = t["no such article"] + t["does not fit"]
        print(f"{name:<18} named {t['named']:>4} · act not in the index {t['act not in the index']:>3} · "
              f"no such article {t['no such article']:>3} · fits {t['fits']}/{checked} = {t['fits'] / max(1, checked):.0%}"
              f" · wrong {bad}/{checked + t['no such article']}"
              + (f" · unchecked {t['unchecked']}" if t["unchecked"] else ""))
    if wrong:
        print("\nsome that do not fit:\n" + "\n".join(f"  {w}" for w in wrong))


if __name__ == "__main__":
    asyncio.run(main())
