"""The model alone on LEXam multiple choice, the way the LEXam leaderboard scores it: no retrieval, no
tools, no harness. The prompt and the letter extraction are LEXam's own (litellm_eval.py MCQ_PROMPT
and evaluation.py extract_letter, github.com/LEXam-Benchmark/LEXam), so the accuracy is comparable
with https://lexam-benchmark.github.io/ (mcq_4_choices, all 1,655 questions).

    uv run python agent-eval/lexam_bare.py                        # all 1,655, thinking on
    uv run python agent-eval/lexam_bare.py --thinking off
    uv run python agent-eval/lexam_bare.py --ids agent-eval/cases/lexam/mcq.yaml   # only the agent's 100

Writes agent-eval/runs/bare-<timestamp>/answers.jsonl (one line per question, resumable with --resume)
and prints accuracy overall, on the Swiss subset and on the agent's sample, with a bootstrap interval.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import json
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import polars as pl
import yaml
from huggingface_hub import snapshot_download
from openai import AsyncOpenAI

HERE = Path(__file__).resolve().parent
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
URL = os.environ.get("SCA_LLM_URL", "http://localhost:9100/v1")

# verbatim from LEXam litellm_eval.py
MCQ_PROMPT = """You are an expert in {course_name} and address legal issues in a structured, exam-style manner.
You are given a multiple-choice question, where only one choice (e.g., A, B, C, etc.) is correct.
Assume Swiss law applies unless specifically stated otherwise. If the context of the course justifies it, consider legal frameworks beyond Swiss law as well.

Please reason through the question step by step, using a chain-of-thought approach:
- Clarify the facts: Briefly restate or highlight the key facts in the question to anchor your reasoning.
- Issue Identification: What legal issue(s) arise from the facts?
- Rule Explanation: What legal rules or principles are relevant, and what are their sources (e.g., statutes, case law, doctrine)?
- Application and Reasoning: Apply the relevant rules to the facts, carefully weighing any ambiguities, exceptions, or competing interpretations.
- Eliminate Incorrect Answers: Briefly explain why each incorrect answer is wrong or less convincing.
- Conclusion: Clearly state the correct answer choice (e.g., A, B, C, etc.) with a brief justification for why it best fits the legal analysis.

Format your final answer as follows:
 Correct Answer: ###C###

Question:
 {question}

Answer:"""


def extract_letter(response: str) -> str | None:  # verbatim from LEXam evaluation.py
    matches = re.findall(r"###([A-Z])###", str(response))
    return matches[-1] if matches else None


def prompt(row: dict) -> str:
    question = row["question"]
    for i, choice in enumerate(ast.literal_eval(row["choices"])):
        question += f"\n{LETTERS[i]}. {choice}"
    return MCQ_PROMPT.format(course_name=row["course"], question=question)


def interval(hits: list[bool], rounds: int = 2000) -> tuple[float, float, float]:
    rng = random.Random(0)
    n = len(hits)
    boots = sorted(sum(rng.choice(hits) for _ in range(n)) / n for _ in range(rounds))
    return sum(hits) / n, boots[int(0.025 * rounds)], boots[int(0.975 * rounds)]


def summary(rows: list[dict], sample: set[str]) -> str:
    lines = []
    for name, subset in (("all", rows), ("Swiss", [r for r in rows if r["jurisdiction"] == "Swiss"]),
                         ("agent sample", [r for r in rows if r["id"] in sample])):
        if subset:
            acc, lo, hi = interval([r["correct"] for r in subset])
            missing = sum(r["choice"] is None for r in subset)
            lines.append(f"{name:<13} n={len(subset):>4}  accuracy {acc:6.1%}  (95% {lo:.1%}–{hi:.1%})  no letter {missing}")
    return "\n".join(lines)


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--thinking", choices=["on", "off"], default="on")
    ap.add_argument("--ids", type=Path, help="only the questions of this case file (by LEXam id)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--max-tokens", type=int, default=32768)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--resume", type=Path, help="a previous run directory to complete")
    args = ap.parse_args()

    root = Path(snapshot_download("LEXam-Benchmark/LEXam", repo_type="dataset", allow_patterns=["*.parquet"]))
    df = pl.read_parquet(root / "mcq_4_choices" / "test-00000-of-00001.parquet")
    sample = {c["source"]["id"] for c in yaml.safe_load((HERE / "cases/lexam/mcq.yaml").read_text())}
    if args.ids:
        keep = {c["source"]["id"] for c in yaml.safe_load(args.ids.read_text())}
        df = df.filter(pl.col("id").is_in(list(keep)))
    rows = list(df.iter_rows(named=True))[: args.limit or None]

    out = args.resume or HERE / "runs" / f"bare-{args.thinking}-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "answers.jsonl"
    done = {json.loads(l)["id"]: json.loads(l) for l in path.read_text().splitlines()} if path.exists() else {}
    todo = [r for r in rows if r["id"] not in done]
    llm = AsyncOpenAI(base_url=URL, api_key="nim", timeout=1800, max_retries=2)
    model = (await llm.models.list()).data[0].id
    print(f"{model} · thinking {args.thinking} · {len(todo)} to ask, {len(done)} done · {out}", flush=True)

    semaphore, started, lock = asyncio.Semaphore(args.concurrency), time.monotonic(), asyncio.Lock()

    async def one(row: dict) -> None:
        async with semaphore:
            t = time.monotonic()
            try:
                chat = await llm.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt(row)}],
                    temperature=args.temperature, max_tokens=args.max_tokens,
                    extra_body={"chat_template_kwargs": {"enable_thinking": args.thinking == "on"}})
                text = chat.choices[0].message.content or ""
                message = chat.choices[0].message
                reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or ""
                tokens, error = chat.usage.completion_tokens, None
            except Exception as e:  # noqa: BLE001 - one failure is one wrong answer, not a dead run
                text, reasoning, tokens, error = "", "", 0, str(e)[:300]
        choice = extract_letter(text)
        record = {"id": row["id"], "jurisdiction": row["jurisdiction"], "language": row["language"],
                  "area": row["area"], "gold": LETTERS[row["gold"]], "choice": choice,
                  "correct": choice == LETTERS[row["gold"]], "tokens": tokens,
                  "seconds": round(time.monotonic() - t, 1), "error": error, "answer": text,
                  "reasoning_chars": len(reasoning), "truncated": tokens >= args.max_tokens}
        async with lock:
            done[row["id"]] = record
            with path.open("a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            if len(done) % 50 == 0:
                rate = (time.monotonic() - started) / max(1, len(done) - (len(rows) - len(todo)))
                print(f"[{len(done)}/{len(rows)}] {sum(r['correct'] for r in done.values()) / len(done):.1%} "
                      f"· {rate:.1f}s/question", flush=True)

    await asyncio.gather(*(one(r) for r in todo))
    results = [done[r["id"]] for r in rows if r["id"] in done]
    text = summary(results, sample)
    (out / "summary.txt").write_text(f"{model} thinking={args.thinking} temperature={args.temperature}\n{text}\n")
    print(text)


if __name__ == "__main__":
    asyncio.run(main())
