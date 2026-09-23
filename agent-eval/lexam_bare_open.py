"""The model alone on the LEXam open questions, graded by the same judge as the agent.

    uv run python agent-eval/lexam_bare_open.py --compare agent-eval/runs/lexam-ab/open-D/latest

The prompt is LEXam's own (litellm_eval.py QA_PROMPT). Only what applies to both is compared: the share
of the marking scheme an answer makes (rubric coverage) and whether the judge could make a legal error
stick. Grounding and citation behaviour are left out - a bare model cites nothing - so "passes" here is
coverage >= RUBRIC_PASS with no legal error, a looser bar than the agent's own report applies.
Not comparable with the LEXam leaderboard's open-question scores, which use another judge ensemble.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import yaml
from openai import AsyncOpenAI

from judge import Judge
from metrics import ACCURACY_PASS, RUBRIC_PASS

HERE = Path(__file__).resolve().parent
URL = os.environ.get("SCA_LLM_URL", "http://localhost:9100/v1")

# verbatim from LEXam litellm_eval.py
QA_PROMPT = """You are an expert in {course_name} and address legal issues in a structured, exam-style manner.
Assume Swiss law applies unless specifically mentioned; if the course context justifies, address legal issues beyond Swiss law as well.
Use precise legal language and formal "Sie" when answering.
Do NOT state any disclaimer or refer to the need for external legal advice.
Do NOT request the user to consult laws or to research on their own.
Offer focused legal analyses and individualized advice.
Speak directly and authoritatively without mentioning that your response is merely for general information.
Incorporate Swiss-specific legal terminology.
If you have discovered relevant legal considerations (Erwägungen), respond with a concise, clear legal analysis.
Cite only from your identified considerations.
Always cite the specific legal provision, explicitly indicating paragraphs (Abs.), numbers (Ziff.), or letters (lit.) where available (e.g., “'Art. 74 Abs. 2 Ziff. 2 OR”, “Art. 336 lit. a StGB”). Avoid general references (such as 'Art. 3 ZGB') without mentioning the specific paragraph, number, or letter, if applicable.
If no relevant considerations are found, explicitly state that no pertinent information is available.
If you do have reliable sources, share practical guidance or insights from them.
Respond in the same language as the question.
If the question specifically requests a short answer, provide a concise response.
If the prompt asks you to analyze a specific case provided in the exam, but the text or details of that case have not been provided in the prompt, explicitly flag that the required case material is missing.

Question:
{question}

Answer:"""


def coverage(judgment: dict) -> float:
    verdicts = [r["verdict"] for r in judgment.get("rubric", [])]
    return sum({"covered": 2, "partial": 1}.get(v, 0) for v in verdicts) / (2 * len(verdicts)) if verdicts else 0.0


def line(name: str, rows: list[tuple[float, int | None]]) -> str:
    n = len(rows)
    cov = sum(c for c, _ in rows) / n
    error = sum((a or 5) < ACCURACY_PASS for _, a in rows)
    passed = sum(c >= RUBRIC_PASS and (a or 5) >= ACCURACY_PASS for c, a in rows)
    return f"{name:<12} n={n}  rubric coverage {cov:5.1%}  legal error {error:>2}  passes {passed}/{n} ({passed / n:.0%})"


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, default=HERE / "cases" / "lexam" / "open.yaml")
    ap.add_argument("--compare", type=Path, nargs="*", default=[], help="agent runs of the same cases")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args()

    cases = yaml.safe_load(args.cases.read_text())
    llm = AsyncOpenAI(base_url=URL, api_key="nim", timeout=1800, max_retries=2)
    model = (await llm.models.list()).data[0].id
    judge = Judge(llm, model)
    out = HERE / "runs" / f"bare-open-{datetime.now():%Y%m%d-%H%M%S}"
    out.mkdir(parents=True)
    semaphore = asyncio.Semaphore(args.concurrency)

    async def one(case: dict) -> dict:
        async with semaphore:
            prompt = QA_PROMPT.format(course_name=case["source"]["course"], question=case["question"])
            try:
                chat = await llm.chat.completions.create(
                    model=model, messages=[{"role": "user", "content": prompt}], temperature=0.6,
                    max_tokens=32768, extra_body={"chat_template_kwargs": {"enable_thinking": True}})
                answer = chat.choices[0].message.content or ""
            except Exception as e:  # noqa: BLE001
                answer = f"(failed: {e})"
            judgment = await judge.judge(case, answer, [])
        return {"id": case["id"], "answer": answer, "judgment": judgment,
                "rubric_coverage": coverage(judgment), "legal_accuracy": judgment.get("legal_accuracy")}

    rows = await asyncio.gather(*(one(c) for c in cases))
    (out / "results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(line("bare", [(r["rubric_coverage"], r["legal_accuracy"]) for r in rows]))
    for run in args.compare:
        agent = {c["id"]: c for c in json.loads((run / "results.json").read_text())["cases"]}
        print(line(run.parent.name if run.name == "latest" else run.name,
                   [(agent[r["id"]]["rubric_coverage"] or 0.0, agent[r["id"]]["legal_accuracy"])
                    for r in rows if r["id"] in agent]))
    print(f"wrote {out}/results.json")


if __name__ == "__main__":
    asyncio.run(main())
