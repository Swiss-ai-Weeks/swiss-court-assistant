"""Compare two runs case by case: did a change to the agent help?

    uv run python agent-eval/compare.py runs/20260917-231746 runs/latest

Prints the headline numbers side by side, then every case whose verdict flipped. Read it against the
judge's own noise: judging identical answers twice flips about one case in eight and moves the mean
score by a few points (README.md), so a single flip is not evidence and a small delta is not a change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


def load(run: Path) -> dict[str, dict[str, Any]]:
    return {c["id"]: c for c in json.loads((run / "results.json").read_text())["cases"]}


def summary(cases: list[dict[str, Any]]) -> dict[str, float | None]:
    def avg(pick) -> float | None:
        values = [v for v in map(pick, cases) if v is not None]
        return mean(values) if values else None
    return {
        "correct": avg(lambda c: float(c["correct"])),
        "score": avg(lambda c: c["score"]),
        "rubric": avg(lambda c: c["rubric_coverage"]),
        "no legal error": avg(lambda c: None if c["legal_accuracy"] is None else float(c["legal_accuracy"] == 5)),
        "grounding": avg(lambda c: c["grounding"]),
        "abstention": avg(lambda c: float(c["abstain_ok"])),
        "quote verbatim": avg(lambda c: c["metrics"]["verified_rate"]),
        "tool calls": avg(lambda c: c["metrics"]["n_tool_calls"]),
        "seconds": avg(lambda c: c["metrics"]["seconds"]),
        "empty answers": float(sum(c["error"] == "the stream ended without an answer" for c in cases)),
    }


def _fmt(key: str, value: float | None) -> str:
    if value is None:
        return "–"
    if key in ("correct", "rubric", "no legal error", "abstention", "quote verbatim"):
        return f"{value:.0%}"
    return f"{value:.0f}" if key == "empty answers" else f"{value:.1f}" if key != "grounding" else f"{value:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", type=Path)
    ap.add_argument("after", type=Path)
    ap.add_argument("--suite", help="only this suite")
    args = ap.parse_args()
    before, after = load(args.before), load(args.after)
    ids = [i for i in before if i in after and (not args.suite or before[i]["suite"] == args.suite)]

    print(f"{len(ids)} cases in both runs\n")
    print(f"{'':16}{'before':>10}{'after':>10}")
    b, a = summary([before[i] for i in ids]), summary([after[i] for i in ids])
    for key in b:
        print(f"{key:16}{_fmt(key, b[key]):>10}{_fmt(key, a[key]):>10}")

    gained = [i for i in ids if after[i]["correct"] and not before[i]["correct"]]
    lost = [i for i in ids if before[i]["correct"] and not after[i]["correct"]]
    for title, group in (("now correct", gained), ("no longer correct", lost)):
        if group:
            print(f"\n{title} ({len(group)}):")
            for i in group:
                x, y = before[i], after[i]
                print(f"  {i:<40} {x['score'] or 0:5.1f} → {y['score'] or 0:5.1f}"
                      f"  calls {x['metrics']['n_tool_calls']}→{y['metrics']['n_tool_calls']}"
                      f"  {y['why'] or x['why']}")


if __name__ == "__main__":
    main()
