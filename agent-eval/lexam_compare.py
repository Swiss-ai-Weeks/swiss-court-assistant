"""Compare multiple-choice accuracy across agent runs and the bare model, on the same questions.

    uv run python agent-eval/lexam_compare.py runs/lexam-ab/A/latest runs/lexam-ab/B/latest \
        --bare runs/bare-on-20260922-215349

The first run is the reference: every other one is compared with it question by question (won, lost,
and the exact McNemar p-value on the questions where they differ), which is far more sensitive than
comparing two accuracies with ±9-point intervals each.
"""

from __future__ import annotations

import argparse
import json
import random
from math import comb
from pathlib import Path


def agent_run(path: Path) -> dict[str, bool | None]:
    """LEXam id -> correct, for the MCQ cases of one run (None: the turn failed)."""
    out = {}
    for t in (path / "transcripts").glob("lexam-mcq-*.json"):
        case = json.loads(t.read_text())
        out[case["source"]["id"]] = None if case.get("error") else case["judgment"].get("choice") == case["gold"]
    return out


def bare_run(path: Path) -> dict[str, bool]:
    return {r["id"]: r["correct"] for r in map(json.loads, (path / "answers.jsonl").read_text().splitlines())}


def interval(hits: list[bool], rounds: int = 2000) -> tuple[float, float]:
    rng, n = random.Random(0), len(hits)
    boots = sorted(sum(rng.choice(hits) for _ in range(n)) / n for _ in range(rounds))
    return boots[int(0.025 * rounds)], boots[int(0.975 * rounds)]


def mcnemar(won: int, lost: int) -> float:
    """Exact two-sided p-value: of the questions the two got differently, is the split beyond chance."""
    n = won + lost
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(won, lost) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", type=Path, nargs="+")
    ap.add_argument("--bare", type=Path, nargs="*", default=[], help="lexam_bare.py run directories")
    args = ap.parse_args()

    arms = {str(p): agent_run(p.resolve()) for p in args.runs}
    arms |= {f"bare:{p.name}": bare_run(p) for p in args.bare}
    ids = set.intersection(*(set(a) for a in list(arms.values())[:len(args.runs)]))  # the agent's questions
    ref_name, ref = next(iter(arms.items()))
    print(f"{len(ids)} questions in every agent run\n")
    print(f"{'run':<52} {'acc':>6} {'95% CI':>13} {'failed':>6}   vs first: won lost  p")
    for name, arm in arms.items():
        hits = [bool(arm.get(i)) for i in ids]
        lo, hi = interval(hits)
        failed = sum(arm.get(i) is None for i in ids)
        won = sum(bool(arm.get(i)) and not ref.get(i) for i in ids)
        lost = sum(not arm.get(i) and bool(ref.get(i)) for i in ids)
        vs = "" if arm is ref else f"{won:>9} {lost:>4}  {mcnemar(won, lost):.3f}"
        print(f"{name[-52:]:<52} {sum(hits) / len(ids):6.1%} {lo:6.1%}–{hi:5.1%} {failed:>6}   {vs}")


if __name__ == "__main__":
    main()
