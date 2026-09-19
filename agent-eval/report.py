"""Turn a run's results into the report: the headline, the breakdowns, and every case with the
reason it failed. Written as Markdown so it reads in a terminal, in the repo, and in a pull request.

    uv run python agent-eval/report.py agent-eval/runs/latest/results.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

SCORE_COLUMNS = [
    ("cases", lambda rows: len(rows)),
    ("correct", lambda rows: _pct(_share(rows, lambda r: r["correct"]))),
    ("score", lambda rows: _num(_avg(rows, lambda r: r["score"]), 1)),
    ("rubric", lambda rows: _pct(_avg(rows, lambda r: r["rubric_coverage"]))),
    ("accuracy", lambda rows: _num(_avg(rows, lambda r: r["legal_accuracy"]), 2)),
    ("grounding", lambda rows: _num(_avg(rows, lambda r: r["grounding"]), 2)),
    ("useful", lambda rows: _num(_avg(rows, lambda r: r["usefulness"]), 2)),
]
GROUNDING_COLUMNS = [
    ("cases", lambda rows: len(rows)),
    ("citations/answer", lambda rows: _num(_avg(rows, lambda r: r["metrics"]["n_citations"]), 1)),
    ("quote verbatim", lambda rows: _pct(_avg(rows, lambda r: r["metrics"]["verified_rate"]))),
    ("passage supports", lambda rows: _pct(_avg(rows, lambda r: r["metrics"]["support_rate"]))),
    ("no legal error", lambda rows: _pct(_share(rows, lambda r: r["legal_accuracy"] == 5))),
    ("language", lambda rows: _pct(_share(rows, lambda r: r["metrics"]["language_ok"]))),
    ("abstention", lambda rows: _pct(_share(rows, lambda r: r["abstain_ok"]))),
]
COST_COLUMNS = [
    ("cases", lambda rows: len(rows)),
    ("median s", lambda rows: _num(_median(rows, lambda r: r["metrics"]["seconds"]), 0)),
    ("mean s", lambda rows: _num(_avg(rows, lambda r: r["metrics"]["seconds"]), 0)),
    ("tool calls", lambda rows: _num(_avg(rows, lambda r: r["metrics"]["n_tool_calls"]), 1)),
    ("repeated searches", lambda rows: _num(_avg(rows, lambda r: r["metrics"]["repeated_searches"]), 2)),
    ("tool errors", lambda rows: str(sum(r["metrics"]["tool_errors"] for r in rows))),
    ("answer chars", lambda rows: _num(_avg(rows, lambda r: r["metrics"]["answer_chars"]), 0)),
]
Rows = list[dict[str, Any]]


def _avg(rows: Rows, pick: Callable[[dict], Any]) -> float | None:
    values = [v for v in (pick(r) for r in rows) if v is not None]
    return mean(values) if values else None


def _median(rows: Rows, pick: Callable[[dict], Any]) -> float | None:
    values = sorted(v for v in (pick(r) for r in rows) if v is not None)
    return values[len(values) // 2] if values else None


def _share(rows: Rows, hit: Callable[[dict], Any]) -> float | None:
    return mean(bool(hit(r)) for r in rows) if rows else None


def _pct(rate: float | None) -> str:
    """Every rate in this report is a share between 0 and 1."""
    return "–" if rate is None else f"{rate * 100:.0f}%"


def _num(value: float | None, digits: int) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def _table(header: str, groups: list[tuple[str, Rows]], columns: list[tuple[str, Callable]]) -> str:
    names = [header, *(name for name, _ in columns)]
    lines = ["| " + " | ".join(names) + " |", "|" + "|".join(["---"] * len(names)) + "|"]
    for label, rows in groups:
        cells = [str(render(rows)) for _, render in columns]
        lines.append("| " + " | ".join([label, *cells]) + " |")
    return "\n".join(lines)


def _grouped(results: Rows, key: str) -> list[tuple[str, Rows]]:
    groups: dict[str, Rows] = defaultdict(list)
    for result in results:
        groups[str(result[key])].append(result)
    return sorted(groups.items())


def _suites(results: Rows) -> list[tuple[str, Rows]]:
    return [("**all**", results), *_grouped(results, "suite")]


def render(run: dict[str, Any], results: Rows) -> str:
    judged = [r for r in results if r["score"] is not None]
    failed = [r for r in results if not r["correct"]]
    errors = [r for r in results if r["error"]]
    weights = ", ".join(f"{k} {v:.0%}" for k, v in run["weights"].items())

    out = [
        f"# Agent evaluation - {run['started'][:16].replace('T', ' ')} UTC",
        "",
        f"`{run['agent']}` agent over {run['decisions']:,} Swiss court decisions, "
        f"{run['cases']} cases, judged by `{run['judge_model'] or 'nobody (--no-judge)'}`. "
        + (f"The answers come from the earlier run in `{run['rejudged_from']}` and were graded again "
           f"here; only the judging is new." if run.get("rejudged_from")
           else f"The run took {run['seconds'] / 60:.0f} min at concurrency {run['concurrency']}."),
        "",
        f"**{_pct(_share(results, lambda r: r['correct']))} of cases correct** "
        f"(rubric coverage ≥ {run['rubric_pass']:.0%}, no legal error the judge could make stick, "
        f"and the expected behaviour), mean score {_num(_avg(results, lambda r: r['score']), 1)}/100 "
        f"(weights: {weights}).",
        "",
        "## Scores",
        "",
        _table("suite", _suites(results), SCORE_COLUMNS),
        "",
        "Rubric is the share of the reference answer's points the answer makes; accuracy, grounding "
        "and usefulness are the judge's 1–5 scales.",
        "",
        "## Grounding and behaviour",
        "",
        _table("suite", _suites(results), GROUNDING_COLUMNS),
        "",
        "*No legal error* is the share of answers in which the judge could not name a wrong statement "
        "and have it stick. *Quote verbatim* and *passage supports* come from the assistant itself: the first is the "
        "share of citations whose quote was located character for character in the decision, the "
        "second the share its own grounding check confirmed as stating the sentence they back. "
        "*Abstention* is how often it refused exactly when it should have.",
        "",
        "## By area",
        "",
        _table("area", _grouped(results, "area"), SCORE_COLUMNS),
        "",
        "## By language and difficulty",
        "",
        _table("language", _grouped(results, "language"), SCORE_COLUMNS),
        "",
        _table("level", _grouped(results, "level"), SCORE_COLUMNS),
        "",
        "## Cost",
        "",
        _table("suite", _suites(results), COST_COLUMNS),
        "",
        "## Every case",
        "",
        "| case | area | lang | score | rubric | acc | gnd | ok | note |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in sorted(results, key=lambda r: (r["suite"], -(r["score"] or -1))):
        out.append(f"| `{r['id']}` | {r['area']} | {r['language']} | "
                   f"{_num(r['score'], 1)} | {_pct(r['rubric_coverage'])} | "
                   f"{r['legal_accuracy'] or '–'} | {r['grounding'] or '–'} | "
                   f"{'✅' if r['correct'] else '❌'} | {r['why'] or ''} |")

    if failed:
        out += ["", f"## What went wrong ({len(failed)} of {len(results)})", ""]
        for r in sorted(failed, key=lambda r: r["score"] or -1):
            comment = (r["judgment"].get("comment") or "").strip()
            missed = [str(i["point"]) for i in r["judgment"].get("rubric", []) if i["verdict"] == "missing"]
            out += [f"**`{r['id']}`** - {r['why'] or 'see below'}  ",
                    f"{comment}" + (f" Rubric points missed: {', '.join(missed)}." if missed else ""), ""]
    if errors:
        out += ["", "## Turns that failed to run", ""] + [f"- `{r['id']}`: {r['error']}" for r in errors]
    if len(judged) < len(results):
        out += ["", f"*{len(results) - len(judged)} case(s) carry no judge score.*"]
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("results", type=Path, nargs="?",
                    default=Path(__file__).resolve().parent / "runs" / "latest" / "results.json")
    ap.add_argument("--out", type=Path, help="write the Markdown here as well as printing it")
    args = ap.parse_args()
    data = json.loads(args.results.read_text())
    markdown = render(data["run"], data["cases"])
    if args.out:
        args.out.write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
