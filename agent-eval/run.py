"""Run the evaluation: ask the assistant every case, grade the answers, write the results.

    # the server on :8090 and the LLM NIM on :9100 must be up
    uv run python agent-eval/run.py                       # every case, both suites
    uv run python agent-eval/run.py --suite behaviour      # only the behavioural cases
    uv run python agent-eval/run.py --only de-or-337 fr-   # cases whose id contains one of these
    uv run python agent-eval/run.py --no-judge             # mechanical checks only, no LLM judge
    uv run python agent-eval/run.py --rejudge runs/latest  # grade a finished run again, agent untouched

Each run writes a directory under agent-eval/runs/<timestamp>/:
    results.json      every case with its metrics, its judgment and its scores
    report.md         the tables, also printed to the terminal
    transcripts/      per case: the answer, the passages it cited, the tools it called, the judgment

The conversations the run creates in the app are deleted again unless --keep-conversations is given.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
import yaml
from openai import AsyncOpenAI

import report
from client import ask, delete_conversation, delete_document, delete_matter, health, prepare_matter, upload
from judge import Judge
from metrics import WEIGHTS, ACCURACY_PASS, RUBRIC_PASS, failure, mechanical, score

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
UNJUDGED = {"rubric": [], "legal_accuracy": None, "grounding": None, "usefulness": None,
            "abstained": None, "comment": ""}  # a turn that failed, or a run with --no-judge
SERVER = os.environ.get("SCA_SERVER", "http://localhost:8090")
JUDGE_URL = os.environ.get("SCA_JUDGE_URL", os.environ.get("SCA_LLM_URL", "http://localhost:9100/v1"))
JUDGE_KEY = os.environ.get("SCA_JUDGE_KEY", os.environ.get("SCA_LLM_KEY", "nim"))


def load_cases(paths: list[Path], suite: str | None, only: list[str] | None) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for path in sorted(p for path in paths for p in ([path] if path.is_file() else sorted(path.glob("*.yaml")))):
        loaded = yaml.safe_load(path.read_text()) or []
        for case in loaded:
            case.setdefault("suite", path.stem)
        cases += loaded
    if suite:
        cases = [c for c in cases if c["suite"] == suite]
    if only:
        cases = [c for c in cases if any(fragment in c["id"] for fragment in only)]
    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise SystemExit(f"duplicate case ids: {sorted({i for i in ids if ids.count(i) > 1})}")
    return cases


async def run_case(case: dict[str, Any], client: httpx.AsyncClient, judge: Judge | None,
                   keep: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    """One case: the setup turns, the judged turn, the judgment, the scores.

    A case with `attachments` uploads those files from fixtures/<fixtures>/ first. With `mode: matter`
    it runs them through Case Prep instead of asking a question, and the researched issues are graded
    as one answer."""
    documents = [await upload(client, FIXTURES / case["fixtures"] / name) for name in case.get("attachments") or []]
    if case.get("mode") == "matter":
        turn, matter_id = await prepare_matter(client, documents)
        if matter_id and not keep:
            await delete_matter(client, matter_id)  # its case file and index collection go with it
    else:
        conversation: str | None = None
        for question in case.get("setup") or []:       # earlier turns, asked but not graded
            setup_turn = await ask(client, question, conversation)
            conversation = setup_turn.conversation_id
        turn = await ask(client, case["question"].strip(), conversation, documents)
        if turn.conversation_id and not keep:
            await delete_conversation(client, turn.conversation_id)
        if not keep:
            for document_id in documents:
                await delete_document(client, document_id)

    mech = mechanical(case, turn)
    if turn.error:
        judgment = UNJUDGED | {"error": turn.error}
    elif judge is None:
        judgment = UNJUDGED | {"error": "not judged"}
    else:
        judgment = await judge.judge(case, turn.answer, turn.sources)
    scored = score(case, mech, judgment)
    result = {k: case[k] for k in ("id", "suite", "area", "language", "level")} | {
        "question": case["question"].strip(), "expect": case.get("expect") or {},
        "answer": turn.answer, "error": turn.error, "metrics": mech, "judgment": judgment,
        **scored, "why": "" if scored["correct"] else (turn.error or failure(case, mech, scored)),
    }
    transcript = result | {"reference": case["reference"].strip(), "rubric": case["rubric"],
                           "setup": case.get("setup") or [], "sources": turn.sources,
                           "tool_calls": turn.tool_calls, "conversation_id": turn.conversation_id}
    return result, transcript


def stored(run_dir: Path) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """The answers of a finished run, as (case, transcript) pairs - everything needed to grade them
    again without asking the agent anything. Judge changes are checked against the same answers."""
    pairs = []
    for path in sorted((run_dir / "transcripts").glob("*.json")):
        transcript = json.loads(path.read_text())
        case = {k: transcript[k] for k in ("id", "suite", "area", "language", "level", "question",
                                           "reference", "rubric", "setup", "expect")}
        pairs.append((case, transcript))
    if not pairs:
        raise SystemExit(f"no transcripts in {run_dir}")
    return pairs


async def rejudge_case(case: dict[str, Any], transcript: dict[str, Any],
                       judge: Judge) -> tuple[dict[str, Any], dict[str, Any]]:
    mech = transcript["metrics"]
    judgment = (UNJUDGED | {"error": transcript["error"]} if transcript["error"] else
                await judge.judge(case, transcript["answer"], transcript["sources"]))
    scored = score(case, mech, judgment)
    result = {k: transcript[k] for k in ("id", "suite", "area", "language", "level", "question",
                                         "expect", "answer", "error")} | {
        "metrics": mech, "judgment": judgment, **scored,
        "why": "" if scored["correct"] else (transcript["error"] or failure(case, mech, scored))}
    return result, transcript | result


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cases", type=Path, nargs="+", default=[HERE / "cases"],
                    help="case files or directories (default: agent-eval/cases)")
    ap.add_argument("--suite", help="only this suite (exam | behaviour | casefile)")
    ap.add_argument("--only", nargs="+", help="only cases whose id contains one of these fragments")
    ap.add_argument("--server", default=SERVER, help=f"the running assistant (default: {SERVER})")
    ap.add_argument("--judge-url", default=JUDGE_URL, help=f"OpenAI-compatible judge (default: {JUDGE_URL})")
    ap.add_argument("--judge-model", help="default: the first model the judge server lists")
    ap.add_argument("--no-judge", action="store_true", help="mechanical checks only")
    ap.add_argument("--rejudge", type=Path, help="grade a finished run's stored answers again, "
                                                 "without asking the agent anything")
    ap.add_argument("--concurrency", type=int, default=1, help="cases in flight (default: 1)")
    ap.add_argument("--timeout", type=float, default=900, help="seconds for one turn (default: 900)")
    ap.add_argument("--keep-conversations", action="store_true",
                    help="leave the conversations this run creates in the app")
    ap.add_argument("--out", type=Path, default=HERE / "runs", help="where the run directory goes")
    args = ap.parse_args()

    work = stored(args.rejudge) if args.rejudge else [(case, None) for case in load_cases(
        args.cases, args.suite, args.only)]
    if args.suite or args.only:  # --rejudge selects from the stored run with the same filters
        work = [(c, t) for c, t in work
                if (not args.suite or c["suite"] == args.suite)
                and (not args.only or any(f in c["id"] for f in args.only))]
    if not work:
        raise SystemExit("no cases selected")

    judge, model = None, ""
    if not args.no_judge:
        llm = AsyncOpenAI(base_url=args.judge_url, api_key=JUDGE_KEY, timeout=300, max_retries=0)
        model = args.judge_model or (await llm.models.list()).data[0].id
        judge = Judge(llm, model)

    started = datetime.now(timezone.utc)
    semaphore = asyncio.Semaphore(max(1, args.concurrency))
    done = 0

    def progress(result: dict[str, Any]) -> None:
        nonlocal done
        done += 1
        mark = "ok  " if result["correct"] else "FAIL"
        print(f"[{done:2d}/{len(work)}] {mark} {result['id']:<38} "
              f"score {result['score'] if result['score'] is not None else '  -':>5} · "
              f"{result['metrics']['seconds']:>5.0f}s · {result['why'][:60]}", flush=True)

    if args.rejudge:
        if judge is None:
            raise SystemExit("--rejudge needs a judge")
        source = json.loads((args.rejudge / "results.json").read_text())["run"]
        print(f"re-judging {len(work)} answers from {args.rejudge} "
              f"({source['agent']} agent, {source['started']}) · judge={model}\n")

        async def one(case: dict[str, Any], transcript: dict[str, Any] | None):
            async with semaphore:
                result, out = await rejudge_case(case, transcript, judge)
            progress(result)
            return result, out
    else:
        timeout = httpx.Timeout(args.timeout, connect=10.0)
        client = httpx.AsyncClient(base_url=args.server, timeout=timeout)
        try:
            source = await health(client)
        except httpx.HTTPError as e:
            await client.aclose()
            raise SystemExit(f"no assistant at {args.server}: {e}") from e
        print(f"agent={source['agent']} · {source['decisions']:,} decisions · {len(work)} cases · "
              f"judge={model or 'none'}\n")

        async def one(case: dict[str, Any], transcript: dict[str, Any] | None):
            async with semaphore:
                result, out = await run_case(case, client, judge, args.keep_conversations)
            progress(result)
            return result, out

    try:
        pairs = await asyncio.gather(*(one(case, transcript) for case, transcript in work))
    finally:
        if not args.rejudge:
            await client.aclose()
    finished = datetime.now(timezone.utc)

    results = [r for r, _ in pairs]
    run = {
        "started": started.isoformat(timespec="seconds"), "finished": finished.isoformat(timespec="seconds"),
        "seconds": round((finished - started).total_seconds(), 1),
        "server": None if args.rejudge else args.server,
        "agent": source["agent"], "decisions": source["decisions"], "judge_model": model or None,
        "judge_url": args.judge_url if model else None, "concurrency": args.concurrency,
        "rejudged_from": str(args.rejudge) if args.rejudge else None,
        "cases": len(results), "weights": WEIGHTS, "rubric_pass": RUBRIC_PASS,
        "accuracy_pass": ACCURACY_PASS,
    }
    out = args.out / started.strftime("%Y%m%d-%H%M%S")
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    for result, transcript in pairs:
        (out / "transcripts" / f"{result['id']}.json").write_text(
            json.dumps(transcript, indent=2, ensure_ascii=False))
    (out / "results.json").write_text(json.dumps({"run": run, "cases": results}, indent=2, ensure_ascii=False))
    markdown = report.render(run, results)
    (out / "report.md").write_text(markdown)
    latest = args.out / "latest"
    latest.unlink(missing_ok=True)
    latest.symlink_to(out.name)
    print("\n" + markdown)
    print(f"wrote {out}/results.json, {out}/report.md and {len(results)} transcripts")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
