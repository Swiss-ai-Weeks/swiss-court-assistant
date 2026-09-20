"""Build eval cases from LEXam, the benchmark of 340 Swiss and international law exams.

    uv run python agent-eval/lexam.py                       # 60 open + 100 multiple-choice, Swiss, test split
    uv run python agent-eval/lexam.py --split dev --open 40 # tune on dev, report on test
    uv run python agent-eval/run.py --cases agent-eval/cases/lexam

LEXam (Fan et al., ICLR 2026, CC BY 4.0, https://huggingface.co/datasets/LEXam-Benchmark/LEXam) has
two kinds of question, and they measure different things:

* **Open questions** are exam questions — usually a fact pattern ending in "Wie beurteilen Sie die
  Rechtslage?" — with the examiners' marking scheme as the answer. The marking scheme becomes the
  case's `reference`, and its gradable units become the `rubric`: split after the examiners' own
  point markers ("[0.5 Punkte]") where there are any, else on bullets, else on numbered items, else
  into runs of sentences. `rubric_source` records which, so a bad split can be found and fixed.
* **Multiple-choice questions** have one correct option, so they are scored exactly: no judge, no
  judge noise. The options are appended to the question with an instruction to end on
  "Answer: X"; `run.py` reads the letter back.

Only Swiss questions by default: the assistant searches Swiss case law and statutes, and the
international and "generic" ones (a third of the open questions) are not what it is for. Questions
that share a fact pattern (two sub-questions of one exam problem) are sampled at most once, and the
sample is stratified by legal area and language in proportion to the Swiss subset.

The dataset is downloaded once into the Hugging Face cache; running the eval needs no network.
Numbers are not comparable with the LEXam leaderboard for open questions (a different judge and a
sample), but multiple-choice accuracy is scored the same way as there.
"""

from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import polars as pl
import yaml
from huggingface_hub import snapshot_download

HERE = Path(__file__).resolve().parent
REPO = "LEXam-Benchmark/LEXam"
LETTERS = "ABCDEFGH"
MAX_POINTS, MIN_POINT_CHARS, POINT_CHARS = 8, 20, 350  # POINT_CHARS: target size of a sentence-run point

INSTRUCTION = {
    "de": "Wählen Sie eine der Optionen {first}–{last}. Schliessen Sie Ihre Antwort mit der Zeile "
          "«Antwort: X» ab, wobei X der Buchstabe der gewählten Option ist.",
    "en": "Choose one of the options {first}–{last}. End your answer with the line \"Answer: X\", "
          "where X is the letter of the option you choose.",
}

# the examiners' point allocations: "[0.5 Punkte]", "(1 Punkt)", "(max. 3 Punkte)", "[2 points]"
_POINTS = re.compile(r"[\[(]\s*(?:max(?:imal)?\.?\s*)?\d+(?:[.,]\d+)?\s*(?:Punkte?|Pkt\.?|points?)\s*[\])]", re.I)
_BULLET = re.compile(r"(?:^|\s)[-•]\s+(?=\S)")  # not "–": that is the German dash in running text
_MONTHS = ("Januar|Februar|März|April|Mai|Juni|Juli|August|September|Oktober|November|Dezember|"
           "January|February|March|June|July|October|December")
_NUMBERED = re.compile(rf"(?:^|\s)(?:\d{{1,2}}|[a-h]|[ivx]{{1,4}})[.)]\s+(?!(?:{_MONTHS})\b)(?=[A-ZÄÖÜ])")
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-ZÄÖÜ(])")
# a full stop that does not end a sentence: "Art. 61", "2. Aufl.", "vgl. BGE", "LOCHER/GÄCHTER, § 65 Rz."
_ABBREVIATION = re.compile(r"(?:\b(?:Art|Abs|lit|let|Ziff|Rz|Aufl|vgl|bzw|insb|ca|resp|Nr|No|Bd|Hrsg|ff|al|ch|cf|"
                           r"S|N|E|Erw|consid|p|pp|z\.B|u\.a|d\.h|i\.V\.m|m\.w\.H)\.|\b\d{1,3}\.|\b[A-ZÄÖÜ]\.)$")


def _clean(text: str) -> str:
    """One line per paragraph. The answers come from PDFs, with line breaks inside words
    ("Kausalzusammen\\nhang"); joining them with a space leaves them readable to the judge, and guessing
    which breaks were hyphens would join real word boundaries too."""
    return re.sub(r"[ \t]*\n[ \t]*", " ", re.sub(r"[ \t]+", " ", text)).strip()


def _pieces(answer: str) -> tuple[list[str], str]:
    text = _clean(answer)
    if len(_POINTS.findall(text)) >= 2:  # split after each marker: it closes the unit it scores
        cuts = [m.end() for m in _POINTS.finditer(text)]
        return [text[a:b] for a, b in zip([0, *cuts], [*cuts, len(text)])], "points"
    for pattern, source in ((_BULLET, "bullets"), (_NUMBERED, "numbered")):
        starts = [m.start() for m in pattern.finditer(text)]
        if len(starts) >= 2:
            return [text[a:b] for a, b in zip([0, *starts], [*starts, len(text)])], source
    sentences: list[str] = []
    for s in _SENTENCE.split(text):
        if sentences and _ABBREVIATION.search(sentences[-1]):
            sentences[-1] += " " + s
        else:
            sentences.append(s)
    runs: list[str] = []  # runs of whole sentences, each about POINT_CHARS long
    for s in sentences:
        if runs and len(runs[-1]) + len(s) < POINT_CHARS:
            runs[-1] += " " + s
        else:
            runs.append(s)
    return runs, "sentences"


def rubric(answer: str) -> tuple[list[str], str]:
    """The marking scheme as at most MAX_POINTS gradable points, and how it was split."""
    pieces, source = _pieces(answer)
    points = [p for p in (re.sub(r"^[-•\s]+", "", p).strip() for p in pieces) if p]
    merged: list[str] = []
    for p in points:  # fragments too short to grade on their own belong to their neighbour
        if merged and len(p) < MIN_POINT_CHARS:
            merged[-1] += " " + p
        elif merged and len(merged[-1]) < MIN_POINT_CHARS:
            merged[-1] += " " + p
        else:
            merged.append(p)
    while len(merged) > MAX_POINTS:  # join the shortest adjacent pair until few enough
        i = min(range(len(merged) - 1), key=lambda j: len(merged[j]) + len(merged[j + 1]))
        merged[i:i + 2] = [merged[i] + " " + merged[i + 1]]
    return merged, source  # never truncated: a point cut short cannot be graded


def _stratified(df: pl.DataFrame, n: int, seed: int) -> pl.DataFrame:
    """n rows, spread over area x language in proportion to the frame."""
    if n >= df.height:
        return df
    groups = df.partition_by(["area", "language"], maintain_order=True)
    quotas = [max(1, round(n * g.height / df.height)) for g in groups]
    while sum(quotas) > n:  # rounding up every stratum can overshoot
        quotas[quotas.index(max(quotas))] -= 1
    parts = [g.sample(min(q, g.height), seed=seed) for g, q in zip(groups, quotas)]
    return pl.concat(parts).sample(fraction=1.0, shuffle=True, seed=seed)


def _meta(row: dict) -> dict:
    return {"dataset": REPO, "id": row["id"], "course": row["course"], "year": row["year"],
            "jurisdiction": row["jurisdiction"]}


def open_cases(df: pl.DataFrame, n: int, seed: int) -> list[dict]:
    # one sub-question per exam problem: they repeat the same fact pattern
    df = (df.with_columns(pattern=pl.col("question").str.slice(0, 200))
          .sample(fraction=1.0, shuffle=True, seed=seed).unique("pattern", keep="first", maintain_order=True))
    cases = []
    for row in _stratified(df, n, seed).iter_rows(named=True):
        points, source = rubric(row["answer"])
        cases.append({
            "id": f"lexam-open-{row['id'][:8]}", "suite": "lexam_open", "area": row["area"],
            "language": row["language"], "level": "exam", "question": row["question"].strip(),
            "reference": _clean(row["answer"]), "rubric": points, "rubric_source": source,
            "expect": {"abstain": False, "cites": True}, "source": _meta(row),
        })
    return cases


def mcq_cases(df: pl.DataFrame, n: int, seed: int) -> list[dict]:
    cases = []
    for row in _stratified(df, n, seed).iter_rows(named=True):
        choices = ast.literal_eval(row["choices"])
        letters = LETTERS[:len(choices)]
        options = "\n".join(f"{letter}) {choice}" for letter, choice in zip(letters, choices))
        lang = row["language"] if row["language"] in INSTRUCTION else "en"
        instruction = INSTRUCTION[lang].format(first=letters[0], last=letters[-1])
        cases.append({
            "id": f"lexam-mcq-{row['id'][:8]}", "suite": "lexam_mcq", "area": row["area"],
            "language": row["language"], "level": "exam",
            "question": f"{row['question'].strip()}\n\n{options}\n\n{instruction}",
            "choices": choices, "gold": letters[row["gold"]],
            "expect": {"abstain": False, "cites": "optional"},
            "source": _meta(row) | {"n_statements": row["n_statements"],
                                    "negative_question": row["negative_question"]},
        })
    return cases


class _Dumper(yaml.SafeDumper):
    pass


_Dumper.add_representer(str, lambda d, s: d.represent_scalar(
    "tag:yaml.org,2002:str", s, style="|" if "\n" in s else None))


def write(cases: list[dict], path: Path, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.dump(cases, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=100)
    path.write_text(header + body)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--open", type=int, default=60, help="open questions (default 60)")
    ap.add_argument("--mcq", type=int, default=100, help="multiple-choice questions (default 100)")
    ap.add_argument("--choices", type=int, default=4, choices=[4, 8], help="options per MCQ (4 or 8)")
    ap.add_argument("--split", default="test", choices=["test", "dev"],
                    help="open questions: dev to tune on, test to report (MCQs only have test)")
    ap.add_argument("--jurisdiction", nargs="+", default=["Swiss"],
                    help="Swiss, International, Generic (default: Swiss)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "cases" / "lexam")
    args = ap.parse_args()

    root = Path(snapshot_download(REPO, repo_type="dataset", allow_patterns=["*.parquet"]))
    keep = pl.col("jurisdiction").is_in(args.jurisdiction)
    header = (f"# Generated by agent-eval/lexam.py from {REPO} (CC BY 4.0) — do not edit by hand;\n"
              f"# rebuild instead. jurisdiction={','.join(args.jurisdiction)} seed={args.seed}\n")
    if args.open:
        df = pl.read_parquet(root / "open_question" / f"{args.split}-00000-of-00001.parquet").filter(keep)
        cases = open_cases(df, args.open, args.seed)
        write(cases, args.out / "open.yaml", header + f"# open questions, {args.split} split\n")
        sources = pl.Series([c["rubric_source"] for c in cases]).value_counts().rows()
        print(f"{len(cases)} open questions ({args.split}) → {args.out / 'open.yaml'}; rubric split by {dict(sources)}")
    if args.mcq:
        df = pl.read_parquet(root / f"mcq_{args.choices}_choices" / "test-00000-of-00001.parquet").filter(keep)
        cases = mcq_cases(df, args.mcq, args.seed)
        write(cases, args.out / "mcq.yaml", header + f"# multiple choice, {args.choices} options, test split\n")
        print(f"{len(cases)} multiple-choice questions → {args.out / 'mcq.yaml'}")


if __name__ == "__main__":
    main()
