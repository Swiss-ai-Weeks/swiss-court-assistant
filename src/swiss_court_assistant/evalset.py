"""Generate a multilingual retrieval test set with a local LLM.

For a set of gold decisions balanced over language x period, an LLM reads one
reasoning passage (plus the Regeste) and writes a question that passage
answers, in the decision's own language. Each question is then translated
into de / fr / it / en (rm optional), so every gold decision is queried from
every language (cross-lingual retrieval). Questions are verified by the LLM
against the passage + headnote and regenerated (up to 3 attempts) if they leak
docket numbers, copy the passage, or are judged not answered by it.

Relevance is known-item: the gold decision (and gold chunk). Other decisions
may also be relevant, so the metrics are lower bounds.

Requires an OpenAI-compatible server (vLLM), e.g.:
    docker run --gpus '"device=0"' -p 8000:8000 \
      -v ~/.cache/huggingface:/root/.cache/huggingface \
      vllm/vllm-openai --model Qwen/Qwen3-30B-A3B-Instruct-2507 --max-model-len 16384

Usage:
    uv run python -m swiss_court_assistant.evalset data/subset/decisions_50k_seed42.parquet
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from pathlib import Path

import polars as pl
from openai import AsyncOpenAI

from swiss_court_assistant.statutes import GLOSSARY, STATUTE_GLOSSARY

MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen3-30B-A3B-Instruct-2507")
BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
# Romansh is supported but off by default: in the pilot, Qwen3-30B's "Romansh"
# was an Italo-Lombard mix, not Rumantsch Grischun - rm needs a better translator.
QUERY_LANGS = ["de", "fr", "it", "en"]
LANG_NAMES = {"de": "German", "fr": "French", "it": "Italian",
              "rm": "Romansh (Rumantsch Grischun)", "en": "English"}
CELLS = ["language", "period"]
STYLES = {
    "situation": "a layperson describing their concrete situation and asking what the law says "
                 "(no legal jargon, no article numbers)",
    "legal_issue": "a lawyer asking a precise legal question about the rule or its interpretation "
                   "(may mention the statute, but not this case)",
}
_DOCKET = re.compile(r"\b\d+[A-Z]?[_.]\d+/\d{4}\b|\bBGE\s+\d+|\bATF\s+\d+|\bDTF\s+\d+")


# ── gold selection ──────────────────────────────────────────────────────
def pick_gold(subset_path: Path, per_cell: int, seed: int) -> pl.DataFrame:
    """One gold passage per decision; `per_cell` decisions per language x period."""
    chunks_path = subset_path.with_name(subset_path.stem + ".chunks.parquet")
    docs = pl.read_parquet(
        subset_path,
        columns=["decision_id", "language", "period", "branch", "jurisdiction", "regeste"],
    ).filter(pl.col("period") != "unknown")
    ch = pl.read_parquet(chunks_path, columns=["chunk_id", "decision_id", "section", "chunk_index", "text"])
    ch = ch.with_columns(n=pl.len().over("decision_id"))
    eligible = ch.filter(
        (pl.col("section") != "regeste")
        & (pl.col("text").str.len_chars() >= 800)
        # skip the rubrum (court, judges, parties) at the top of every decision
        & (pl.col("chunk_index") >= pl.when(pl.col("n") >= 5).then(2).otherwise(1))
    ).with_columns(
        pref=(pl.col("section") == "erwaegung"),
        h=pl.col("chunk_id").hash(seed),
    )
    gold = (
        eligible.sort(["decision_id", "pref", "h"], descending=[False, True, False])
        .group_by("decision_id").first()
        .select("decision_id", gold_chunk_id="chunk_id", passage="text")
    )
    return (
        docs.join(gold, on="decision_id")
        .with_columns(_r=pl.col("decision_id").hash(seed).rank("ordinal").over(CELLS))
        .filter(pl.col("_r") <= per_cell)
        .drop("_r")
        .with_columns(
            style=pl.when(pl.col("decision_id").hash(seed + 1) % 2 == 0)
            .then(pl.lit("situation")).otherwise(pl.lit("legal_issue"))
        )
    )


# ── LLM calls ───────────────────────────────────────────────────────────
async def chat_json(client: AsyncOpenAI, sem: asyncio.Semaphore, prompt: str, schema: dict,
                    temperature: float = 0.3) -> dict | None:
    async with sem:
        try:
            r = await client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=1024,
                response_format={"type": "json_schema", "json_schema": {"name": "out", "schema": schema}},
            )
            return json.loads(r.choices[0].message.content)
        except Exception as e:  # one bad item must not kill a long batch
            print("LLM error:", type(e).__name__, str(e)[:200])
            return None


def gen_prompt(row: dict) -> str:
    regeste = (row["regeste"] or "")[:1500]
    lang = LANG_NAMES[row["language"]]
    return f"""You are building a test set for a Swiss case-law search engine.

Below is a passage from a Swiss court decision{" and its headnote (Regeste)" if regeste else ""}.
Write ONE search question in {lang}, phrased as if written by {STYLES[row["style"]]}.

Rules:
- The passage must contain the answer or the decisive legal reasoning.
- Do NOT mention the court, docket number, date, party names, or cited decisions.
- Do NOT copy phrases of more than four consecutive words from the passage; paraphrase.
- The question must make sense on its own, to someone who has never seen this decision.
- 1-3 sentences.
{f"{chr(10)}HEADNOTE:{chr(10)}{regeste}{chr(10)}" if regeste else ""}
PASSAGE:
{row["passage"]}"""


GEN_SCHEMA = {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}
def tr_schema(langs: list[str]) -> dict:
    return {"type": "object", "properties": {l: {"type": "string"} for l in langs}, "required": langs}


VERIFY_SCHEMA = {"type": "object", "properties": {"supported": {"type": "boolean"}}, "required": ["supported"]}


_GLOSSARY = GLOSSARY


def normalize_statutes(text: str, lang: str) -> str:
    """Rewrite statute abbreviations into the official form of `lang`.

    The prompt alone doesn't stop the LLM from leaving e.g. "LAVS" in a German
    translation, which breaks lexical matching. English (and Romansh) use the
    German forms, as Swiss English legal texts usually do.
    """
    target = lang if lang in ("de", "fr", "it") else "de"
    for row in _GLOSSARY:
        want = row[target][0]
        # other languages' forms and the target's own old aliases ("AuG" -> "AIG")
        for f in {f for forms in row.values() for f in forms} - {want}:
            text = re.sub(rf"(?<![\w.]){re.escape(f)}(?!\w)", want, text)
    return text


def tr_prompt(question: str, langs: list[str]) -> str:
    targets = ", ".join(f'"{l}": {LANG_NAMES[l]}' for l in langs)
    return f"""Translate this legal search question into each language. Keep the meaning and
register; use the standard Swiss legal terminology of each language. Keys: {targets}.
Statute abbreviations must use the official form of the target language (for English,
keep the German abbreviation). Official equivalents:
{STATUTE_GLOSSARY}

QUESTION: {question}"""


def verify_prompt(question: str, row: dict) -> str:
    regeste = (row["regeste"] or "")[:1500]
    return f"""You are checking a test set for a Swiss case-law search engine. Would a lawyer
researching the question below want to find this decision, because the passage (read
together with the headnote, if given) answers it or contains the decisive legal
reasoning? Answer false if the question asks about something this decision does not address.

QUESTION: {question}
{f"{chr(10)}HEADNOTE:{chr(10)}{regeste}{chr(10)}" if regeste else ""}
PASSAGE:
{row["passage"]}"""


def copies_passage(question: str, passage: str, n: int = 6) -> bool:
    words = lambda s: re.findall(r"\w+", s.lower())
    q, p = words(question), " ".join(words(passage))
    return any(" ".join(q[i:i + n]) in p for i in range(len(q) - n + 1))


async def gen_one(client: AsyncOpenAI, sem: asyncio.Semaphore, row: dict,
                  attempts: int = 3) -> tuple[dict | None, list[str]]:
    """Generate, leak-filter and verify one question; retry hotter on failure."""
    fails = []
    for a in range(attempts):
        g = await chat_json(client, sem, gen_prompt(row), GEN_SCHEMA, temperature=0.3 + 0.3 * a)
        q = (g or {}).get("question", "").strip()
        if not q:
            fails.append("empty")
        elif _DOCKET.search(q):
            fails.append("docket")
        elif copies_passage(q, row["passage"]):
            fails.append("copy")
        else:
            v = await chat_json(client, sem, verify_prompt(q, row), VERIFY_SCHEMA, temperature=0.0)
            if v and v.get("supported"):
                return row | {"question": q, "attempts": a + 1}, fails
            fails.append("unsupported")
    return None, fails


async def build(gold: pl.DataFrame, concurrency: int, langs: list[str]) -> pl.DataFrame:
    client = AsyncOpenAI(base_url=BASE_URL, api_key="local")
    sem = asyncio.Semaphore(concurrency)
    rows = gold.to_dicts()

    results = await asyncio.gather(*(gen_one(client, sem, r) for r in rows))
    kept = [r for r, _ in results if r]
    reasons: dict[str, int] = {}
    for _, fails in results:
        for f in fails:
            reasons[f] = reasons.get(f, 0) + 1
    print(f"kept {len(kept)}/{len(rows)} gold; failed attempts by reason: {reasons}")

    trs = await asyncio.gather(*(chat_json(client, sem, tr_prompt(r["question"], langs), tr_schema(langs)) for r in kept))
    out = []
    for r, t in zip(kept, trs):
        if not t:
            continue
        t[r["language"]] = r["question"]  # keep the original, not a round-trip
        for ql in langs:
            if t.get(ql, "").strip():
                out.append({
                    "qid": f"{r['decision_id']}::{ql}", "query_lang": ql,
                    "query": normalize_statutes(t[ql].strip(), ql),
                    "translated": ql != r["language"], "style": r["style"],
                    "gold_decision_id": r["decision_id"], "gold_chunk_id": r["gold_chunk_id"],
                    "doc_language": r["language"], "period": r["period"],
                    "branch": r["branch"], "jurisdiction": r["jurisdiction"],
                })
    return pl.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("subset", type=Path)
    ap.add_argument("--per-cell", type=int, default=50, help="gold decisions per language x period")
    ap.add_argument("--langs", nargs="+", default=QUERY_LANGS, choices=list(LANG_NAMES),
                    help="query languages (add rm only with a translator that really handles Romansh)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--out", type=Path, default=Path("data/eval/queries.parquet"))
    args = ap.parse_args()

    gold = pick_gold(args.subset, args.per_cell, args.seed)
    print(f"{gold.height} gold decisions")
    print(gold.group_by(CELLS).len().pivot(on="language", index="period", values="len").sort("period"))
    queries = asyncio.run(build(gold, args.concurrency, args.langs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    queries.write_parquet(args.out)
    print(f"wrote {args.out}: {queries.height} queries, {queries['gold_decision_id'].n_unique()} gold decisions")
    print(queries.group_by("query_lang").len().sort("query_lang"))


if __name__ == "__main__":
    main()
