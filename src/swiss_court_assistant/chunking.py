"""Split decisions into citable passages.

Every chunk keeps char offsets into the decision's ``full_text`` so an answer
can always quote the exact source span, plus the Erwägung numbers (e.g.
"E. 2.1") of the reasoning paragraphs it contains, taken from the upstream
``structure/erwaegungen_paragraphs.parquet``. The Regeste (headnote), when
present, becomes its own chunk.

Usage:
    uv run python -m swiss_court_assistant.chunking data/subset/decisions_50k_seed42.parquet
"""

from __future__ import annotations

import argparse
import re
from multiprocessing import Pool
from pathlib import Path

import polars as pl

PARAGRAPHS = "data/raw/structure/erwaegungen_paragraphs.parquet"
TARGET_CHARS = 1400   # ~350 tokens: small enough to be a precise citation
MAX_CHARS = 2200      # hard cap for a single chunk
OVERLAP_CHARS = 200   # carried into the next chunk so no sentence is orphaned
_SENT_END = re.compile(r"[.:;!?»\"')]\s*$")
_WORD = re.compile(r"\S+")


def line_spans(text: str) -> list[tuple[int, int]]:
    spans, start = [], 0
    for m in re.finditer(r"\n", text):
        spans.append((start, m.end()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def split_text(text: str) -> list[tuple[int, int]]:
    """Greedy line-aligned windows, preferring cuts after a sentence end."""
    lines = line_spans(text)
    chunks: list[tuple[int, int]] = []
    i = 0
    while i < len(lines):
        start = lines[i][0]
        j, best = i, None
        while j < len(lines) and lines[j][1] - start <= MAX_CHARS:
            size = lines[j][1] - start
            if size >= TARGET_CHARS and _SENT_END.search(text[lines[j][0]:lines[j][1]]):
                best = j
                break
            if size >= TARGET_CHARS and best is None:
                best = j  # fallback: first line past the target
            j += 1
        if best is None:
            best = max(j - 1, i)  # a single overlong line, or the end of the text
        end = lines[best][1]
        if end - start > MAX_CHARS:  # one giant line: hard-split it
            end = start + MAX_CHARS
            chunks.append((start, end))
            lines[best] = (end, lines[best][1])
            i = best
            continue
        chunks.append((start, end))
        if best + 1 >= len(lines):
            break
        # step back ~OVERLAP_CHARS worth of whole lines for the next window
        k = best + 1
        while k - 1 > i and end - lines[k - 1][0] <= OVERLAP_CHARS:
            k -= 1
        # inside the overlap, prefer to start right after a sentence end
        for kk in range(k, best + 1):
            if _SENT_END.search(text[lines[kk - 1][0]:lines[kk - 1][1]]):
                k = kk
                break
        i = k
    return [(s, e) for s, e in chunks if text[s:e].strip()]


def locate_paragraphs(text: str, paras: list[tuple[str, str]]) -> list[tuple[int, str]]:
    """Find where each Erwägung paragraph starts in full_text (sequential search).

    Matching is on the first words joined by flexible whitespace, because the
    upstream text and full_text differ in line wrapping.
    """
    found, pos = [], 0
    for e_number, ptext in paras:
        words = _WORD.findall(ptext)[:8]
        if len(words) < 3:
            continue
        m = re.compile(r"\s+".join(map(re.escape, words))).search(text, pos)
        if m:
            found.append((m.start(), e_number))
            pos = m.start() + 1
    return found


def chunk_decision(row: dict) -> list[dict]:
    text = row["full_text"]
    base = {
        "decision_id": row["decision_id"],
        "language": row["language"],
    }
    out = []
    regeste = (row["regeste"] or "").strip()
    # Regeste chunks get negative indices; offsets refer to full_text only.
    for k, (s, e) in enumerate(split_text(regeste) if regeste else []):
        out.append(base | {
            "chunk_index": -1 - k, "section": "regeste", "text": regeste[s:e].strip(),
            "char_start": None, "char_end": None, "erwaegungen": [],
        })
    anchors = locate_paragraphs(text, row["paras"] or [])
    for idx, (s, e) in enumerate(split_text(text)):
        # Erwägungen starting inside this chunk, plus the one still running at its start
        inside = [n for off, n in anchors if s <= off < e]
        running = [n for off, n in anchors if off < s]
        labels = ([running[-1]] if running else []) + inside
        out.append(base | {
            "chunk_index": idx, "section": "erwaegung" if labels else "body",
            "text": text[s:e].strip(), "char_start": s, "char_end": e,
            "erwaegungen": labels,
        })
    return out


def _chunk_batch(rows: list[dict]) -> list[dict]:
    return [c for r in rows for c in chunk_decision(r)]


def build_chunks(decisions_path: str | Path, workers: int = 64) -> pl.DataFrame:
    docs = pl.read_parquet(
        decisions_path, columns=["decision_id", "language", "full_text", "regeste"]
    )
    paras = (
        pl.scan_parquet(PARAGRAPHS)
        .join(docs.lazy().select("decision_id"), on="decision_id", how="semi")
        .with_row_index("_ord")  # keep document order for the sequential search
        .group_by("decision_id")
        .agg(pl.struct("e_number", "text").sort_by("_ord").alias("paras"))
        .collect()
    )
    docs = docs.join(paras, on="decision_id", how="left").with_columns(
        pl.col("paras").list.eval(pl.concat_list(
            pl.element().struct.field("e_number"), pl.element().struct.field("text")
        ))
    )
    rows = docs.to_dicts()
    batches = [rows[i:i + 200] for i in range(0, len(rows), 200)]
    with Pool(workers) as pool:
        chunks = [c for part in pool.imap(_chunk_batch, batches) for c in part]
    df = pl.DataFrame(chunks, schema_overrides={"char_start": pl.Int32, "char_end": pl.Int32})
    return df.with_columns(
        chunk_id=pl.format("{}#{}", "decision_id", "chunk_index")
    ).select("chunk_id", pl.exclude("chunk_id"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("decisions", help="subset parquet produced by sampling.py")
    args = ap.parse_args()
    src = Path(args.decisions)
    chunks = build_chunks(src)
    out = src.with_name(src.stem + ".chunks.parquet")
    chunks.write_parquet(out, compression="zstd")
    lens = chunks["text"].str.len_chars()
    print(f"wrote {out}: {chunks.height:,} chunks from {chunks['decision_id'].n_unique():,} decisions")
    print(chunks["section"].value_counts(sort=True))
    print("chars p5/p50/p95:", [int(lens.quantile(q)) for q in (0.05, 0.5, 0.95)])


if __name__ == "__main__":
    main()
