"""Build train/val/test manifests from the extracted SwissDial corpus.

Each SwissDial sentence exists once per dialect, so the split is made over *sentence ids*,
never over clips: putting the Zurich reading of a sentence in train and the Bernese reading
of the same sentence in test would let the model score on text it has already memorised.

The target text is the High German reference ("de"), which is what makes the fine-tuned
model a Swiss German -> High German speech translator rather than a dialect transcriber.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

DIALECTS = ["ag", "be", "bs", "gr", "lu", "sg", "vs", "zh"]
CLIP_RE = re.compile(r"ch_(?P<dialect>[a-z]{2})_(?P<sentence_id>\d+)\.wav$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, default=Path("../data/asr/data1.1"),
                   help="extracted SwissDial 1.1 directory (holds the dialect folders and the sentence JSONs)")
    p.add_argument("--out-dir", type=Path, default=Path("manifests"))
    p.add_argument("--sentences", choices=["transcribed", "numerics"], default="transcribed",
                   help="'transcribed' spells every number out in words (matches what is spoken); "
                        "'numerics' keeps digits where the original had them")
    p.add_argument("--val-sentences", type=int, default=200, help="sentence ids held out for validation")
    p.add_argument("--test-sentences", type=int, default=300, help="sentence ids held out for the test set")
    p.add_argument("--cap-per-dialect", type=int, default=0,
                   help="max training clips per dialect (0 = no cap). SwissDial 1.1 has ~10.5k Grisons "
                        "clips against ~2.7k for the others; a cap of 3000 gives a dialect-balanced set")
    p.add_argument("--min-duration", type=float, default=0.5, help="drop clips shorter than this (seconds)")
    p.add_argument("--max-duration", type=float, default=30.0, help="drop clips longer than Whisper's 30 s window")
    p.add_argument("--workers", type=int, default=32, help="threads used to read wav headers")
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def load_sentences(dataset_dir: Path, variant: str) -> dict[int, dict]:
    path = dataset_dir / f"sentences_ch_de_{variant}.json"
    if not path.is_file():
        sys.exit(f"transcript file not found: {path}\nDid you extract data1.1.tar.gz into {dataset_dir}?")
    return {int(s["id"]): s for s in json.loads(path.read_text(encoding="utf-8")) if "id" in s}


def collect_clips(dataset_dir: Path, sentences: dict[int, dict], workers: int) -> list[dict]:
    """One record per wav file, with its High German target and its measured duration."""
    found: list[tuple[Path, int, str]] = []
    for dialect in DIALECTS:
        folder = dataset_dir / dialect
        if not folder.is_dir():
            print(f"  ! no folder for dialect {dialect}, skipping", file=sys.stderr)
            continue
        for wav in folder.glob(f"ch_{dialect}_*.wav"):
            match = CLIP_RE.search(wav.name)
            if match:
                found.append((wav, int(match.group("sentence_id")), dialect))

    def probe(item: tuple[Path, int, str]) -> dict | None:
        wav, sentence_id, dialect = item
        sentence = sentences.get(sentence_id)
        if sentence is None:
            return None
        text = str(sentence.get("de", "")).strip()
        if not text:
            return None
        try:
            info = sf.info(wav)  # header only, no decoding
        except Exception:  # noqa: BLE001 - a corrupt clip should not abort the whole run
            return None
        return {
            "audio": str(wav.resolve()),
            "text": text,
            "dialect": dialect,
            "dialect_text": str(sentence.get(f"ch_{dialect}", "")).strip(),
            "sentence_id": sentence_id,
            "duration": round(info.frames / info.samplerate, 3),
            "topic": sentence.get("thema", ""),
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        records = list(tqdm(pool.map(probe, found), total=len(found), desc="reading wav headers"))
    return [r for r in records if r is not None]


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    print(f"dataset: {dataset_dir}")

    sentences = load_sentences(dataset_dir, args.sentences)
    print(f"sentences: {len(sentences)} ({args.sentences})")

    records = collect_clips(dataset_dir, sentences, args.workers)
    print(f"clips with a High German target: {len(records)}")

    kept, dropped = [], Counter()
    for r in records:
        if r["duration"] < args.min_duration:
            dropped["too short"] += 1
        elif r["duration"] > args.max_duration:
            dropped["longer than 30 s"] += 1
        else:
            kept.append(r)
    for reason, n in dropped.items():
        print(f"  dropped {n} clips: {reason}")

    by_sentence: dict[int, list[dict]] = defaultdict(list)
    for r in kept:
        by_sentence[r["sentence_id"]].append(r)

    # Hold out only sentences every dialect recorded, so val/test stay dialect-balanced.
    full_coverage = sorted(sid for sid, clips in by_sentence.items() if len(clips) == len(DIALECTS))
    needed = args.val_sentences + args.test_sentences
    if len(full_coverage) < needed:
        sys.exit(f"only {len(full_coverage)} sentences are recorded in all {len(DIALECTS)} dialects, "
                 f"but {needed} were requested for val+test")

    rng = random.Random(args.seed)
    rng.shuffle(full_coverage)
    test_ids = set(full_coverage[:args.test_sentences])
    val_ids = set(full_coverage[args.test_sentences:needed])

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for sid, clips in by_sentence.items():
        name = "test" if sid in test_ids else "val" if sid in val_ids else "train"
        splits[name].extend(clips)

    if args.cap_per_dialect:
        per_dialect: dict[str, list[dict]] = defaultdict(list)
        for r in splits["train"]:
            per_dialect[r["dialect"]].append(r)
        capped = []
        for dialect, clips in per_dialect.items():
            clips.sort(key=lambda r: r["sentence_id"])  # deterministic before shuffling
            rng.shuffle(clips)
            capped.extend(clips[:args.cap_per_dialect])
        splits["train"] = capped

    for name in splits:
        splits[name].sort(key=lambda r: (r["sentence_id"], r["dialect"]))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = args.out_dir / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
        hours = sum(r["duration"] for r in rows) / 3600
        per_dialect = Counter(r["dialect"] for r in rows)
        print(f"\n{name}: {len(rows)} clips, {hours:.1f} h -> {path}")
        print("  " + "  ".join(f"{d}={per_dialect.get(d, 0)}" for d in DIALECTS))

    overlap = {n: {r["sentence_id"] for r in rows} for n, rows in splits.items()}
    assert not overlap["train"] & overlap["test"], "sentence leaked between train and test"
    assert not overlap["train"] & overlap["val"], "sentence leaked between train and val"
    print("\nno sentence id appears in more than one split.")


if __name__ == "__main__":
    main()
