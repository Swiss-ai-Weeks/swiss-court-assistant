"""Score a Whisper checkpoint on the held-out SwissDial test split, overall and per dialect.

Run it once on the stock model before training to get a baseline, then on the fine-tuned one:

    python evaluate.py --model-dir openai/whisper-large-v3 --out results/baseline.json
    python evaluate.py --model-dir runs/whisper-swiss/final --out results/finetuned.json

WER/CER are reported on normalised text (casing and punctuation removed); BLEU is reported
because the target is a translation into High German, not a literal transcript.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import jiwer
import sacrebleu
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from data import WHISPER_SR, has_processor, load_audio, read_manifest


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default="runs/whisper-swiss/final",
                   help="a fine-tuned directory, a LoRA adapter directory, or a Hub id for the baseline")
    p.add_argument("--base-model", default="openai/whisper-large-v3",
                   help="base weights to load a LoRA adapter on top of")
    p.add_argument("--manifest", type=Path, default=Path("manifests/test.jsonl"))
    p.add_argument("--out", type=Path, default=None, help="write the full report here as JSON")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=0, help="score only the first N clips (0 = all)")
    p.add_argument("--beams", type=int, default=1)
    p.add_argument("--save-predictions", type=Path, default=None)
    return p.parse_args()


def load_model(args: argparse.Namespace) -> tuple[torch.nn.Module, WhisperProcessor]:
    model_dir = Path(args.model_dir)
    is_adapter = (model_dir / "adapter_config.json").is_file()
    source = args.base_model if is_adapter else args.model_dir

    model = WhisperForConditionalGeneration.from_pretrained(source, dtype=torch.bfloat16)
    if is_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(model_dir))
        model = model.merge_and_unload()
        print(f"loaded LoRA adapter {model_dir} onto {args.base_model}")

    processor_source = args.model_dir if has_processor(model_dir) else args.base_model
    processor = WhisperProcessor.from_pretrained(processor_source, language="german", task="transcribe")

    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = "de"
    model.generation_config.task = "transcribe"
    model.config.use_cache = True
    return model.eval().cuda(), processor


def main() -> None:
    args = parse_args()
    rows = read_manifest(args.manifest)
    if args.limit:
        rows = rows[:args.limit]
    print(f"scoring {len(rows)} clips from {args.manifest}")

    model, processor = load_model(args)

    def collate(batch: list[dict]) -> tuple[torch.Tensor, list[dict]]:
        features = processor.feature_extractor(
            [load_audio(r["audio"]) for r in batch], sampling_rate=WHISPER_SR,
            return_tensors="pt").input_features
        return features, batch

    loader = DataLoader(rows, batch_size=args.batch_size, num_workers=args.workers,
                        collate_fn=collate, shuffle=False)

    predictions: list[dict] = []
    with torch.inference_mode():
        for features, batch in tqdm(loader, desc="transcribing"):
            generated = model.generate(
                features.to("cuda", dtype=torch.bfloat16), max_new_tokens=225, num_beams=args.beams)
            texts = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)
            for row, text in zip(batch, texts):
                predictions.append({**row, "prediction": text.strip()})

    normalizer = BasicTextNormalizer()

    def score(items: list[dict]) -> dict[str, float]:
        refs = [normalizer(i["text"]) for i in items]
        hyps = [normalizer(i["prediction"]) for i in items]
        pairs = [(r, h) for r, h in zip(refs, hyps) if r.strip()]
        if not pairs:
            return {}
        refs, hyps = [r for r, _ in pairs], [h for _, h in pairs]
        return {
            "clips": len(pairs),
            "wer": round(100 * jiwer.wer(refs, hyps), 2),
            "cer": round(100 * jiwer.cer(refs, hyps), 2),
            "bleu": round(sacrebleu.corpus_bleu(
                [i["prediction"] for i in items], [[i["text"] for i in items]]).score, 2),
        }

    by_dialect: dict[str, list[dict]] = defaultdict(list)
    for item in predictions:
        by_dialect[item["dialect"]].append(item)

    report = {
        "model": args.model_dir,
        "manifest": str(args.manifest),
        "overall": score(predictions),
        "per_dialect": {d: score(items) for d, items in sorted(by_dialect.items())},
    }

    print(f"\n{'':6} {'clips':>6} {'WER':>7} {'CER':>7} {'BLEU':>7}")
    overall = report["overall"]
    print(f"{'ALL':6} {overall['clips']:>6} {overall['wer']:>7} {overall['cer']:>7} {overall['bleu']:>7}")
    for dialect, s in report["per_dialect"].items():
        print(f"{dialect:6} {s['clips']:>6} {s['wer']:>7} {s['cer']:>7} {s['bleu']:>7}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nreport -> {args.out}")
    if args.save_predictions:
        args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
        args.save_predictions.write_text(
            "".join(json.dumps(p, ensure_ascii=False) + "\n" for p in predictions), encoding="utf-8")
        print(f"predictions -> {args.save_predictions}")


if __name__ == "__main__":
    main()
