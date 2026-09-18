"""Merge the LoRA adapter into the base weights and write a standalone Whisper model.

The result is an ordinary Whisper checkpoint: anything that loads whisper-large-v3 loads this,
including faster-whisper once converted with ct2-transformers-converter.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from data import has_processor


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", type=Path, default=Path("runs/whisper-swiss/final"))
    p.add_argument("--base-model", default="openai/whisper-large-v3")
    p.add_argument("--out-dir", type=Path, default=Path("export/whisper-swiss-de"))
    p.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="float16",
                   help="float16 is what most Whisper servers expect")
    p.add_argument("--ctranslate2", action="store_true",
                   help="also emit a CTranslate2 model for faster-whisper (needs ctranslate2 installed)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    dtype = getattr(torch, args.dtype)
    is_adapter = (args.model_dir / "adapter_config.json").is_file()

    model = WhisperForConditionalGeneration.from_pretrained(
        args.base_model if is_adapter else args.model_dir, dtype=dtype)
    if is_adapter:
        from peft import PeftModel

        print(f"merging {args.model_dir} into {args.base_model}")
        model = PeftModel.from_pretrained(model, str(args.model_dir)).merge_and_unload()
    else:
        print(f"copying full fine-tune from {args.model_dir}")

    model.generation_config.forced_decoder_ids = None
    model.generation_config.language = "de"
    model.generation_config.task = "transcribe"

    processor_source = args.model_dir if has_processor(args.model_dir) else args.base_model
    processor = WhisperProcessor.from_pretrained(processor_source, language="german", task="transcribe")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out_dir)
    processor.save_pretrained(args.out_dir)
    print(f"standalone model -> {args.out_dir}")

    if args.ctranslate2:
        if shutil.which("ct2-transformers-converter") is None:
            print("! ctranslate2 is not installed; skipping (uv pip install ctranslate2)")
            return
        ct2_dir = args.out_dir.parent / f"{args.out_dir.name}-ct2"
        extras = [n for n in ("tokenizer.json", "processor_config.json", "preprocessor_config.json",
                              "tokenizer_config.json") if (args.out_dir / n).is_file()]
        subprocess.run([
            "ct2-transformers-converter", "--model", str(args.out_dir), "--output_dir", str(ct2_dir),
            "--copy_files", *extras, "--quantization", "float16", "--force"], check=True)
        print(f"faster-whisper model -> {ct2_dir}")


if __name__ == "__main__":
    main()
