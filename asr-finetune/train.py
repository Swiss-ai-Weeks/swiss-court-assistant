"""Fine-tune Whisper to turn Swiss German speech into High German text.

The audio is Swiss German dialect, the target is the Standard German reference sentence, so the
model learns speech translation while running as an ordinary German transcription task
(language=de, task=transcribe) - which is what lets it drop into a normal Whisper serving path.

  LoRA (default)  ~25 GB of VRAM, trains adapters only, exports by merging into the base model.
  Full           ~55 GB of VRAM, updates all 1.55B parameters, usually a point or two better.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jiwer
import numpy as np
import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

from data import SpeechSeq2SeqCollator, SwissDialDataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="openai/whisper-large-v3")
    p.add_argument("--train-manifest", type=Path, default=Path("manifests/train.jsonl"))
    p.add_argument("--val-manifest", type=Path, default=Path("manifests/val.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("runs/whisper-swiss"))
    p.add_argument("--method", choices=["lora", "full"], default="lora")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max-steps", type=int, default=-1, help="overrides --epochs when > 0")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--eval-batch-size", type=int, default=8)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=None, help="default: 1e-3 for LoRA, 1e-5 for full")
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", default="q_proj,v_proj")
    p.add_argument("--workers", type=int, default=12, help="dataloader workers (they do the resampling)")
    p.add_argument("--eval-steps", type=int, default=500)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--logging-steps", type=int, default=25)
    p.add_argument("--eval-wer", action=argparse.BooleanOptionalAction, default=True,
                   help="decode the validation set each eval to report WER (slower than loss alone)")
    p.add_argument("--no-gradient-checkpointing", dest="gradient_checkpointing",
                   action="store_false", default=True)
    p.add_argument("--resume", action="store_true", help="resume from the last checkpoint in --out-dir")
    p.add_argument("--seed", type=int, default=13)
    return p.parse_args()


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    # fp32 on purpose, and stated explicitly: whisper-large-v3 ships a float16 checkpoint and
    # transformers would otherwise honour that dtype. Trainer's bf16=True supplies the
    # mixed-precision autocast; half-precision *weights* would mismatch the fp32 input features
    # at the encoder convolutions and give Adam nothing stable to update.
    model = WhisperForConditionalGeneration.from_pretrained(args.model, dtype=torch.float32)

    # Let the decoder prompt come from the generation config rather than a hard-coded prefix.
    if hasattr(model.config, "forced_decoder_ids"):
        model.config.forced_decoder_ids = None
    model.generation_config.forced_decoder_ids = None
    model.generation_config.suppress_tokens = []
    model.generation_config.language = "de"
    model.generation_config.task = "transcribe"
    model.config.use_cache = not args.gradient_checkpointing

    if args.method == "lora":
        from peft import LoraConfig, get_peft_model

        if args.gradient_checkpointing:
            # Without this the frozen bf16 base produces activations with no grad history.
            model.enable_input_require_grads()
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=[t.strip() for t in args.lora_targets.split(",") if t.strip()],
            bias="none"))
        model.print_trainable_parameters()
    return model


def main() -> None:
    args = parse_args()
    lr = args.lr if args.lr is not None else (1e-3 if args.method == "lora" else 1e-5)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "train_args.json").write_text(
        json.dumps({**vars(args), "lr": lr}, indent=2, default=str), encoding="utf-8")

    processor = WhisperProcessor.from_pretrained(args.model, language="german", task="transcribe")
    model = build_model(args)

    train_set = SwissDialDataset(args.train_manifest, processor)
    val_set = SwissDialDataset(args.val_manifest, processor)
    print(f"train clips: {len(train_set)}   val clips: {len(val_set)}")

    collator = SpeechSeq2SeqCollator(
        processor=processor,
        decoder_start_token_id=processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>"))

    normalizer = BasicTextNormalizer()

    def compute_metrics(pred) -> dict[str, float]:
        label_ids = np.where(pred.label_ids != -100, pred.label_ids, processor.tokenizer.pad_token_id)
        predictions = processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)
        references = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        pairs = [(normalizer(p), normalizer(r)) for p, r in zip(predictions, references)]
        pairs = [(p, r) for p, r in pairs if r.strip()]
        if not pairs:
            return {"wer": float("nan")}
        return {"wer": 100 * jiwer.wer([r for _, r in pairs], [p for p, _ in pairs])}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=lr,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        lr_scheduler_type="linear",
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        predict_with_generate=args.eval_wer,
        generation_max_length=225,
        load_best_model_at_end=True,
        metric_for_best_model="wer" if args.eval_wer else "loss",
        greater_is_better=False,
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=args.workers,
        dataloader_pin_memory=True,
        seed=args.seed,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=val_set,
        data_collator=collator,
        compute_metrics=compute_metrics if args.eval_wer else None,
        processing_class=processor,
    )

    trainer.train(resume_from_checkpoint=args.resume or None)

    final = args.out_dir / "final"
    trainer.save_model(str(final))
    processor.save_pretrained(str(final))
    print(f"\nsaved to {final}")
    print("next: python evaluate.py --model-dir", final)


if __name__ == "__main__":
    main()
