"""Manifest-backed dataset and collator shared by train.py and evaluate.py.

SwissDial ships 22.05 kHz 24-bit mono wavs; Whisper wants 16 kHz float, so every clip is
resampled in the dataloader workers (there are plenty of cores, and it keeps the GPU fed).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

WHISPER_SR = 16000


def has_processor(directory: str | Path) -> bool:
    """Whether a directory carries its own Whisper processor. transformers 5 writes
    processor_config.json; older checkpoints carry preprocessor_config.json."""
    directory = Path(directory)
    return any((directory / name).is_file()
               for name in ("processor_config.json", "preprocessor_config.json"))


def read_manifest(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_audio(path: str, target_sr: int = WHISPER_SR) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr, res_type="soxr_hq")
    return audio


class SwissDialDataset(Dataset):
    """Yields Whisper log-mel features and tokenised High German targets."""

    def __init__(self, manifest: str | Path, processor: Any, with_labels: bool = True):
        self.rows = read_manifest(manifest)
        self.processor = processor
        self.with_labels = with_labels

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        audio = load_audio(row["audio"])
        features = self.processor.feature_extractor(
            audio, sampling_rate=WHISPER_SR, return_tensors="np").input_features[0]
        item: dict[str, Any] = {"input_features": features}
        if self.with_labels:
            item["labels"] = self.processor.tokenizer(row["text"]).input_ids
        return item


@dataclass
class SpeechSeq2SeqCollator:
    """Pads a batch and masks label padding so it is ignored by the loss."""

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        batch = self.processor.feature_extractor.pad(
            [{"input_features": f["input_features"]} for f in features], return_tensors="pt")

        labels_batch = self.processor.tokenizer.pad(
            [{"input_ids": f["labels"]} for f in features], return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # The model prepends the decoder start token itself; drop it if the tokenizer added one.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
