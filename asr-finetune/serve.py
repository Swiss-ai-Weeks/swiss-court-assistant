"""A small self-hosted HTTP endpoint for the fine-tuned model, for wiring into voice mode.

    uv run --group serve python serve.py --model-dir export/whisper-swiss-de --port 9003
    curl -F audio=@clip.wav http://localhost:9003/v1/transcribe

Everything stays on this machine, so it satisfies the same sovereignty constraint as the NIMs.
The streaming Nemotron ASR NIM is still the right path for English; this covers Swiss German,
where that NIM has no model.
"""

from __future__ import annotations

import argparse
import io

import librosa
import soundfile as sf
import torch
import uvicorn
from fastapi import FastAPI, File, UploadFile
from transformers import WhisperForConditionalGeneration, WhisperProcessor

WHISPER_SR = 16000
app = FastAPI(title="Swiss German ASR")
state: dict = {}


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": state.get("model_dir")}


@app.post("/v1/transcribe")
async def transcribe(audio: UploadFile = File(...)) -> dict:
    raw, sr = sf.read(io.BytesIO(await audio.read()), dtype="float32", always_2d=False)
    if raw.ndim > 1:
        raw = raw.mean(axis=1)
    if sr != WHISPER_SR:
        raw = librosa.resample(raw, orig_sr=sr, target_sr=WHISPER_SR, res_type="soxr_hq")

    processor, model = state["processor"], state["model"]
    features = processor.feature_extractor(
        raw, sampling_rate=WHISPER_SR, return_tensors="pt").input_features
    with torch.inference_mode():
        generated = model.generate(features.to(model.device, dtype=model.dtype), max_new_tokens=225)
    text = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
    return {"text": text, "seconds": round(len(raw) / WHISPER_SR, 2)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default="export/whisper-swiss-de")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=9003)
    args = p.parse_args()

    state["model_dir"] = args.model_dir
    state["processor"] = WhisperProcessor.from_pretrained(args.model_dir, language="german", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_dir, dtype=torch.float16)
    model.generation_config.language = "de"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    state["model"] = model.eval().cuda()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
