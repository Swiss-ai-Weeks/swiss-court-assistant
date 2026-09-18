# Swiss German ASR fine-tuning

Fine-tunes Whisper on **SwissDial** so voice mode can take Swiss German dialect speech and
produce **High German text**. The streaming Nemotron ASR NIM has no Swiss German model, which
is the gap this fills.

The trick is in the target text. SwissDial gives, for every recorded sentence, both a dialect
transcript (`ch_zh`, `ch_be`, …) and the Standard German reference (`de`). Training against
`de` makes the model a speech *translator* while it still runs as a plain German transcription
task (`language=de`, `task=transcribe`) — so the result is an ordinary Whisper checkpoint that
any Whisper serving path loads unchanged.

Everything runs locally. The corpus and the base weights are already on this machine, so
training needs no network, and the exported model is served by `serve.py` on this host.

## The data

`data/asr/data1.1` (extracted from `data1.1.tar.gz`) — 30,921 clips, mono 22.05 kHz 24-bit,
8 dialects, CC BY-NC 4.0 from ETH Zürich. `data1.1` supersedes `data.tar.gz`: the two are
identical except that Grisons grows from 2,749 to 10,475 clips, so only v1.1 is used.

The splits are cut over **sentence ids**, never over clips. Each sentence was recorded by up to
eight speakers, so a clip-level split would put the Zurich reading of a sentence in train and
the Bernese reading of the same sentence in test, and the model would be scored on text it had
already memorised. Val and test draw only from sentences all eight dialects recorded, which is
what makes them exactly balanced:

| split | clips | hours | per dialect |
|-------|-------|-------|-------------|
| train | 26,921 | 31.4 | ag 2248 · be 2200 · bs 2213 · gr 9975 · lu 2215 · sg 2252 · vs 2253 · zh 3565 |
| val   | 1,600 | 1.8 | 200 each |
| test  | 2,400 | 2.8 | 300 each |

Grisons is 37% of the training clips. Pass `--cap-per-dialect 3000` to `prepare_data.py` for a
dialect-balanced ~19k-clip set if that skew shows up in the per-dialect scores.

## Running it

`whisper-large-v3` is already in the local HF cache and the manifests are already built in
`manifests/`, so nothing downloads at training time. To rebuild the manifests:

```bash
cd asr-finetune && uv run python prepare_data.py
```

Get the baseline first — it is the number the fine-tune has to beat:

```bash
cd asr-finetune && uv run python evaluate.py --model-dir openai/whisper-large-v3 --out results/baseline.json
```

Train (LoRA, the default):

```bash
cd asr-finetune && uv run python train.py --out-dir runs/whisper-swiss
```

Then score it, merge the adapter into standalone weights, and serve:

```bash
cd asr-finetune && uv run python evaluate.py --model-dir runs/whisper-swiss/final --out results/finetuned.json
```

```bash
cd asr-finetune && uv run python export.py --model-dir runs/whisper-swiss/final --out-dir export/whisper-swiss-de
```

```bash
cd asr-finetune && uv run --group serve python serve.py --model-dir export/whisper-swiss-de --port 9003
```

Progress is in TensorBoard under `runs/whisper-swiss`, and `--resume` picks up the last checkpoint.

## GPU budget

Both H100s are currently occupied by the NIM stack, so **a GPU has to be freed before training
starts**. Rough peaks for `whisper-large-v3`, batch 16 with gradient checkpointing:

| method | VRAM | notes |
|--------|------|-------|
| `--method lora` (default) | ~30 GB | trains ~16M adapter params, merged back at export |
| `--method full` | ~55 GB | updates all 1.55B params, usually a little better |

Stopping `sca-guardrails` frees 29 GB on GPU 1; adding `magpie-tts` and `nemotron-asr-multi`
takes it to ~56 GB, which covers a full fine-tune. That leaves the agent's LLM on GPU 0 running,
so the app stays up while training. Pick the GPU with `CUDA_VISIBLE_DEVICES=1`.

If instead you free GPU 0 by stopping `nim-llm`, you get ~91 GB but the agent stops answering.

One thing worth knowing if you change the model loading: `whisper-large-v3` ships a **float16**
checkpoint, and transformers 5 honours that dtype by default. `train.py` therefore asks for fp32
explicitly — half-precision weights mismatch the fp32 input features at the encoder convolutions
and leave Adam updating half-precision masters. Inference scripts load bf16/fp16 deliberately.

## Scripts

| file | what it does |
|------|--------------|
| `prepare_data.py` | joins wavs to their High German reference, filters by duration, writes leak-free splits |
| `data.py` | manifest dataset, 22.05→16 kHz resampling, padding collator |
| `train.py` | LoRA or full fine-tune, bf16, WER on the validation set |
| `evaluate.py` | WER / CER / BLEU on the test split, overall and per dialect |
| `export.py` | merges the adapter into standalone weights, optional CTranslate2 build |
| `serve.py` | local HTTP transcription endpoint for wiring into voice mode |

## Wiring into voice mode

`server/voice.py` streams to the Riva ASR NIM over gRPC. This model is not a Riva model, so
Swiss German goes through `serve.py` instead: post the buffered utterance to `/v1/transcribe`
and feed the returned High German text into the existing agent path. That is a separate change
from this directory, which stops at a trained and exported model.

Whisper is not streaming, so the natural fit is to transcribe each utterance once the client
stops sending audio, rather than emitting interim results the way the Nemotron NIM does.
