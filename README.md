Swiss Courts Assistant

Conversational search over Swiss case law ([voilaj/swiss-caselaw](https://huggingface.co/datasets/voilaj/swiss-caselaw)).

## Data

Download the corpus (~7.5 GB decisions + citation/statute graph):

```bash
uv run hf download voilaj/swiss-caselaw --repo-type dataset --include "data/*.parquet" --local-dir data/raw
uv run hf download voilaj/swiss-caselaw --repo-type dataset --include "graph/*.parquet" --local-dir data/raw
```

Build a stratified development subset (keeps the joint distribution over
language × jurisdiction × legal branch × period; deterministic per seed):

```bash
uv run python -m swiss_court_assistant.sampling --n 50000 --seed 42
```

Output: `data/subset/decisions_50k_seed42.parquet` plus a `.manifest.json`
comparing full-corpus vs subset shares for every stratum.
