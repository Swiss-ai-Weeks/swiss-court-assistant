"""Hybrid passage retrieval: BM25 + dense embeddings, RRF fusion, cross-encoder rerank.

Indexes are built over the chunk table produced by chunking.py; row order of
that parquet is the index position everywhere.

Usage:
    uv run python -m swiss_court_assistant.retrieval build bm25 data/subset/decisions_50k_seed42.chunks.parquet
    uv run python -m swiss_court_assistant.retrieval build bge-m3 <chunks> --device cuda:1
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
from multiprocessing import Pool
from pathlib import Path

import bm25s
import numpy as np
import polars as pl
import Stemmer
import torch
from bm25s import stopwords as sw
from bm25s.tokenization import Tokenized

INDEX_ROOT = Path("data/index")

# ── lexical ─────────────────────────────────────────────────────────────
_TOKEN = re.compile(r"(?u)\b\w\w+\b")
_STEMMER_LANG = {"de": "german", "fr": "french", "it": "italian", "en": "english"}
_STOP = {*sw.STOPWORDS_EN, *sw.STOPWORDS_GERMAN, *sw.STOPWORDS_FRENCH, *sw.STOPWORDS_ITALIAN}
_stemmers: dict[str, Stemmer.Stemmer | None] = {}


def tokenize(text: str, lang: str) -> list[str]:
    """Lowercase, drop de/fr/it/en stopwords, stem with the text's own language."""
    if lang not in _stemmers:
        _stemmers[lang] = Stemmer.Stemmer(_STEMMER_LANG[lang]) if lang in _STEMMER_LANG else None
    toks = [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]
    st = _stemmers[lang]
    return st.stemWords(toks) if st else toks


def _tok_pair(args: tuple[str, str]) -> list[str]:
    return tokenize(*args)


# ── dense ───────────────────────────────────────────────────────────────
DENSE_MODELS = {
    "bge-m3": {"name": "BAAI/bge-m3", "query_prompt": ""},
    "qwen3-4b": {
        "name": "Qwen/Qwen3-Embedding-4B",
        "query_prompt": "Instruct: Given a question about Swiss law, retrieve passages "
                        "from Swiss court decisions that answer it\nQuery: ",
    },
    # NVIDIA NIM served over HTTP (docker, see README); no local weights
    "nemotron-embed": {
        "backend": "nim", "model": "nvidia/nemotron-3-embed-1b",
        "url_env": "NIM_EMBED_URL", "url": "http://localhost:8000/v1",
    },
}
MAX_SEQ = 1024  # chunks are <= 2200 chars, well under this


class NimEncoder:
    """Embeddings from an NVIDIA NIM (OpenAI-compatible /v1/embeddings).

    NIM retrieval embedders take input_type=query|passage instead of a prompt.
    """

    # Small batches, many in flight: best measured on one H100 (~130 passages/s,
    # GPU-bound; 64 x 16 gave ~100/s).
    def __init__(self, cfg: dict, batch_size: int = 16, concurrency: int = 128):
        # one NIM saturates one GPU; a comma-separated list (one instance per GPU) is used round-robin
        self.urls = [u.strip() for u in os.environ.get(cfg["url_env"], cfg["url"]).split(",") if u.strip()]
        self.url = self.urls[0]
        self.model, self.batch_size, self.concurrency = cfg["model"], batch_size, concurrency

    def embed(self, texts: list[str], input_type: str) -> np.ndarray:
        from contextlib import AsyncExitStack

        from openai import AsyncOpenAI
        from tqdm.asyncio import tqdm_asyncio

        async def run() -> list[np.ndarray]:
            sem = asyncio.Semaphore(self.concurrency)
            # close the clients inside the loop, or their transports are torn down after it
            async with AsyncExitStack() as stack:
                clients = [await stack.enter_async_context(
                    AsyncOpenAI(base_url=u, api_key="nim", timeout=600, max_retries=5)) for u in self.urls]

                async def one(i: int, batch: list[str]) -> np.ndarray:
                    async with sem:
                        # no encoding_format: the SDK then asks for base64 and decodes it with numpy,
                        # which measured 165 vs 100 passages/s - parsing JSON floats was the bottleneck
                        r = await clients[i % len(clients)].embeddings.create(
                            model=self.model, input=batch,
                            extra_body={"input_type": input_type, "truncate": "END"},
                        )
                    # to float32 right away: 800k vectors as Python floats would not fit
                    return np.asarray([d.embedding for d in sorted(r.data, key=lambda d: d.index)],
                                      dtype=np.float32)

                batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
                return await tqdm_asyncio.gather(*(one(i, b) for i, b in enumerate(batches)),
                                                 desc=f"embed {input_type}", disable=len(batches) < 10)

        emb = np.concatenate(asyncio.run(run()))
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)


def load_encoder(key: str, device: str):
    if DENSE_MODELS[key].get("backend") == "nim":
        return NimEncoder(DENSE_MODELS[key])
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(
        DENSE_MODELS[key]["name"], device=device, model_kwargs={"dtype": torch.bfloat16}
    )
    m.max_seq_length = MAX_SEQ
    return m


def index_dir(chunks_path: Path) -> Path:
    return INDEX_ROOT / Path(chunks_path).name.removesuffix(".chunks.parquet")


def build(what: str, chunks_path: Path, device: str, batch_size: int) -> None:
    out = index_dir(chunks_path)
    out.mkdir(parents=True, exist_ok=True)
    chunks = pl.read_parquet(chunks_path, columns=["chunk_id", "language", "text"])
    if what == "bm25":
        with Pool(64) as pool:
            tokens = pool.map(_tok_pair, zip(chunks["text"], chunks["language"]), chunksize=2000)
        vocab: dict[str, int] = {}
        ids = [[vocab.setdefault(t, len(vocab)) for t in toks] for toks in tokens]
        model = bm25s.BM25()
        model.index(Tokenized(ids=ids, vocab=vocab), show_progress=False)
        model.save(out / "bm25")
        print(f"bm25: {len(vocab):,} terms over {chunks.height:,} chunks")
    else:
        enc = load_encoder(what, device)
        if isinstance(enc, NimEncoder):
            emb = enc.embed(chunks["text"].to_list(), "passage")
        else:
            emb = enc.encode(
                chunks["text"].to_list(), batch_size=batch_size, normalize_embeddings=True,
                convert_to_numpy=True, show_progress_bar=True,
            )
        np.save(out / f"{what}.npy", emb.astype(np.float16))
        print(f"{what}: {emb.shape}")
    chunks.select("chunk_id").write_parquet(out / "chunk_ids.parquet")


# ── search ──────────────────────────────────────────────────────────────
class BM25Retriever:
    def __init__(self, chunks_path: Path):
        self.model = bm25s.BM25.load(index_dir(chunks_path) / "bm25")
        self.vocab = self.model.vocab_dict

    def search(self, queries: list[str], langs: list[str], k: int) -> tuple[np.ndarray, np.ndarray]:
        idx = np.full((len(queries), k), -1, dtype=np.int64)
        scores = np.zeros((len(queries), k), dtype=np.float32)
        toks = [[t for t in tokenize(q, l) if t in self.vocab] for q, l in zip(queries, langs)]
        live = [i for i, t in enumerate(toks) if t]
        if live:
            d, s = self.model.retrieve([toks[i] for i in live], k=k, show_progress=False, n_threads=32)
            idx[live], scores[live] = d, s
        return idx, scores


class DenseRetriever:
    def __init__(self, chunks_path: Path, key: str, device: str = "cuda:0"):
        self.key, self.device = key, device
        # fp16 matmul on GPU; fp32 on CPU (used when NIM containers hold the GPUs)
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.encoder = load_encoder(key, device)
        emb = np.load(index_dir(chunks_path) / f"{key}.npy")
        self.emb = torch.from_numpy(emb).to(device, self.dtype)

    def search(self, queries: list[str], langs: list[str] | None, k: int) -> tuple[np.ndarray, np.ndarray]:
        if isinstance(self.encoder, NimEncoder):
            q = torch.from_numpy(self.encoder.embed(queries, "query"))
        else:
            q = self.encoder.encode(
                queries, prompt=DENSE_MODELS[self.key]["query_prompt"] or None,
                normalize_embeddings=True, convert_to_tensor=True, batch_size=64,
            )
        q = q.to(self.device, self.dtype)
        idx, scores = [], []
        for i in range(0, len(q), 256):
            s, j = torch.topk(q[i:i + 256] @ self.emb.T, k, dim=1)
            idx.append(j.cpu().numpy()), scores.append(s.float().cpu().numpy())
        return np.concatenate(idx), np.concatenate(scores)


def rrf(rankings: list[np.ndarray], k: int, c: int = 60, weights: list[float] | None = None) -> np.ndarray:
    """Reciprocal-rank fusion of per-query chunk rankings (-1 = padding)."""
    weights = weights or [1.0] * len(rankings)
    fused = np.full((rankings[0].shape[0], k), -1, dtype=np.int64)
    for qi in range(rankings[0].shape[0]):
        score: dict[int, float] = {}
        for w, r in zip(weights, rankings):
            for rank, ci in enumerate(r[qi]):
                if ci >= 0:
                    score[int(ci)] = score.get(int(ci), 0.0) + w / (c + rank + 1)
        top = sorted(score, key=score.__getitem__, reverse=True)[:k]
        fused[qi, :len(top)] = top
    return fused


def candidate_pool(rankings: dict[str, np.ndarray], dense_depth: int = 100,
                   bm25_depth: int = 30) -> list[list[int]]:
    """Per-query union of each retriever's top candidates: the reranker's input.

    On the test set this reaches the gold decision far more often than a fused
    top-50 (0.70 vs 0.62): dense covers cross-lingual queries, BM25 adds
    same-language hits that dense misses.
    """
    order = [k for k in rankings if k != "bm25"] + (["bm25"] if "bm25" in rankings else [])
    n = next(iter(rankings.values())).shape[0]
    pools = []
    for i in range(n):
        seen: dict[int, None] = {}
        for k in order:
            for c in rankings[k][i, :bm25_depth if k == "bm25" else dense_depth]:
                if c >= 0:
                    seen.setdefault(int(c), None)
        pools.append(list(seen))
    return pools


RERANK_INSTRUCTION = ("Given a question about Swiss law, judge whether this passage from a "
                      "Swiss court decision answers it or contains the decisive legal reasoning")
_QWEN_SYSTEM = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the '
                'Query and the Instruct provided. Note that the answer can only be "yes" or "no".'
                '<|im_end|>\n<|im_start|>user\n')
RERANKERS = {
    "bge": {"name": "BAAI/bge-reranker-v2-m3", "batch_size": 128},
    # sequence-classification conversion of Qwen/Qwen3-Reranker-4B; needs its chat template.
    # Scored with plain transformers: sentence-transformers 6 rejects its pair template.
    "qwen": {
        "name": "tomaarsen/Qwen3-Reranker-4B-seq-cls", "batch_size": 64, "max_length": 2048,
        "raw_transformers": True,
        "query_template": _QWEN_SYSTEM + "<Instruct>: {instruction}\n<Query>: {query}\n",
        "doc_template": "<Document>: {doc}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
    },
    # NVIDIA NIM served over HTTP (docker, see README): POST /v1/ranking
    "nemotron": {
        "backend": "nim", "model": "nvidia/llama-nemotron-rerank-vl-1b-v2",
        "url_env": "NIM_RERANK_URL", "url": "http://localhost:8001/v1/ranking", "batch_size": 64,
    },
}


class Reranker:
    def __init__(self, key: str = "bge", device: str = "cuda:0"):
        cfg = RERANKERS[key]
        self.batch_size, self.device = cfg["batch_size"], device
        self.max_length = cfg.get("max_length", MAX_SEQ)
        self.query_template = cfg.get("query_template", "{query}")
        self.doc_template = cfg.get("doc_template", "{doc}")
        self.model = self.hf = self.nim = None
        if cfg.get("backend") == "nim":
            self.nim = cfg | {"url": os.environ.get(cfg["url_env"], cfg["url"])}
        elif cfg.get("raw_transformers"):
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # left padding so the scored (last) token is always at position -1
            self.tok = AutoTokenizer.from_pretrained(cfg["name"], padding_side="left")
            self.hf = AutoModelForSequenceClassification.from_pretrained(
                cfg["name"], dtype=torch.bfloat16).to(device).eval()
            self.hf.config.pad_token_id = self.tok.pad_token_id
        else:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(cfg["name"], max_length=self.max_length, device=device,
                                      model_kwargs={"dtype": torch.bfloat16})

    def _score_nim(self, queries: list[str], pools: list[list[int]], texts: list[str]) -> np.ndarray:
        """Scores from a ranking NIM, flattened in (query, pool) order."""
        import httpx
        from tqdm.asyncio import tqdm_asyncio

        async def post(client: httpx.AsyncClient, body: dict) -> dict:
            for attempt in range(5):  # a loaded NIM may answer 429/503
                try:
                    r = await client.post(self.nim["url"], json=body)
                    if r.status_code not in (429, 502, 503, 504):
                        r.raise_for_status()
                        return r.json()
                except httpx.TransportError:
                    if attempt == 4:
                        raise
                await asyncio.sleep(2 ** attempt)
            r.raise_for_status()
            return r.json()

        async def one(client: httpx.AsyncClient, sem: asyncio.Semaphore, q: str, pool: list[int]) -> np.ndarray:
            out = np.empty(len(pool), dtype=np.float32)
            for s in range(0, len(pool), self.batch_size):
                part = pool[s:s + self.batch_size]
                body = {"model": self.nim["model"], "query": {"text": q},
                        "passages": [{"text": texts[c]} for c in part], "truncate": "END"}
                async with sem:
                    res = await post(client, body)
                for item in res["rankings"]:
                    out[s + item["index"]] = item["logit"]
            return out

        async def run() -> list[np.ndarray]:
            sem = asyncio.Semaphore(16)
            async with httpx.AsyncClient(timeout=600) as client:
                return await tqdm_asyncio.gather(*(one(client, sem, q, p) for q, p in zip(queries, pools)),
                                                 desc="rerank")

        return np.concatenate(asyncio.run(run()))

    @torch.inference_mode()
    def _score_raw(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        from tqdm import tqdm

        texts = [a + b for a, b in pairs]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # less padding
        scores = np.empty(len(texts), dtype=np.float32)
        for s in tqdm(range(0, len(order), self.batch_size), desc="rerank"):
            ids = order[s:s + self.batch_size]
            enc = self.tok([texts[i] for i in ids], padding=True, truncation=True,
                           max_length=self.max_length, return_tensors="pt").to(self.device)
            scores[ids] = self.hf(**enc).logits[:, 0].float().cpu().numpy()
        return scores

    def rerank(self, queries: list[str], pools: list[list[int]], texts: list[str]) -> np.ndarray:
        """Score every (query, candidate chunk) pair; pools may differ in size."""
        if self.nim is not None:
            owner = [(qi, ci) for qi, pool in enumerate(pools) for ci in pool]
            scores = self._score_nim(queries, pools, texts)
        else:
            pairs, owner = [], []
            for qi, (q, pool) in enumerate(zip(queries, pools)):
                fq = self.query_template.format(query=q, instruction=RERANK_INSTRUCTION)
                for ci in pool:
                    pairs.append((fq, self.doc_template.format(doc=texts[ci]))), owner.append((qi, ci))
            scores = (self._score_raw(pairs) if self.hf is not None
                      else self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=True))
        per_q: dict[int, list[tuple[float, int]]] = {}
        for (qi, ci), s in zip(owner, scores):
            per_q.setdefault(qi, []).append((float(s), ci))
        out = np.full((len(queries), max(map(len, pools))), -1, dtype=np.int64)
        for qi, lst in per_q.items():
            ranked = [ci for _, ci in sorted(lst, reverse=True)]
            out[qi, :len(ranked)] = ranked
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("what", choices=["bm25", *DENSE_MODELS])
    b.add_argument("chunks", type=Path)
    b.add_argument("--device", default="cuda:0")
    b.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    build(args.what, args.chunks, args.device, args.batch_size)


if __name__ == "__main__":
    main()
