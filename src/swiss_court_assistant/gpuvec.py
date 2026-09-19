"""The vector matrix on the GPU, searched with NVIDIA cuVS: the same exact search as vecmatrix.py, in
float16. Measured on the full-corpus index (4.9M x 2048, H100 NVL): 4 ms instead of 90 ms for one
language, 5 ms instead of 530 ms with a facet filter, identical top-40.

vecmatrix.py already moved search out of sqlite-vec into a float32 matrix in memory (40 GB for the
full-corpus index). This module converts that matrix to float16 (20 GB) and serves it from GPU memory
with cuVS brute-force inner-product search. The search stays exact; only the vectors lose precision.
Unit-length 2048-dim embeddings in float16 keep ~3 significant digits, which can reorder only ties
among near-duplicate passages. `bench` measures the overlap against the float32 search.

The facet filters (canton, court, year …) stay exact too: the row mask becomes a cuVS prefilter
bitset, so the GPU scores only the rows that pass, not a top-k that is filtered afterwards.

sqlite remains the source of truth. `export` reads the float32 matrix, re-exporting it from sqlite
first when it is older than the index, and records the index state it came from, like vecmatrix.py.
The app uses this matrix unless SCA_VECTORS=cpu, and falls back to the float32 one when cuVS is
missing, the float16 copy is stale, or it does not fit in GPU memory.

    uv sync                                                  # cuVS is in the default `gpu` group
    uv run python -m swiss_court_assistant.gpuvec export     # after `index build` / `index update`
    uv run python -m swiss_court_assistant.gpuvec bench      # GPU vs CPU: speed and top-k overlap
    uv run python -m swiss_court_assistant.gpuvec status

Files next to the float32 matrix, for DB = data/vectordb/corpus.sqlite and model nemotron-embed:
    corpus.nemotron-embed.f16         float16 rows, same order as the .f32 file (same .ids)
    corpus.nemotron-embed.f16.json    the .json it was converted from
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import threading
import time
from pathlib import Path

import numpy as np

from swiss_court_assistant import vecmatrix as M
from swiss_court_assistant import vectordb as V

log = logging.getLogger(__name__)

_STEP = 1 << 16  # rows converted and uploaded at a time: 256 MB of float32


def export(db: Path, model: str) -> None:
    """The float16 copy of the float32 matrix, re-exporting that from sqlite first when it is stale."""
    t0 = time.time()
    if not M.current(db, model):
        print("float32 matrix missing or older than the index; exporting it from sqlite first …")
        M.export(db, model)
    path = M.base(db, model)
    meta = json.loads(M._file(path, ".json").read_text())
    rows, dim = meta["rows"], meta["index"]["dim"]
    src = np.memmap(M._file(path, ".f32"), dtype=np.float32, mode="r", shape=(rows, dim))
    tmp = M._file(path, ".f16.tmp")
    dst = np.memmap(tmp, dtype=np.float16, mode="w+", shape=(rows, dim))
    for a in range(0, rows, _STEP):
        dst[a:a + _STEP] = src[a:a + _STEP]
        if (a // _STEP) % 16 == 0:
            print(f"  {a:,}/{rows:,} rows ({time.time() - t0:.0f} s)")
    dst.flush()
    del dst
    tmp.replace(M._file(path, ".f16"))
    M._file(path, ".f16.json").write_text(json.dumps(meta, indent=1))
    print(f"done: {rows:,} x {dim} float16 in {time.time() - t0:.0f} s -> {path}.f16")


def current(db: Path, model: str) -> bool:
    path = M.base(db, model)
    if not M._file(path, ".f16.json").exists() or not M.current(db, model):
        return False
    return json.loads(M._file(path, ".f16.json").read_text()) == json.loads(M._file(path, ".json").read_text())


class GpuVectorMatrix:
    """VectorMatrix's interface (`ids`, `segments`, `meta`, `search`, `warm`) over float16 vectors in GPU
    memory, one cuVS brute-force index per segment (kind/language)."""

    def __init__(self, path: Path, meta: dict, device: int):
        import cupy as cp
        from cuvs.neighbors import brute_force

        self.meta = meta
        self.device = device
        self.ids = np.fromfile(M._file(path, ".ids"), dtype=np.int64)
        self.segments = {k: tuple(v) for k, v in meta["segments"].items()}
        dim = meta["index"]["dim"]
        host = np.memmap(M._file(path, ".f16"), dtype=np.float16, mode="r", shape=(meta["rows"], dim))
        t0 = time.time()
        self._index: dict[str, object] = {}
        self._vectors = []  # the index only views its dataset: the arrays must outlive the loop
        with cp.cuda.Device(device):
            for key, (a, b) in self.segments.items():
                vectors = cp.empty((b - a, dim), dtype=cp.float16)
                for i in range(a, b, _STEP):
                    vectors[i - a:min(i + _STEP, b) - a].set(np.ascontiguousarray(host[i:min(i + _STEP, b)]))
                self._vectors.append(vectors)
                self._index[key] = brute_force.build(vectors, metric="inner_product")
        log.info("vectors on GPU %d: %d x %d float16 (%.1f GB) in %.0f s", device, meta["rows"], dim,
                 meta["rows"] * dim * 2 / 1e9, time.time() - t0)
        # cuVS and CuPy calls hop between the knn worker threads; one at a time, each a few ms
        self._lock = threading.Lock()
        self.backend = f"cuVS brute-force float16, GPU {device}"
        self.searches, self._ms = 0, 0.0  # shown by /api/health: proof the GPU is doing the searching

    @classmethod
    def open(cls, db: Path, model: str, con: sqlite3.Connection, device: int = 1) -> GpuVectorMatrix | None:
        """The GPU matrix, or None when cuVS is not installed or the float16 copy is missing or stale."""
        path = M.base(db, model)
        if not M._file(path, ".f16.json").exists():
            log.warning("no float16 matrix at %s.f16; run `python -m swiss_court_assistant.gpuvec export`", path)
            return None
        meta = json.loads(M._file(path, ".f16.json").read_text())
        try:
            now = M._state(con, model)
        except SystemExit:
            return None
        if meta["index"] != now:
            log.warning("float16 matrix %s is stale; run `python -m swiss_court_assistant.gpuvec export`", path)
            return None
        try:
            return cls(path, meta, device)
        except ImportError:
            log.warning("cuVS is not installed (`uv sync`); searching on the CPU")
            return None
        except Exception as e:  # typically out of GPU memory: another model took the space
            log.warning("vectors do not fit on GPU %d (%s); searching on the CPU", device, e)
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
            return None

    def stats(self) -> dict:
        return {"searches": self.searches, "avg_ms": round(self._ms / self.searches, 2) if self.searches else None}

    def warm(self) -> None:
        """Nothing to page in: the vectors were uploaded when the matrix was opened."""

    def search(self, query: np.ndarray, k: int, kind: str = "decision",
               language: str | None = None, mask: np.ndarray | None = None) -> list[tuple[int, float]]:
        """(chunk id, cosine similarity) of the k nearest rows, best first — as VectorMatrix.search."""
        import cupy as cp
        from cuvs.neighbors import brute_force, filters

        q = np.asarray(query, dtype=np.float16).reshape(1, -1)
        best: list[tuple[int, float]] = []
        with self._lock, cp.cuda.Device(self.device):
            t0 = time.perf_counter()
            d_q = cp.asarray(q)
            for key, (a, b) in self.segments.items():
                if not key.startswith(f"{kind}/") or (language is not None and key != f"{kind}/{language}"):
                    continue
                keep = None if mask is None else mask[a:b]
                n = (b - a) if keep is None else int(keep.sum())
                if n == 0:
                    continue
                prefilter = None
                if keep is not None:
                    bits = np.packbits(keep, bitorder="little")
                    bits = np.pad(bits, (0, -len(bits) % 4)).view(np.uint32)
                    prefilter = filters.from_bitset(cp.asarray(bits))
                dist, nbr = brute_force.search(self._index[key], d_q, min(k, n), prefilter=prefilter)
                dist, nbr = dist.copy_to_host()[0], nbr.copy_to_host()[0]
                ok = (nbr >= 0) & (nbr < b - a)
                if keep is not None:
                    ok[ok] &= keep[nbr[ok]]
                best += [(int(self.ids[a + r]), float(s)) for r, s in zip(nbr[ok], dist[ok])]
            self.searches += 1
            self._ms += (time.perf_counter() - t0) * 1000
        return sorted(best, key=lambda x: -x[1])[:k]


def open_matrix(db: Path, model: str, con: sqlite3.Connection):
    """The matrix the app searches: on the GPU unless SCA_VECTORS=cpu or it cannot be, else in memory."""
    if os.environ.get("SCA_VECTORS", "gpu").lower() == "gpu":
        gpu = GpuVectorMatrix.open(db, model, con, int(os.environ.get("SCA_VECTORS_GPU", "1")))
        if gpu is not None:
            return gpu
    return M.VectorMatrix.open(db, model, con)


def bench(db: Path, model: str, device: int, n: int = 50, k: int = 40) -> None:
    """GPU (float16) against CPU (float32) on real passage vectors as queries: time and top-k overlap."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    t0 = time.time()
    gpu = GpuVectorMatrix.open(db, model, con, device)
    if gpu is None:
        raise SystemExit("no usable float16 matrix; run `gpuvec export` (and `uv sync`)")
    print(f"GPU matrix loaded in {time.time() - t0:.0f} s")
    cpu = M.VectorMatrix.open(db, model, con)
    rng = np.random.default_rng(0)
    a, b = gpu.segments["decision/de"]
    queries = cpu.vectors[np.sort(rng.choice(np.arange(a, b), n, replace=False))]
    # a filter like "canton ZH": every 7th decision row
    mask = np.zeros(len(gpu.ids), dtype=bool)
    mask[::7] = True
    for label, lang, m in (("decision/de", "de", None), ("all decisions", None, None),
                           ("decision/de, 1/7 of rows", "de", mask)):
        gpu.search(queries[0], k, "decision", lang, m)
        t = time.time()
        g = [gpu.search(q, k, "decision", lang, m) for q in queries]
        tg = (time.time() - t) / n * 1000
        t = time.time()
        c = [cpu.search(q, k, "decision", lang, m) for q in queries[:10]]
        tc = (time.time() - t) / 10 * 1000
        overlap = np.mean([len({i for i, _ in x} & {i for i, _ in y}) / k for x, y in zip(g, c)])
        print(f"{label:26s} GPU {tg:7.1f} ms   CPU {tc:7.1f} ms   top-{k} overlap {overlap:.1%}")


def status(db: Path, model: str) -> None:
    path = M.base(db, model)
    if not M._file(path, ".f16.json").exists():
        print(f"no float16 matrix at {path}.f16")
        return
    meta = json.loads(M._file(path, ".f16.json").read_text())
    print(f"{path}.f16: {meta['rows']:,} x {meta['index']['dim']} "
          f"({'current' if current(db, model) else 'STALE'})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["export", "bench", "status"])
    ap.add_argument("--db", type=Path, default=V.DB_DIR / "corpus.sqlite")
    ap.add_argument("--model", default="nemotron-embed")
    ap.add_argument("--gpu", type=int, default=int(os.environ.get("SCA_VECTORS_GPU", "1")))
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.command == "export":
        export(args.db, args.model)
    elif args.command == "bench":
        bench(args.db, args.model, args.gpu)
    else:
        status(args.db, args.model)


if __name__ == "__main__":
    main()
