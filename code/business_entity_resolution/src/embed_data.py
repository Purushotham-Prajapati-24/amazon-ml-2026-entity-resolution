"""Stage 2a: fit the lexical embedder on a label-free sample and embed every record.

Writes data_cache/vec/embedder.pkl and data_cache/vec/{split}_s{k}.npy (float16,
row order = normalised parquet order).

    python src/embed_data.py
"""
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from er.config import CACHE_DIR, SOURCES, SPLITS
from er.io import load_norm
from er.vectorize import RecordEmbedder

VEC_DIR = CACHE_DIR / "vec"
MODEL_PATH = VEC_DIR / "embedder.pkl"
FIT_ROWS_PER_SOURCE = 120_000
CHUNK = 100_000

_embedder = None


def _init():
    global _embedder
    _embedder = RecordEmbedder.load(MODEL_PATH)


def _work(chunk):
    names, addrs = chunk
    return _embedder.dense(names, addrs)


def fit():
    t0 = time.time()
    parts = [
        load_norm(split, k, ["name_core", "addr_norm"]).sample(FIT_ROWS_PER_SOURCE, random_state=k)
        for split in SPLITS for k in SOURCES
    ]
    sample = pd.concat(parts, ignore_index=True)
    RecordEmbedder().fit(sample.name_core.tolist(), sample.addr_norm.tolist()).save(MODEL_PATH)
    print(f"fitted embedder on {len(sample):,} rows  {time.time() - t0:.0f}s")


def main():
    VEC_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        fit()
    workers = min(12, max(1, (os.cpu_count() or 2) - 4))
    with Pool(workers, initializer=_init) as pool:
        for split in SPLITS:
            for k in SOURCES:
                t0 = time.time()
                df = load_norm(split, k, ["name_core", "addr_norm"])
                names, addrs = df.name_core.tolist(), df.addr_norm.tolist()
                chunks = [(names[i:i + CHUNK], addrs[i:i + CHUNK]) for i in range(0, len(df), CHUNK)]
                vec = np.vstack(list(pool.imap(_work, chunks)))
                np.save(VEC_DIR / f"{split}_s{k}.npy", vec)
                print(f"{split} s{k}: {vec.shape}  {time.time() - t0:5.1f}s")


if __name__ == "__main__":
    main()
