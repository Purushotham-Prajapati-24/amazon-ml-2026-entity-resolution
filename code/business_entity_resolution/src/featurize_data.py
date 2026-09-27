"""Stage 2b: hashed bag features for the learned encoder (data_cache/feat/).

Per split/source/field writes ``*_indptr.npy`` (int64) and ``*_idx.npy`` (int32).

    python src/featurize_data.py
"""
import os
import time
from multiprocessing import Pool

import numpy as np

from er.config import CACHE_DIR, SOURCES, SPLITS
from er.encoder import featurize
from er.io import load_norm

FEAT_DIR = CACHE_DIR / "feat"
CHUNK = 100_000


def _work(chunk):
    n, a = featurize(*chunk)
    return (np.diff(n.indptr), n.indices.astype(np.int32),
            np.diff(a.indptr), a.indices.astype(np.int32))


def _save(prefix, lens, idx):
    indptr = np.zeros(len(lens) + 1, dtype=np.int64)
    np.cumsum(lens, out=indptr[1:])
    np.save(FEAT_DIR / f"{prefix}_indptr.npy", indptr)
    np.save(FEAT_DIR / f"{prefix}_idx.npy", idx)


def main():
    FEAT_DIR.mkdir(parents=True, exist_ok=True)
    workers = min(12, max(1, (os.cpu_count() or 2) - 4))
    with Pool(workers) as pool:
        for split in SPLITS:
            for k in SOURCES:
                t0 = time.time()
                df = load_norm(split, k, ["name_core", "addr_norm", "numbers", "states"])
                cols = [df[c].tolist() for c in ("name_core", "addr_norm", "numbers", "states")]
                chunks = [tuple(c[i:i + CHUNK] for c in cols) for i in range(0, len(df), CHUNK)]
                parts = list(pool.imap(_work, chunks))
                _save(f"{split}_s{k}_name", np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
                _save(f"{split}_s{k}_addr", np.concatenate([p[2] for p in parts]), np.concatenate([p[3] for p in parts]))
                nnz = sum(len(p[1]) + len(p[3]) for p in parts)
                print(f"{split} s{k}: {len(df):>10,} rows, {nnz / len(df):.1f} feats/row  {time.time() - t0:5.1f}s")


if __name__ == "__main__":
    main()
