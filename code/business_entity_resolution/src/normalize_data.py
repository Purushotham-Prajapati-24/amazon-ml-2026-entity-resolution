"""Stage 1: normalise every record once and cache it (data_cache/norm/{split}_s{k}.parquet).

Columns: entity_id, country, name_norm, name_core, name_key, name_url, addr_norm,
states, numbers (space-joined), addr_empty.

    python src/normalize_data.py
"""
import os
import time
from multiprocessing import Pool

import pandas as pd

from er.config import CACHE_DIR, SOURCES, SPLITS
from er.io import load_source
from er.normalize import normalize_record

NORM_DIR = CACHE_DIR / "norm"
CHUNK = 50_000


def _work(chunk):
    names, addrs = chunk
    return pd.DataFrame([normalize_record(n, a) for n, a in zip(names, addrs)])


def main():
    NORM_DIR.mkdir(parents=True, exist_ok=True)
    workers = max(1, (os.cpu_count() or 2) - 4)
    with Pool(workers) as pool:
        for split in SPLITS:
            for k in SOURCES:
                t0 = time.time()
                df = load_source(split, k)
                names, addrs = df.business_name.tolist(), df.business_address.tolist()
                chunks = [(names[i:i + CHUNK], addrs[i:i + CHUNK]) for i in range(0, len(df), CHUNK)]
                out = pd.concat(pool.imap(_work, chunks), ignore_index=True)
                out.insert(0, "entity_id", df.entity_id.values)
                out.insert(1, "country", df.country.values)
                out["addr_empty"] = out.addr_norm == ""
                out.to_parquet(NORM_DIR / f"{split}_s{k}.parquet", compression="zstd", index=False)
                print(f"{split} s{k}: {len(out):>10,} rows  {time.time() - t0:5.1f}s")


if __name__ == "__main__":
    main()
