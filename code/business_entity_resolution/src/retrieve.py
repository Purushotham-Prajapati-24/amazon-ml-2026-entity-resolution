"""Stage 2d: candidate generation. For every S2/S3 record, retrieve S1 candidates in the
same country from four retrievers and keep the union:

    enc   learned encoder              top K_ENC
    comb  SVD name+address             top K_SVD
    name  SVD name half                top K_SVD
    addr  SVD address half             top K_SVD   (skipped for empty addresses)

Writes data_cache/cand/{split}_s{k}.parquet with one row per (query, S1) pair:
q_row, s1_row, enc_cos, name_cos, addr_cos, and each retriever's rank (99 = not retrieved).

    python src/retrieve.py --split train
    python src/retrieve.py --split test
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch

from er.config import CACHE_DIR
from er.io import load_norm
from er.knn import topk_search

VEC_DIR = CACHE_DIR / "vec"
CAND_DIR = CACHE_DIR / "cand"
K_ENC, K_SVD = 16, 4
NOT_RETRIEVED = 99
RETRIEVERS = ("enc", "comb", "name", "addr")


def pair_dot(a: np.ndarray, b: np.ndarray, ia: np.ndarray, ib: np.ndarray, chunk: int = 4_000_000) -> np.ndarray:
    """Row-wise dot products a[ia] . b[ib] on the GPU, in chunks."""
    out = np.empty(len(ia), dtype=np.float32)
    ta, tb = torch.from_numpy(a).cuda(), torch.from_numpy(b).cuda()
    for i in range(0, len(ia), chunk):
        x = ta[torch.from_numpy(ia[i:i + chunk]).cuda()].float()
        y = tb[torch.from_numpy(ib[i:i + chunk]).cuda()].float()
        out[i:i + chunk] = (x * y).sum(1).cpu().numpy()
    del ta, tb
    torch.cuda.empty_cache()
    return out


def retrieve_country(e1, v1, eq, vq, addr_empty):
    """Union of retrievers for one country. Inputs are row-aligned local arrays."""
    h = v1.shape[1] // 2
    hits = {
        "enc": topk_search(e1, eq, K_ENC)[1],
        "comb": topk_search(v1, vq, K_SVD)[1],
        "name": topk_search(np.ascontiguousarray(v1[:, :h]), np.ascontiguousarray(vq[:, :h]), K_SVD)[1],
    }
    addr_idx = np.full((len(vq), K_SVD), -1, dtype=np.int32)
    ne = np.flatnonzero(~addr_empty)
    if len(ne):
        addr_idx[ne] = topk_search(np.ascontiguousarray(v1[:, h:]), np.ascontiguousarray(vq[ne, h:]), K_SVD)[1]
    hits["addr"] = addr_idx

    n1 = np.int64(len(e1))
    keys = {r: (np.arange(len(eq), dtype=np.int64)[:, None] * n1 + ix).ravel() for r, ix in hits.items()}
    keys["addr"] = keys["addr"][hits["addr"].ravel() >= 0]
    uniq = np.unique(np.concatenate(list(keys.values())))
    out = pd.DataFrame({"q": (uniq // n1).astype(np.int32), "s1": (uniq % n1).astype(np.int32)})
    for r, ix in hits.items():
        rank = np.full(len(uniq), NOT_RETRIEVED, dtype=np.int8)
        k = ix.shape[1]
        flat_rank = np.tile(np.arange(k, dtype=np.int8), len(ix))
        valid = ix.ravel() >= 0
        pos = np.searchsorted(uniq, keys[r])
        rank[pos] = flat_rank[valid] if r == "addr" else flat_rank
        out[f"{r}_rank"] = rank
    q, s = out.q.to_numpy(), out.s1.to_numpy()
    out["enc_cos"] = pair_dot(e1, eq, s, q)
    out["name_cos"] = 2 * pair_dot(np.ascontiguousarray(v1[:, :h]), np.ascontiguousarray(vq[:, :h]), s, q)
    out["addr_cos"] = 2 * pair_dot(np.ascontiguousarray(v1[:, h:]), np.ascontiguousarray(vq[:, h:]), s, q)
    return out


def run(split: str, tag: str = ""):
    CAND_DIR.mkdir(parents=True, exist_ok=True)
    s1 = load_norm(split, 1, ["entity_id", "country"])
    e1_all, v1_all = np.load(VEC_DIR / f"{split}_s1_enc{tag}.npy"), np.load(VEC_DIR / f"{split}_s1.npy")
    for k in (2, 3):
        t0 = time.time()
        q = load_norm(split, k, ["entity_id", "country", "addr_empty"])
        eq_all, vq_all = np.load(VEC_DIR / f"{split}_s{k}_enc{tag}.npy"), np.load(VEC_DIR / f"{split}_s{k}.npy")
        parts = []
        for c in s1.country.unique():  # open set of country labels, no hard-coding
            i_rows = np.flatnonzero((s1.country == c).to_numpy())
            q_rows = np.flatnonzero((q.country == c).to_numpy())
            if len(q_rows) == 0:
                continue
            part = retrieve_country(e1_all[i_rows], v1_all[i_rows], eq_all[q_rows], vq_all[q_rows],
                                    q.addr_empty.to_numpy()[q_rows])
            part["q"] = q_rows[part.q.to_numpy()].astype(np.int32)
            part["s1"] = i_rows[part.s1.to_numpy()].astype(np.int32)
            parts.append(part)
            print(f"  S{k} {c}: {len(q_rows):,} queries -> {len(part):,} pairs ({len(part) / len(q_rows):.1f}/query)  {time.time() - t0:.0f}s")
        cand = pd.concat(parts, ignore_index=True).rename(columns={"q": "q_row", "s1": "s1_row"})
        cand.to_parquet(CAND_DIR / f"{split}_s{k}{tag}.parquet", index=False)
        print(f"{split} S{k}: {len(cand):,} candidate pairs written  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--tag", default="", help="suffix selecting an experimental encoder, e.g. _loco_us")
    a = ap.parse_args()
    run(a.split, a.tag)
