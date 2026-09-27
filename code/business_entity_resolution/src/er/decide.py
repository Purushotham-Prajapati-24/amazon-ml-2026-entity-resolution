"""Turn pair probabilities into per-S1 match lists.

1. Assignment: every S2/S3 record goes to at most one S1 (the ground truth never links
   a record to two), its highest-probability candidate, if p >= tau.
2. Per-S1 selection (optional): sort an S1's assigned records by p and keep the prefix
   k that maximises expected F0.5,
        EF(k) = 1.25 * S_k / (0.25 * T + k),   S_k = sum of top-k p,  T = sum of all p,
   against EF(0) = prod(1 - p), the chance the S1 is really a singleton. This adapts the
   cut-off per entity instead of using one global threshold.
"""
import numpy as np
import pandas as pd

from .labels import pair_key

BETA2 = 0.25


def assign(s1_row, src, q_row, p, tau: float) -> pd.DataFrame:
    qk = pair_key(src, q_row)
    order = np.lexsort((-p, qk))
    first = np.r_[True, qk[order][1:] != qk[order][:-1]]
    best = order[first]
    best = best[p[best] >= tau]
    return pd.DataFrame({"s1_row": np.asarray(s1_row)[best], "qkey": qk[best], "p": p[best]})


def expected_f_select(a: pd.DataFrame) -> pd.DataFrame:
    s1, p = a.s1_row.to_numpy(), a.p.to_numpy().astype(np.float64)
    order = np.lexsort((-p, s1))
    s1, p = s1[order], p[order]
    start = np.r_[True, s1[1:] != s1[:-1]]
    gid = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    cs = np.cumsum(p)
    base = np.r_[0.0, cs[first[1:] - 1]] if len(first) else np.zeros(0)
    s_k = cs - base[gid]
    total = np.add.reduceat(p, first)
    k = np.arange(len(p)) - first[gid] + 1
    ef = (1 + BETA2) * s_k / (BETA2 * total[gid] + k)
    ef0 = np.exp(np.add.reduceat(np.log1p(-np.clip(p, 0, 1 - 1e-9)), first))
    best = np.maximum.reduceat(ef, first)
    # first position reaching the group max = chosen prefix length
    is_best = ef >= best[gid] - 1e-12
    best_k = np.full(len(first), np.iinfo(np.int64).max)
    np.minimum.at(best_k, gid[is_best], k[is_best])
    keep = (k <= best_k[gid]) & (best[gid] > ef0[gid])
    return a.iloc[order[keep]]
