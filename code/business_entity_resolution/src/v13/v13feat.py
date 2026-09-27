"""v13 candidate features (label-free, country-agnostic), on top of v12.

dw_share : for words the record DROPS from the S1 name, per-country share of top-1 pairs dropping that word that keep an
           S1 house number (min over dropped words; -1 if none)
aw_addr  : for words the record ADDS, per-country share of top-1 pairs adding it whose street part (address without
           digits) equals the S1's (min; -1 if none)
hn_delta : signed delta of the FIRST number of each address (house number), clipped to +-1000; hn_same flag
street_eq: address with digits removed, token-set equal (-1 if either empty)
"""
import re
from collections import defaultdict
import numpy as np

V13 = ["dw_share", "aw_addr", "hn_delta", "hn_same", "street_eq"]
_D = re.compile(r"\d+")


def street(a):
    return frozenset(t for t in _D.sub(" ", a).split() if len(t) > 1)


def first_int(nums):
    for t in nums.split():
        if t.isdigit() and len(t) <= 7:
            return int(t)
    return None


def word_stats(a_core, b_core, a_num, b_num, a_addr, b_addr, country, min_n=30):
    drop = defaultdict(lambda: [0, 0]); adda = defaultdict(lambda: [0, 0])
    for ac, bc, an, bn, aa, ba, c in zip(a_core, b_core, a_num, b_num, a_addr, b_addr, country):
        A, B = set(ac.split()), set(bc.split())
        xa, xb = A - B, B - A
        NA, NB = set(an.split()), set(bn.split())
        if xa and len(xa) <= 2 and NA:
            sh = int(bool(NA & NB))
            for w in xa:
                k = drop[(c, w)]; k[0] += sh; k[1] += 1
        if xb and len(xb) <= 2 and aa and ba:
            eq = int(street(aa) == street(ba))
            for w in xb:
                k = adda[(c, w)]; k[0] += eq; k[1] += 1
    out = {"drop": defaultdict(dict), "adda": defaultdict(dict)}
    for name, d in (("drop", drop), ("adda", adda)):
        for (c, w), (s, n) in d.items():
            if n >= min_n:
                out[name][c][w] = s / n
    return {k: dict(v) for k, v in out.items()}


def v13_features(a_core, b_core, a_num, b_num, a_addr, b_addr, country, stats):
    n = len(a_core)
    out = np.zeros((n, len(V13)), dtype=np.float32)
    D, AA = stats["drop"], stats["adda"]
    for i in range(n):
        A, B = set(a_core[i].split()), set(b_core[i].split())
        xa, xb = A - B, B - A
        sd = D.get(country[i], {}); sa = AA.get(country[i], {})
        dws = [sd[w] for w in xa if w in sd]
        aws = [sa[w] for w in xb if w in sa]
        x, y = first_int(a_num[i]), first_int(b_num[i])
        hd, hs = (0.0, 0.0) if x is None or y is None else (float(np.clip(y - x, -1000, 1000)), float(x == y))
        if x is None or y is None:
            hd = -2000.0
        sa_, sb_ = street(a_addr[i]), street(b_addr[i])
        se = -1.0 if not sa_ or not sb_ else float(sa_ == sb_)
        out[i] = (min(dws) if dws else -1.0, min(aws) if aws else -1.0, hd, hs, se)
    return out


def v13_parallel(*cols, stats, workers=12, chunk=400_000):
    n = len(cols[0])
    if n <= chunk:
        return v13_features(*cols, stats)
    from multiprocessing import Pool
    parts = [tuple(c[i:i + chunk] for c in cols) + (stats,) for i in range(0, n, chunk)]
    with Pool(workers) as pool:
        return np.vstack(pool.starmap(v13_features, parts))
