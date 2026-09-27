"""v12 candidate features (country-agnostic, label-free).

legal_eq_c / legal_conf_c / legal_add_c / legal_drop_c : canonical legal forms (corp=corporation, ltd=limited ...)
num_delta_abs / num_delta_pos : smallest |b-a| between house numbers when none is shared (look-alikes use +1..+11)
aw_share / aw_n : for the words the record ADDS relative to the S1, the per-country rate at which records adding
                  that word keep one of the S1's house numbers (computed on the split's top-1 retrieval pairs);
                  look-alike markers ~0-13%, generator noise words ~85%. min over added words; -1 if none.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from collections import defaultdict
import numpy as np

CANON = {"inc": "inc", "incorporated": "inc", "lnc": "inc", "corp": "corp", "corporation": "corp", "co": "co", "company": "co",
         "ltd": "ltd", "limited": "ltd", "pvt": "pvt", "private": "pvt", "llc": "llc", "plc": "plc", "llp": "llp", "lp": "lp",
         "pllc": "pllc", "pc": "pc", "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "sci": "sci", "eurl": "eurl",
         "snc": "snc", "cie": "cie", "ei": "ei", "scop": "scop", "selarl": "selarl"}
V12 = ["legal_eq_c", "legal_conf_c", "legal_add_c", "legal_drop_c", "num_delta_abs", "num_delta_pos", "aw_share", "aw_n"]


def canon(s):
    return frozenset(CANON[t] for t in s.split() if t in CANON)


def ints(s):
    return [int(t) for t in s.split() if t.isdigit() and len(t) <= 7]


def added_word_stats(a_core, b_core, a_num, b_num, country, min_n=30):
    """{country: {word: (share_num, n)}} from pairs (each target's top-1 retrieval candidate)."""
    cnt = defaultdict(lambda: [0, 0])
    for ac, bc, an, bn, c in zip(a_core, b_core, a_num, b_num, country):
        xb = set(bc.split()) - set(ac.split())
        if not xb or len(xb) > 2:
            continue
        A, B = set(an.split()), set(bn.split())
        if not A:
            continue
        sh = int(bool(A & B))
        for w in xb:
            k = cnt[(c, w)]
            k[0] += sh
            k[1] += 1
    out = defaultdict(dict)
    for (c, w), (s, n) in cnt.items():
        if n >= min_n:
            out[c][w] = (s / n, n)
    return dict(out)


def v12_features(a_norm, b_norm, a_core, b_core, a_num, b_num, country, stats):
    n = len(a_norm)
    out = np.zeros((n, len(V12)), dtype=np.float32)
    for i in range(n):
        la, lb = canon(a_norm[i]), canon(b_norm[i])
        eq = float(la == lb)
        conf = float(bool(la) and bool(lb) and la != lb)
        add = float(bool(lb) and not la)
        drop = float(bool(la) and not lb)
        A, B = set(a_num[i].split()), set(b_num[i].split())
        dabs, dpos = -1.0, 0.0
        if A and B and not (A & B):
            xs, ys = ints(a_num[i]), ints(b_num[i])
            if xs and ys:
                best = min(((y - x) for x in xs for y in ys), key=abs)
                dabs, dpos = float(min(abs(best), 100000)), float(best > 0)
        xb = set(b_core[i].split()) - set(a_core[i].split())
        share, nn = -1.0, 0.0
        st = stats.get(country[i], {})
        for w in xb:
            if w in st:
                s, k = st[w]
                if share < 0 or s < share:
                    share, nn = s, float(k)
        out[i] = (eq, conf, add, drop, dabs, dpos, share, np.log1p(nn))
    return out


def v12_parallel(*cols, stats, workers=12, chunk=400_000):
    n = len(cols[0])
    if n <= chunk:
        return v12_features(*cols, stats)
    from multiprocessing import Pool
    parts = [tuple(c[i:i + chunk] for c in cols) + (stats,) for i in range(0, n, chunk)]
    with Pool(workers) as pool:
        return np.vstack(pool.starmap(v12_features, parts))
