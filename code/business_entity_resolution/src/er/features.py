"""Pairwise features for the matching model.

Three groups, all country-agnostic (the test set adds France):
- retrieval: encoder / SVD cosines and each retriever's rank
- context:   how the pair competes with the other candidates of the same S2/S3
             record and of the same S1 entity (computed over the full pruned set)
- string:    name / address similarities, house numbers, states, legal forms, filler
             words. Hard negatives in this data are "siblings": same core name and a
             near-identical address but a different house number, a changed legal
             form or an added filler word ("Group", "Holdings").
"""
from collections import Counter

import numpy as np
import pandas as pd
import pyarrow.compute as pc
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein

from .config import CACHE_DIR
from .io import load_norm
from .normalize import LEGAL_WORDS

NORM_COLS = ["country", "name_norm", "name_core", "name_key", "addr_norm", "numbers", "states", "addr_empty", "name_url"]
LEGAL_FORMS = LEGAL_WORDS - {"the", "and", "of", "et"}
# Words appended to generated hard negatives (siblings) vs. noise words appended to true
# positive records; mixing them in one feature blurs opposite signals.
# The French slice (test only) uses a translated vocabulary: extra tokens on confident
# France matches show "fils", "associes" (replacing "partners"), "france", "developpement",
# "frs"; the sibling markers are taken as the translations "groupe" / "holding".
HARD_NEG_FILLER = frozenset({"group", "holdings", "groupe", "holding"})
POSITIVE_NOISE = frozenset(
    "services service center partners international enterprise solution solutions associate "
    "associates trust global worldwide consulting consultants shop store brothers sons agency "
    "management ventures trading labs sys "
    "fils freres frs associes france developpement".split()
)
RETRIEVAL = ["enc_cos", "name_cos", "addr_cos", "enc_rank", "comb_rank", "name_rank", "addr_rank"]
CONTEXT = ["q_n", "q_rank", "q_margin", "q_gap", "s1_n", "s1_rank", "s1_margin", "s1_load",
           "q_name_rank", "q_name_margin", "q_n_name95"]
# how common a name key is: chains / ambiguous names (S1 side) and mutated-name sibling
# clusters that exist only among S2/S3 records (target side)
FREQ = ["s1_kf_s1", "s1_kf_tgt", "q_kf_tgt", "q_kf_s1", "name_key_eq"]
PY_FEATS = [
    "num_shared", "num_jacc", "num_conflict", "a_has_num", "b_has_num", "state_agree",
    "legal_a", "legal_b", "legal_eq", "legal_conflict", "extra_a", "extra_b",
    "extra_a_hardneg", "extra_b_hardneg", "extra_a_noise", "extra_b_noise",
    "core_jacc", "acronym", "len_a", "len_b", "ntok_a", "ntok_b",
]
STR_FEATS = [
    "name_ratio", "name_tset", "name_tsort", "name_partial", "name_jw", "name_nospace",
    "addr_ratio", "addr_tset", "addr_partial",
]
# b_pseudo: the record name is a single invented word (>= 5 letters, never used in any S1
# name). Such records are almost always true matches whose only evidence is the address
# (1.7% of hold-out pairs but ~8% of misses and of false merges). b_short: 2-4 letter
# single-token names (acronyms such as "UA" for "Union des Allo").
FLAGS = ["src", "b_addr_empty", "b_name_url", "b_name_ascii", "b_pseudo", "b_short"]
# v11: which words / numbers the record introduces relative to the S1 (hold-out A/B, stage 1:
# +0.0014 macro F0.5). A record that brings in a genuine name word of its country (one the
# variant generator never adds) is a look-alike 49% of the time among negatives vs 0.5% of
# true pairs; look-alikes also add or change house numbers (typo-like changes are noise).
V11 = ["num_added", "num_dropped", "num_typo", "intro_name_word", "gen_word_added", "marker_added"]
MARKERS = frozenset({"group", "holdings", "downtown", "uptown", "riverside", "central", "west",
                     "groupe", "holding", "participations"})
FEATURES = RETRIEVAL + CONTEXT + FREQ + PY_FEATS + STR_FEATS + FLAGS + V11
LEET = str.maketrans("01345", "olaes")


def related(a: str, b: str) -> bool:
    """Could word b be a spelling / abbreviation / leetspeak variant of word a?"""
    a2, b2 = a.translate(LEET), b.translate(LEET)
    if a2 == b2:
        return True
    s, l = sorted((a2, b2), key=len)
    if l.startswith(s) or l.endswith(s):
        return True
    if len(s) <= 4 and s[:1] == l[:1] and all(ch in iter(l) for ch in s):
        return True
    return Levenshtein.normalized_similarity(a2, b2) >= 0.5


# --------------------------------------------------------------------------- records
class Records:
    """Normalised columns of S1 and both target sources for one split, as numpy arrays."""

    def __init__(self, split: str):
        self.s1 = {c: v.to_numpy() for c, v in load_norm(split, 1, NORM_COLS).items()}
        self.tgt = {}
        for k in (2, 3):
            d = {c: v.to_numpy() for c, v in load_norm(split, k, NORM_COLS).items()}
            raw = pd.read_parquet(CACHE_DIR / f"{split}_source{k}.parquet", columns=["business_name"])
            d["name_ascii"] = pc.string_is_ascii(raw.business_name.array._pa_array).to_numpy(zero_copy_only=False)
            d["name_handle"] = raw.business_name.str.match(r"^\s*[@#]").to_numpy()
            self.tgt[k] = d
        self._name_key_counts()
        self._name_kinds()
        self._country_vocab()

    def _country_vocab(self, min_s1: int = 20, ratio_name: float = 1.3, ratio_gen: float = 3.0):
        """Per country, from this split's own names (no labels): name vocabulary = words in
        >= min_s1 S1 names not over-represented among S2/S3 names; generator words = words
        >= 3x over-represented among S2/S3 names."""
        self.vocab, self.genw = {}, {}
        for c in np.unique(self.s1["country"]):
            sc, tc, nt = Counter(), Counter(), 0
            m1 = self.s1["country"] == c
            for n in self.s1["name_core"][m1]:
                sc.update(set(n.split()))
            for k in (2, 3):
                mt = self.tgt[k]["country"] == c
                nt += int(mt.sum())
                for n in self.tgt[k]["name_core"][mt]:
                    tc.update(set(n.split()))
            n1, nt = max(int(m1.sum()), 1), max(nt, 1)
            self.vocab[c] = {w for w, n in sc.items() if n >= min_s1 and (tc.get(w, 0) / nt) / (n / n1) <= ratio_name}
            self.genw[c] = {w for w, n in tc.items() if n >= 50 and (n / nt) / (max(sc.get(w, 0), 1) / n1) >= ratio_gen}

    def _name_kinds(self):
        vocab = set()
        for n in self.s1["name_norm"]:
            vocab.update(n.split())
        for k in (2, 3):
            t = self.tgt[k]
            pseudo = np.zeros(len(t["name_norm"]), dtype=np.float32)
            short = np.zeros_like(pseudo)
            # transliterated Indic names, URLs and @handles also collapse to one unknown token
            plain = t["name_ascii"] & ~t["name_url"].astype(bool) & ~t["name_handle"]
            for i, n in enumerate(t["name_norm"]):
                if plain[i] and n and " " not in n and n.isalpha():
                    if len(n) >= 5 and n not in vocab:
                        pseudo[i] = 1.0
                    elif 2 <= len(n) <= 4:
                        short[i] = 1.0
            self.tgt[k]["pseudo"], self.tgt[k]["short"] = pseudo, short

    def _name_key_counts(self):
        """Counts of (country, name_key) among S1 and among all S2/S3 records of the split."""
        h = lambda d: pd.util.hash_array((d["country"] + "|" + d["name_key"]).astype(object))
        hs1, ht = h(self.s1), {k: h(self.tgt[k]) for k in (2, 3)}
        c_s1 = pd.Series(hs1).value_counts()
        c_t = pd.Series(np.concatenate([ht[2], ht[3]])).value_counts()
        look = lambda counts, keys: counts.reindex(keys).fillna(0).to_numpy(np.float32)
        self.s1["kf_s1"], self.s1["kf_tgt"] = look(c_s1, hs1), look(c_t, hs1)
        for k in (2, 3):
            self.tgt[k]["kf_tgt"], self.tgt[k]["kf_s1"] = look(c_t, ht[k]), look(c_s1, ht[k])

    def target(self, col: str, src: np.ndarray, q_row: np.ndarray) -> np.ndarray:
        out = np.empty(len(q_row), dtype=self.tgt[2][col].dtype)
        for k in (2, 3):
            m = src == k
            out[m] = self.tgt[k][col][q_row[m]]
        return out


# --------------------------------------------------------------------------- context
def _group_rank(key: np.ndarray, score: np.ndarray):
    """Rank (0 = best) of each row's score within its key group, plus group size,
    group best and group second best, all aligned to the input order."""
    order = np.lexsort((-score, key))
    k, s = key[order], score[order]
    start = np.r_[True, k[1:] != k[:-1]]
    gid = np.cumsum(start) - 1
    first = np.flatnonzero(start)
    size = np.diff(np.r_[first, len(k)])
    rank = np.arange(len(k)) - first[gid]
    best = s[first]
    second = np.where(size > 1, s[np.minimum(first + 1, len(s) - 1)], -1.0)
    out = [np.empty_like(x) for x in (rank, size[gid], best[gid], second[gid])]
    for o, x in zip(out, (rank, size[gid], best[gid], second[gid])):
        o[order] = x
    return out


def add_context(c: pd.DataFrame) -> pd.DataFrame:
    """Competition features over the full pruned candidate set of a split."""
    enc = c.enc_cos.to_numpy().astype(np.float32)
    qkey = c.src.to_numpy().astype(np.int64) << 32 | c.q_row.to_numpy().astype(np.int64)
    rank, size, best, second = _group_rank(qkey, enc)
    c["q_n"], c["q_rank"] = size.astype(np.int16), rank.astype(np.int16)
    c["q_margin"] = np.where(rank == 0, enc - second, enc - best).astype(np.float32)
    c["q_gap"] = (best - second).astype(np.float32)
    rank, size, best, second = _group_rank(c.s1_row.to_numpy().astype(np.int64), enc)
    c["s1_n"], c["s1_rank"] = size.astype(np.int32), rank.astype(np.int32)
    c["s1_margin"] = np.where(rank == 0, enc - second, enc - best).astype(np.float32)
    top1 = c.s1_row.to_numpy()[c.q_rank.to_numpy() == 0]
    load = np.bincount(top1, minlength=int(c.s1_row.max()) + 1)
    c["s1_load"] = load[c.s1_row.to_numpy()].astype(np.int32)
    return c


def add_name_context(c: pd.DataFrame, rec: Records, chunk: int = 5_000_000) -> pd.DataFrame:
    """Name-similarity competition per query over the full pruned set: rank of this S1's
    name similarity, margin over the best other candidate, and how many candidates have a
    near-identical name (ambiguity, decisive for records without an address)."""
    tset = np.empty(len(c), dtype=np.float32)
    s1r, q, src = c.s1_row.to_numpy(), c.q_row.to_numpy(), c.src.to_numpy()
    for i in range(0, len(c), chunk):
        sl = slice(i, i + chunk)
        tset[sl] = _cp(list(rec.s1["name_core"][s1r[sl]]), list(rec.target("name_core", src[sl], q[sl])),
                       fuzz.token_set_ratio)
    qkey = src.astype(np.int64) << 32 | q.astype(np.int64)
    rank, size, best, second = _group_rank(qkey, tset)
    c["q_name_rank"] = rank.astype(np.int16)
    c["q_name_margin"] = np.where(rank == 0, tset - second, tset - best).astype(np.float32)
    _, inv = np.unique(qkey, return_inverse=True)
    c["q_n_name95"] = np.bincount(inv, weights=(tset >= 95))[inv].astype(np.int16)
    return c


# --------------------------------------------------------------------------- string features
def _py_features(a_norm, b_norm, a_core, b_core, a_num, b_num, a_st, b_st) -> np.ndarray:
    out = np.zeros((len(a_norm), len(PY_FEATS)), dtype=np.float32)
    for i in range(len(a_norm)):
        an, bn = set(a_num[i].split()), set(b_num[i].split())
        shared = len(an & bn)
        union = len(an | bn)
        sa, sb = set(a_st[i].split()), set(b_st[i].split())
        state = 0.5 if not sa or not sb else float(bool(sa & sb))
        la = {t for t in a_norm[i].split() if t in LEGAL_FORMS}
        lb = {t for t in b_norm[i].split() if t in LEGAL_FORMS}
        ta, tb = a_core[i].split(), b_core[i].split()
        seta, setb = set(ta), set(tb)
        xa, xb = seta - setb, setb - seta
        ia = "".join(t[0] for t in ta)
        ib = "".join(t[0] for t in tb)
        acro = float((len(ta) >= 2 and ia == b_core[i].replace(" ", "")) or
                     (len(tb) >= 2 and ib == a_core[i].replace(" ", "")))
        out[i] = (
            shared, shared / union if union else 0.0, float(bool(an) and bool(bn) and not shared),
            float(bool(an)), float(bool(bn)), state,
            len(la), len(lb), float(la == lb), float(bool(la) and bool(lb) and not (la & lb)),
            len(xa), len(xb), float(bool(xa & HARD_NEG_FILLER)), float(bool(xb & HARD_NEG_FILLER)),
            float(bool(xa) and xa <= POSITIVE_NOISE), float(bool(xb) and xb <= POSITIVE_NOISE),
            len(seta & setb) / max(len(seta | setb), 1), acro,
            len(a_core[i]), len(b_core[i]), len(ta), len(tb),
        )
    return out


def _py_features_parallel(*cols, workers: int = 12, chunk: int = 400_000) -> np.ndarray:
    n = len(cols[0])
    if n <= chunk:
        return _py_features(*cols)
    from multiprocessing import Pool
    parts = [tuple(c[i:i + chunk] for c in cols) for i in range(0, n, chunk)]
    with Pool(workers) as pool:
        return np.vstack(pool.starmap(_py_features, parts))


def _v11_features(a_core, b_core, a_num, b_num, cty, vocab, genw) -> np.ndarray:
    out = np.zeros((len(a_core), len(V11)), dtype=np.float32)
    for i in range(len(a_core)):
        an, bn = set(a_num[i].split()), set(b_num[i].split())
        typo = -1.0
        if an and bn and not (an & bn):
            typo = max(Levenshtein.normalized_similarity(x, y) for x in an for y in bn)
        A, B = set(a_core[i].split()), set(b_core[i].split())
        xa, xb = A - B, B - A
        intro = gen = marker = 0.0
        if xb:
            V, G = vocab.get(cty[i], ()), genw.get(cty[i], ())
            intro = float(any(w in V and not any(related(w, v) for v in xa) for w in xb))
            gen = float(sum(w in G for w in xb))
            marker = float(bool(xb & MARKERS))
        out[i] = (len(bn - an), len(an - bn), typo, intro, gen, marker)
    return out


def _v11_parallel(a_core, b_core, a_num, b_num, cty, vocab, genw, workers: int = 12, chunk: int = 400_000):
    n = len(a_core)
    if n <= chunk:
        return _v11_features(a_core, b_core, a_num, b_num, cty, vocab, genw)
    from multiprocessing import Pool
    parts = [(a_core[i:i + chunk], b_core[i:i + chunk], a_num[i:i + chunk], b_num[i:i + chunk], cty[i:i + chunk], vocab, genw)
             for i in range(0, n, chunk)]
    with Pool(workers) as pool:
        return np.vstack(pool.starmap(_v11_features, parts))


def _cp(a, b, scorer) -> np.ndarray:
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def pair_features(c: pd.DataFrame, rec: Records) -> pd.DataFrame:
    """Feature frame for candidate rows ``c`` (needs retrieval + context columns)."""
    s1r, q, src = c.s1_row.to_numpy(), c.q_row.to_numpy(), c.src.to_numpy()
    A = {k: rec.s1[k][s1r] for k in ("name_norm", "name_core", "addr_norm", "numbers", "states")}
    B = {k: rec.target(k, src, q) for k in ("name_norm", "name_core", "addr_norm", "numbers", "states",
                                             "addr_empty", "name_url", "name_ascii")}
    f = pd.DataFrame({k: c[k].to_numpy() for k in RETRIEVAL + CONTEXT})
    f["s1_kf_s1"], f["s1_kf_tgt"] = rec.s1["kf_s1"][s1r], rec.s1["kf_tgt"][s1r]
    f["q_kf_tgt"], f["q_kf_s1"] = rec.target("kf_tgt", src, q), rec.target("kf_s1", src, q)
    f["name_key_eq"] = (rec.s1["name_key"][s1r] == rec.target("name_key", src, q)).astype(np.float32)
    py = _py_features_parallel(A["name_norm"], B["name_norm"], A["name_core"], B["name_core"],
                               A["numbers"], B["numbers"], A["states"], B["states"])
    for j, name in enumerate(PY_FEATS):
        f[name] = py[:, j]
    an, bn = list(A["name_core"]), list(B["name_core"])
    f["name_ratio"] = _cp(an, bn, fuzz.ratio)
    f["name_tset"] = _cp(an, bn, fuzz.token_set_ratio)
    f["name_tsort"] = _cp(an, bn, fuzz.token_sort_ratio)
    f["name_partial"] = _cp(an, bn, fuzz.partial_ratio)
    f["name_jw"] = _cp(an, bn, JaroWinkler.normalized_similarity)
    f["name_nospace"] = _cp([s.replace(" ", "") for s in an], [s.replace(" ", "") for s in bn], fuzz.ratio)
    aa, ba = list(A["addr_norm"]), list(B["addr_norm"])
    f["addr_ratio"] = _cp(aa, ba, fuzz.ratio)
    f["addr_tset"] = _cp(aa, ba, fuzz.token_set_ratio)
    f["addr_partial"] = _cp(aa, ba, fuzz.partial_ratio)
    f["src"] = src.astype(np.float32)
    f["b_addr_empty"] = B["addr_empty"].astype(np.float32)
    f["b_name_url"] = B["name_url"].astype(np.float32)
    f["b_name_ascii"] = B["name_ascii"].astype(np.float32)
    f["b_pseudo"] = rec.target("pseudo", src, q)
    f["b_short"] = rec.target("short", src, q)
    v11 = _v11_parallel(A["name_core"], B["name_core"], A["numbers"], B["numbers"],
                        rec.s1["country"][s1r], rec.vocab, rec.genw)
    for j, name in enumerate(V11):
        f[name] = v11[:, j]
    return f[FEATURES]
