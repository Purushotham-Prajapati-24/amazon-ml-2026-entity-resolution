"""Second-stage "cluster" features from first-stage match probabilities.

A true S1 entity has ~3.5 matching records. For each candidate pair (S1 s, record q),
look at the *other* records that stage 1 already links to s with p >= CONF and ask how
well q agrees with that cluster: name / address similarity to its best member, shared
house numbers, cluster size and mass. Address-less records can join a cluster by name;
mutated-name siblings disagree with it.

Only pairs of the same S1 are used, so the features are identical whether a split
contains every candidate (test) or only hold-out S1 rows (training); training uses
out-of-fold stage-1 probabilities, which are OOF per S1 because folds are grouped by S1.
"""
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

from .features import Records

CONF = 0.5
CLUSTER = ["cl_n", "cl_sum_p", "cl_max_p", "cl_name_max", "cl_addr_max", "cl_num_share", "cl_same_src"]


def _cp(a, b, scorer):
    return process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)


def cluster_features(s1_row, src, q_row, p, rec: Records) -> pd.DataFrame:
    n = len(s1_row)
    pairs = pd.DataFrame({"s1_row": s1_row, "src": src, "q_row": q_row, "p": p, "i": np.arange(n)})
    members = pairs[pairs.p >= CONF][["s1_row", "src", "q_row", "p", "i"]]
    j = pairs.merge(members, on="s1_row", suffixes=("", "_m"))
    j = j[j.i != j.i_m]                                  # exclude the pair itself
    out = pd.DataFrame(0.0, index=np.arange(n), columns=CLUSTER, dtype=np.float32)
    if len(j):
        s_q, r_q = j.src.to_numpy(), j.q_row.to_numpy()
        s_m, r_m = j.src_m.to_numpy(), j.q_row_m.to_numpy()
        j["name"] = _cp(list(rec.target("name_core", s_q, r_q)), list(rec.target("name_core", s_m, r_m)), fuzz.token_set_ratio)
        a_q, a_m = rec.target("addr_norm", s_q, r_q), rec.target("addr_norm", s_m, r_m)
        addr = _cp(list(a_q), list(a_m), fuzz.token_set_ratio)
        j["addr"] = np.where((a_q == "") | (a_m == ""), -1.0, addr)
        n_q, n_m = rec.target("numbers", s_q, r_q), rec.target("numbers", s_m, r_m)
        j["num"] = [float(bool(a) and bool(b) and not set(a.split()).isdisjoint(b.split())) for a, b in zip(n_q, n_m)]
        j["same_src"] = (s_q == s_m).astype(np.float32)
        g = j.groupby("i")
        agg = pd.DataFrame({
            "cl_n": g.size(), "cl_sum_p": g.p_m.sum(), "cl_max_p": g.p_m.max(),
            "cl_name_max": g.name.max(), "cl_addr_max": g.addr.max(), "cl_num_share": g.num.max(),
            "cl_same_src": g.same_src.max(),
        })
        out.loc[agg.index, CLUSTER] = agg[CLUSTER].to_numpy(np.float32)
    empty = out.cl_n.to_numpy() == 0
    out.loc[empty, ["cl_name_max", "cl_addr_max", "cl_num_share"]] = -1.0  # no cluster to compare with
    return out
