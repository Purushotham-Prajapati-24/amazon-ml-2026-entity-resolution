"""Stage 3c: score every pruned test candidate with the matcher(s) and write the submission.

Candidates are processed in chunks that each contain complete S1 groups, so stage-1
probabilities and the stage-2 cluster features (which need every pair of an S1) are
computed in one pass. Context features are computed over the full pruned set first.
Writes output/candidate_pairs.tsv (exactly the scored pairs) and
output/matching_results.tsv; stage-1 and stage-2 scores are kept in data_cache/model/.

    python src/predict.py              # stage 1 + stage 2 (default)
    python src/predict.py --stage1     # stage-1 decision only
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from er.candidates import load_candidates, prune
from er.cluster import cluster_features
from er.config import CACHE_DIR, OUTPUT_DIR
from er.decide import assign, expected_f_select
from er.features import FEATURES, Records, add_context, add_name_context, pair_features
from er.io import load_norm, write_id_lists
from train_stage2 import FEATURES2

MODEL_DIR = CACHE_DIR / "model"
CHUNK = 6_000_000


def ids_for(split, s1_row, qkey):
    s1_ids = load_norm(split, 1, ["entity_id"]).entity_id.to_numpy()
    src, row = qkey >> 32, qkey & 0xFFFFFFFF
    match = np.empty(len(qkey), dtype=object)
    for k in (2, 3):
        m = src == k
        match[m] = load_norm(split, k, ["entity_id"]).entity_id.to_numpy()[row[m]]
    return pd.DataFrame({"s1": s1_ids[s1_row], "match": match})


def s1_chunks(s1_sorted: np.ndarray, size: int):
    """Chunk boundaries of ~size rows that never split an S1 group."""
    cuts = [0]
    for i in range(size, len(s1_sorted), size):
        b = int(np.searchsorted(s1_sorted, s1_sorted[i], side="left"))
        if b > cuts[-1]:
            cuts.append(b)
    cuts.append(len(s1_sorted))
    return list(zip(cuts[:-1], cuts[1:]))


def decide_and_write(c, p, cfg, s1):
    a = assign(c.s1_row.to_numpy(), c.src.to_numpy(), c.q_row.to_numpy(), p, cfg["tau"])
    if cfg["selection"] == "expected_f":
        a = expected_f_select(a)
    write_id_lists(ids_for("test", a.s1_row.to_numpy(), a.qkey.to_numpy()), s1.entity_id, "match",
                   OUTPUT_DIR / "matching_results.tsv", "matched_entity_ids")
    n = np.bincount(a.s1_row.to_numpy(), minlength=len(s1))
    st = pd.DataFrame({"country": s1.country.to_numpy(), "n": n}).groupby("country").n
    print(f"predicted {len(a):,} pairs (tau={cfg['tau']}, selection={cfg['selection']})")
    print("share of S1 with >=1 match:", st.apply(lambda s: round((s > 0).mean(), 3)).to_dict())
    print("mean matches per S1:", st.mean().round(3).to_dict())


def main(stage2: bool, tag: str = "", model_tag=None):
    t0 = time.time()
    mt = tag if model_tag is None else model_tag   # matchers can differ from the candidate/encoder tag
    m1 = lgb.Booster(model_file=str(MODEL_DIR / f"matcher{mt}.txt"))
    m2 = lgb.Booster(model_file=str(MODEL_DIR / f"matcher_stage2{mt}.txt")) if stage2 else None
    cfg = json.load(open(MODEL_DIR / (f"decision_stage2{mt}.json" if stage2 else f"decision{mt}.json")))
    rec = Records("test")
    c = add_name_context(add_context(load_candidates("test", prune, tag=tag)), rec)
    c = c.iloc[np.argsort(c.s1_row.to_numpy(), kind="stable")].reset_index(drop=True)
    print(f"pruned test candidates with context: {len(c):,}  ({time.time() - t0:.0f}s)")

    p1 = np.empty(len(c), dtype=np.float32)
    p2 = np.empty(len(c), dtype=np.float32) if stage2 else None
    for lo, hi in s1_chunks(c.s1_row.to_numpy(), CHUNK):
        part = c.iloc[lo:hi]
        f = pair_features(part, rec)
        p1[lo:hi] = m1.predict(f[FEATURES], num_threads=0)
        if stage2:
            f["p1"] = p1[lo:hi]
            cl = cluster_features(part.s1_row.to_numpy(), part.src.to_numpy(), part.q_row.to_numpy(), p1[lo:hi], rec)
            for col in cl.columns:
                f[col] = cl[col].to_numpy()
            p2[lo:hi] = m2.predict(f[FEATURES2], num_threads=0)
        print(f"  scored {hi:,}/{len(c):,}  ({time.time() - t0:.0f}s)")

    scores = pd.DataFrame({"s1_row": c.s1_row, "src": c.src, "q_row": c.q_row, "p1": p1})
    if stage2:
        scores["p2"] = p2
    scores.to_parquet(MODEL_DIR / f"test_scores{mt}.parquet", index=False)

    s1 = load_norm("test", 1, ["entity_id", "country"])
    qkey = c.src.to_numpy().astype(np.int64) << 32 | c.q_row.to_numpy().astype(np.int64)
    write_id_lists(ids_for("test", c.s1_row.to_numpy(), qkey), s1.entity_id, "match",
                   OUTPUT_DIR / "candidate_pairs.tsv", "candidate_entity_ids")
    decide_and_write(c, p2 if stage2 else p1, cfg, s1)
    print(f"done  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1", action="store_true", help="use the stage-1 matcher only")
    ap.add_argument("--tag", default="", help="suffix selecting an experimental encoder/matcher set, e.g. _st")
    ap.add_argument("--model-tag", help="matcher tag if different from --tag (e.g. _big)")
    a = ap.parse_args()
    main(not a.stage1, a.tag, a.model_tag)
