"""Stage 4: average two matchers' test probabilities and write the submission.

The two runs retrieve partly different candidates (different encoders), so the average is
taken over the union of pairs; a pair proposed by only one run keeps that run's score.
On hold-out out-of-fold predictions this averaging adds ~+0.001 macro F0.5 over either
run alone. candidate_pairs.tsv is the union (exactly the pairs the decision step sees).

With --adapted-for-unlabelled, S1 entities of countries that have no training labels
(France) take only the LAST run's scores (the run self-trained on those countries), since
averaging with a run that never adapted to them dilutes that gain. The set of labelled
countries is read from the training data, not hard-coded.

    python src/ensemble_predict.py --scores test_scores.parquet test_scores_st.parquet --tau 0.5
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from er.config import CACHE_DIR, OUTPUT_DIR
from er.decide import assign, expected_f_select
from er.io import load_norm, write_id_lists
from predict import ids_for

MODEL_DIR = CACHE_DIR / "model"
KEY = ["s1_row", "src", "q_row"]


def main(score_files, tau, col, adapted_for_unlabelled=False, out_dir=OUTPUT_DIR, weights=None):
    t0 = time.time()
    runs = []
    for i, name in enumerate(score_files):
        s = pd.read_parquet(name if "/" in name or "\\" in name else MODEL_DIR / name, columns=KEY + [col])
        runs.append(s.rename(columns={col: f"p{i}"}))
    u = runs[0]
    for r in runs[1:]:
        u = u.merge(r, on=KEY, how="outer")
    # weighted mean over the runs that proposed the pair (weights renormalised per pair)
    w = np.asarray(weights if weights else [1.0] * len(runs), dtype=np.float64)
    P = u[[f"p{i}" for i in range(len(runs))]].to_numpy(np.float64)
    have = ~np.isnan(P)
    p = (np.nan_to_num(P) * w).sum(1) / (have * w).sum(1)
    print(f"union of {len(runs)} runs: {len(u):,} pairs ({time.time() - t0:.0f}s)")

    s1 = load_norm("test", 1, ["entity_id", "country"])
    if adapted_for_unlabelled:
        labelled = set(load_norm("train", 1, ["country"]).country.unique())
        unl = ~np.isin(s1.country.to_numpy()[u.s1_row.to_numpy()], list(labelled))
        last = u[f"p{len(runs) - 1}"].to_numpy()
        keep = ~unl | ~np.isnan(last)                    # unlabelled countries: last run's pairs only
        p = np.where(unl, last, p)
        u, p = u[keep].reset_index(drop=True), p[keep]
        print(f"countries without training labels use the last run only: {sorted(set(s1.country) - labelled)}; "
              f"{len(u):,} pairs kept")
    out_dir.mkdir(parents=True, exist_ok=True)
    qkey = u.src.to_numpy().astype(np.int64) << 32 | u.q_row.to_numpy().astype(np.int64)
    write_id_lists(ids_for("test", u.s1_row.to_numpy(), qkey), s1.entity_id, "match",
                   out_dir / "candidate_pairs.tsv", "candidate_entity_ids")
    a = expected_f_select(assign(u.s1_row.to_numpy(), u.src.to_numpy(), u.q_row.to_numpy(), p, tau))
    write_id_lists(ids_for("test", a.s1_row.to_numpy(), a.qkey.to_numpy()), s1.entity_id, "match",
                   out_dir / "matching_results.tsv", "matched_entity_ids")
    n = np.bincount(a.s1_row.to_numpy(), minlength=len(s1))
    st = pd.DataFrame({"country": s1.country.to_numpy(), "n": n}).groupby("country").n
    print(f"predicted {len(a):,} pairs (tau={tau})")
    print("share of S1 with >=1 match:", st.apply(lambda x: round((x > 0).mean(), 3)).to_dict())
    print("mean matches per S1:", st.mean().round(3).to_dict())
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", nargs="+", required=True, help="test_scores parquet files (in data_cache/model or paths)")
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--col", default="p2", help="probability column to average (p2 = stage 2)")
    ap.add_argument("--adapted-for-unlabelled", action="store_true")
    ap.add_argument("--out", type=Path, default=OUTPUT_DIR)
    ap.add_argument("--weights", nargs="+", type=float, help="one weight per score file (default: equal)")
    a = ap.parse_args()
    main(a.scores, a.tau, a.col, a.adapted_for_unlabelled, a.out, a.weights)
