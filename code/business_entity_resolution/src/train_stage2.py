"""Stage 3d: second-stage matcher with cluster features built from stage-1 probabilities.

Inputs are the hold-out features and the stage-1 out-of-fold probabilities written by
train_matcher.py. Evaluated with the same 5-fold GroupKFold by S1 and decision grid.

    python src/train_stage2.py
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from er.cluster import CLUSTER, cluster_features
from er.config import CACHE_DIR
from er.features import FEATURES, Records
from er.holdout import holdout_mask
from er.io import load_norm
from er.labels import pair_key, truth_rows
from train_matcher import PARAMS, decision_grid, fit, report

MODEL_DIR = CACHE_DIR / "model"
FEATURES2 = FEATURES + ["p1"] + CLUSTER


def main(tag: str = ""):
    t0 = time.time()
    f = pd.read_parquet(MODEL_DIR / f"train_holdout_features{tag}.parquet")
    oof = pd.read_parquet(MODEL_DIR / f"oof{tag}.parquet")
    assert (f.s1_row.to_numpy() == oof.s1_row.to_numpy()).all() and (f.q_row.to_numpy() == oof.q_row.to_numpy()).all()
    f["p1"] = oof.p.to_numpy().astype(np.float32)
    cl = cluster_features(f.s1_row.to_numpy(), f.src_id.to_numpy(), f.q_row.to_numpy(), f.p1.to_numpy(), Records("train"))
    for c in CLUSTER:
        f[c] = cl[c].to_numpy()
    print(f"cluster features built  ({time.time() - t0:.0f}s); pairs with a cluster: {(f.cl_n > 0).mean():.1%}")

    X, y, groups = f[FEATURES2], f.label.to_numpy(), f.s1_row.to_numpy()
    oof2, iters = np.zeros(len(f)), []
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        m = fit(X, y, tr, va)
        oof2[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        print(f"  fold {fold}: best_iter {m.best_iteration}, logloss {m.best_score['valid_0']['binary_logloss']:.4f}  ({time.time() - t0:.0f}s)")

    pd.DataFrame({"s1_row": f.s1_row, "src_id": f.src_id, "q_row": f.q_row, "label": f.label, "p": oof2}) \
        .to_parquet(MODEL_DIR / f"oof2{tag}.parquet", index=False)

    s1 = load_norm("train", 1, ["country"])
    country = s1.country.to_numpy()
    held_rows = np.flatnonzero(holdout_mask(len(s1)))
    tr_rows = truth_rows()
    tr_rows = tr_rows[np.isin(tr_rows.s1_row, held_rows)]
    truth = pd.DataFrame({"s1_row": tr_rows.s1_row, "qkey": pair_key(tr_rows.src, tr_rows.t_row)})
    print()
    _, tau, sel = decision_grid(f, oof2, truth, held_rows, " stage 2 (OOF)")
    print("\nbest:", end=" ")
    r = report(f, oof2, tau, sel, truth, held_rows, country)

    n_iter = int(np.mean(iters) * 1.1)
    final = lgb.train(PARAMS, lgb.Dataset(X, y), num_boost_round=n_iter)
    final.save_model(str(MODEL_DIR / f"matcher_stage2{tag}.txt"))
    json.dump({"tau": tau, "selection": sel, "n_iter": n_iter, "oof_macro_f05": r["macro_f05"]},
              open(MODEL_DIR / f"decision_stage2{tag}.json", "w"), indent=2)
    imp = pd.Series(final.feature_importance("gain"), index=FEATURES2).sort_values(ascending=False)
    print("\ntop features by gain:", ", ".join(f"{k} {v / imp.sum():.1%}" for k, v in imp.head(15).items()))
    print(f"saved stage-2 matcher ({n_iter} trees)  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    main(ap.parse_args().tag)
