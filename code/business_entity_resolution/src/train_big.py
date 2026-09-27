"""Stage 3 on cross-fitted training rows (~8x the hold-out).

Rows come from build_features.py --fold f: pairs of S1 fold f scored with an encoder that
never trained on that fold, so their encoder features behave like test features. On the
hold-out, each doubling of matcher training rows gained ~+0.0014 macro F0.5.

Stage 1: a model fit on one fold's rows predicts the other fold (out-of-fold p1 for
stage 2), then a final model is fit on all rows. Stage 2 adds p1 and per-S1 cluster
features. Both stages are evaluated on the usual S1 hold-out (features from the production
encoder, which never trained on it), so scores compare with train_matcher / train_stage2.

Memory: features are loaded column by column into one float32 matrix (no DataFrame copy),
predictions are chunked, and rows are capped (whole S1 groups) at --max-rows.

    python src/train_big.py --folds 0 1 --eval-tag _st --out-tag _big
"""
import argparse
import gc
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from er.cluster import CLUSTER, cluster_features
from er.config import CACHE_DIR
from er.features import FEATURES, Records
from er.holdout import holdout_mask
from er.io import load_norm
from er.labels import pair_key, truth_rows
from predict import s1_chunks
from train_matcher import PARAMS, decision_grid, report

MODEL_DIR = CACHE_DIR / "model"
FEATURES2 = FEATURES + ["p1"] + CLUSTER
ID_COLS = ["s1_row", "q_row", "src_id", "label"]


def load_matrix(files, keep_s1=None):
    """(ids DataFrame sorted by S1, float32 feature matrix) without holding a DataFrame copy."""
    ids = pd.concat([pq.read_table(p, columns=ID_COLS).to_pandas().assign(_fold=i) for i, p in enumerate(files)],
                    ignore_index=True)
    rows = np.arange(len(ids)) if keep_s1 is None else np.flatnonzero(np.isin(ids.s1_row.to_numpy(), keep_s1))
    order = rows[np.argsort(ids.s1_row.to_numpy()[rows], kind="stable")]
    ids = ids.iloc[order].reset_index(drop=True)
    X = np.empty((len(order), len(FEATURES)), dtype=np.float32)
    for j, c in enumerate(FEATURES):
        col = np.concatenate([pq.read_table(p, columns=[c]).column(c).to_numpy() for p in files])
        X[:, j] = col[order]
    return ids, X


def predict_chunked(model, X, idx=None, n_iter=None, chunk=2_000_000):
    idx = np.arange(len(X)) if idx is None else idx
    out = np.empty(len(idx), dtype=np.float32)
    for i in range(0, len(idx), chunk):
        out[i:i + chunk] = model.predict(X[idx[i:i + chunk]], num_iteration=n_iter)
    return out


def cluster_matrix(ids, p1, rec):
    parts, s1 = [], ids.s1_row.to_numpy()
    for lo, hi in s1_chunks(s1, 3_000_000):
        parts.append(cluster_features(s1[lo:hi], ids.src_id.to_numpy()[lo:hi], ids.q_row.to_numpy()[lo:hi],
                                      p1[lo:hi], rec).to_numpy(np.float32))
    return np.vstack(parts)


def train_early(X, y, tr, va):
    d_tr = lgb.Dataset(X[tr], y[tr], free_raw_data=True)
    d_va = lgb.Dataset(X[va], y[va], reference=d_tr, free_raw_data=True)
    return lgb.train(PARAMS, d_tr, num_boost_round=4000, valid_sets=[d_va],
                     callbacks=[lgb.early_stopping(100, verbose=False)])


def main(folds, eval_tag, out_tag, max_rows, oof_groups=None):
    t0 = time.time()
    files = [MODEL_DIR / f"train_fold{f}_features_f{f}.parquet" for f in folds]
    # cap rows by keeping whole S1 groups: all of the first fold, a sample of the rest
    ids_all = pd.concat([pq.read_table(p, columns=["s1_row"]).to_pandas().assign(_fold=i) for i, p in enumerate(files)])
    keep_s1 = None
    if len(ids_all) > max_rows:
        first = ids_all.s1_row[ids_all._fold == 0].unique()
        rest = ids_all.s1_row[ids_all._fold != 0].unique()
        n_first = int((ids_all._fold == 0).sum())
        frac = max(0.0, (max_rows - n_first) / max(len(ids_all) - n_first, 1))
        keep_s1 = np.concatenate([first, np.random.default_rng(0).choice(rest, int(len(rest) * frac), replace=False)])
    del ids_all
    ids, X = load_matrix(files, keep_s1)
    y, fold = ids.label.to_numpy().astype(np.float32), ids._fold.to_numpy()
    if oof_groups:  # fewer, larger out-of-fold groups: e.g. 4 folds -> 2 groups = 3 big fits, not 5
        fold = fold % oof_groups
    n_groups = int(fold.max()) + 1
    print(f"cross-fitted training rows: {len(ids):,} (positives {y.mean():.1%})  ({time.time() - t0:.0f}s)", flush=True)
    ev_ids, Xev = load_matrix([MODEL_DIR / f"train_holdout_features{eval_tag}.parquet"])
    ev = ev_ids.rename(columns={"_fold": "fold"})

    s1 = load_norm("train", 1, ["country"])
    country = s1.country.to_numpy()
    held_rows = np.flatnonzero(holdout_mask(len(s1)))
    tr = truth_rows()
    tr = tr[np.isin(tr.s1_row, held_rows)]
    truth = pd.DataFrame({"s1_row": tr.s1_row, "qkey": pair_key(tr.src, tr.t_row)})

    # ---- stage 1
    p1_oof, iters = np.zeros(len(ids), dtype=np.float32), []
    for i in range(n_groups):
        tr_idx, va_idx = np.flatnonzero(fold != i), np.flatnonzero(fold == i)
        m = train_early(X, y, tr_idx, va_idx)
        p1_oof[va_idx] = predict_chunked(m, X, va_idx, m.best_iteration)
        iters.append(m.best_iteration)
        print(f"  stage 1 group {i}: best_iter {m.best_iteration}  ({time.time() - t0:.0f}s)", flush=True)
        del m; gc.collect()
    n1 = int(np.mean(iters) * 1.1)
    m1 = lgb.train(PARAMS, lgb.Dataset(X, y, free_raw_data=True), num_boost_round=n1)
    p1_ev = predict_chunked(m1, Xev)
    m1.save_model(str(MODEL_DIR / f"matcher{out_tag}.txt"))
    print(f"stage 1 final ({n1} trees)  ({time.time() - t0:.0f}s)", flush=True)
    _, tau1, sel1 = decision_grid(ev, p1_ev, truth, held_rows, " stage 1 on hold-out")
    print("stage 1 best:", end=" ")
    r1 = report(ev, p1_ev, tau1, sel1, truth, held_rows, country)

    # ---- stage 2: cluster features, then a (N, 68) matrix built after freeing the records
    rec = Records("train")
    cl, cl_ev = cluster_matrix(ids, p1_oof, rec), cluster_matrix(ev, p1_ev, rec)
    del rec; gc.collect()
    X2 = np.empty((len(X), len(FEATURES2)), dtype=np.float32)
    X2[:, :len(FEATURES)] = X
    del X; gc.collect()
    X2[:, len(FEATURES)] = p1_oof
    X2[:, len(FEATURES) + 1:] = cl
    Xev2 = np.hstack([Xev, p1_ev[:, None], cl_ev]).astype(np.float32)
    del cl; gc.collect()
    print(f"cluster features built  ({time.time() - t0:.0f}s)", flush=True)
    tr_idx, va_idx = np.flatnonzero(fold != 0), np.flatnonzero(fold == 0)
    m = train_early(X2, y, tr_idx, va_idx)
    n2 = int(m.best_iteration * 1.1)
    print(f"  stage 2 best_iter {m.best_iteration}  ({time.time() - t0:.0f}s)", flush=True)
    del m; gc.collect()
    m2 = lgb.train(PARAMS, lgb.Dataset(X2, y, free_raw_data=True), num_boost_round=n2)
    p2_ev = predict_chunked(m2, Xev2)
    m2.save_model(str(MODEL_DIR / f"matcher_stage2{out_tag}.txt"))
    _, tau2, sel2 = decision_grid(ev, p2_ev, truth, held_rows, " stage 2 on hold-out")
    print("stage 2 best:", end=" ")
    r2 = report(ev, p2_ev, tau2, sel2, truth, held_rows, country)

    json.dump({"tau": tau1, "selection": sel1, "n_iter": n1, "holdout_macro_f05": r1["macro_f05"]},
              open(MODEL_DIR / f"decision{out_tag}.json", "w"), indent=2)
    json.dump({"tau": tau2, "selection": sel2, "n_iter": n2, "holdout_macro_f05": r2["macro_f05"]},
              open(MODEL_DIR / f"decision_stage2{out_tag}.json", "w"), indent=2)
    print(f"saved matcher{out_tag}.txt, matcher_stage2{out_tag}.txt  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--eval-tag", default="_st", help="tag of the hold-out features to evaluate on")
    ap.add_argument("--out-tag", default="_big")
    ap.add_argument("--max-rows", type=int, default=22_000_000)
    ap.add_argument("--oof-groups", type=int, help="group folds (fold %% n) for the stage-1 out-of-fold fits")
    a = ap.parse_args()
    main(a.folds, a.eval_tag, a.out_tag, a.max_rows, a.oof_groups)
