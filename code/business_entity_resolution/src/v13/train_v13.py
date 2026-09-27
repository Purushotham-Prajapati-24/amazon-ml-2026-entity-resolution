"""v13 trainer: train_big flow with v12+v13 side-car features, tuned LightGBM params, and an extra CatBoost stage 2
(averaged with LightGBM stage 2; weight chosen on the hold-out). Outputs matcher_v13.txt, matcher_stage2_v13.txt,
cat_stage2_v13.cbm, decision_stage2_v13.json (tau, cat weight)."""
import sys, gc, json, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, pyarrow.parquet as pq, lightgbm as lgb
from catboost import CatBoostClassifier
import train_big as tb
from er.cluster import CLUSTER
from er.features import FEATURES as BASE, Records
from er.holdout import holdout_mask
from er.io import load_norm
from er.labels import pair_key, truth_rows
from train_matcher import PARAMS, decision_grid, report
from v12feat import V12
from v13feat import V13

HERE = str(__import__("pathlib").Path(__file__).resolve().parents[4] / "data_cache" / "v13") + "/"
M = tb.MODEL_DIR
ALL = BASE + V12 + V13
ALL2 = ALL + ["p1"] + CLUSTER
hp = json.load(open(HERE + "hp_results.json"))
LGBP = {**PARAMS, **hp["best_lgb"]["params"]}
CATP = hp["best_cat"]["params"]
tb.PARAMS = LGBP
print("LightGBM params:", LGBP, "\nCatBoost params:", CATP, flush=True)


def side(p, tag):
    return p.with_name(p.stem + f"_{tag}side.parquet")


def load_matrix(files, keep_s1=None):
    ids = pd.concat([pq.read_table(p, columns=tb.ID_COLS).to_pandas().assign(_fold=i) for i, p in enumerate(files)], ignore_index=True)
    rows = np.arange(len(ids)) if keep_s1 is None else np.flatnonzero(np.isin(ids.s1_row.to_numpy(), keep_s1))
    order = rows[np.argsort(ids.s1_row.to_numpy()[rows], kind="stable")]
    ids = ids.iloc[order].reset_index(drop=True)
    X = np.empty((len(order), len(ALL)), dtype=np.float32)
    for j, c in enumerate(ALL):
        src = [side(p, "v12") if c in V12 else side(p, "v13") if c in V13 else p for p in files]
        X[:, j] = np.concatenate([pq.read_table(s, columns=[c]).column(c).to_numpy() for s in src])[order]
    return ids, X


def main():
    t0 = time.time()
    files = [M / f"train_fold{f}_features_f{f}.parquet" for f in (0, 1)]
    # cap at 22M rows keeping whole S1 groups (as train_big / v12): all of fold 0, a sample of fold 1
    ids_all = pd.concat([pq.read_table(p, columns=["s1_row"]).to_pandas().assign(_fold=i) for i, p in enumerate(files)])
    first, rest = ids_all.s1_row[ids_all._fold == 0].unique(), ids_all.s1_row[ids_all._fold != 0].unique()
    n_first = int((ids_all._fold == 0).sum())
    frac = max(0.0, (22_000_000 - n_first) / max(len(ids_all) - n_first, 1))
    keep_s1 = np.concatenate([first, np.random.default_rng(0).choice(rest, int(len(rest) * frac), replace=False)])
    del ids_all
    ids, X = load_matrix(files, keep_s1)
    y, fold = ids.label.to_numpy().astype(np.float32), ids._fold.to_numpy()
    print(f"rows {len(ids):,} x {X.shape[1]} ({time.time()-t0:.0f}s)", flush=True)
    ev, Xev = load_matrix([M / "train_holdout_features_st.parquet"])
    s1 = load_norm("train", 1, ["country"]); country = s1.country.to_numpy()
    held = np.flatnonzero(holdout_mask(len(s1)))
    tr = truth_rows(); tr = tr[np.isin(tr.s1_row, held)]
    truth = pd.DataFrame({"s1_row": tr.s1_row, "qkey": pair_key(tr.src, tr.t_row)})
    # stage 1 (LightGBM): OOF over the two folds, then a final fit
    p1_oof, iters = np.zeros(len(ids), dtype=np.float32), []
    for i in (0, 1):
        tri, vai = np.flatnonzero(fold != i), np.flatnonzero(fold == i)
        m = tb.train_early(X, y, tri, vai)
        p1_oof[vai] = tb.predict_chunked(m, X, vai, m.best_iteration); iters.append(m.best_iteration)
        print(f"  stage 1 group {i}: best_iter {m.best_iteration} ({time.time()-t0:.0f}s)", flush=True)
        del m; gc.collect()
    n1 = int(np.mean(iters) * 1.1)
    m1 = lgb.train(LGBP, lgb.Dataset(X, y, free_raw_data=True), num_boost_round=n1)
    m1.save_model(str(M / "matcher_v13.txt"))
    p1_ev = tb.predict_chunked(m1, Xev)
    _, tau1, sel1 = decision_grid(ev, p1_ev, truth, held, " stage 1 on hold-out")
    print("stage 1 best:", end=" "); report(ev, p1_ev, tau1, sel1, truth, held, country)
    # stage 2 matrix
    rec = Records("train")
    cl, cl_ev = tb.cluster_matrix(ids, p1_oof, rec), tb.cluster_matrix(ev, p1_ev, rec)
    del rec; gc.collect()
    X2 = np.empty((len(X), len(ALL2)), dtype=np.float32)
    X2[:, :len(ALL)] = X; del X; gc.collect()
    X2[:, len(ALL)] = p1_oof; X2[:, len(ALL) + 1:] = cl; del cl; gc.collect()
    Xev2 = np.hstack([Xev, p1_ev[:, None], cl_ev]).astype(np.float32)
    tri, vai = np.flatnonzero(fold != 0), np.flatnonzero(fold == 0)
    m = tb.train_early(X2, y, tri, vai)
    n2 = int(m.best_iteration * 1.1)
    print(f"  stage 2 LGB best_iter {m.best_iteration} ({time.time()-t0:.0f}s)", flush=True)
    del m; gc.collect()
    # CatBoost stage 2 on a whole-S1 sample: train fold 1 rows (6M), early-stop on fold 0 rows (1.2M)
    rng = np.random.default_rng(0)
    s_tr = rng.choice(tri, min(6_000_000, len(tri)), replace=False)
    s_va = rng.choice(vai, min(1_200_000, len(vai)), replace=False)
    cat = CatBoostClassifier(iterations=6000, loss_function="Logloss", thread_count=-1, random_seed=0, od_type="Iter",
                             od_wait=100, verbose=500, **CATP)
    cat.fit(X2[s_tr], y[s_tr], eval_set=(X2[s_va], y[s_va]), use_best_model=True)
    cat.save_model(str(M / "cat_stage2_v13.cbm"))
    print(f"  stage 2 CatBoost best_iter {cat.get_best_iteration()} ({time.time()-t0:.0f}s)", flush=True)
    m2 = lgb.train(LGBP, lgb.Dataset(X2, y, free_raw_data=True), num_boost_round=n2)
    m2.save_model(str(M / "matcher_stage2_v13.txt"))
    p2l = tb.predict_chunked(m2, Xev2)
    p2c = cat.predict_proba(Xev2)[:, 1].astype(np.float32)
    best = None
    for w in (0.0, 0.3, 0.5, 0.7):
        p = (1 - w) * p2l + w * p2c
        r, tau2, sel2 = decision_grid(ev, p, truth, held, f" stage 2 cat weight {w}")
        print(f"cat weight {w}:", end=" "); rr = report(ev, p, tau2, sel2, truth, held, country)
        if best is None or rr["macro_f05"] > best[0]:
            best = (rr["macro_f05"], w, tau2, sel2)
    print(f"BEST stage 2: macro {best[0]:.5f} cat weight {best[1]} tau {best[2]}", flush=True)
    json.dump({"tau": best[2], "selection": best[3], "cat_weight": best[1], "n_iter": n2, "holdout_macro_f05": best[0]},
              open(M / "decision_stage2_v13.json", "w"), indent=2)
    json.dump({"tau": tau1, "selection": sel1, "n_iter": n1}, open(M / "decision_v13.json", "w"), indent=2)
    print(f"saved v13 models ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
