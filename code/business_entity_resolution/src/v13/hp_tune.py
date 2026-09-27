"""Time-boxed hyper-parameter search (LightGBM + CatBoost) on the S1-grouped hold-out slice, v12+v13 features."""
import sys, time, json
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import log_loss
from catboost import CatBoostClassifier
from er.config import CACHE_DIR
from er.decide import assign, expected_f_select
from er.features import FEATURES
from er.labels import pair_key, truth_rows
from er.metrics import macro_f05
from v12feat import V12
from v13feat import V13

M = CACHE_DIR / "model"
HERE = str(__import__("pathlib").Path(__file__).resolve().parents[4] / "data_cache" / "v13") + "/"
COLS = FEATURES + V12 + V13
f = pd.read_parquet(M / "train_holdout_features_st.parquet")
for tag, cs in (("v12", V12), ("v13", V13)):
    side = pd.read_parquet(M / f"train_holdout_features_st_{tag}side.parquet")
    for c in cs:
        f[c] = side[c].to_numpy()
s1 = np.unique(f.s1_row.to_numpy())
perm = np.random.default_rng(42).permutation(s1)
n = len(s1)
tr_s1, te_s1 = perm[: int(0.2 * n)], perm[int(0.2 * n): int(0.4 * n)]
va_s1 = tr_s1[: int(0.1 * len(tr_s1))]
g = f.s1_row.to_numpy()
tr, va, te = f[np.isin(g, tr_s1) & ~np.isin(g, va_s1)], f[np.isin(g, va_s1)], f[np.isin(g, te_s1)]
Xtr, ytr = tr[COLS].to_numpy(np.float32), tr.label.to_numpy().astype(int)
Xva, yva = va[COLS].to_numpy(np.float32), va.label.to_numpy().astype(int)
Xte, yte = te[COLS].to_numpy(np.float32), te.label.to_numpy().astype(int)
t = truth_rows(); t = t[np.isin(t.s1_row, te_s1)]
truth = pd.DataFrame({"s1": t.s1_row.to_numpy(), "match": pair_key(t.src, t.t_row)})


def f05(p):
    return max(macro_f05(expected_f_select(assign(te.s1_row.to_numpy(), te.src_id.to_numpy(), te.q_row.to_numpy(), p, tau))
                         .rename(columns={"s1_row": "s1", "qkey": "match"}), truth, pd.Series(te_s1))["macro_f05"]
               for tau in (0.5, 0.6, 0.7))


BASE = dict(objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100, feature_fraction=0.8,
            bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=0, seed=0)
LGB = [("base (current)", {}), ("leaves 255 / mdl 50", dict(num_leaves=255, min_data_in_leaf=50)),
       ("leaves 63 / mdl 200", dict(num_leaves=63, min_data_in_leaf=200)), ("ff 0.6", dict(feature_fraction=0.6)),
       ("ff 0.5 / l2 5", dict(feature_fraction=0.5, lambda_l2=5.0)), ("l2 10", dict(lambda_l2=10.0)),
       ("lr 0.03", dict(learning_rate=0.03)), ("mdl 30", dict(min_data_in_leaf=30)),
       ("leaves 255 / ff 0.6 / l2 5", dict(num_leaves=255, feature_fraction=0.6, lambda_l2=5.0, min_data_in_leaf=50)),
       ("max_bin 511", dict(max_bin=511)), ("depth 10 / leaves 255", dict(max_depth=10, num_leaves=255))]
CAT = [("depth 8 lr .08 l2 3", dict(depth=8, learning_rate=0.08, l2_leaf_reg=3)),
       ("depth 6 lr .1 l2 3", dict(depth=6, learning_rate=0.1, l2_leaf_reg=3)),
       ("depth 10 lr .08 l2 3", dict(depth=10, learning_rate=0.08, l2_leaf_reg=3)),
       ("depth 8 lr .08 l2 10", dict(depth=8, learning_rate=0.08, l2_leaf_reg=10)),
       ("depth 8 lr .05 l2 3 border 254", dict(depth=8, learning_rate=0.05, l2_leaf_reg=3, border_count=254))]
res = {"lgb": [], "cat": []}
best_lgb = best_cat = None
for name, extra in LGB:
    t0 = time.time()
    P = {**BASE, **extra}
    m = lgb.train(P, lgb.Dataset(Xtr, ytr), 8000, valid_sets=[lgb.Dataset(Xva, yva)], callbacks=[lgb.early_stopping(100, verbose=False)])
    p = np.clip(m.predict(Xte, num_iteration=m.best_iteration), 1e-6, 1 - 1e-6)
    r = dict(name=name, params=extra, f05=f05(p), logloss=log_loss(yte, p), iters=m.best_iteration)
    res["lgb"].append(r)
    if best_lgb is None or r["logloss"] < best_lgb[0]["logloss"]:
        best_lgb = (r, p)
    print(f"LGB {name:30s} F0.5 {r['f05']:.5f} logloss {r['logloss']:.5f} ({r['iters']} it, {time.time()-t0:.0f}s)", flush=True)
for name, extra in CAT:
    t0 = time.time()
    m = CatBoostClassifier(iterations=6000, loss_function="Logloss", thread_count=-1, random_seed=0, od_type="Iter",
                           od_wait=100, verbose=False, **extra)
    m.fit(Xtr, ytr, eval_set=(Xva, yva), use_best_model=True)
    p = np.clip(m.predict_proba(Xte)[:, 1], 1e-6, 1 - 1e-6)
    r = dict(name=name, params=extra, f05=f05(p), logloss=log_loss(yte, p), iters=m.get_best_iteration())
    res["cat"].append(r)
    if best_cat is None or r["logloss"] < best_cat[0]["logloss"]:
        best_cat = (r, p)
    print(f"CAT {name:30s} F0.5 {r['f05']:.5f} logloss {r['logloss']:.5f} ({r['iters']} it, {time.time()-t0:.0f}s)", flush=True)
pa = (best_lgb[1] + best_cat[1]) / 2
print(f"BEST LGB {best_lgb[0]['name']} + BEST CAT {best_cat[0]['name']}: avg F0.5 {f05(pa):.5f} logloss {log_loss(yte, pa):.5f}", flush=True)
res["best_lgb"], res["best_cat"] = best_lgb[0], best_cat[0]
json.dump(res, open(HERE + "hp_results.json", "w"), indent=1, default=float)
print("HP DONE", flush=True)
