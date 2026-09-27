"""Stage 3b: train the LightGBM matcher on hold-out pairs and tune the decision rule.

Default mode: 5-fold GroupKFold by S1 gives out-of-fold probabilities for every hold-out
pair (saved for error analysis); the decision rule (assignment threshold, with/without
expected-F0.5 selection) is tuned on them for macro F0.5 over all hold-out S1 entities
(singletons included). The final model is refit on all hold-out pairs.

Leave-country-out mode (``--train-countries US --eval-countries India``): fit on one
country's rows only and score another country as if it were unseen, a proxy for the
zero-shot France test slice. Use with features built from an encoder trained on the same
countries (``--tag _loco_us``).

    python src/train_matcher.py
    python src/train_matcher.py --tag _loco_us --train-countries US --eval-countries India
"""
import argparse
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from er.config import CACHE_DIR
from er.decide import assign, expected_f_select
from er.features import FEATURES
from er.holdout import holdout_mask
from er.io import load_norm
from er.labels import pair_key, truth_rows
from er.metrics import macro_f05

MODEL_DIR = CACHE_DIR / "model"
PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=127, min_data_in_leaf=100,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    metric="binary_logloss", verbose=-1, num_threads=0, seed=0,
)
TAUS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)


def score(pred: pd.DataFrame, truth: pd.DataFrame, s1_rows) -> dict:
    return macro_f05(pred.rename(columns={"s1_row": "s1", "qkey": "match"}),
                     truth.rename(columns={"s1_row": "s1", "qkey": "match"}), pd.Series(s1_rows))


def fit(X, y, tr, va):
    m = lgb.train(PARAMS, lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=4000,
                  valid_sets=[lgb.Dataset(X.iloc[va], y[va])],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def decide(f, p, tau, sel):
    a = assign(f.s1_row.to_numpy(), f.src_id.to_numpy(), f.q_row.to_numpy(), p, tau)
    return expected_f_select(a) if sel == "expected_f" else a


def decision_grid(f, p, truth, s1_rows, label=""):
    best = None
    print(f"decision grid{label} (macro F0.5):")
    for tau in TAUS:
        a = assign(f.s1_row.to_numpy(), f.src_id.to_numpy(), f.q_row.to_numpy(), p, tau)
        plain = score(a, truth, s1_rows)["macro_f05"]
        ef = score(expected_f_select(a), truth, s1_rows)["macro_f05"]
        print(f"  tau={tau:.1f}: threshold only {plain:.4f} | + expected-F selection {ef:.4f}")
        for sel, v in (("threshold", plain), ("expected_f", ef)):
            if best is None or v > best[0]:
                best = (v, tau, sel)
    return best


def report(f, p, tau, sel, truth, s1_rows, country):
    a = decide(f, p, tau, sel)
    r = score(a, truth, s1_rows)
    print(f"tau={tau} selection={sel}: " + ", ".join(
        f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()))
    for c in pd.unique(country[s1_rows]):
        rows = s1_rows[country[s1_rows] == c]
        rc = score(a[np.isin(a.s1_row, rows)], truth[np.isin(truth.s1_row, rows)], rows)
        print(f"  {c:6s} macro F0.5 {rc['macro_f05']:.4f}  P {rc['micro_precision']:.4f}  R {rc['micro_recall']:.4f}  singleton acc {rc['singleton_acc']:.4f}")
    return r


def main(tag: str, train_countries, eval_countries):
    t0 = time.time()
    f = pd.read_parquet(MODEL_DIR / f"train_holdout_features{tag}.parquet")
    X, y, groups = f[FEATURES], f.label.to_numpy(), f.s1_row.to_numpy()
    print(f"{len(f):,} rows, {len(FEATURES)} features, positives {y.mean():.1%}")

    s1 = load_norm("train", 1, ["country"])
    country = s1.country.to_numpy()
    held_rows = np.flatnonzero(holdout_mask(len(s1)))
    tr_rows = truth_rows()
    tr_rows = tr_rows[np.isin(tr_rows.s1_row, held_rows)]
    truth = pd.DataFrame({"s1_row": tr_rows.s1_row, "qkey": pair_key(tr_rows.src, tr_rows.t_row)})
    row_country = country[f.s1_row.to_numpy()]

    if eval_countries:  # ---------------------------------------------- leave-country-out
        src_idx = np.flatnonzero(np.isin(row_country, train_countries))
        tgt_idx = np.flatnonzero(np.isin(row_country, eval_countries))
        # early-stopping split inside the source countries, grouped by S1
        tr, va = next(GroupKFold(n_splits=10).split(src_idx, groups=groups[src_idx]))
        m = fit(X, y, src_idx[tr], src_idx[va])
        print(f"fit on {train_countries}: best_iter {m.best_iteration}  ({time.time() - t0:.0f}s)")
        src_p = m.predict(X.iloc[src_idx], num_iteration=m.best_iteration)
        tgt_p = m.predict(X.iloc[tgt_idx], num_iteration=m.best_iteration)
        fs, ft = f.iloc[src_idx], f.iloc[tgt_idx]
        src_rows = held_rows[np.isin(country[held_rows], train_countries)]
        tgt_rows = held_rows[np.isin(country[held_rows], eval_countries)]
        _, tau, sel = decision_grid(fs, src_p, truth[np.isin(truth.s1_row, src_rows)], src_rows, " on source (in-sample)")
        print(f"\nunseen {eval_countries} with the source-tuned rule:")
        report(ft, tgt_p, tau, sel, truth[np.isin(truth.s1_row, tgt_rows)], tgt_rows, country)
        best = decision_grid(ft, tgt_p, truth[np.isin(truth.s1_row, tgt_rows)], tgt_rows, " on unseen country (oracle tau)")
        print(f"oracle best on unseen: {best[0]:.4f} (tau={best[1]}, {best[2]})")
        return

    oof, iters = np.zeros(len(f)), []   # ------------------------------------------- default
    for fold, (tr, va) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        m = fit(X, y, tr, va)
        oof[va] = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        iters.append(m.best_iteration)
        print(f"  fold {fold}: best_iter {m.best_iteration}, logloss {m.best_score['valid_0']['binary_logloss']:.4f}  ({time.time() - t0:.0f}s)")
    pd.DataFrame({"s1_row": f.s1_row, "src_id": f.src_id, "q_row": f.q_row, "label": f.label, "p": oof}) \
        .to_parquet(MODEL_DIR / f"oof{tag}.parquet", index=False)

    print()
    _, tau, sel = decision_grid(f, oof, truth, held_rows, " on hold-out (OOF)")
    print("\nbest:", end=" ")
    r = report(f, oof, tau, sel, truth, held_rows, country)

    n_iter = int(np.mean(iters) * 1.1)
    final = lgb.train(PARAMS, lgb.Dataset(X, y), num_boost_round=n_iter)
    final.save_model(str(MODEL_DIR / f"matcher{tag}.txt"))
    json.dump({"tau": tau, "selection": sel, "n_iter": n_iter, "oof_macro_f05": r["macro_f05"]},
              open(MODEL_DIR / f"decision{tag}.json", "w"), indent=2)
    imp = pd.Series(final.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    print("\ntop features by gain:", ", ".join(f"{k} {v / imp.sum():.1%}" for k, v in imp.head(15).items()))
    print(f"saved matcher ({n_iter} trees) + decision{tag}.json  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--train-countries", nargs="+")
    ap.add_argument("--eval-countries", nargs="+")
    a = ap.parse_args()
    main(a.tag, a.train_countries, a.eval_countries)
