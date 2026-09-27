"""Choose the ensemble weight of the base-encoder run on the S1 hold-out.

Scores the hold-out pairs with a big-data stage-1 matcher (its features come from the
self-trained encoder, `train_holdout_features_st.parquet`), unions them with the base run's
out-of-fold stage-1 probabilities (`oof.parquet`), and picks the base weight in {0.1, 0.2, 0.3}
that maximises macro F0.5 after the usual decision rule. Weight 0 is not allowed: pairs proposed
only by the base run would have no score. Writes data_cache/model/blend_weight<tag>.txt.

    python src/blend_weight.py --matcher matcher_v11.txt
"""
import argparse

import lightgbm as lgb
import numpy as np
import pandas as pd

from er.config import CACHE_DIR
from er.decide import assign, expected_f_select
from er.holdout import holdout_mask
from er.io import load_norm
from er.labels import pair_key, truth_rows
from er.metrics import macro_f05
from train_big import load_matrix, predict_chunked

MODEL_DIR = CACHE_DIR / "model"
KEY = ["s1_row", "src_id", "q_row"]


def main(model_file, eval_tag, weights):
    ev, Xev = load_matrix([MODEL_DIR / f"train_holdout_features{eval_tag}.parquet"])
    big = lgb.Booster(model_file=str(MODEL_DIR / model_file))
    b = pd.DataFrame({"s1_row": ev.s1_row, "src_id": ev.src_id, "q_row": ev.q_row, "p": predict_chunked(big, Xev)})
    a = pd.read_parquet(MODEL_DIR / "oof.parquet")[KEY + ["p"]]
    u = a.merge(b, on=KEY, how="outer", suffixes=("_a", "_b"))
    held = np.flatnonzero(holdout_mask(len(load_norm("train", 1, ["entity_id"]))))
    tr = truth_rows()
    tr = tr[np.isin(tr.s1_row, held)]
    truth = pd.DataFrame({"s1": tr.s1_row.to_numpy(), "match": pair_key(tr.src, tr.t_row)})
    pa, pb = u.p_a.to_numpy(), u.p_b.to_numpy()

    def score(p):
        m = ~np.isnan(p)
        return max(macro_f05(expected_f_select(assign(u.s1_row.to_numpy()[m], u.src_id.to_numpy()[m],
                                                      u.q_row.to_numpy()[m], p[m], t))
                             .rename(columns={"s1_row": "s1", "qkey": "match"}), truth, pd.Series(held))["macro_f05"]
                   for t in (0.5, 0.6, 0.7))

    print(f"{model_file}: big run alone (base fills its gaps) {score(np.where(np.isnan(pb), pa, pb)):.5f}", flush=True)
    res = {}
    for w in weights:
        res[w] = score(np.where(np.isnan(pa), pb, np.where(np.isnan(pb), pa, w * pa + (1 - w) * pb)))
        print(f"  base weight {w:.1f}: {res[w]:.5f}", flush=True)
    best = max(res, key=res.get)
    tag = model_file.replace("matcher", "").replace(".txt", "")
    (MODEL_DIR / f"blend_weight{tag}.txt").write_text(f"{best}")
    print(f"best base weight {best} -> blend_weight{tag}.txt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--matcher", required=True, help="stage-1 big-data matcher file in data_cache/model")
    ap.add_argument("--eval-tag", default="_st")
    ap.add_argument("--weights", nargs="+", type=float, default=[0.1, 0.2, 0.3])
    a = ap.parse_args()
    main(a.matcher, a.eval_tag, a.weights)
