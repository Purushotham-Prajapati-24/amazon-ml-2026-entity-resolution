"""Stage 3a: pairwise features + labels for the matching model's training set.

Training rows are the pruned candidate pairs whose S1 is in the hold-out: the encoder
never saw those pairs, so their retrieval features behave like test features. Context
features are computed over the full pruned candidate set first, so competition with
non-hold-out S1 entities is still reflected.

    python src/build_features.py
"""
import time

import pandas as pd

from er.candidates import load_candidates, prune
from er.config import CACHE_DIR
from er.features import Records, add_context, add_name_context, pair_features
from er.holdout import holdout_mask, s1_fold
from er.io import load_norm
from er.labels import pair_key, truth_rows

MODEL_DIR = CACHE_DIR / "model"


def main(tag: str = "", fold=None):
    t0 = time.time()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rec = Records("train")
    c = add_name_context(add_context(load_candidates("train", prune, tag=tag)), rec)
    print(f"pruned train candidates with context: {len(c):,}  ({time.time() - t0:.0f}s)")

    n_s1 = len(load_norm("train", 1, ["entity_id"]))
    held = holdout_mask(n_s1)
    if fold is not None:  # cross-fitted training rows: S1 of this fold, never the hold-out
        held = (s1_fold(n_s1) == fold) & ~held
    c = c[held[c.s1_row.to_numpy()]].reset_index(drop=True)

    # label: is (s1, target) a ground-truth pair?
    truth = truth_rows()
    truth = truth[held[truth.s1_row.to_numpy()]]
    pos = pd.MultiIndex.from_arrays([truth.s1_row, pair_key(truth.src, truth.t_row)])
    label = pd.MultiIndex.from_arrays([c.s1_row, pair_key(c.src, c.q_row)]).isin(pos)
    print(f"hold-out pairs: {len(c):,}, positives {label.sum():,} ({label.mean():.1%}), "
          f"covering {label.sum() / len(truth):.1%} of hold-out true pairs")

    chunk = 4_000_000  # cross-fitted folds are ~17M rows; bound peak memory
    f = pd.concat([pair_features(c.iloc[i:i + chunk], rec) for i in range(0, len(c), chunk)], ignore_index=True)
    f["label"] = label
    f["s1_row"], f["q_row"], f["src_id"] = c.s1_row.to_numpy(), c.q_row.to_numpy(), c.src.to_numpy()
    name = f"train_fold{fold}_features{tag}.parquet" if fold is not None else f"train_holdout_features{tag}.parquet"
    f.to_parquet(MODEL_DIR / name, index=False)
    print(f"wrote {len(f):,} rows x {f.shape[1]} cols  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--fold", type=int, help="build cross-fitted training rows for this S1 fold")
    a = ap.parse_args()
    main(a.tag, a.fold)
