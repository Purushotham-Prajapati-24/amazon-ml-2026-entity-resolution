"""Re-weight the two stage-2 models for the unlabelled country only (France).

predict_v13.py stores both stage-2 probabilities (p2_lgb, p2_cat) next to their hold-out-weighted
mix p2. French records only compete with French S1s, so changing the French mix leaves every
US / India row untouched, and the leaderboard difference measures France alone. Measured
(v13, 27 Sep): CatBoost weight 0.3 (hold-out choice) 0.981542, 0.6 0.981687, 1.0 0.981577.
CatBoost generalises better to the unseen country, but the mix beats either model alone.

    python src/v13/france_mix.py --cat-weight 0.6     # -> test_scores_v13f06.parquet
    python src/ensemble_predict.py --scores test_scores.parquet test_scores_v13f06.parquet \
        --weights 0.1 0.9 --tau 0.5 --adapted-for-unlabelled
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd

from er.config import CACHE_DIR
from er.io import load_norm

UNLABELLED = "France"

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cat-weight", type=float, default=0.6)
    ap.add_argument("--scores", default="test_scores_v13.parquet")
    a = ap.parse_args()
    M = CACHE_DIR / "model"
    s = pd.read_parquet(M / a.scores)
    fr = load_norm("test", 1, ["country"]).country.to_numpy()[s.s1_row.to_numpy()] == UNLABELLED
    w = a.cat_weight
    s["p2"] = np.where(fr, (1 - w) * s.p2_lgb + w * s.p2_cat, s.p2).astype(np.float32)
    out = M / f"test_scores_v13f{round(w * 10):02d}.parquet"
    s[["s1_row", "src", "q_row", "p1", "p2"]].to_parquet(out, index=False)
    print(f"wrote {out.name}: {int(fr.sum()):,} French candidate pairs at CatBoost weight {w}")
