"""Pseudo-labels for self-training the encoder on a country without training labels.

Takes the saved test probabilities of the full matcher (stage 2 if present), assigns each
test record to its best S1, and keeps confident assignments in the chosen countries.
The matcher uses evidence the encoder does not (string similarity, house numbers,
cluster agreement), so these labels teach the encoder pairs it is unsure about.

    python src/make_pseudo.py --countries France --min-p 0.98
"""
import argparse

import numpy as np
import pandas as pd

from er.config import CACHE_DIR
from er.decide import assign
from er.io import load_norm

MODEL_DIR = CACHE_DIR / "model"


def main(countries, min_p, out_name):
    s = pd.read_parquet(MODEL_DIR / "test_scores.parquet")
    p = (s.p2 if "p2" in s else s.p1).to_numpy()
    a = assign(s.s1_row.to_numpy(), s.src.to_numpy(), s.q_row.to_numpy(), p, min_p)
    country = load_norm("test", 1, ["country"]).country.to_numpy()
    a = a[np.isin(country[a.s1_row.to_numpy()], countries)]
    qkey = a.qkey.to_numpy()
    out = pd.DataFrame({"split": "test", "s1_row": a.s1_row.to_numpy().astype(np.int64),
                        "src": (qkey >> 32).astype(np.int8), "t_row": (qkey & 0xFFFFFFFF).astype(np.int64)})
    out.to_parquet(CACHE_DIR / "encoder" / out_name, index=False)
    n_s1 = out.s1_row.nunique()
    n_country = int(np.isin(country, countries).sum())
    print(f"{len(out):,} pseudo pairs (p >= {min_p}) covering {n_s1:,} of {n_country:,} S1 in {countries}"
          f" ({len(out) / max(n_s1, 1):.2f} per S1) -> {out_name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="+", default=["France"])
    ap.add_argument("--min-p", type=float, default=0.98)
    ap.add_argument("--out", default="pseudo_test_france.parquet")
    a = ap.parse_args()
    main(a.countries, a.min_p, a.out)
