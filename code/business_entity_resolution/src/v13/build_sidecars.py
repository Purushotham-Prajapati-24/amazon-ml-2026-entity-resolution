"""Build v12 side-car feature files (8 columns, same row order) for the existing training / hold-out
feature parquets, plus the per-country added-word stats for train and test (label-free)."""
import sys, time, pickle
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, pyarrow.parquet as pq
from er.config import CACHE_DIR
from er.io import load_norm
from v12feat import V12, v12_parallel, added_word_stats

M = CACHE_DIR / "model"
OUT = str(__import__("pathlib").Path(__file__).resolve().parents[4] / "data_cache" / "v13") + "/"


def split_stats(split, s1, tg):
    parts = []
    for k in (2, 3):
        c = pd.read_parquet(CACHE_DIR / "cand" / f"{split}_s{k}_st.parquet", columns=["q_row", "s1_row", "enc_rank"])
        c = c[c.enc_rank == 0]
        parts.append(pd.DataFrame({"s": c.s1_row.to_numpy(), "k": k, "q": c.q_row.to_numpy()}))
    top = pd.concat(parts, ignore_index=True)
    k2, qq, ss = top.k.to_numpy() == 2, top.q.to_numpy(), top.s.to_numpy()
    bc = np.where(k2, tg[2].name_core.to_numpy()[np.where(k2, qq, 0)], tg[3].name_core.to_numpy()[np.where(~k2, qq, 0)])
    bn = np.where(k2, tg[2].numbers.to_numpy()[np.where(k2, qq, 0)], tg[3].numbers.to_numpy()[np.where(~k2, qq, 0)])
    return added_word_stats(s1.name_core.to_numpy()[ss], bc, s1.numbers.to_numpy()[ss], bn, s1.country.to_numpy()[ss])


if __name__ == "__main__":
    t0 = time.time()
    cols = ["country", "name_norm", "name_core", "numbers"]
    # test stats first (small), saved for prediction
    s1t = load_norm("test", 1, cols)
    tgt = {k: load_norm("test", k, cols[1:]) for k in (2, 3)}
    st_test = split_stats("test", s1t, tgt)
    pickle.dump(st_test, open(OUT + "aw_stats_test.pkl", "wb"))
    for c, st in st_test.items():
        print("test", c, len(st), {w: (round(st[w][0], 3), st[w][1]) for w in ("groupe", "holding", "developpement", "international",
              "france", "distribution", "participations", "fils", "associes", "service", "group", "center") if w in st}, flush=True)
    del s1t, tgt
    s1 = load_norm("train", 1, cols)
    tg = {k: load_norm("train", k, cols[1:]) for k in (2, 3)}
    st = split_stats("train", s1, tg)
    pickle.dump(st, open(OUT + "aw_stats_train.pkl", "wb"))
    print(f"stats done ({time.time()-t0:.0f}s)", flush=True)
    for name in ["train_holdout_features_st", "train_fold0_features_f0", "train_fold1_features_f1"]:
        ids = pq.read_table(M / f"{name}.parquet", columns=["s1_row", "q_row", "src_id"]).to_pandas()
        sr, src, q = ids.s1_row.to_numpy(), ids.src_id.to_numpy(), ids.q_row.to_numpy()
        def pick(col):
            return np.where(src == 2, tg[2][col].to_numpy()[np.where(src == 2, q, 0)], tg[3][col].to_numpy()[np.where(src == 3, q, 0)])
        new = v12_parallel(s1.name_norm.to_numpy()[sr], pick("name_norm"), s1.name_core.to_numpy()[sr], pick("name_core"),
                           s1.numbers.to_numpy()[sr], pick("numbers"), s1.country.to_numpy()[sr], stats=st)
        pd.DataFrame(new, columns=V12).to_parquet(M / f"{name}_v12side.parquet", index=False)
        print(f"{name}: {len(new):,} rows ({time.time()-t0:.0f}s)", flush=True)
    print("DONE", flush=True)
