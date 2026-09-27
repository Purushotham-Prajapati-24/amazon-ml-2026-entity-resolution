"""v13 side-car columns (5) for hold-out / fold 0 / fold 1, plus v13 word stats for test."""
import sys, time, pickle
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, pyarrow.parquet as pq
from er.config import CACHE_DIR
from er.io import load_norm
from v13feat import V13, v13_parallel, word_stats

M = CACHE_DIR / "model"
HERE = str(__import__("pathlib").Path(__file__).resolve().parents[4] / "data_cache" / "v13") + "/"
COLS = ["country", "name_core", "numbers", "addr_norm"]


def stats_for(split, s1, tg):
    parts = []
    for k in (2, 3):
        c = pd.read_parquet(CACHE_DIR / "cand" / f"{split}_s{k}_st.parquet", columns=["q_row", "s1_row", "enc_rank"])
        c = c[c.enc_rank == 0]
        parts.append(pd.DataFrame({"s": c.s1_row.to_numpy(), "k": k, "q": c.q_row.to_numpy()}))
    top = pd.concat(parts, ignore_index=True)
    k2, qq, ss = top.k.to_numpy() == 2, top.q.to_numpy(), top.s.to_numpy()
    tp = lambda col: np.where(k2, tg[2][col].to_numpy()[np.where(k2, qq, 0)], tg[3][col].to_numpy()[np.where(~k2, qq, 0)])
    return word_stats(s1.name_core.to_numpy()[ss], tp("name_core"), s1.numbers.to_numpy()[ss], tp("numbers"),
                      s1.addr_norm.to_numpy()[ss], tp("addr_norm"), s1.country.to_numpy()[ss])


if __name__ == "__main__":
    t0 = time.time()
    s1t = load_norm("test", 1, COLS); tgt = {k: load_norm("test", k, COLS[1:]) for k in (2, 3)}
    pickle.dump(stats_for("test", s1t, tgt), open(HERE + "v13_stats_test.pkl", "wb"))
    del s1t, tgt
    s1 = load_norm("train", 1, COLS); tg = {k: load_norm("train", k, COLS[1:]) for k in (2, 3)}
    st = pickle.load(open(HERE + "v13_stats_train.pkl", "rb"))
    print(f"stats ready ({time.time()-t0:.0f}s)", flush=True)
    for name in ["train_holdout_features_st", "train_fold0_features_f0", "train_fold1_features_f1"]:
        ids = pq.read_table(M / f"{name}.parquet", columns=["s1_row", "q_row", "src_id"]).to_pandas()
        sr, src, q = ids.s1_row.to_numpy(), ids.src_id.to_numpy(), ids.q_row.to_numpy()
        pick = lambda col: np.where(src == 2, tg[2][col].to_numpy()[np.where(src == 2, q, 0)], tg[3][col].to_numpy()[np.where(src == 3, q, 0)])
        new = v13_parallel(s1.name_core.to_numpy()[sr], pick("name_core"), s1.numbers.to_numpy()[sr], pick("numbers"),
                           s1.addr_norm.to_numpy()[sr], pick("addr_norm"), s1.country.to_numpy()[sr], stats=st)
        pd.DataFrame(new, columns=V13).to_parquet(M / f"{name}_v13side.parquet", index=False)
        print(f"{name}: {len(new):,} ({time.time()-t0:.0f}s)", flush=True)
    print("DONE", flush=True)
