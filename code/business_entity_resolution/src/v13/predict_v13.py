"""v13 test prediction: base + v12 + v13 features, LightGBM stage 1, cluster features, LightGBM + CatBoost stage 2
(weighted by decision_stage2_v13.json). Writes test_scores_v13.parquet (p1, p2, p2_lgb, p2_cat) and the output TSVs."""
import sys, json, pickle, time
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, lightgbm as lgb
from catboost import CatBoostClassifier
import predict as pr
from er.candidates import load_candidates, prune
from er.cluster import CLUSTER, cluster_features
from er.features import FEATURES as BASE, Records, add_context, add_name_context, pair_features
from er.io import load_norm, write_id_lists
from er.config import OUTPUT_DIR
from v12feat import V12, v12_parallel
from v13feat import V13, v13_parallel

HERE = str(__import__("pathlib").Path(__file__).resolve().parents[4] / "data_cache" / "v13") + "/"
ALL = BASE + V12 + V13
ALL2 = ALL + ["p1"] + CLUSTER

if __name__ == "__main__":
    t0 = time.time()
    M = pr.MODEL_DIR
    st12 = pickle.load(open(HERE + "aw_stats_test.pkl", "rb")); st13 = pickle.load(open(HERE + "v13_stats_test.pkl", "rb"))
    m1 = lgb.Booster(model_file=str(M / "matcher_v13.txt")); m2 = lgb.Booster(model_file=str(M / "matcher_stage2_v13.txt"))
    cat = CatBoostClassifier(); cat.load_model(str(M / "cat_stage2_v13.cbm"))
    cfg = json.load(open(M / "decision_stage2_v13.json")); w = cfg["cat_weight"]
    print("config", cfg, flush=True)
    rec = Records("test")
    c = add_name_context(add_context(load_candidates("test", prune, tag="_st")), rec)
    c = c.iloc[np.argsort(c.s1_row.to_numpy(), kind="stable")].reset_index(drop=True)
    print(f"test candidates {len(c):,} ({time.time()-t0:.0f}s)", flush=True)
    p1 = np.empty(len(c), np.float32); p2l = np.empty(len(c), np.float32); p2c = np.empty(len(c), np.float32)
    for lo, hi in pr.s1_chunks(c.s1_row.to_numpy(), pr.CHUNK):
        part = c.iloc[lo:hi]
        f = pair_features(part, rec)
        s1r, q, src = part.s1_row.to_numpy(), part.q_row.to_numpy(), part.src.to_numpy()
        a_norm, b_norm = rec.s1["name_norm"][s1r], rec.target("name_norm", src, q)
        a_core, b_core = rec.s1["name_core"][s1r], rec.target("name_core", src, q)
        a_num, b_num = rec.s1["numbers"][s1r], rec.target("numbers", src, q)
        a_addr, b_addr = rec.s1["addr_norm"][s1r], rec.target("addr_norm", src, q)
        cty = rec.s1["country"][s1r]
        n12 = v12_parallel(a_norm, b_norm, a_core, b_core, a_num, b_num, cty, stats=st12)
        n13 = v13_parallel(a_core, b_core, a_num, b_num, a_addr, b_addr, cty, stats=st13)
        for j, name in enumerate(V12): f[name] = n12[:, j]
        for j, name in enumerate(V13): f[name] = n13[:, j]
        p1[lo:hi] = m1.predict(f[ALL], num_threads=0)
        f["p1"] = p1[lo:hi]
        cl = cluster_features(s1r, src, q, p1[lo:hi], rec)
        for col in cl.columns: f[col] = cl[col].to_numpy()
        X2 = f[ALL2].to_numpy(np.float32)
        p2l[lo:hi] = m2.predict(X2, num_threads=0)
        p2c[lo:hi] = cat.predict_proba(X2)[:, 1]
        print(f"  scored {hi:,}/{len(c):,} ({time.time()-t0:.0f}s)", flush=True)
    p2 = (1 - w) * p2l + w * p2c
    pd.DataFrame({"s1_row": c.s1_row, "src": c.src, "q_row": c.q_row, "p1": p1, "p2": p2, "p2_lgb": p2l, "p2_cat": p2c}) \
        .to_parquet(M / "test_scores_v13.parquet", index=False)
    s1 = load_norm("test", 1, ["entity_id", "country"])
    qkey = c.src.to_numpy().astype(np.int64) << 32 | c.q_row.to_numpy().astype(np.int64)
    write_id_lists(pr.ids_for("test", c.s1_row.to_numpy(), qkey), s1.entity_id, "match",
                   OUTPUT_DIR / "candidate_pairs.tsv", "candidate_entity_ids")
    pr.decide_and_write(c, p2, cfg, s1)
    print(f"done ({time.time()-t0:.0f}s)", flush=True)
