"""Post-processing for the unlabelled French slice: name-word / look-alike veto.

Leaderboard evidence: applied to v9g (0.975568) it scored 0.977334 (+0.0018), US/India
rows unchanged, so France's own F0.5 rose by roughly +0.012.

French records only compete with French S1s, so the French decision is recomputed in
isolation with the same rule as the base file (assign + expected-F0.5 at tau), from the
same scores the base file used for France. A 'reproduction' check confirms that without
the veto the French rows equal the base file's exactly; non-French rows are copied
byte-identically.

Veto (label-free): a French pair is dropped when the matched record introduces a name word
the S1 lacks that
 (a) is French name vocabulary: in >= 20 French S1 names and not over-represented in
     French S2/S3 names (ratio <= --ratio), i.e. the variant generator never adds it, and
     is not a spelling / abbreviation / leetspeak of any S1 word it could replace; or
 (b) is a look-alike marker: groupe / holding / participations (translations of the US
     markers Group / Holdings, which have 0 true matches in the US labels).
On labelled US/India data true pairs introduce a name word only 0.12-0.22% of the time.
Adapted from the version first built for v9g (france_refine/out/v9g_france_namevocab).

--scope refined (for matchers with the v11 features, which already reject most look-alikes):
the name-word part (a) only applies to swaps at the identical address and to records with a
different address and no shared house number. Size test on v11 (a rejected genuine variant
leaves its S1 one match short, a look-alike does not): vetoed additions at the same address /
number and vetoed swaps sharing a number behave like known noise words (fils, associes), while
on v9g the vetoed swaps behaved like look-alikes - consistent with the +0.0018 above.

    python src/france_veto.py --scores test_scores_big4.parquet \
        --base <submissions>/v10t_blend3_v5_v9_v10/output/matching_results.tsv \
        --out  <submissions>/v10tf_blend3_france_veto/output
"""
import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from er.config import CACHE_DIR
from er.decide import assign, expected_f_select
from er.features import related

MARKERS = {"groupe", "holding", "participations"}
UNLABELLED = "France"


def france_vocab(ratio_max, min_s1=20):
    def df(k):
        t = pq.read_table(CACHE_DIR / "norm" / f"test_s{k}.parquet", columns=["name_core", "country"])
        names = t.column(0).to_numpy(zero_copy_only=False)[t.column(1).to_numpy(zero_copy_only=False) == UNLABELLED]
        c = Counter()
        for n in names:
            c.update(set(n.split()))
        return c, len(names)
    s1c, n1 = df(1)
    tc, nt = Counter(), 0
    for k in (2, 3):
        c, n = df(k)
        tc.update(c)
        nt += n
    return {w for w, c in s1c.items() if c >= min_s1 and (tc.get(w, 0) / nt) / (c / n1) <= ratio_max}


def main(scores, base_file, out_dir, ratio_max, tau, scope="all"):
    s1n = pq.read_table(CACHE_DIR / "norm" / "test_s1.parquet", columns=["entity_id", "country", "name_core", "addr_norm", "numbers"])
    country = s1n.column("country").to_numpy(zero_copy_only=False)
    s1_ids = s1n.column("entity_id").to_numpy(zero_copy_only=False)
    s1_core = s1n.column("name_core").to_numpy(zero_copy_only=False)
    s1_addr = s1n.column("addr_norm").to_numpy(zero_copy_only=False)
    s1_num = s1n.column("numbers").to_numpy(zero_copy_only=False)
    fr = country == UNLABELLED
    parts = []
    path = scores if ("/" in scores or "\\" in scores) else CACHE_DIR / "model" / scores
    for b in pq.ParquetFile(path).iter_batches(columns=["s1_row", "src", "q_row", "p2"], batch_size=4_000_000):
        t = b.to_pandas()
        parts.append(t[fr[t.s1_row.to_numpy()]])
    sc = pd.concat(parts, ignore_index=True)
    print(f"France candidate pairs ({Path(str(path)).name}): {len(sc):,}")
    V = france_vocab(ratio_max)
    print(f"French name vocabulary: {len(V):,} words (ratio <= {ratio_max})")
    t_core, t_ids, t_addr, t_num = {}, {}, {}, {}
    for k in (2, 3):
        t = pq.read_table(CACHE_DIR / "norm" / f"test_s{k}.parquet", columns=["entity_id", "name_core", "addr_norm", "numbers"])
        t_ids[k] = t.column(0).to_numpy(zero_copy_only=False)
        t_core[k] = t.column(1).to_numpy(zero_copy_only=False)
        t_addr[k] = t.column(2).to_numpy(zero_copy_only=False)
        t_num[k] = t.column(3).to_numpy(zero_copy_only=False)
    cand = np.flatnonzero(sc.p2.to_numpy() >= tau)
    flag = np.zeros(len(sc), bool)
    kinds = Counter()
    for i in cand:
        r, k, q = sc.s1_row.iat[i], int(sc.src.iat[i]), sc.q_row.iat[i]
        A = set(s1_core[r].split())
        B = set(t_core[k][q].split())
        xa, xb = A - B, B - A
        if xb & MARKERS:
            flag[i] = True; kinds["marker"] += 1; continue
        intro = [w for w in xb if w in V and not any(related(w, v) for v in xa)]
        if not intro:
            continue
        if scope == "refined":
            same_addr = s1_addr[r] == t_addr[k][q]
            far = not same_addr and not (set(s1_num[r].split()) & set(t_num[k][q].split()))
            if not (far or (xa and same_addr)):
                kinds["exempt " + ("swap" if xa else "added")] += 1
                continue
        flag[i] = True; kinds["swap" if xa else "added"] += 1
    print(f"France pairs with p >= {tau}: {len(cand):,}; vetoed: {int(flag.sum()):,} {dict(kinds)}")

    def decide(p):
        return expected_f_select(assign(sc.s1_row.to_numpy(), sc.src.to_numpy(), sc.q_row.to_numpy(), p, tau))

    def lists(a):
        src, row = a.qkey.to_numpy() >> 32, a.qkey.to_numpy() & 0xFFFFFFFF
        tid = np.where(src == 2, t_ids[2][np.where(src == 2, row, 0)], t_ids[3][np.where(src == 3, row, 0)])
        d = pd.DataFrame({"s1": s1_ids[a.s1_row.to_numpy()], "t": tid, "p": a.p.to_numpy()}).sort_values(["s1", "p"], ascending=[True, False])
        return d.groupby("s1").t.apply(lambda x: ",".join(x)).to_dict()

    base = lists(decide(sc.p2.to_numpy()))
    p = sc.p2.to_numpy().copy()
    p[flag] = 0.0
    new = lists(decide(p))

    fr_ids = set(s1_ids[fr])
    out_dir.mkdir(parents=True, exist_ok=True)
    n_diff_repro = n_changed = removed = 0
    with open(base_file, encoding="utf-8") as fin, open(out_dir / "matching_results.tsv", "w", encoding="utf-8", newline="\n") as fout:
        fout.write(fin.readline())
        for line in fin:
            s, m = line.rstrip("\n").split("\t")
            if s in fr_ids:
                if set(filter(None, m.split(","))) != set(filter(None, base.get(s, "").split(","))):
                    n_diff_repro += 1
                nm = new.get(s, "")
                if set(filter(None, nm.split(","))) != set(filter(None, m.split(","))):
                    n_changed += 1
                    removed += len(set(filter(None, m.split(","))) - set(filter(None, nm.split(","))))
                fout.write(f"{s}\t{nm}\n")
            else:
                fout.write(line if line.endswith("\n") else line + "\n")
    print(f"reproduction check: French rows differing from the base file = {n_diff_repro} (must be 0)")
    print(f"French S1 rows changed: {n_changed:,} of {int(fr.sum()):,} ({n_changed / fr.sum():.1%}); pairs removed {removed:,}")
    h = hashlib.sha256((out_dir / "matching_results.tsv").read_bytes()).hexdigest().upper()
    print(f"wrote {out_dir / 'matching_results.tsv'}  sha256 {h}")
    if n_diff_repro:
        raise SystemExit("reproduction check failed: the scores file does not match the base file's French rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True, help="test scores the base file used for France (p2 column)")
    ap.add_argument("--base", type=Path, required=True, help="matching_results.tsv to modify (France rows only)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--ratio", type=float, default=1.3)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--scope", choices=["all", "refined"], default="all",
                    help="refined: name-word veto only for same-address swaps and far records (see docstring)")
    a = ap.parse_args()
    main(a.scores, a.base, a.out, a.ratio, a.tau, a.scope)
