"""Stage 2c: train the retrieval encoder on train ground-truth pairs, then embed all records.

Contrastive (symmetric InfoNCE) with in-batch negatives. Batches are grouped by
(country, S1 state) so negatives are geographically close, i.e. hard. Each epoch uses
one random match per S1, so an S1 never appears twice in a batch (no false negatives).
5% of S1 entities are held out to measure retrieval recall honestly.

Self-training: ``--pseudo file.parquet`` adds pseudo-labelled pairs (columns split,
s1_row, src, t_row) to the training pairs. Pairs from the test split are supported by
appending the test feature bags after the train bags, so the encoder can adapt to a
country that has no labels (France).

    python src/train_encoder.py            # train + embed
    python src/train_encoder.py --embed    # embed only, with the saved model
    python src/train_encoder.py --pseudo data_cache/encoder/pseudo_test_france.parquet --tag _st
"""
import argparse
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from er.config import CACHE_DIR, SOURCES, SPLITS
from er.encoder import CSRBags, RecordEncoder, to_device
from er.holdout import holdout_mask, s1_fold
from er.io import load_ground_truth, load_norm
from er.knn import topk_search

FEAT_DIR = CACHE_DIR / "feat"
ENC_DIR = CACHE_DIR / "encoder"
VEC_DIR = CACHE_DIR / "vec"
DEVICE = "cuda"
BATCH = 4096
EPOCHS = 8
SEED = 0


def load_bags(split, k):
    return {f: CSRBags(np.load(FEAT_DIR / f"{split}_s{k}_{f}_indptr.npy"),
                       np.load(FEAT_DIR / f"{split}_s{k}_{f}_idx.npy")) for f in ("name", "addr")}


def concat_bags(a, b):
    """Rows of ``b`` appended after the rows of ``a``."""
    return {f: CSRBags(np.concatenate([a[f].indptr, b[f].indptr[1:] + a[f].indptr[-1]]),
                       np.concatenate([a[f].indices, b[f].indices])) for f in ("name", "addr")}


def n_rows(bags):
    return len(bags["name"].indptr) - 1


def pseudo_pairs(paths, s1_bags, tgt):
    """Load pseudo-labelled pairs; test-split rows are re-indexed onto concatenated bags.
    Returns (pairs, s1_bags, tgt) with bags extended when test rows are present."""
    ps = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    out = []
    for split, g in ps.groupby("split"):
        s1n = load_norm(split, 1, ["country", "states"])
        g = g.copy()
        g["country"] = s1n.country.to_numpy()[g.s1_row.to_numpy()]
        g["group"] = (s1n.country + "|" + s1n.states).to_numpy()[g.s1_row.to_numpy()]
        if split == "test":
            n1 = {k: n_rows(tgt[k]) for k in (2, 3)}
            g["s1_row"] = g.s1_row + n_rows(s1_bags)
            g["t_row"] = g.t_row + g.src.map(n1)
            s1_bags = concat_bags(s1_bags, load_bags("test", 1))
            tgt = {k: concat_bags(tgt[k], load_bags("test", k)) for k in (2, 3)}
        out.append(g)
    pairs = pd.concat(out, ignore_index=True)
    pairs["held"] = False
    return pairs[["s1_row", "src", "t_row", "held", "country", "group"]], s1_bags, tgt


def gather(bags, rows):
    return tuple(bags[f].gather(rows) for f in ("name", "addr"))


def gather_two(bags2, bags3, rows, src):
    """Gather target rows that live in two files; returns bags in the given row order."""
    order = np.argsort(src, kind="stable")  # S2 rows first, then S3
    parts = []
    for f in ("name", "addr"):
        m2, m3 = order[src[order] == 2], order[src[order] == 3]
        i2, o2, l2 = bags2[f].gather(rows[m2])
        i3, o3, l3 = bags3[f].gather(rows[m3])
        parts.append((np.concatenate([i2, i3]), np.concatenate([o2, o3 + len(i2)]), np.concatenate([l2, l3])))
    return parts, order


@torch.no_grad()
def embed(model, bags, n_rows, chunk=65536):
    model.eval()
    out = np.empty((n_rows, model.skip.out_features), dtype=np.float16)
    for i in range(0, n_rows, chunk):
        rows = np.arange(i, min(i + chunk, n_rows))
        name, addr = gather(bags, rows)
        out[i:i + len(rows)] = model(to_device(name, DEVICE), to_device(addr, DEVICE)).half().cpu().numpy()
    model.train()
    return out


def build_pairs():
    s1 = load_norm("train", 1, ["entity_id", "country", "states"])
    pos = {k: pd.Index(load_norm("train", k, ["entity_id"]).entity_id) for k in (2, 3)}
    gt = load_ground_truth()
    gt["s1_row"] = pd.Index(s1.entity_id).get_indexer(gt.s1)
    gt["src"] = gt.match.str[1].astype(int)
    gt["t_row"] = -1
    for k in (2, 3):
        m = gt.src == k
        gt.loc[m, "t_row"] = pos[k].get_indexer(gt.match[m])
    assert (gt.s1_row >= 0).all() and (gt.t_row >= 0).all()
    gt["held"] = holdout_mask(len(s1))[gt.s1_row]
    gt["country"] = s1.country.to_numpy()[gt.s1_row]
    gt["group"] = (s1.country + "|" + s1.states).to_numpy()[gt.s1_row]
    return s1, gt


def epoch_batches(train_pairs, rng):
    """One random match per S1, grouped by (country, state), batches shuffled."""
    p = train_pairs.sample(frac=1.0, random_state=int(rng.integers(1 << 31)))
    p = p.drop_duplicates("s1_row")
    p = p.sort_values("group", kind="stable")
    starts = np.arange(0, len(p), BATCH)
    rng.shuffle(starts)
    arr = p[["s1_row", "src", "t_row"]].to_numpy()
    for st in starts:
        b = arr[st:st + BATCH]
        if len(b) >= 256:
            yield b


def evaluate(model, s1, s1_bags, tgt_bags, pairs, label, n=40000):
    v1 = embed(model, s1_bags, len(s1))
    out = []
    for c in ("US", "India"):
        pc = pairs[pairs.country == c]
        pc = pc.sample(min(n, len(pc)), random_state=1)
        imask = (s1.country == c).to_numpy()
        index_rows = np.flatnonzero(imask)
        qv = np.empty((len(pc), v1.shape[1]), dtype=np.float16)
        for k in (2, 3):
            m = (pc.src == k).to_numpy()
            qv[m] = embed_rows(model, tgt_bags[k], pc.t_row.to_numpy()[m])
        _, ix = topk_search(v1[imask], qv, 32)
        found = index_rows[ix] == pc.s1_row.to_numpy()[:, None]
        rank = np.where(found.any(1), found.argmax(1), 999)
        out.append(f"{c}: R@1 {np.mean(rank < 1):.4f} R@8 {np.mean(rank < 8):.4f} R@32 {np.mean(rank < 32):.4f}")
    print(f"   [{label}] " + " | ".join(out))


@torch.no_grad()
def embed_rows(model, bags, rows, chunk=65536):
    model.eval()
    out = np.empty((len(rows), model.skip.out_features), dtype=np.float16)
    for i in range(0, len(rows), chunk):
        name, addr = gather(bags, rows[i:i + chunk])
        out[i:i + chunk] = model(to_device(name, DEVICE), to_device(addr, DEVICE)).half().cpu().numpy()
    model.train()
    return out


def train(countries=None, tag: str = "", pseudo=None, exclude_fold=None):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    s1, gt = build_pairs()
    s1_bags = load_bags("train", 1)
    tgt = {k: load_bags("train", k) for k in (2, 3)}
    train_pairs, held_pairs = gt[~gt.held], gt[gt.held]
    if countries:  # leave-country-out experiments train on a subset of countries only
        train_pairs = train_pairs[train_pairs.country.isin(countries)]
    if exclude_fold is not None:  # cross-fitting: this encoder never sees the fold's S1
        fold = s1_fold(len(s1))
        train_pairs = train_pairs[fold[train_pairs.s1_row.to_numpy()] != exclude_fold]
        print(f"excluding S1 fold {exclude_fold}: {len(train_pairs):,} true pairs left")
    if pseudo:
        pp, s1_bags, tgt = pseudo_pairs(pseudo, s1_bags, tgt)
        print(f"pseudo pairs: {len(pp):,} from {pseudo}")
        train_pairs = pd.concat([train_pairs[pp.columns], pp], ignore_index=True)
    print(f"pairs: train {len(train_pairs):,} (S1 {train_pairs.s1_row.nunique():,}) | held-out {len(held_pairs):,}")

    model = RecordEncoder().to(DEVICE)
    opt_sparse = torch.optim.SparseAdam(list(model.emb.parameters()), lr=3e-3)
    dense_params = [p for n, p in model.named_parameters() if not n.startswith("emb.")]
    opt_dense = torch.optim.Adam(dense_params, lr=1e-3)

    for ep in range(EPOCHS):
        t0, losses = time.time(), []
        for b in epoch_batches(train_pairs, rng):
            s1_rows, src, t_rows = b[:, 0], b[:, 1], b[:, 2]
            (tn, ta), order = gather_two(tgt[2], tgt[3], t_rows, src)
            qn, qa = gather(s1_bags, s1_rows[order])
            zq = model(to_device(qn, DEVICE), to_device(qa, DEVICE))
            zt = model(to_device(tn, DEVICE), to_device(ta, DEVICE))
            logits = zq @ zt.T / model.log_temp.exp()
            labels = torch.arange(len(zq), device=DEVICE)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            opt_sparse.zero_grad(); opt_dense.zero_grad()
            loss.backward()
            opt_sparse.step(); opt_dense.step()
            with torch.no_grad():
                model.log_temp.clamp_(np.log(0.01), np.log(0.5))
            losses.append(loss.item())
        print(f"epoch {ep + 1}/{EPOCHS} loss {np.mean(losses):.4f} temp {model.log_temp.exp().item():.4f} ({time.time() - t0:.0f}s)")
        if ep % 2 == 1 or ep == EPOCHS - 1:
            evaluate(model, s1, s1_bags, tgt, held_pairs, "held-out S1")
    evaluate(model, s1, s1_bags, tgt, train_pairs, "train S1   ")
    ENC_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), ENC_DIR / f"encoder{tag}.pt")
    return model


def embed_all(model, tag: str = "", splits=SPLITS):
    for split in splits:
        for k in SOURCES:
            t0 = time.time()
            bags = load_bags(split, k)
            n = len(bags["name"].indptr) - 1
            np.save(VEC_DIR / f"{split}_s{k}_enc{tag}.npy", embed(model, bags, n))
            print(f"embedded {split} s{k}: {n:,}  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", action="store_true", help="skip training, load saved model")
    ap.add_argument("--countries", nargs="+", help="train only on pairs of these S1 countries")
    ap.add_argument("--tag", default="", help="suffix for the model and embedding files, e.g. _loco_us")
    ap.add_argument("--splits", nargs="*", default=list(SPLITS), help="splits to embed (none: skip)")
    ap.add_argument("--pseudo", nargs="+", help="parquet files of pseudo-labelled pairs to add")
    ap.add_argument("--exclude-fold", type=int, help="leave this S1 fold out of training (cross-fitting)")
    args = ap.parse_args()
    if args.embed:
        model = RecordEncoder().to(DEVICE)
        model.load_state_dict(torch.load(ENC_DIR / f"encoder{args.tag}.pt"))
    else:
        model = train(args.countries, args.tag, args.pseudo, args.exclude_fold)
    embed_all(model, args.tag, args.splits)
