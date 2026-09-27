"""Competition metric: macro F0.5 over Source-1 entities, singletons included."""
import numpy as np
import pandas as pd


def macro_f05(pred: pd.DataFrame, truth: pd.DataFrame, s1_ids: pd.Series, beta: float = 0.5) -> dict:
    """Score predicted pairs against true pairs.

    pred, truth: long frames with columns ``s1`` and ``match`` (one row per pair).
    s1_ids: every Source-1 entity in the evaluation set; entities absent from both
    frames are true singletons predicted empty and score 1.0.
    """
    pred = pred[["s1", "match"]].drop_duplicates()
    truth = truth[["s1", "match"]].drop_duplicates()
    tp = pred.merge(truth, on=["s1", "match"]).groupby("s1").size()
    n_pred = pred.groupby("s1").size()
    n_true = truth.groupby("s1").size()

    idx = pd.Index(s1_ids.values, name="s1")
    tp, n_pred, n_true = (s.reindex(idx, fill_value=0).to_numpy(float) for s in (tp, n_pred, n_true))

    b2 = beta * beta
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(n_pred > 0, tp / n_pred, 0.0)
        r = np.where(n_true > 0, tp / n_true, 0.0)
        f = np.where(tp > 0, (1 + b2) * p * r / (b2 * p + r), 0.0)
    singleton = n_true == 0
    f = np.where(singleton, (n_pred == 0).astype(float), f)

    return {
        "macro_f05": float(f.mean()),
        "micro_precision": float(tp.sum() / max(n_pred.sum(), 1)),
        "micro_recall": float(tp.sum() / max(n_true.sum(), 1)),
        "singleton_acc": float(f[singleton].mean()) if singleton.any() else float("nan"),
        "non_singleton_f05": float(f[~singleton].mean()) if (~singleton).any() else float("nan"),
        "n_entities": int(len(idx)),
    }


def pair_recall(candidates: pd.DataFrame, truth: pd.DataFrame) -> float:
    """Share of true pairs present in the candidate set (the blocking recall ceiling)."""
    hit = truth.merge(candidates[["s1", "match"]].drop_duplicates(), on=["s1", "match"])
    return len(hit) / max(len(truth), 1)
