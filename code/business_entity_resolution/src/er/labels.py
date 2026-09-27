"""Ground truth expressed in row space: (s1_row, src, t_row) triples for the train split."""
import numpy as np
import pandas as pd

from .io import load_ground_truth, load_norm


def truth_rows() -> pd.DataFrame:
    s1_ids = pd.Index(load_norm("train", 1, ["entity_id"]).entity_id)
    gt = load_ground_truth()
    src = gt.match.str[1].astype(int).to_numpy()
    t_row = np.empty(len(gt), dtype=np.int64)
    for k in (2, 3):
        ids = pd.Index(load_norm("train", k, ["entity_id"]).entity_id)
        t_row[src == k] = ids.get_indexer(gt.match[src == k])
    return pd.DataFrame({"s1_row": s1_ids.get_indexer(gt.s1), "src": src.astype(np.int8), "t_row": t_row})


def pair_key(src, row) -> np.ndarray:
    """Single int64 id for a target record across both sources."""
    return np.asarray(src, dtype=np.int64) << 32 | np.asarray(row, dtype=np.int64)
