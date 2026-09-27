"""Loading and pruning the retrieval candidates written by retrieve.py."""
import numpy as np
import pandas as pd

from .config import CACHE_DIR
from .io import load_norm

CAND_DIR = CACHE_DIR / "cand"
K_ENC_KEEP, K_SVD_KEEP = 5, 1
RANK_COLS = ["enc_rank", "comb_rank", "name_rank", "addr_rank"]
ALL_COLS = ["q_row", "s1_row", *RANK_COLS, "enc_cos", "name_cos", "addr_cos"]


def prune(c: pd.DataFrame, k_enc: int = K_ENC_KEEP, k_svd: int = K_SVD_KEEP) -> pd.DataFrame:
    """Keep encoder top-k_enc or top-k_svd of any SVD retriever (the candidate_pairs.tsv set).

    Records without an address keep every retrieved candidate: name is their only signal,
    so same-name chains push the true S1 down the ranking (74% of their misses were
    candidates dropped here).
    """
    keep = (c.enc_rank < k_enc) | (c.comb_rank < k_svd) | (c.name_rank < k_svd) | (c.addr_rank < k_svd)
    if "q_empty" in c:
        keep |= c.q_empty.to_numpy()
    return c[keep]


def load_candidates(split: str, row_filter=None, columns=None, tag: str = "") -> pd.DataFrame:
    """Candidates of both sources with ``src`` and ``q_empty`` (query has no address)
    columns; ``row_filter`` runs per file to bound memory."""
    parts = []
    for k in (2, 3):
        c = pd.read_parquet(CAND_DIR / f"{split}_s{k}{tag}.parquet", columns=columns or ALL_COLS)
        c["q_empty"] = load_norm(split, k, ["addr_empty"]).addr_empty.to_numpy()[c.q_row.to_numpy()]
        if row_filter is not None:
            c = row_filter(c)
        c["src"] = np.int8(k)
        parts.append(c)
    return pd.concat(parts, ignore_index=True)
