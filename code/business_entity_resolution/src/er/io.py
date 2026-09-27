"""Loading cached parquet data and writing submission TSVs."""
import pandas as pd

from .config import CACHE_DIR


def source_path(split: str, k: int):
    return CACHE_DIR / f"{split}_source{k}.parquet"


def load_source(split: str, k: int) -> pd.DataFrame:
    return pd.read_parquet(source_path(split, k))


def load_norm(split: str, k: int, columns=None) -> pd.DataFrame:
    """Normalised records written by ``normalize_data.py``."""
    return pd.read_parquet(CACHE_DIR / "norm" / f"{split}_s{k}.parquet", columns=columns)


def load_ground_truth() -> pd.DataFrame:
    """Positive pairs as a long frame: columns s1, match."""
    gt = pd.read_parquet(CACHE_DIR / "train_ground_truth.parquet")
    pairs = gt.assign(match=gt.matched_entity_ids.str.split(",")).explode("match")
    pairs = pairs[pairs.match.notna() & (pairs.match != "")]
    return pairs.rename(columns={"source1_entity_id": "s1"})[["s1", "match"]].reset_index(drop=True)


def write_id_lists(pairs: pd.DataFrame, s1_ids: pd.Series, id_col: str, path, list_header: str):
    """Write one row per S1 id with a comma-joined, de-duplicated list of ``pairs[id_col]``.

    Uses LF line endings: the official validator only strips ``\\n``, so a CRLF file
    would leave ``\\r`` glued to the last id of every row.
    """
    pairs = pairs[["s1", id_col]].drop_duplicates().sort_values(["s1", id_col])
    lists = pairs.groupby("s1", sort=False)[id_col].agg(",".join)
    lists = lists.reindex(s1_ids.values).fillna("")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"source1_entity_id\t{list_header}\n")
        for s1, ids in zip(lists.index, lists.values):
            f.write(f"{s1}\t{ids}\n")
