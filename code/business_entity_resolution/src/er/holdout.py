"""The fixed S1 hold-out used to evaluate every learned stage honestly."""
import numpy as np

HOLDOUT_FRAC = 0.05
HOLDOUT_SEED = 0


def holdout_mask(n_s1: int) -> np.ndarray:
    """Boolean mask over train S1 rows (in train_source1 order) that are held out."""
    return np.random.default_rng(HOLDOUT_SEED).random(n_s1) < HOLDOUT_FRAC


N_FOLDS = 4


def s1_fold(n_s1: int, n_folds: int = N_FOLDS) -> np.ndarray:
    """Fold id per train S1 row, used to give every S1 features from an encoder that never
    trained on it (cross-fitting), so the matcher can train on far more than the hold-out."""
    return np.random.default_rng(HOLDOUT_SEED + 1).integers(0, n_folds, n_s1)
