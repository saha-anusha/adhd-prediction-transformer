"""Participant-level folds, saved to disk so every model uses identical splits."""
import numpy as np
import pandas as pd

from src.config import FOLDS_DIR, SEEDS, K_FOLDS
from src.data.cohort import load_labels


def make_folds(force=False):
    FOLDS_DIR.mkdir(parents=True, exist_ok=True)
    subs = sorted(load_labels()["sub"].unique())
    for seed in SEEDS:
        path = FOLDS_DIR / f"folds_seed{seed}.csv"
        if path.exists() and not force:
            continue
        perm = np.random.default_rng(seed).permutation(subs)
        pd.DataFrame({"sub": perm, "fold": np.arange(len(perm)) % K_FOLDS}).to_csv(path, index=False)


def load_folds(seed, subs):
    """Fold id for each subject in `subs`, in the same order."""
    folds = pd.read_csv(FOLDS_DIR / f"folds_seed{seed}.csv").set_index("sub")["fold"]
    out = folds.reindex(subs)
    assert out.notna().all(), "Some cohort subjects are missing from the fold files"
    return out.astype(int).to_numpy()
