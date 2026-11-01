"""Shared definition of the modeling cohort so every model uses the same subjects and target.

One row per subject. Visits = sessions with a label (and an embedding, if embeddings are given),
sorted by age. Target y = E-SWAN score at the subject's LAST such visit.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import LABELS_CSV, EMB_DIR, EMB_INDEX, COHORT_CSV, EMB_DIM


def load_labels():
    return pd.read_csv(LABELS_CSV)


def load_embeddings(dummy=False, dim=EMB_DIM, seed=0):
    """Returns {(sub, ses): np.ndarray of shape (dim,)}."""
    if dummy:
        labels = load_labels()
        rng = np.random.default_rng(seed)
        return {(s, e): rng.normal(size=dim) for s, e in zip(labels["sub"], labels["ses"])}
    idx = pd.read_csv(EMB_INDEX)
    emb = {}
    for r in idx.itertuples():
        p = Path(r.path)
        if not p.is_absolute() and not p.exists():
            p = EMB_DIR / p.name
        emb[(r.sub, r.ses)] = np.load(p).reshape(-1)
    return emb


def build_cohort(emb=None):
    labels = load_labels()
    if emb is not None:
        keep = [(s, e) in emb for s, e in zip(labels["sub"], labels["ses"])]
        labels = labels[keep]
    rows = []
    for sub, g in labels.sort_values("age").groupby("sub", sort=True):
        rows.append({"sub": sub,
                     "sessions": list(g["ses"]),
                     "ages": list(g["age"]),
                     "y": g["y"].iloc[-1],
                     "n_visits": len(g)})
    cohort = pd.DataFrame(rows)
    COHORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    cohort[["sub", "n_visits", "y"]].to_csv(COHORT_CSV, index=False)
    return cohort
