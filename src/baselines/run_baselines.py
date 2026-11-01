"""Baselines:
  mean_predictor     - predicts the training-fold mean (labels only; works before embeddings exist)
  single_scan_ridge  - last visit's BrainIAC embedding -> ridge regression
  emb_diff_ridge     - [last, last - first] embeddings -> ridge (diff = 0 for 1-visit subjects)

python -m src.baselines.run_baselines           
"""
import argparse

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.config import EMB_INDEX, PREDS_DIR, SEEDS, K_FOLDS
from src.data.cohort import load_embeddings, build_cohort
from src.eval.folds import make_folds, load_folds
from src.eval.harness import save_preds, summarize


def ridge():
    # Scaler + alpha search are fit on the training fold only, so there's no test leakage
    return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 15)))


def build_features(cohort, emb):
    last = np.stack([emb[(r.sub, r.sessions[-1])] for r in cohort.itertuples()])
    first = np.stack([emb[(r.sub, r.sessions[0])] for r in cohort.itertuples()])
    return {"single_scan_ridge": last,
            "emb_diff_ridge": np.hstack([last, last - first])}


def run(dummy=False):
    make_folds()
    emb = load_embeddings(dummy=True) if dummy else (load_embeddings() if EMB_INDEX.exists() else None)
    if emb is None:
        print("No embeddings yet: running mean_predictor only.")
    cohort = build_cohort(emb)
    feats = build_features(cohort, emb) if emb is not None else {}
    out_dir = PREDS_DIR / "dummy" if dummy else PREDS_DIR

    subs, y = cohort["sub"].to_numpy(), cohort["y"].to_numpy()
    print(f"Cohort: {len(subs)} subjects, {(cohort['n_visits'] >= 2).sum()} with 2+ visits")

    for seed in SEEDS:
        fold = load_folds(seed, subs)
        preds = {name: np.zeros(len(y)) for name in ["mean_predictor", *feats]}
        for f in range(K_FOLDS):
            tr, te = fold != f, fold == f
            if te.sum() == 0:
                continue
            preds["mean_predictor"][te] = y[tr].mean()
            for name, X in feats.items():
                preds[name][te] = ridge().fit(X[tr], y[tr]).predict(X[te])
        for name, p in preds.items():
            save_preds(name, seed, subs, y, p, out_dir)

    print(summarize(out_dir).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dummy", action="store_true")
    run(ap.parse_args().dummy)
