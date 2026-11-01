"""Save per-model predictions and summarize them into one results table.

Every model writes results/preds/<model>_seed<N>.csv (sub, y_true, y_pred) via save_preds().
Then: python -m src.eval.harness  ->  results/tables/results.csv
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from src.config import PREDS_DIR, TABLES_DIR, COHORT_CSV


def save_preds(model_name, seed, subs, y_true, y_pred, out_dir=PREDS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sub": subs, "y_true": y_true, "y_pred": y_pred}).to_csv(
        out_dir / f"{model_name}_seed{seed}.csv", index=False)


def pearson(a, b):
    return np.nan if np.std(a) == 0 or np.std(b) == 0 else float(np.corrcoef(a, b)[0, 1])


def summarize(preds_dir=PREDS_DIR):
    cohort = pd.read_csv(COHORT_CSV)
    multi = set(cohort.loc[cohort["n_visits"] >= 2, "sub"])
    records = []
    for p in sorted(preds_dir.glob("*_seed*.csv")):
        model, seed = p.stem.rsplit("_seed", 1)
        df = pd.read_csv(p)
        for subset, d in [("all", df), ("multi_visit", df[df["sub"].isin(multi)])]:
            if len(d) < 3:
                continue
            records.append({"model": model, "seed": int(seed), "subset": subset, "n": len(d),
                            "mae": mean_absolute_error(d["y_true"], d["y_pred"]),
                            "r2": r2_score(d["y_true"], d["y_pred"]),
                            "r": pearson(d["y_true"], d["y_pred"])})
    res = pd.DataFrame(records)
    s = (res.groupby(["subset", "model"])
            .agg(n=("n", "first"),
                 mae=("mae", "mean"), mae_sd=("mae", "std"),
                 r2=("r2", "mean"), r2_sd=("r2", "std"),
                 r=("r", "mean"), r_sd=("r", "std"))
            .reset_index())
    table = pd.DataFrame({
        "subset": s["subset"], "model": s["model"], "n": s["n"],
        "MAE": [f"{m:.2f} ± {sd:.2f}" for m, sd in zip(s["mae"], s["mae_sd"])],
        "R2": [f"{m:.3f} ± {sd:.3f}" for m, sd in zip(s["r2"], s["r2_sd"])],
        "Pearson r": [f"{m:.3f} ± {sd:.3f}" for m, sd in zip(s["r"], s["r_sd"])],
    })
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    name = "results.csv" if preds_dir == PREDS_DIR else f"results_{preds_dir.name}.csv"
    table.to_csv(TABLES_DIR / name, index=False)
    return table


if __name__ == "__main__":
    print(summarize().to_string(index=False))
