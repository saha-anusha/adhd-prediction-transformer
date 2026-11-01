"""Build data/labels.csv with columns: sub, ses, age, y (E-SWAN ADHD score).

Step 1: python -m src.data.labels --explore   (see what files/columns exist)
Step 2: fill in the TODOs below
Step 3: python -m src.data.labels
"""
import argparse
import glob
from pathlib import Path

import pandas as pd

from src.config import RAW, LABELS_CSV

# ---- TODO: fill these in after running --explore ----
PHENO_FILE = RAW / "phenotype" / "eswan.tsv"
PHENO_SUB_COL = "participant_id"
PHENO_SES_COL = "session_id"
SCORE_COLS = ["eswan_adhd_total"]  # a single total column, or a list of item columns to sum
AGE_COL = "age"                    # age column inside each sub-*/sub-*_sessions.tsv
# -----------------------------------------------------


def norm_id(x, prefix):
    """Make IDs consistent with BIDS folder names, e.g. '001' -> 'sub-001'."""
    x = str(x).strip()
    return x if x.startswith(prefix) else f"{prefix}{x}"


def explore():
    print("=== Phenotype files ===")
    for p in sorted(glob.glob(str(RAW / "phenotype" / "*.tsv"))):
        df = pd.read_csv(p, sep="\t", nrows=5)
        print(f"\n{Path(p).name}  ->  columns: {list(df.columns)[:30]}")
    ses_files = sorted(glob.glob(str(RAW / "sub-*" / "sub-*_sessions.tsv")))
    print(f"\n=== {len(ses_files)} sessions.tsv files ===")
    if ses_files:
        print(pd.read_csv(ses_files[0], sep="\t").head())


def load_ages():
    rows = []
    for p in glob.glob(str(RAW / "sub-*" / "sub-*_sessions.tsv")):
        df = pd.read_csv(p, sep="\t")
        df["sub"] = Path(p).name.split("_")[0]
        df = df.rename(columns={"session_id": "ses", AGE_COL: "age"})
        rows.append(df[["sub", "ses", "age"]])
    ages = pd.concat(rows, ignore_index=True)
    ages["ses"] = ages["ses"].map(lambda x: norm_id(x, "ses-"))
    return ages


def build():
    ph = pd.read_csv(PHENO_FILE, sep="\t")
    ph["sub"] = ph[PHENO_SUB_COL].map(lambda x: norm_id(x, "sub-"))
    ph["ses"] = ph[PHENO_SES_COL].map(lambda x: norm_id(x, "ses-"))
    # NaN if any item is missing, so partial questionnaires don't produce fake low scores
    ph["y"] = ph[SCORE_COLS].sum(axis=1, min_count=len(SCORE_COLS))

    df = (ph[["sub", "ses", "y"]]
          .merge(load_ages(), on=["sub", "ses"], how="inner")
          .dropna(subset=["y", "age"])
          .sort_values(["sub", "age"]))
    LABELS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(LABELS_CSV, index=False)

    visits = df.groupby("sub").size()
    print(f"Saved {LABELS_CSV}")
    print(f"{df['sub'].nunique()} subjects, {len(df)} labeled sessions")
    print("Visits per subject:", visits.value_counts().sort_index().to_dict())
    print(f"y range: {df['y'].min():.2f} to {df['y'].max():.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--explore", action="store_true")
    args = ap.parse_args()
    explore() if args.explore else build()
