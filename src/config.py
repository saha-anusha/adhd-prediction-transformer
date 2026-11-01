from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
LABELS_CSV = DATA / "labels.csv"
EMB_DIR = DATA / "embeddings"
EMB_INDEX = EMB_DIR / "index.csv"

RESULTS = ROOT / "results"
FOLDS_DIR = RESULTS / "folds"
PREDS_DIR = RESULTS / "preds"
TABLES_DIR = RESULTS / "tables"
COHORT_CSV = RESULTS / "cohort.csv"

SEEDS = [0, 1, 2, 3, 4]
K_FOLDS = 5
EMB_DIM = 768  # TODO: confirm with Vineet once BrainIAC is running
