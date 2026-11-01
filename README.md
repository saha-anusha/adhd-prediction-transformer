# adhd-prediction-transformer

Temporal transformer over BrainIAC 3D vision-transformer foundation-model embeddings to predict ADHD symptom severity from longitudinal brain MRI.

CS 7643 group project: Anusha Saha, Arnav Dantuluri, Meredith Gideon, Vineet Upadhya.

We encode each visit's T1-weighted MRI from Penn LEAD (OpenNeuro ds007116) with the frozen BrainIAC
foundation model, then train a temporal transformer over a participant's visit embeddings, age, and
inter-visit intervals to predict E-SWAN ADHD scores. We compare against single-scan and
embedding-difference baselines.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
Data is not stored in this repo. Download Penn LEAD from OpenNeuro into `data/raw/`.
Cached embeddings are shared via [TODO: Drive / ICE path].

## Layout
| Path | Owner | Contents |
|---|---|---|
| `src/data/` | Vineet / Anusha | preprocessing; `labels.py` (labels table), `cohort.py` (shared cohort loader) |
| `src/embeddings/` | Vineet | BrainIAC embedding extraction |
| `src/eval/` | Anusha | `folds.py` (shared splits), `harness.py` (metrics + results table) |
| `src/baselines/` | Anusha | mean, single-scan, embedding-difference baselines |
| `src/models/` | Arnav | temporal transformer |
| `src/probe/` | Vineet | cortical-thickness probe |
| `results/` | everyone | folds, predictions, results tables (small CSVs, tracked in git) |

## Interfaces (do not change without telling the team)

**Embeddings** (produced by `src/embeddings/`):
- One file per session: `data/embeddings/sub-<ID>_ses-<ID>.npy`, shape `(D,)`, D = [TODO: confirm]
- Index: `data/embeddings/index.csv` with columns `sub, ses, path`

**Labels** (produced by `src/data/labels.py`):
- `data/labels.csv` with columns `sub, ses, age, y` (y = E-SWAN ADHD score)

**Folds** (produced by `src/eval/folds.py`): every model must use these.
- `results/folds/folds_seed<N>.csv` with columns `sub, fold`, for seeds 0–4, 5 folds each

**Predictions** (produced by every model):
- `results/preds/<model_name>_seed<N>.csv` with columns `sub, y_true, y_pred`
- Target: each participant's E-SWAN score at their last visit with a label (and an embedding)
- Then run `python -m src.eval.harness` to rebuild `results/tables/results.csv`

## Running the baselines
```bash
python -m src.data.labels --explore          # inspect phenotype files; fill TODOs in labels.py
python -m src.data.labels                    # build data/labels.csv
python -m src.baselines.run_baselines --dummy   # pipeline test on random embeddings
python -m src.baselines.run_baselines        # real run (mean-only until embeddings exist)
```
Sanity check: on the dummy run, ridge baselines should have R² at or slightly below 0.
A clearly positive R² on random features means something is leaking.

## Adding a model to the results table
```python
from src.config import SEEDS
from src.data.cohort import load_embeddings, build_cohort
from src.eval.folds import load_folds
from src.eval.harness import save_preds

emb = load_embeddings()
cohort = build_cohort(emb)   # one row per subject: sessions, ages (sorted), y = last-visit score
subs, y = cohort["sub"].to_numpy(), cohort["y"].to_numpy()
for seed in SEEDS:
    fold = load_folds(seed, subs)
    preds = ...  # for f in range(5): train on fold != f, predict fold == f
    save_preds("temporal_transformer", seed, subs, y, preds)
```
Then `python -m src.eval.harness`. Use a distinct model name per variant
(e.g. `temporal_transformer_no_time` for the H2 ablation).

When real embeddings arrive, rerun the baselines and every model against them so all rows in the
table use the same cohort (`results/cohort.csv` is rebuilt by `build_cohort`).

## Status
| Component | Owner | Status |
|---|---|---|
| Preprocessing + embeddings | Vineet | in progress |
| Labels, folds, harness, baselines | Anusha | code done; waiting on phenotype columns + embeddings |
| Temporal transformer | Arnav | in progress |
| Report | Meredith | in progress |
