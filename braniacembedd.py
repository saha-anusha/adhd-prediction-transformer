from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
WORK_DIR = PROJECT_DIR / "work"
MANIFEST = WORK_DIR / "manifest_preproc.csv"
OUTPUT_DIR = WORK_DIR

# BrainIAC
BRAINIAC_SRC = PROJECT_DIR / "BrainIAC" / "src"
CHECKPOINT = BRAINIAC_SRC / "checkpoints" / "BrainIAC.ckpt"

sys.path.insert(0, str(BRAINIAC_SRC))

from load_brainiac import load_brainiac
from dataset import get_validation_transform

# Settings
DEVICE = "auto"
BATCH_SIZE = 1
NUM_WORKERS = 0
LIMIT = None
OVERWRITE = False

LOG = logging.getLogger("brainiac_embeddings")


class ScanDataset(Dataset):
    def __init__(self, records: pd.DataFrame, transform):
        self.records = records.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        image = self.transform({"image": str(row.preproc_path)})["image"]
        return {"image": image, "row_idx": idx}

#gets embeddings
def extract():
    manifest_path = MANIFEST.expanduser().resolve()
    output_dir = OUTPUT_DIR.expanduser().resolve()

    data = pd.read_csv(manifest_path, dtype={"subject": str, "session": str})
    required = {"subject", "session", "preproc_path", "qc_ok"}


    data = data.loc[data.qc_ok.astype(str).str.lower().isin(("true", "1", "yes"))].copy()
    data = data.sort_values(["subject", "session"]).reset_index(drop=True)
    if LIMIT is not None:
        data = data.head(LIMIT).copy()

    def find_scan(row):
        path = Path(str(row.preproc_path)).expanduser()
        candidates = [path]
        if not path.is_absolute():
            candidates.extend((manifest_path.parent / path, PROJECT_DIR / path))
        candidates.append(
            WORK_DIR / "preproc" / row.subject / row.session
            / f"{row.subject}_{row.session}_desc-preproc_T1w.nii.gz"
        )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate.resolve())
        return None

    data["preproc_path"] = [find_scan(row) for row in data.itertuples()]
    embedding_root = output_dir / "embeddings"
    data["embedding_path"] = [
        str(embedding_root / row.subject / f"{row.subject}_{row.session}_brainiac.npy")
        for row in data.itertuples()
    ]

    def valid_embedding(path):
        try:
            array = np.load(path, allow_pickle=False)
            return array.shape == (EXPECTED_DIM,) and np.isfinite(array).all()
        except (OSError, ValueError, EOFError):
            return False

    pending = data if OVERWRITE else data.loc[~data.embedding_path.map(valid_embedding)]
    LOG.info("Found %d scans; %d need extraction", len(data), len(pending))
    errors = []

    if not pending.empty:
        load_brainiac, get_validation_transform, checkpoint = load_brainiac_components()
        device = torch.device("cuda" if DEVICE == "auto" and torch.cuda.is_available()
                              else "cpu" if DEVICE == "auto" else DEVICE)
        LOG.info("Loading BrainIAC checkpoint %s on %s", checkpoint, device)
        model = load_brainiac(str(checkpoint), str(device)).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        transform = get_validation_transform(image_size=(96, 96, 96))
        dataset = ScanDataset(pending, transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(device.type == "cuda"))
        with torch.inference_mode():
            for batch in loader:
                x = batch["image"].to(device=device, dtype=torch.float32, non_blocking=True)
                features = model(x)
                vectors = features.cpu().numpy().astype(np.float32)
                for idx, vector in zip(batch["row_idx"].tolist(), vectors):
                    row = dataset.records.iloc[idx]
                    if not np.isfinite(vector).all():
                        errors.append({"subject": row.subject, "session": row.session, "error": "Nonfinite embedding"})
                        continue
                    dest = Path(row.embedding_path)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    temp = dest.with_suffix(".tmp.npy")
                    np.save(temp, vector)
                    temp.replace(dest)
                    LOG.info("Saved %s", dest)

    data["embedding_ok"] = data.embedding_path.map(valid_embedding)
    output_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_dir / "embeddings_manifest.csv", index=False)
    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "embedding_errors.csv", index=False)
    LOG.info("Valid embeddings: %d/%d", int(data.embedding_ok.sum()), len(data))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    extract()
