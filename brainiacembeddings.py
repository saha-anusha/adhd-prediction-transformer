
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# Paths
PROJECT_DIR = Path(__file__).resolve().parent
WORK_DIR = PROJECT_DIR / "work"
BRAINIAC_SRC = PROJECT_DIR / "BrainIAC" / "src"
CHECKPOINT = BRAINIAC_SRC / "checkpoints" / "BrainIAC.ckpt"

sys.path.insert(0, str(BRAINIAC_SRC))

from load_brainiac import load_brainiac
from dataset import get_validation_transform

# Settings
BATCH_SIZE = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ScanDataset(Dataset):
    def __init__(self, data, transform):
        self.data = data.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data.iloc[idx].preproc_path
        image = self.transform({"image": str(path)})["image"]
        return image, idx


def extract():
    data = pd.read_csv(
        WORK_DIR / "manifest_preproc.csv",
        dtype={"subject": str, "session": str}
    )

    data = data[data.qc_ok.astype(str).str.lower().isin(["true", "1", "yes"])].copy()
    data = data.reset_index(drop=True)

    
    data["embedding_path"] = [
        str(WORK_DIR / "embeddings" / row.subject /
            f"{row.subject}_{row.session}_brainiac.npy")
        for row in data.itertuples()
    ]

    
    pending = data[~data.embedding_path.map(lambda p: Path(p).exists())].copy()

    print(f"Found {len(data)} scans, {len(pending)} need extraction")

    if not pending.empty:
        model = load_brainiac(str(CHECKPOINT), str(DEVICE)).eval()
        transform = get_validation_transform(image_size=(96, 96, 96))

        dataset = ScanDataset(pending, transform)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

        with torch.inference_mode():
            for images, indices in loader:
                images = images.to(DEVICE, dtype=torch.float32)
                embeddings = model(images).cpu().numpy()

                for idx, embedding in zip(indices.tolist(), embeddings):
                    path = Path(dataset.data.iloc[idx].embedding_path)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(path, embedding.astype(np.float32))
                    print(f"Saved {path}")

    data["embedding_ok"] = data.embedding_path.map(lambda p: Path(p).exists())
    data.to_csv(WORK_DIR / "embeddings_manifest.csv", index=False)


if __name__ == "__main__":
    extract()
