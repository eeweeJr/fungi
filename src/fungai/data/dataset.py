"""PyTorch dataset for FungiTastic images + metadata."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from fungai.data.ingest import DATA_DIR, DEFAULT_DEST, load_metadata

PROCESSED_DIR = DATA_DIR / "processed"

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

METADATA_FIELDS = ("habitat", "substrate", "latitude", "longitude",
                   "elevation", "month", "biogeo_zone", "land_cover", "toxicity")


def load_or_build_species_map(subset: str, dest: Path = DEFAULT_DEST) -> dict[str, int]:
    """Species -> int mapping built from the train split and persisted,
    so train/val/test share one label space."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    map_path = PROCESSED_DIR / f"species_map_{subset}.json"
    if map_path.exists():
        return json.loads(map_path.read_text())
    train = load_metadata(subset, "train", dest=dest)
    if "species" not in train.columns:
        raise KeyError("no species column found in train metadata")
    species = sorted(train["species"].dropna().unique().tolist())
    mapping = {name: i for i, name in enumerate(species)}
    map_path.write_text(json.dumps(mapping, indent=0))
    return mapping


class FungiDataset(Dataset):
    """Loads fungi images with labels and contextual metadata."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        transform: transforms.Compose | None = None,
        label_map: dict[str, int] | None = None,
    ):
        if "image_path" not in metadata.columns:
            raise KeyError("metadata needs an 'image_path' column (are images downloaded?)")
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform or DEFAULT_TRANSFORM
        self.label_map = label_map or {}

    @classmethod
    def from_split(
        cls,
        subset: str = "mini",
        split: str = "train",
        size: str = "300",
        transform: transforms.Compose | None = None,
    ) -> FungiDataset:
        df = load_metadata(subset, split, size)
        label_map = load_or_build_species_map(subset)
        return cls(df, transform=transform, label_map=label_map)

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx: int) -> dict:
        row = self.metadata.iloc[idx]
        img = Image.open(row["image_path"]).convert("RGB")
        img = self.transform(img)
        species = row.get("species")
        # -1 = unknown species (open-set val/test classes absent from train)
        label = self.label_map.get(species, -1)
        return {
            "image": img,
            "label": label,
            "species": species,
            "metadata": {k: row[k] for k in METADATA_FIELDS if k in row.index},
        }
