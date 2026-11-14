"""PyTorch dataset for FungiTastic images + metadata."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from fungai.data.ingest import DATA_DIR, DEFAULT_DEST, load_metadata

PROCESSED_DIR = DATA_DIR / "processed"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

METADATA_FIELDS = ("habitat", "substrate", "latitude", "longitude",
                   "elevation", "month", "biogeo_zone", "land_cover", "toxicity")

CATEGORICAL_META = ("habitat", "substrate", "biogeo_zone", "land_cover", "toxicity")
CONTINUOUS_META = ("latitude", "longitude", "elevation", "month")


def build_train_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def build_eval_transform(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(int(image_size * 256 / 224)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


@dataclass
class MetadataPreprocessor:
    """Fit on train, transform any split. Categoricals → int codes, continuous → z-score."""

    categorical_maps: dict[str, dict[str, int]] = field(default_factory=dict)
    continuous_stats: dict[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def fit(cls, train_df: pd.DataFrame, fields: tuple[str, ...] = METADATA_FIELDS) -> MetadataPreprocessor:
        cat_maps: dict[str, dict[str, int]] = {}
        cont_stats: dict[str, tuple[float, float]] = {}

        for f in fields:
            if f not in train_df.columns:
                continue
            if f in CATEGORICAL_META:
                vals = sorted(train_df[f].dropna().unique().tolist(), key=str)
                cat_maps[f] = {str(v): i + 1 for i, v in enumerate(vals)}  # 0 reserved for unknown
            elif f in CONTINUOUS_META:
                col = pd.to_numeric(train_df[f], errors="coerce")
                mean = float(col.mean()) if not col.isna().all() else 0.0
                std = float(col.std()) if not col.isna().all() else 1.0
                std = std if std > 0 else 1.0
                cont_stats[f] = (mean, std)

        return cls(categorical_maps=cat_maps, continuous_stats=cont_stats)

    def transform(self, row_metadata: dict) -> torch.Tensor:
        """Convert metadata dict to fixed-length float tensor."""
        values: list[float] = []
        for f in METADATA_FIELDS:
            if f in self.categorical_maps:
                raw = str(row_metadata.get(f, ""))
                values.append(float(self.categorical_maps[f].get(raw, 0)))
            elif f in self.continuous_stats:
                raw = row_metadata.get(f)
                mean, std = self.continuous_stats[f]
                try:
                    val = (float(raw) - mean) / std
                except (TypeError, ValueError):
                    val = 0.0
                values.append(val)
            else:
                values.append(0.0)
        return torch.tensor(values, dtype=torch.float32)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"categorical_maps": self.categorical_maps, "continuous_stats": self.continuous_stats}
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> MetadataPreprocessor:
        data = json.loads(path.read_text())
        cont = {k: tuple(v) for k, v in data["continuous_stats"].items()}
        return cls(categorical_maps=data["categorical_maps"], continuous_stats=cont)


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
        label = self.label_map.get(species, -1)
        meta = {k: row[k] for k in METADATA_FIELDS if k in row.index}
        result = {
            "image": img,
            "label": label,
            "species": species,
            "metadata": meta,
            "image_path": str(row["image_path"]),
        }
        if "observation_id" in row.index:
            result["observation_id"] = str(row["observation_id"])
        return result


class ObservationDataset(Dataset):
    """Groups images by observation_id for multi-image aggregation at eval time."""

    def __init__(
        self,
        metadata: pd.DataFrame,
        transform: transforms.Compose | None = None,
        label_map: dict[str, int] | None = None,
    ):
        if "observation_id" not in metadata.columns:
            raise KeyError("needs observation_id for observation-level grouping")
        if "image_path" not in metadata.columns:
            raise KeyError("metadata needs an 'image_path' column")
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform or DEFAULT_TRANSFORM
        self.label_map = label_map or {}
        self._groups = metadata.groupby("observation_id")
        self._obs_ids = list(self._groups.groups.keys())

    def __len__(self) -> int:
        return len(self._obs_ids)

    def __getitem__(self, idx: int) -> dict:
        obs_id = self._obs_ids[idx]
        rows = self._groups.get_group(obs_id)
        images = []
        for _, row in rows.iterrows():
            img = Image.open(row["image_path"]).convert("RGB")
            images.append(self.transform(img))
        first = rows.iloc[0]
        species = first.get("species")
        label = self.label_map.get(species, -1)
        meta = {k: first[k] for k in METADATA_FIELDS if k in first.index}
        return {
            "images": images,
            "label": label,
            "species": species,
            "metadata": meta,
            "observation_id": str(obs_id),
            "n_images": len(images),
        }


def collate_observations(batch: list[dict]) -> dict:
    """Custom collate for ObservationDataset — pads variable-length image lists."""
    max_imgs = max(b["n_images"] for b in batch)
    c, h, w = batch[0]["images"][0].shape

    padded = torch.zeros(len(batch), max_imgs, c, h, w)
    mask = torch.zeros(len(batch), max_imgs, dtype=torch.bool)
    labels = []
    obs_ids = []

    for i, b in enumerate(batch):
        for j, img in enumerate(b["images"]):
            padded[i, j] = img
            mask[i, j] = True
        labels.append(b["label"])
        obs_ids.append(b["observation_id"])

    return {
        "images": padded,
        "image_mask": mask,
        "label": torch.tensor(labels, dtype=torch.long),
        "observation_id": obs_ids,
    }
