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


COLOR_AUGS = ("full", "brightness_contrast", "none")


def backbone_normalization(backbone: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """(mean, std) the backbone's pretrained weights expect, from timm's config.

    Falls back to ImageNet statistics for names timm does not know. Note that
    `vit_base_patch16_224` expects (0.5, 0.5, 0.5), not ImageNet statistics.
    """
    import timm

    try:
        cfg = timm.models.get_pretrained_cfg(backbone)
    except Exception:  # noqa: BLE001 - unknown or custom name
        cfg = None
    if cfg is None or cfg.mean is None or cfg.std is None:
        return IMAGENET_MEAN, IMAGENET_STD
    return tuple(cfg.mean), tuple(cfg.std)


def build_train_transform(
    image_size: int = 224,
    crop_scale_min: float = 0.8,
    color_aug: str = "full",
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> transforms.Compose:
    """Training augmentation. Defaults reproduce the original transform.

    color_aug: full = brightness, contrast, saturation and hue jitter;
    brightness_contrast = no saturation or hue change (cap and gill colour are
    diagnostic); none = no colour jitter.
    """
    if color_aug not in COLOR_AUGS:
        raise ValueError(f"color_aug must be one of {COLOR_AUGS}, got {color_aug!r}")
    steps = [
        transforms.RandomResizedCrop(image_size, scale=(crop_scale_min, 1.0)),
        transforms.RandomHorizontalFlip(),
    ]
    if color_aug == "full":
        steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05))
    elif color_aug == "brightness_contrast":
        steps.append(transforms.ColorJitter(brightness=0.2, contrast=0.2))
    steps += [transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)]
    return transforms.Compose(steps)


def build_eval_transform(
    image_size: int = 224,
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> transforms.Compose:
    """Resize the shorter side to image_size * 256/224, then centre-crop."""
    return transforms.Compose([
        transforms.Resize(eval_resize_size(image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def eval_resize_size(image_size: int) -> int:
    """Shorter-side length the eval transform resizes to before centre-cropping."""
    return int(image_size * 256 / 224)


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
        genus_map: dict[str, int] | None = None,
    ):
        if "image_path" not in metadata.columns:
            raise KeyError("metadata needs an 'image_path' column (are images downloaded?)")
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform or DEFAULT_TRANSFORM
        self.label_map = label_map or {}
        # Genus is labelled separately because a held-out species (label -1)
        # still has a known genus.
        self.genus_map = genus_map

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
        if self.genus_map is not None:
            result["genus_label"] = self.genus_map.get(row.get("genus"), -1)
        if "revisited_site" in row.index:
            result["revisited_site"] = bool(row["revisited_site"])
        return result


class ObservationDataset(Dataset):
    """One item per observation: up to `max_photos` photos of the same mushroom.

    train=True draws k photos uniformly from 1..min(n, max_photos), without
    replacement and in random order, so the model sees every photo count and never
    learns an order. train=False uses `select_eval_photos`: a deterministic first
    `max_photos` by filename, the same set the per-image baseline is scored on.
    max_photos=None keeps every photo (original behaviour).

    Photos carry no view labels and no position information.
    """

    def __init__(
        self,
        metadata: pd.DataFrame,
        transform: transforms.Compose | None = None,
        label_map: dict[str, int] | None = None,
        genus_map: dict[str, int] | None = None,
        max_photos: int | None = None,
        train: bool = False,
        seed: int = 0,
    ):
        from fungai.data.observations import ObservationIndex, select_eval_photos

        if "observation_id" not in metadata.columns:
            raise KeyError("needs observation_id for observation-level grouping")
        if "image_path" not in metadata.columns:
            raise KeyError("metadata needs an 'image_path' column")
        if "filename" not in metadata.columns:
            metadata = metadata.assign(filename=metadata["image_path"].map(lambda p: Path(p).name))
        self.n_photos_available = len(metadata)
        self.n_truncated = 0
        if not train and max_photos is not None:
            per_obs = metadata.groupby("observation_id").size()
            self.n_truncated = int((per_obs > max_photos).sum())
            metadata = select_eval_photos(metadata, max_photos)
        self.metadata = metadata.reset_index(drop=True)
        self.transform = transform or DEFAULT_TRANSFORM
        self.label_map = label_map or {}
        self.genus_map = genus_map
        self.max_photos = max_photos
        self.train = train
        self._rng = np.random.default_rng(seed)
        self._index = ObservationIndex(self.metadata)

        self.observation_ids = [str(o) for o in self._index.observation_ids]
        first_rows = self._index.order[self._index.starts]
        self.observation_species = self.metadata["species"].astype(object).to_numpy()[first_rows]
        self._paths = self.metadata["image_path"].astype(str).to_numpy()
        self._has_revisited = "revisited_site" in self.metadata.columns
        self._genus = (self.metadata["genus"].astype(object).to_numpy()
                       if "genus" in self.metadata.columns else None)

    def __len__(self) -> int:
        return len(self._index)

    def photo_rows(self, idx: int) -> np.ndarray:
        """Row positions of the photos used for this item (random in train mode)."""
        rows = self._index.rows(idx)
        if not self.train:
            return rows
        cap = len(rows) if self.max_photos is None else min(len(rows), self.max_photos)
        k = int(self._rng.integers(1, cap + 1))
        return self._rng.choice(rows, size=k, replace=False)

    def __getitem__(self, idx: int) -> dict:
        rows = self.photo_rows(idx)
        images = [self.transform(Image.open(self._paths[r]).convert("RGB")) for r in rows]
        first = self.metadata.iloc[int(self._index.rows(idx)[0])]
        species = first.get("species")
        label = self.label_map.get(species, -1)
        meta = {k: first[k] for k in METADATA_FIELDS if k in first.index}
        item = {
            "images": images,
            "label": label,
            "species": species,
            "metadata": meta,
            "observation_id": self.observation_ids[idx],
            "n_images": len(images),
            "image_paths": [self._paths[r] for r in rows],
        }
        if self.genus_map is not None and self._genus is not None:
            item["genus_label"] = self.genus_map.get(self._genus[rows[0]], -1)
        if self._has_revisited:
            item["revisited_site"] = bool(first["revisited_site"])
        return item


def collate_observations(batch: list[dict]) -> dict:
    """Pad variable-length photo lists into the observation batch contract.

    images [B, P, 3, H, W] zero-padded, image_mask [B, P] (True = real photo),
    label [B], n_images [B], observation_id list, plus genus_label [B],
    image_paths list[list[str]] and revisited_site [B] when the items carry them.
    """
    max_imgs = max(b["n_images"] for b in batch)
    c, h, w = batch[0]["images"][0].shape

    padded = torch.zeros(len(batch), max_imgs, c, h, w)
    mask = torch.zeros(len(batch), max_imgs, dtype=torch.bool)
    for i, b in enumerate(batch):
        for j, img in enumerate(b["images"]):
            padded[i, j] = img
            mask[i, j] = True

    out = {
        "images": padded,
        "image_mask": mask,
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "n_images": torch.tensor([b["n_images"] for b in batch], dtype=torch.long),
        "observation_id": [b["observation_id"] for b in batch],
    }
    if all("genus_label" in b for b in batch):
        out["genus_label"] = torch.tensor([b["genus_label"] for b in batch], dtype=torch.long)
    if all("image_paths" in b for b in batch):
        out["image_paths"] = [b["image_paths"] for b in batch]
    if all("revisited_site" in b for b in batch):
        out["revisited_site"] = torch.tensor([b["revisited_site"] for b in batch], dtype=torch.bool)
    return out
