"""Shared fixtures for fungai test suite. All network-free, synthetic data only."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from fungai.model.classifier import FungiClassifier


@pytest.fixture
def dummy_species_map():
    return {"species_A": 0, "species_B": 1, "species_C": 2, "species_D": 3, "species_E": 4}


@pytest.fixture
def dummy_probs():
    rng = np.random.default_rng(42)
    logits = rng.standard_normal((100, 5))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


@pytest.fixture
def dummy_labels():
    rng = np.random.default_rng(42)
    labels = rng.integers(0, 5, size=100)
    labels[95:] = -1
    return labels


@pytest.fixture
def dummy_toxicity_map():
    return {
        "species_A": True,
        "species_B": False,
        "species_C": False,
        "species_D": True,
        "species_E": False,
    }


@pytest.fixture
def tiny_model():
    return FungiClassifier(
        num_classes=5, backbone="mobilenetv3_small_050", pretrained=False, dropout=0.0
    )


class _DictDataset(Dataset):
    def __init__(self, n=20):
        self.images = torch.randn(n, 3, 224, 224)
        self.labels = torch.randint(0, 5, (n,))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        lbl = int(self.labels[idx])
        return {
            "image": self.images[idx],
            "label": self.labels[idx],
            "species": f"species_{chr(65 + lbl)}",
            "metadata": {},
            "image_path": f"/fake/img_{idx}.jpg",
            "observation_id": f"obs_{idx // 2}",
        }


def _test_collate(batch: list[dict]) -> dict:
    """Same as _fungi_collate but avoids importing from the module under test."""
    images = torch.stack([b["image"] for b in batch])
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    result = {"image": images, "label": labels}
    if "image_path" in batch[0]:
        result["image_path"] = [b["image_path"] for b in batch]
    if "observation_id" in batch[0]:
        result["observation_id"] = [b["observation_id"] for b in batch]
    if "species" in batch[0]:
        result["species"] = [b["species"] for b in batch]
    return result


@pytest.fixture
def dummy_dataloader():
    return DataLoader(_DictDataset(20), batch_size=4, collate_fn=_test_collate)
