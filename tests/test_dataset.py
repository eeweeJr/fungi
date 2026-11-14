"""Tests for fungai.data.dataset — transforms, MetadataPreprocessor, ObservationDataset."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from fungai.data.dataset import (
    METADATA_FIELDS,
    MetadataPreprocessor,
    build_eval_transform,
    build_train_transform,
    collate_observations,
)


@pytest.fixture
def sample_image():
    return Image.new("RGB", (300, 300), color="red")


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "habitat": pd.Categorical(["forest", "meadow", "forest", "meadow"]),
        "substrate": pd.Categorical(["wood", "soil", "wood", "soil"]),
        "biogeo_zone": pd.Categorical(["atlantic", "continental", "atlantic", "continental"]),
        "land_cover": pd.Categorical(["deciduous", "grassland", "deciduous", "grassland"]),
        "toxicity": pd.Categorical(["edible", "poisonous", "edible", "poisonous"]),
        "latitude": [55.0, 56.0, 57.0, 58.0],
        "longitude": [10.0, 11.0, 12.0, 13.0],
        "elevation": [100.0, 200.0, 300.0, 400.0],
        "month": [6, 7, 8, 9],
    })


def test_train_transform_shape(sample_image):
    t = build_train_transform(224)
    out = t(sample_image)
    assert out.shape == (3, 224, 224)


def test_eval_transform_shape(sample_image):
    t = build_eval_transform(224)
    out = t(sample_image)
    assert out.shape == (3, 224, 224)


def test_train_transform_augments(sample_image):
    t = build_train_transform(224)
    torch.manual_seed(0)
    a = t(sample_image)
    torch.manual_seed(999)
    b = t(sample_image)
    assert not torch.equal(a, b)


def test_eval_transform_deterministic(sample_image):
    t = build_eval_transform(224)
    a = t(sample_image)
    b = t(sample_image)
    assert torch.equal(a, b)


def test_imagenet_normalization(sample_image):
    t = build_eval_transform(224)
    out = t(sample_image)
    assert out.min() < 0 or out.max() > 1.0


def test_metadata_preprocessor_fit_transform(sample_df):
    pp = MetadataPreprocessor.fit(sample_df)
    row = {
        "habitat": "forest", "substrate": "wood", "biogeo_zone": "atlantic",
        "land_cover": "deciduous", "toxicity": "edible",
        "latitude": 55.0, "longitude": 10.0, "elevation": 100.0, "month": 6,
    }
    tensor = pp.transform(row)
    assert tensor.shape == (len(METADATA_FIELDS),)
    assert tensor.dtype == torch.float32


def test_metadata_preprocessor_unknown_categorical(sample_df):
    pp = MetadataPreprocessor.fit(sample_df)
    row = {"habitat": "UNKNOWN_VALUE", "substrate": "wood"}
    tensor = pp.transform(row)
    habitat_idx = list(METADATA_FIELDS).index("habitat")
    assert tensor[habitat_idx].item() == 0.0


def test_metadata_preprocessor_missing_continuous(sample_df):
    pp = MetadataPreprocessor.fit(sample_df)
    row = {"latitude": None, "longitude": None}
    tensor = pp.transform(row)
    lat_idx = list(METADATA_FIELDS).index("latitude")
    assert tensor[lat_idx].item() == 0.0


def test_metadata_preprocessor_save_load(sample_df, tmp_path):
    pp = MetadataPreprocessor.fit(sample_df)
    path = tmp_path / "preprocessor.json"
    pp.save(path)
    pp2 = MetadataPreprocessor.load(path)
    row = {
        "habitat": "forest", "substrate": "wood", "latitude": 55.0,
        "longitude": 10.0, "elevation": 100.0, "month": 6,
    }
    t1 = pp.transform(row)
    t2 = pp2.transform(row)
    assert torch.equal(t1, t2)


def test_collate_observations():
    batch = [
        {
            "images": [torch.randn(3, 224, 224), torch.randn(3, 224, 224)],
            "label": 0,
            "species": "sp_A",
            "metadata": {},
            "observation_id": "obs_1",
            "n_images": 2,
        },
        {
            "images": [torch.randn(3, 224, 224)],
            "label": 1,
            "species": "sp_B",
            "metadata": {},
            "observation_id": "obs_2",
            "n_images": 1,
        },
    ]
    collated = collate_observations(batch)
    assert collated["images"].shape == (2, 2, 3, 224, 224)
    assert collated["image_mask"].shape == (2, 2)
    assert collated["image_mask"][0].all()
    assert collated["image_mask"][1, 0] and not collated["image_mask"][1, 1]
    assert collated["label"].shape == (2,)
