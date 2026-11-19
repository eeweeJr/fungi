"""Tests for fungai.data.dataset — transforms, MetadataPreprocessor, ObservationDataset."""

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


# --------------------------------------------------------------------------- #
# Observation dataset, transforms and sampler (babys-first-model)
# --------------------------------------------------------------------------- #

from fungai.data.dataset import ObservationDataset, backbone_normalization
from fungai.data.sampler import ObservationSampler, build_observation_sampler


@pytest.fixture
def obs_frame(tmp_path):
    rows = []
    counts = {101: 6, 102: 1, 103: 2, 104: 3, 105: 1, 106: 1}
    species = {101: "sp_a", 102: "sp_a", 103: "sp_b", 104: "sp_b", 105: "sp_b", 106: "sp_c"}
    for obs, n in counts.items():
        for i in range(n):
            path = tmp_path / f"{i}-{obs}.JPG"
            Image.new("RGB", (40, 30), color=(i * 40 % 255, obs % 255, 0)).save(path)
            rows.append({"observation_id": obs, "filename": path.name, "image_path": str(path),
                         "species": species[obs], "genus": "G", "revisited_site": False})
    return pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)


def _small_tf():
    return build_eval_transform(32)


def test_observation_dataset_eval_is_capped_and_deterministic(obs_frame):
    ds = ObservationDataset(obs_frame, _small_tf(), {"sp_a": 0, "sp_b": 1, "sp_c": 2},
                            genus_map={"G": 0}, max_photos=4, train=False)
    assert len(ds) == 6
    assert ds.n_truncated == 1
    a, b = ds[0], ds[0]
    assert a["n_images"] == 4 and a["image_paths"] == b["image_paths"]
    assert [p.split("/")[-1] for p in a["image_paths"]] == [f"{i}-101.JPG" for i in range(4)]
    assert a["genus_label"] == 0 and a["label"] == 0


def test_observation_dataset_train_draws_varying_subsets(obs_frame):
    ds = ObservationDataset(obs_frame, _small_tf(), {"sp_a": 0}, max_photos=4, train=True, seed=0)
    sizes = {len(ds.photo_rows(0)) for _ in range(60)}
    assert sizes == {1, 2, 3, 4}
    orders = {tuple(ds.photo_rows(2)) for _ in range(40)}
    assert len(orders) > 2  # 2 photos: subsets and orders vary
    for _ in range(20):
        rows = ds.photo_rows(0)
        assert len(set(rows)) == len(rows)  # no repeats


def test_collate_observations_full_contract(obs_frame):
    ds = ObservationDataset(obs_frame, _small_tf(), {"sp_a": 0, "sp_b": 1}, genus_map={"G": 0},
                            max_photos=4)
    batch = collate_observations([ds[0], ds[1], ds[2]])
    assert batch["images"].shape == (3, 4, 3, 32, 32)
    assert batch["image_mask"].sum(1).tolist() == [4, 1, 2]
    assert batch["n_images"].tolist() == [4, 1, 2]
    assert torch.all(batch["images"][1, 1:] == 0)  # padding is exactly zero
    assert batch["genus_label"].tolist() == [0, 0, 0]
    assert len(batch["image_paths"][0]) == 4
    assert batch["revisited_site"].dtype == torch.bool


def test_observation_sampler_matches_image_sampler_class_share(obs_frame):
    ds = ObservationDataset(obs_frame, _small_tf(), max_photos=4, train=True)
    for mode in ("observation", "sqrt", "balanced"):
        obs_level = build_observation_sampler(ds, mode, seed=0).expected_class_share()
        img_level = ObservationSampler(obs_frame["observation_id"].to_numpy(),
                                       obs_frame["species"].to_numpy(),
                                       alpha={"observation": 0, "sqrt": .5, "balanced": 1}[mode]
                                       ).expected_class_share()
        for k in img_level:
            assert obs_level[k] == pytest.approx(img_level[k])
    s = build_observation_sampler(ds, "sqrt", seed=0)
    idx = list(iter(s))
    assert len(idx) == len(ds) and all(0 <= i < len(ds) for i in idx)
    assert build_observation_sampler(ds, "none") is None


def test_brightness_contrast_aug_keeps_hue(sample_image):
    t = build_train_transform(64, crop_scale_min=0.5, color_aug="brightness_contrast")
    assert not any("saturation" in repr(s) and "hue=(-0.05" in repr(s) for s in t.transforms)
    jitter = [s for s in t.transforms if type(s).__name__ == "ColorJitter"][0]
    assert jitter.hue is None and jitter.saturation is None
    assert t(sample_image).shape == (3, 64, 64)
    with pytest.raises(ValueError):
        build_train_transform(64, color_aug="hue")


def test_backbone_normalization_from_timm():
    assert backbone_normalization("vit_base_patch16_224") == ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    mean, _ = backbone_normalization("vit_small_patch16_dinov3.lvd1689m")
    assert mean == pytest.approx((0.485, 0.456, 0.406))
    assert backbone_normalization("not_a_model")[0] == pytest.approx((0.485, 0.456, 0.406))
