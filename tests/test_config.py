"""Tests for fungai.config — YAML loading, validation, seeding."""

import pytest
import torch
import yaml

from fungai.config import ExperimentConfig, config_to_flat_dict, load_config, seed_everything


def test_load_default_config():
    cfg = load_config("configs/default.yaml")
    assert cfg.model.backbone == "vit_base_patch16_224"
    assert cfg.training.epochs == 3
    assert cfg.safety.memory_limit_gb == 16.0
    assert isinstance(cfg, ExperimentConfig)


def test_config_override(tmp_path):
    custom = {
        "seed": 99,
        "model": {"backbone": "resnet18", "num_classes": 10},
        "training": {"lr": 0.001, "epochs": 5},
    }
    p = tmp_path / "custom.yaml"
    p.write_text(yaml.dump(custom))
    cfg = load_config(p)
    assert cfg.seed == 99
    assert cfg.model.backbone == "resnet18"
    assert cfg.model.num_classes == 10
    assert cfg.training.lr == 0.001
    assert cfg.training.epochs == 5
    # defaults still filled for unspecified fields
    assert cfg.data.batch_size == 32


def test_config_validation_rejects_bad_batch_size(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump({"data": {"batch_size": 0}}))
    with pytest.raises(ValueError, match="batch_size"):
        load_config(p)


def test_config_to_flat_dict():
    cfg = ExperimentConfig()
    flat = config_to_flat_dict(cfg)
    assert "model.backbone" in flat
    assert "training.lr" in flat
    assert "safety.memory_limit_gb" in flat
    assert "seed" in flat
    assert isinstance(flat["model.backbone"], str)


def test_seed_everything():
    seed_everything(123)
    a = torch.randn(5)
    seed_everything(123)
    b = torch.randn(5)
    assert torch.equal(a, b)


def test_data_protocol_defaults_and_validation(tmp_path):
    cfg = load_config("configs/default.yaml")
    assert cfg.data.protocol == "open"
    assert cfg.data.sampler == "sqrt"
    for bad in ({"protocol": "half-open"}, {"sampler": "weighted"}):
        p = tmp_path / "bad.yaml"
        p.write_text(yaml.dump({"data": bad}))
        with pytest.raises(ValueError):
            load_config(p)


# --------------------------------------------------------------------------- #
# babys-first-model config files
# --------------------------------------------------------------------------- #

from dataclasses import fields as _fields

from fungai.config import _SECTION_CLASSES

CANONICAL = "configs/babys_first_model.yaml"
VARIANT_ALLOWED_DIFFS = {
    "configs/babys_first_model_small.yaml": {
        "notes", "model.backbone", "data.batch_size", "training.accum_steps",
    },
    "configs/babys_first_model_smoke.yaml": {
        "notes", "model.backbone", "data.size", "data.image_size", "data.batch_size",
        "training.accum_steps", "training.epochs", "training.freeze_epochs",
    },
    "configs/babys_first_model_large.yaml": {
        "notes", "model.backbone", "model.drop_path", "data.batch_size", "training.accum_steps",
        "training.lr_backbone", "training.layer_decay", "safety.memory_limit_gb",
    },
}
ALL_BFM = [CANONICAL, *VARIANT_ALLOWED_DIFFS]


def _flat_yaml(path: str) -> dict:
    raw = yaml.safe_load(open(path))
    out = {}
    for k, v in raw.items():
        if isinstance(v, dict):
            out.update({f"{k}.{sk}": sv for sk, sv in v.items()})
        else:
            out[k] = v
    return out


@pytest.mark.parametrize("path", ALL_BFM)
def test_bfm_config_loads_and_validates(path):
    cfg = load_config(path)
    assert cfg.model.arch == "observation"
    assert cfg.experiment_name == "fungai-dev"


@pytest.mark.parametrize("path", ALL_BFM)
def test_bfm_config_sets_every_field(path):
    """No silent defaults: reading the file tells you the whole experiment."""
    raw = yaml.safe_load(open(path))
    top_scalars = {f.name for f in _fields(ExperimentConfig)} - set(_SECTION_CLASSES)
    assert top_scalars <= set(raw), f"missing top-level: {top_scalars - set(raw)}"
    for section, cls in _SECTION_CLASSES.items():
        assert section in raw, f"missing section {section}"
        expected = {f.name for f in _fields(cls)}
        assert set(raw[section]) == expected, (
            f"{section}: missing {expected - set(raw[section])}, extra {set(raw[section]) - expected}"
        )


@pytest.mark.parametrize("path", list(VARIANT_ALLOWED_DIFFS))
def test_bfm_variant_differs_only_in_allowed_keys(path):
    base, var = _flat_yaml(CANONICAL), _flat_yaml(path)
    assert set(base) == set(var)
    differing = {k for k in base if base[k] != var[k]}
    assert differing <= VARIANT_ALLOWED_DIFFS[path], f"not allowed: {differing - VARIANT_ALLOWED_DIFFS[path]}"


def test_bfm_effective_batch_is_16_observations():
    for path in ALL_BFM:
        cfg = load_config(path)
        assert cfg.data.batch_size * cfg.training.accum_steps == 16, path


@pytest.mark.parametrize("bad", [
    {"bogus_top": 1},
    {"model": {"bogus": 1}},
    {"diagnostics": {"bogus": 1}},
])
def test_unknown_key_fails(tmp_path, bad):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump(bad))
    with pytest.raises(ValueError, match="Unknown"):
        load_config(p)


@pytest.mark.parametrize("bad", [
    {"model": {"arch": "video"}},
    {"model": {"head": "tree"}},
    {"model": {"fusion_dim": 500, "fusion_heads": 8}},
    {"data": {"color_aug": "hue"}},
    {"data": {"max_photos": 0}},
    {"training": {"accum_steps": 0}},
    {"training": {"freeze_epochs": 5, "epochs": 5}},
    {"training": {"layer_decay": 1.5}},
])
def test_bfm_validation_rejects(tmp_path, bad):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.dump(bad))
    with pytest.raises(ValueError):
        load_config(p)


def test_default_config_keeps_original_behaviour():
    cfg = load_config("configs/default.yaml")
    assert cfg.model.arch == "image"
    assert cfg.data.crop_scale_min == 0.8 and cfg.data.color_aug == "full"
    assert cfg.diagnostics.enabled is False
    assert "diagnostics.enabled" in config_to_flat_dict(cfg)
