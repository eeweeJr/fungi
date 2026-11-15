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
