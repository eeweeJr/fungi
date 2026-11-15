"""Typed experiment configuration with YAML loading and reproducibility seeding."""

from __future__ import annotations

import random
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


@dataclass
class DataConfig:
    dataset: str = "fungitastic"
    subset: str = "mini"
    size: str = "300"
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    augmentation: bool = True
    # open: hold some species out of train as unknowns; closed: the 210-class benchmark
    protocol: str = "open"
    split_seed: int = 42  # fixes the held-out species; independent of the run seed
    n_holdout_species: int = 30
    n_holdout_toxic: int = 2
    # none: shuffle images; observation | sqrt | balanced: draw by observation
    sampler: str = "sqrt"


@dataclass
class ModelConfig:
    backbone: str = "vit_base_patch16_224"
    pretrained: bool = True
    dropout: float = 0.1
    num_classes: int = 210


@dataclass
class TrainingConfig:
    lr: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 10
    warmup_epochs: int = 1
    scheduler: str = "cosine"
    grad_clip_norm: float = 1.0
    early_stopping_patience: int = 5
    early_stopping_metric: str = "val_macro_f1"
    cost_sensitive: bool = False
    label_smoothing: float = 0.0


@dataclass
class EvalConfig:
    primary_metric: str = "macro_f1"
    top_k: list[int] = field(default_factory=lambda: [1, 5])
    ece_n_bins: int = 15
    coverage_steps: int = 20
    failure_gallery_n: int = 50
    aggregate_observations: bool = True


@dataclass
class SafetyConfig:
    memory_limit_gb: float = 16.0
    memory_poll_interval_s: float = 2.0
    preflight_enabled: bool = True


@dataclass
class ExperimentConfig:
    seed: int = 42
    experiment_name: str = "fungai-dev"
    run_name: str | None = None
    notes: str = ""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)


_SECTION_CLASSES = {
    "data": DataConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "evaluation": EvalConfig,
    "safety": SafetyConfig,
}


def load_config(path: str | Path) -> ExperimentConfig:
    """Load YAML config, merge with defaults, return typed ExperimentConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}

    top_kwargs: dict[str, Any] = {}
    for f in fields(ExperimentConfig):
        if f.name in _SECTION_CLASSES:
            section_raw = raw.get(f.name, {})
            if not isinstance(section_raw, dict):
                raise ValueError(f"Config section '{f.name}' must be a mapping, got {type(section_raw)}")
            top_kwargs[f.name] = _SECTION_CLASSES[f.name](**section_raw)
        elif f.name in raw:
            top_kwargs[f.name] = raw[f.name]

    cfg = ExperimentConfig(**top_kwargs)
    _validate(cfg)
    return cfg


def _validate(cfg: ExperimentConfig) -> None:
    if cfg.data.batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {cfg.data.batch_size}")
    if cfg.data.image_size < 1:
        raise ValueError(f"image_size must be >= 1, got {cfg.data.image_size}")
    if cfg.data.protocol not in ("open", "closed"):
        raise ValueError(f"protocol must be 'open' or 'closed', got {cfg.data.protocol}")
    if cfg.data.sampler not in ("none", "observation", "sqrt", "balanced"):
        raise ValueError(
            f"sampler must be none, observation, sqrt or balanced, got {cfg.data.sampler}"
        )
    if cfg.training.epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {cfg.training.epochs}")
    if cfg.training.lr <= 0:
        raise ValueError(f"lr must be > 0, got {cfg.training.lr}")
    if cfg.safety.memory_limit_gb <= 0:
        raise ValueError(f"memory_limit_gb must be > 0, got {cfg.safety.memory_limit_gb}")
    if cfg.training.scheduler not in ("cosine", "none"):
        raise ValueError(f"scheduler must be 'cosine' or 'none', got {cfg.training.scheduler}")


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs for reproducibility.

    MPS (Apple Silicon) does not support deterministic mode as of PyTorch 2.4.
    Results on MPS are NOT bitwise reproducible across runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def config_to_flat_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Flatten nested dataclass to dot-separated keys for MLflow param logging."""
    result: dict[str, Any] = {}
    for f in fields(cfg):
        val = getattr(cfg, f.name)
        if hasattr(val, "__dataclass_fields__"):
            for sf in fields(val):
                result[f"{f.name}.{sf.name}"] = getattr(val, sf.name)
        else:
            result[f.name] = val
    return result
