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
    # Augmentation. Defaults reproduce the original per-image transform.
    crop_scale_min: float = 0.8   # RandomResizedCrop lower bound
    color_aug: str = "full"       # full | brightness_contrast | none
    # Observation-level model: most photos used per observation, train and eval.
    max_photos: int = 4
    # Abort an observation run if a split invariant fails (fungai.data.split_gate).
    split_gate: bool = True


@dataclass
class ModelConfig:
    # image: per-image FungiClassifier | observation: ObservationClassifier
    arch: str = "image"
    backbone: str = "vit_base_patch16_224"
    pretrained: bool = True
    drop_path: float = 0.0
    grad_checkpointing: bool = False
    dropout: float = 0.1          # per-image classifier, and the observation model's photo head
    num_classes: int = 210        # overwritten at runtime from the split protocol
    # Observation model only.
    fusion: str = "perceiver"     # perceiver | none (photo head only)
    fusion_dim: int = 512
    n_queries: int = 16
    fusion_blocks: int = 2
    fusion_heads: int = 8
    fusion_dropout: float = 0.1
    head: str = "hierarchical"    # hierarchical | flat
    photo_loss_weight: float = 0.3


@dataclass
class TrainingConfig:
    lr: float = 1e-4              # observation model: LR of the new (non-backbone) layers
    lr_backbone: float = 2e-5     # observation model: LR of the top backbone block
    layer_decay: float = 0.75     # LR multiplier per backbone block below the top
    freeze_epochs: int = 0        # epochs with the backbone frozen
    accum_steps: int = 1          # gradient accumulation; effective batch = batch_size * this
    amp: bool = False             # autocast, CUDA only
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
    temperature_scaling: bool = True
    known_retention: float = 0.95     # sets the species and genus decision thresholds on val
    bootstrap_samples: int = 1000
    bootstrap_seed: int = 0
    # Frozen-feature backbone scaling probe (fungai.baselines.scaling_probe).
    probe_backbones: list[str] = field(default_factory=lambda: [
        "vit_small_patch16_dinov3.lvd1689m",
        "vit_base_patch16_dinov3.lvd1689m",
        "vit_large_patch16_dinov3.lvd1689m",
    ])
    probe_epochs: int = 50
    probe_weight_decays: list[float] = field(default_factory=lambda: [0.0, 1e-4, 1e-2])


@dataclass
class DiagnosticsConfig:
    enabled: bool = False
    every_n_steps: int = 50           # cheap scalars: losses, gradient norms, update ratios
    probe_observations: int = 32      # fixed val observations for the per-epoch heavy probes
    probe_seed: int = 0
    backbone_tap_every: int = 4       # tap every k-th backbone block
    part_attention: bool = True       # attention-on-part measurements from val masks
    save_attention_figures: bool = True
    n_figure_observations: int = 8
    alarm_query_cosine: float = 0.95
    alarm_entropy_low: float = 0.05
    alarm_entropy_high: float = 0.98
    alarm_padding_mass: float = 1e-6
    invariance_tolerance: float = 1e-4


@dataclass
class SafetyConfig:
    memory_limit_gb: float = 16.0
    memory_poll_interval_s: float = 2.0
    preflight_enabled: bool = True


@dataclass
class HubConfig:
    # upload best.pth (plus run metadata) to a private Hugging Face repo after training
    push_best_checkpoint: bool = False
    repo_id: str | None = None  # e.g. "your-username/fungai-checkpoints"


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
    hub: HubConfig = field(default_factory=HubConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


_SECTION_CLASSES = {
    "data": DataConfig,
    "model": ModelConfig,
    "training": TrainingConfig,
    "evaluation": EvalConfig,
    "safety": SafetyConfig,
    "hub": HubConfig,
    "diagnostics": DiagnosticsConfig,
}


def load_config(path: str | Path) -> ExperimentConfig:
    """Load YAML config, merge with defaults, return typed ExperimentConfig."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}

    known_top = {f.name for f in fields(ExperimentConfig)}
    unknown = set(raw) - known_top
    if unknown:
        raise ValueError(f"Unknown config key(s) in {path}: {sorted(unknown)}")

    top_kwargs: dict[str, Any] = {}
    for f in fields(ExperimentConfig):
        if f.name in _SECTION_CLASSES:
            section_raw = raw.get(f.name, {}) or {}
            if not isinstance(section_raw, dict):
                raise ValueError(f"Config section '{f.name}' must be a mapping, got {type(section_raw)}")
            cls = _SECTION_CLASSES[f.name]
            unknown = set(section_raw) - {sf.name for sf in fields(cls)}
            if unknown:
                raise ValueError(f"Unknown key(s) in config section '{f.name}': {sorted(unknown)}")
            top_kwargs[f.name] = cls(**section_raw)
        elif f.name in raw:
            top_kwargs[f.name] = raw[f.name]

    cfg = ExperimentConfig(**top_kwargs)
    _validate(cfg)
    cfg.source_path = str(path)  # not a config field: tagged on the MLflow run as `config`
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
    if cfg.hub.push_best_checkpoint and not cfg.hub.repo_id:
        raise ValueError("hub.push_best_checkpoint needs hub.repo_id (e.g. 'username/fungai-checkpoints')")
    if cfg.training.epochs < 1:
        raise ValueError(f"epochs must be >= 1, got {cfg.training.epochs}")
    if cfg.training.lr <= 0:
        raise ValueError(f"lr must be > 0, got {cfg.training.lr}")
    if cfg.safety.memory_limit_gb <= 0:
        raise ValueError(f"memory_limit_gb must be > 0, got {cfg.safety.memory_limit_gb}")
    if cfg.training.scheduler not in ("cosine", "none"):
        raise ValueError(f"scheduler must be 'cosine' or 'none', got {cfg.training.scheduler}")

    choices = {
        "data.color_aug": (cfg.data.color_aug, ("full", "brightness_contrast", "none")),
        "model.arch": (cfg.model.arch, ("image", "observation")),
        "model.fusion": (cfg.model.fusion, ("perceiver", "none")),
        "model.head": (cfg.model.head, ("hierarchical", "flat")),
    }
    for name, (value, allowed) in choices.items():
        if value not in allowed:
            raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
    if not 0 < cfg.data.crop_scale_min <= 1:
        raise ValueError(f"crop_scale_min must be in (0, 1], got {cfg.data.crop_scale_min}")
    if cfg.data.max_photos < 1:
        raise ValueError(f"max_photos must be >= 1, got {cfg.data.max_photos}")
    if cfg.training.accum_steps < 1:
        raise ValueError(f"accum_steps must be >= 1, got {cfg.training.accum_steps}")
    if not 0 < cfg.training.layer_decay <= 1:
        raise ValueError(f"layer_decay must be in (0, 1], got {cfg.training.layer_decay}")
    if cfg.training.lr_backbone <= 0:
        raise ValueError(f"lr_backbone must be > 0, got {cfg.training.lr_backbone}")
    if not 0 <= cfg.training.freeze_epochs < cfg.training.epochs:
        raise ValueError(
            f"freeze_epochs must be in [0, epochs), got {cfg.training.freeze_epochs} "
            f"with epochs={cfg.training.epochs}"
        )
    if cfg.model.photo_loss_weight < 0:
        raise ValueError(f"photo_loss_weight must be >= 0, got {cfg.model.photo_loss_weight}")
    if cfg.model.fusion_dim % cfg.model.fusion_heads:
        raise ValueError(
            f"fusion_dim ({cfg.model.fusion_dim}) must be divisible by fusion_heads "
            f"({cfg.model.fusion_heads})"
        )
    if cfg.model.n_queries < 1 or cfg.model.fusion_blocks < 1:
        raise ValueError("n_queries and fusion_blocks must be >= 1")
    if cfg.model.arch == "observation" and cfg.model.fusion == "none" and cfg.model.photo_loss_weight == 0:
        raise ValueError("fusion: none trains only the photo head, so photo_loss_weight must be > 0")
    if not 0 < cfg.evaluation.known_retention < 1:
        raise ValueError(f"known_retention must be in (0, 1), got {cfg.evaluation.known_retention}")
    if cfg.diagnostics.every_n_steps < 1 or cfg.diagnostics.backbone_tap_every < 1:
        raise ValueError("diagnostics.every_n_steps and backbone_tap_every must be >= 1")


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
