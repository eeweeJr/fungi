"""Full training loop with validation, scheduling, checkpointing, and MLflow tracking."""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from fungai.config import ExperimentConfig, config_to_flat_dict, load_config, seed_everything
from fungai.eval.cost_matrix import cost_sensitive_weights
from fungai.eval.report import evaluate_model
from fungai.safety.preflight import run_preflight
from fungai.tracking.mlflow_utils import (
    log_epoch_metrics,
    log_evaluation_report,
    log_model_checkpoint,
    start_tracked_run,
)

logger = logging.getLogger("fungai.train")

OUTPUTS_DIR = Path("outputs")


def _next_version(experiment_name: str) -> str:
    """Return the next unused version string like 'experiment_v0.3'."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    prefix = f"{experiment_name}_v0."
    pattern = re.compile(re.escape(prefix) + r"(\d+)$")
    used: set[int] = set()
    for d in OUTPUTS_DIR.iterdir():
        if d.is_dir():
            m = pattern.match(d.name)
            if m:
                used.add(int(m.group(1)))
    counter = 1
    while counter in used:
        counter += 1
    return f"{experiment_name}_v0.{counter}"


def _fungi_collate(batch: list[dict]) -> dict:
    """Custom collate that stacks tensors and collects strings as lists."""
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_norm: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    n_samples = 0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        mask = labels >= 0
        if mask.sum() == 0:
            continue
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits[mask], labels[mask])
        loss.backward()
        if grad_clip_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        total_loss += loss.item() * mask.sum().item()
        n_samples += mask.sum().item()
    return total_loss / max(n_samples, 1)


def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            mask = labels >= 0
            if mask.sum() == 0:
                continue
            logits = model(images)
            loss = criterion(logits[mask], labels[mask])
            total_loss += loss.item() * mask.sum().item()
            n_samples += mask.sum().item()
    return total_loss / max(n_samples, 1)


def build_scheduler(optimizer, config: ExperimentConfig):
    if config.training.scheduler == "cosine":
        warmup = LinearLR(
            optimizer, start_factor=0.01, total_iters=config.training.warmup_epochs
        )
        cosine = CosineAnnealingLR(
            optimizer, T_max=max(config.training.epochs - config.training.warmup_epochs, 1)
        )
        return SequentialLR(
            optimizer, [warmup, cosine], milestones=[config.training.warmup_epochs]
        )
    return None


def build_criterion(
    config: ExperimentConfig,
    species_map: dict[str, int],
    toxicity_map: dict[str, bool],
    device: torch.device,
) -> nn.Module:
    if config.training.cost_sensitive:
        weights = cost_sensitive_weights(species_map, toxicity_map).to(device)
        return nn.CrossEntropyLoss(
            weight=weights, label_smoothing=config.training.label_smoothing
        )
    return nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


class EarlyStopping:
    def __init__(self, patience: int, metric_name: str, mode: str = "max"):
        self.patience = patience
        self.metric_name = metric_name
        self.mode = mode
        self.best_value = float("-inf") if mode == "max" else float("inf")
        self.counter = 0
        self.best_epoch = 0

    def step(self, metrics: dict, epoch: int) -> bool:
        """Returns True if training should stop."""
        value = metrics.get(self.metric_name, 0)
        improved = (value > self.best_value) if self.mode == "max" else (value < self.best_value)
        if improved:
            self.best_value = value
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
        return self.counter >= self.patience


def _build_toxicity_map(species_map: dict[str, int], train_df) -> dict[str, bool]:
    """Build species→is_toxic mapping from the training metadata."""
    tox_map: dict[str, bool] = {}
    if "toxicity" in train_df.columns and "species" in train_df.columns:
        for sp in species_map:
            rows = train_df[train_df["species"] == sp]
            if len(rows) > 0:
                tox_val = rows["toxicity"].mode()
                tox_map[sp] = bool(len(tox_val) > 0 and str(tox_val.iloc[0]).lower() in ("poisonous", "toxic", "true"))
            else:
                tox_map[sp] = False
    else:
        tox_map = {sp: False for sp in species_map}
    return tox_map


def _build_species_to_genus(species_map: dict[str, int], train_df) -> dict[int, str]:
    """Build label_index→genus mapping."""
    s2g: dict[int, str] = {}
    if "genus" in train_df.columns and "species" in train_df.columns:
        sp_genus = train_df.drop_duplicates("species").set_index("species")["genus"].to_dict()
        for sp, idx in species_map.items():
            s2g[idx] = str(sp_genus.get(sp, "unknown"))
    return s2g


def _build_train_species_counts(train_df, species_map: dict[str, int]) -> dict[int, int]:
    """Count training images per species label."""
    counts: dict[int, int] = {}
    if "species" in train_df.columns:
        for sp, cnt in train_df["species"].value_counts().items():
            idx = species_map.get(sp)
            if idx is not None:
                counts[idx] = int(cnt)
    return counts


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ExperimentConfig,
    species_map: dict[str, int],
    toxicity_map: dict[str, bool],
    species_to_genus: dict[int, str],
    genus_names: list[str],
    train_species_counts: dict[int, int],
) -> None:
    """Full training loop with validation, checkpointing, MLflow tracking, and safety."""
    import mlflow

    seed_everything(config.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.lr, weight_decay=config.training.weight_decay
    )
    criterion = build_criterion(config, species_map, toxicity_map, device)
    scheduler = build_scheduler(optimizer, config)
    early_stopping = EarlyStopping(
        patience=config.training.early_stopping_patience,
        metric_name=config.training.early_stopping_metric,
    )

    watchdog = None
    if config.safety.preflight_enabled:
        watchdog = run_preflight(config, model, train_loader, optimizer, criterion, device)

    version_name = _next_version(config.experiment_name)
    run_dir = OUTPUTS_DIR / version_name
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / "best.pth"
    last_path = run_dir / "last.pth"

    import yaml as _yaml
    from fungai.config import config_to_flat_dict as _flat
    (run_dir / "config.yaml").write_text(
        _yaml.dump(_flat(config), default_flow_style=False)
    )

    print(f"Run: {version_name}  →  {run_dir}", flush=True)

    eval_config = {
        "aggregate_observations": config.evaluation.aggregate_observations,
        "failure_gallery_n": config.evaluation.failure_gallery_n,
    }

    run = start_tracked_run(config, run_name=version_name)

    try:
        for epoch in range(config.training.epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                grad_clip_norm=config.training.grad_clip_norm,
            )

            val_loss = validate_one_epoch(model, val_loader, criterion, device)

            eval_report = evaluate_model(
                model, val_loader, device, species_map, toxicity_map,
                species_to_genus, genus_names, train_species_counts, eval_config,
            )

            current_lr = optimizer.param_groups[0]["lr"]
            epoch_metrics = {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
                "val_macro_f1": eval_report["primary_metrics"]["macro_f1"],
                "val_top_1_accuracy": eval_report["primary_metrics"]["top_1_accuracy"],
                "val_top_5_accuracy": eval_report["primary_metrics"]["top_5_accuracy"],
                "val_ece": eval_report["calibration"]["ece"],
            }

            log_epoch_metrics(epoch, epoch_metrics)

            print(
                f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                f"macro_f1={eval_report['primary_metrics']['macro_f1']:.4f}  "
                f"top1={eval_report['primary_metrics']['top_1_accuracy']:.4f}",
                flush=True,
            )

            if scheduler:
                scheduler.step()

            save_checkpoint(model, optimizer, epoch, epoch_metrics, last_path)

            if early_stopping.counter == 0:
                save_checkpoint(model, optimizer, epoch, epoch_metrics, best_path)

            if early_stopping.step(epoch_metrics, epoch):
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best {early_stopping.metric_name}={early_stopping.best_value:.4f} "
                    f"at epoch {early_stopping.best_epoch}",
                    flush=True,
                )
                break

        if best_path.exists():
            checkpoint = torch.load(best_path, weights_only=True)
            model.load_state_dict(checkpoint["model_state_dict"])
            final_report = evaluate_model(
                model, val_loader, device, species_map, toxicity_map,
                species_to_genus, genus_names, train_species_counts, eval_config,
            )
            (run_dir / "eval_report.json").write_text(
                json.dumps(final_report, indent=2, default=str)
            )
            log_evaluation_report(final_report)
            log_model_checkpoint(best_path)
            log_model_checkpoint(last_path)

    finally:
        mlflow.end_run()
        if watchdog:
            print(f"\nPeak memory during training: {watchdog.peak_gb:.2f} GB", flush=True)
            watchdog.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FungAI classifier")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = load_config(args.config)
    seed_everything(config.seed)

    from fungai.data.dataset import (
        FungiDataset,
        build_eval_transform,
        build_train_transform,
        load_or_build_species_map,
    )
    from fungai.data.ingest import load_metadata
    from fungai.model.classifier import FungiClassifier

    species_map = load_or_build_species_map(config.data.subset)
    config.model.num_classes = len(species_map)

    train_df = load_metadata(config.data.subset, "train", config.data.size)
    val_df = load_metadata(config.data.subset, "val", config.data.size)

    toxicity_map = _build_toxicity_map(species_map, train_df)
    species_to_genus = _build_species_to_genus(species_map, train_df)
    genus_names = sorted(set(species_to_genus.values()))
    train_species_counts = _build_train_species_counts(train_df, species_map)

    train_transform = build_train_transform(config.data.image_size) if config.data.augmentation else build_eval_transform(config.data.image_size)
    eval_transform = build_eval_transform(config.data.image_size)

    train_ds = FungiDataset(train_df, transform=train_transform, label_map=species_map)
    val_ds = FungiDataset(val_df, transform=eval_transform, label_map=species_map)

    train_loader = DataLoader(
        train_ds, batch_size=config.data.batch_size, shuffle=True,
        num_workers=config.data.num_workers, pin_memory=config.data.pin_memory,
        collate_fn=_fungi_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.data.batch_size, shuffle=False,
        num_workers=config.data.num_workers, pin_memory=config.data.pin_memory,
        collate_fn=_fungi_collate,
    )

    model = FungiClassifier(
        num_classes=config.model.num_classes,
        backbone=config.model.backbone,
        pretrained=config.model.pretrained,
        dropout=config.model.dropout,
    )

    train(
        model, train_loader, val_loader, config, species_map, toxicity_map,
        species_to_genus, genus_names, train_species_counts,
    )


if __name__ == "__main__":
    main()
