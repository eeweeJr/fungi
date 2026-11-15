"""Full training loop with validation, scheduling, checkpointing, and MLflow tracking."""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from fungai.config import ExperimentConfig, load_config, seed_everything
from fungai.eval.cost_matrix import cost_sensitive_weights
from fungai.eval.report import evaluate_model, predictions_table
from fungai.safety.preflight import run_preflight
from fungai.tracking.mlflow_utils import (
    log_data_snapshot,
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
    if "genus_label" in batch[0]:
        result["genus_label"] = torch.tensor([b["genus_label"] for b in batch], dtype=torch.long)
    if "revisited_site" in batch[0]:
        result["revisited_site"] = torch.tensor([b["revisited_site"] for b in batch], dtype=torch.bool)
    return result


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    grad_clip_norm: float = 0.0,
    scheduler=None,
) -> float:
    """One pass over the loader. The scheduler, if given, steps after every optimizer step."""
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
        if scheduler is not None:
            scheduler.step()
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


def warmup_cosine_factor(step: int, total_steps: int, warmup_steps: int,
                         start_factor: float = 0.01) -> float:
    """LR multiplier at an optimizer step: linear warmup from `start_factor` to 1 over
    `warmup_steps`, then cosine decay to 0 at `total_steps`."""
    if warmup_steps > 0 and step < warmup_steps:
        return start_factor + (1.0 - start_factor) * step / warmup_steps
    decay_steps = max(total_steps - warmup_steps, 1)
    progress = min(max(step - warmup_steps, 0) / decay_steps, 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def build_scheduler(optimizer, config: ExperimentConfig, steps_per_epoch: int = 1):
    """Per-optimizer-step schedule. Warmup lasts `warmup_epochs` epochs measured in
    steps, so a short run still reaches full LR inside its first epoch.

    One multiplicative factor is applied to every parameter group, so the ratios
    between groups (e.g. layer-wise LR decay) are preserved."""
    if config.training.scheduler != "cosine":
        return None
    steps_per_epoch = max(int(steps_per_epoch), 1)
    total = config.training.epochs * steps_per_epoch
    warmup = min(config.training.warmup_epochs * steps_per_epoch, max(total - 1, 0))
    return LambdaLR(optimizer, lambda step: warmup_cosine_factor(step, total, warmup))


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
        self.improved = False  # whether the most recent step set a new best

    def step(self, metrics: dict, epoch: int) -> bool:
        """Record this epoch's metric. Returns True if training should stop.

        Check `improved` after calling this to decide whether to save a best checkpoint.
        """
        value = metrics.get(self.metric_name, 0)
        improved = (value > self.best_value) if self.mode == "max" else (value < self.best_value)
        self.improved = bool(improved)
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
                tox_map[sp] = bool(len(tox_val) > 0 and str(tox_val.iloc[0]).lower() in ("1", "1.0", "poisonous", "toxic", "true"))
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


def _push_best_checkpoint(run_dir: Path, config: ExperimentConfig, early_stopping: EarlyStopping,
                          data_snapshot: Path | None) -> None:
    """Upload best.pth to the Hub and tag the MLflow run with where it went. A failed
    upload never fails the run: the checkpoint stays local and can be pushed later."""
    import mlflow

    from fungai.tracking.hub import build_run_info, log_checkpoint_location, push_best_checkpoint
    from fungai.tracking.mlflow_utils import get_git_hash

    run_info = build_run_info(
        run_dir.name, mlflow.active_run().info.run_id, config.experiment_name, get_git_hash(),
        Path(data_snapshot).name if data_snapshot else None,
        best_epoch=early_stopping.best_epoch,
        best_metric={early_stopping.metric_name: early_stopping.best_value},
        backbone=config.model.backbone, num_classes=config.model.num_classes,
    )
    try:
        record = push_best_checkpoint(run_dir, config.hub.repo_id, run_info)
    except Exception as exc:  # noqa: BLE001 - network, auth, quota: report and carry on
        logger.warning("checkpoint upload failed (%s); push it later with: "
                       "python -m fungai.tracking.hub push %s", exc, run_dir)
        mlflow.set_tag("checkpoint.status", "upload failed")
        return
    log_checkpoint_location(record)
    print(f"Best checkpoint pushed: {record['url']}", flush=True)


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
    data_snapshot: Path | None = None,
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
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=len(train_loader))
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

    start_tracked_run(config, run_name=version_name)
    if data_snapshot is not None:
        log_data_snapshot(data_snapshot)
        (run_dir / "data_snapshot.json").write_text(
            json.dumps({"snapshot": Path(data_snapshot).name}, indent=2)
        )

    try:
        for epoch in range(config.training.epochs):
            train_loss = train_one_epoch(
                model, train_loader, optimizer, criterion, device,
                grad_clip_norm=config.training.grad_clip_norm, scheduler=scheduler,
            )

            val_loss = validate_one_epoch(model, val_loader, criterion, device)

            eval_report = evaluate_model(
                model, val_loader, device, species_map, toxicity_map,
                species_to_genus, genus_names, train_species_counts, eval_config,
            )

            current_lr = optimizer.param_groups[0]["lr"]  # at the end of the epoch
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

            save_checkpoint(model, optimizer, epoch, epoch_metrics, last_path)

            should_stop = early_stopping.step(epoch_metrics, epoch)
            if early_stopping.improved:
                save_checkpoint(model, optimizer, epoch, epoch_metrics, best_path)

            if should_stop:
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
            final_report, predictions = evaluate_model(
                model, val_loader, device, species_map, toxicity_map,
                species_to_genus, genus_names, train_species_counts, eval_config,
                return_predictions=True,
            )
            (run_dir / "eval_report.json").write_text(
                json.dumps(final_report, indent=2, default=str)
            )
            log_evaluation_report(final_report)
            table, obs_probs = predictions_table(
                predictions, species_map, species_to_genus, genus_names
            )
            table.to_parquet(run_dir / "val_predictions.parquet", index=False)
            np.save(run_dir / "val_probs.npy", obs_probs)
            mlflow.log_artifact(str(run_dir / "val_predictions.parquet"), artifact_path="predictions")
            mlflow.log_artifact(str(run_dir / "val_probs.npy"), artifact_path="predictions")
            if config.hub.push_best_checkpoint:
                _push_best_checkpoint(run_dir, config, early_stopping, data_snapshot)
            else:
                log_model_checkpoint(best_path)
                log_model_checkpoint(last_path)

    finally:
        mlflow.end_run()
        if watchdog:
            print(f"\nPeak memory during training: {watchdog.summary()}", flush=True)
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
        backbone_normalization,
        build_eval_transform,
        build_train_transform,
    )
    from fungai.data.observations import select_eval_photos
    from fungai.data.sampler import build_train_sampler
    from fungai.data.snapshot import build_snapshot
    from fungai.model.classifier import FungiClassifier

    snapshot_dir, frames, protocol = build_snapshot(
        config.data.subset, config.data.protocol, config.data.split_seed,
        config.data.n_holdout_species, config.data.n_holdout_toxic, size=config.data.size,
    )
    print(f"Data snapshot: {snapshot_dir.name}", flush=True)
    species_map = protocol.species_map
    config.model.num_classes = len(species_map)

    train_df, val_df = frames["train"], frames["val"]
    # Score on the same photos the observation model sees (first max_photos per
    # observation), so per-image and observation runs are directly comparable.
    val_df = select_eval_photos(val_df, config.data.max_photos)

    toxicity_map = _build_toxicity_map(species_map, train_df)
    species_to_genus = _build_species_to_genus(species_map, train_df)
    genus_names = protocol.genus_names
    train_species_counts = _build_train_species_counts(train_df, species_map)

    mean, std = backbone_normalization(config.model.backbone)
    d = config.data
    train_transform = (build_train_transform(d.image_size, d.crop_scale_min, d.color_aug, mean, std)
                       if d.augmentation else build_eval_transform(d.image_size, mean, std))
    eval_transform = build_eval_transform(d.image_size, mean, std)

    train_ds = FungiDataset(train_df, transform=train_transform, label_map=species_map,
                            genus_map=protocol.genus_map)
    val_ds = FungiDataset(val_df, transform=eval_transform, label_map=species_map,
                          genus_map=protocol.genus_map)

    sampler = build_train_sampler(train_df, config.data.sampler, seed=config.seed)
    train_loader = DataLoader(
        train_ds, batch_size=config.data.batch_size, shuffle=sampler is None, sampler=sampler,
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
        species_to_genus, genus_names, train_species_counts, data_snapshot=snapshot_dir,
    )


if __name__ == "__main__":
    main()
