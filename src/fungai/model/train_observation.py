"""Train the observation-level classifier.

    uv run python -m fungai.model.train_observation --config configs/babys_first_model.yaml

Per optimizer step: gradient accumulation over `training.accum_steps` batches,
global-norm clipping, one LR schedule step. Phase 1 (`freeze_epochs`) trains only
the new layers with the backbone frozen; phase 2 unfreezes the backbone with
layer-wise LR decay and a linear backbone warmup over `warmup_epochs`.

Every setting comes from the config. Outputs go to outputs/<experiment>_v0.N/ and
MLflow, and every run is watched by the memory watchdog (process + GPU memory).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from fungai.config import ExperimentConfig, load_config, seed_everything
from fungai.model.backbone import layer_decay_scale
from fungai.model.losses import observation_loss
from fungai.model.observation import ObservationClassifier, build_observation_model
from fungai.model.train import (
    OUTPUTS_DIR,
    EarlyStopping,
    _build_species_to_genus,
    _build_toxicity_map,
    _build_train_species_counts,
    _next_version,
    save_checkpoint,
    warmup_cosine_factor,
)

logger = logging.getLogger("fungai.train_observation")

NO_DECAY_NAMES = ("cls_token", "reg_token", "pos_embed", "queries", "mask_token")


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- optimizer

def _decay_split(named_params) -> tuple[list, list]:
    decay, no_decay = [], []
    for name, p in named_params:
        if p.ndim <= 1 or name.endswith(".bias") or any(k in name for k in NO_DECAY_NAMES):
            no_decay.append(p)
        else:
            decay.append(p)
    return decay, no_decay


def build_optimizer(model: ObservationClassifier, cfg: ExperimentConfig) -> torch.optim.AdamW:
    """AdamW with one group per (backbone layer | new layers) x (decay | no decay).

    New layers run at `training.lr`. Backbone layer groups run at
    `training.lr_backbone * layer_decay ** depth_below_top`. Biases, norms, class/
    register tokens and fusion queries get no weight decay. Each group records its
    `name`, `base_lr` and whether it is `backbone`.
    """
    t = cfg.training
    groups = []
    names = {id(p): n for n, p in model.named_parameters()}

    def add(name, params, lr, backbone):
        decay, no_decay = _decay_split([(names[id(p)], p) for p in params])
        for suffix, ps, wd in (("", decay, t.weight_decay), ("_no_decay", no_decay, 0.0)):
            if ps:
                groups.append({"params": ps, "lr": lr, "weight_decay": wd, "name": name + suffix,
                               "base_lr": lr, "backbone": backbone})

    add("new", [p for _, p in model.new_parameters()], t.lr, False)
    n_blocks = model.backbone.n_blocks
    for gname, params in model.backbone.layer_groups():
        add(f"backbone_{gname}", params, t.lr_backbone * layer_decay_scale(gname, n_blocks, t.layer_decay),
            True)
    return torch.optim.AdamW(groups)


def build_observation_scheduler(optimizer, cfg: ExperimentConfig, steps_per_epoch: int) -> LambdaLR:
    """One warmup + cosine factor over the whole run for every group. Backbone groups
    are additionally held at 0 while frozen, then ramp linearly from 0 over
    `warmup_epochs` once unfrozen."""
    t = cfg.training
    total = t.epochs * steps_per_epoch
    warmup = min(t.warmup_epochs * steps_per_epoch, max(total - 1, 0))
    unfreeze = t.freeze_epochs * steps_per_epoch
    ramp = max(t.warmup_epochs * steps_per_epoch, 1)

    def new_fn(step):
        return warmup_cosine_factor(step, total, warmup)

    def backbone_fn(step):
        if step < unfreeze:
            return 0.0
        return warmup_cosine_factor(step, total, warmup) * min((step - unfreeze + 1) / ramp, 1.0)

    return LambdaLR(optimizer, [backbone_fn if g["backbone"] else new_fn for g in optimizer.param_groups])


# --------------------------------------------------------------------------- data

def _limit_observations(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    ids = df["observation_id"].drop_duplicates()
    keep = ids.sample(min(n, len(ids)), random_state=seed)
    return df[df["observation_id"].isin(keep)].reset_index(drop=True)


def build_data(cfg: ExperimentConfig, limit: int | None = None):
    """Snapshot, split gate, datasets and loaders. Returns a dict of everything the
    loop and evaluation need. `limit` keeps only that many train and val
    observations, for a quick end-to-end check; it is not a result."""
    from fungai.data.dataset import (
        ObservationDataset,
        backbone_normalization,
        build_eval_transform,
        build_train_transform,
        collate_observations,
    )
    from fungai.data.observations import observation_summary
    from fungai.data.sampler import build_observation_sampler
    from fungai.data.snapshot import build_snapshot
    from fungai.data.split_gate import run_split_gate
    from fungai.model.hierarchy import build_species_genus_index

    d = cfg.data
    snapshot_dir, frames, protocol = build_snapshot(
        d.subset, d.protocol, d.split_seed, d.n_holdout_species, d.n_holdout_toxic, size=d.size,
    )
    if d.split_gate:
        run_split_gate(protocol, frames)
    species_map = protocol.species_map
    cfg.model.num_classes = len(species_map)
    train_df, val_df = frames["train"], frames["val"]
    if limit:
        train_df = _limit_observations(train_df, limit, cfg.seed)
        val_df = _limit_observations(val_df, limit, cfg.seed)
    for name, df in (("train", train_df), ("val", val_df)):
        if "image_path" not in df.columns:
            raise FileNotFoundError(f"no {d.size}p images found for {name}; run the ingest for size {d.size}")

    mean, std = backbone_normalization(cfg.model.backbone)
    train_tf = (build_train_transform(d.image_size, d.crop_scale_min, d.color_aug, mean, std)
                if d.augmentation else build_eval_transform(d.image_size, mean, std))
    eval_tf = build_eval_transform(d.image_size, mean, std)
    train_ds = ObservationDataset(train_df, train_tf, species_map, protocol.genus_map,
                                  max_photos=d.max_photos, train=True, seed=cfg.seed)
    val_ds = ObservationDataset(val_df, eval_tf, species_map, protocol.genus_map,
                                max_photos=d.max_photos, train=False)
    sampler = build_observation_sampler(train_ds, d.sampler, seed=cfg.seed)
    train_loader = DataLoader(train_ds, batch_size=d.batch_size, shuffle=sampler is None, sampler=sampler,
                              num_workers=d.num_workers, pin_memory=d.pin_memory,
                              collate_fn=collate_observations, drop_last=True)
    # Evaluation is forward-only, so it can take a larger batch; this has no effect on results.
    val_loader = DataLoader(val_ds, batch_size=d.batch_size * 2, shuffle=False,
                            num_workers=d.num_workers, pin_memory=d.pin_memory,
                            collate_fn=collate_observations)
    shape = {"train": observation_summary(train_df, d.max_photos),
             "val": observation_summary(val_df, d.max_photos)}
    return {
        "snapshot_dir": snapshot_dir, "protocol": protocol, "frames": frames,
        "species_map": species_map, "train_df": train_df, "val_df": val_df,
        "train_ds": train_ds, "val_ds": val_ds, "train_loader": train_loader, "val_loader": val_loader,
        "species_genus": build_species_genus_index(species_map, protocol.species_to_genus,
                                                   protocol.genus_map),
        "genus_names": protocol.genus_names, "mean": mean, "std": std, "eval_tf": eval_tf,
        "observation_shape": shape,
    }


def build_probe_batch(cfg: ExperimentConfig, data: dict, presence: pd.DataFrame | None) -> dict:
    """Fixed val observations for the per-epoch probes: half single-photo, half
    multi-photo, preferring observations whose photos all have masks."""
    from fungai.data.dataset import ObservationDataset, collate_observations
    from fungai.data.observations import select_eval_photos

    val = select_eval_photos(data["val_df"], cfg.data.max_photos)
    val = val[val["species"].isin(data["species_map"])]  # known species only
    if presence is not None:
        has = set(presence["filename"])
        full = val.groupby("observation_id")["filename"].agg(lambda f: all(x in has for x in f))
        if full.sum() >= cfg.diagnostics.probe_observations:
            val = val[val["observation_id"].isin(full[full].index)]
    counts = val.groupby("observation_id").size()
    rng = np.random.default_rng(cfg.diagnostics.probe_seed)
    n = cfg.diagnostics.probe_observations
    multi = counts[counts >= 2].index.to_numpy()
    single = counts[counts == 1].index.to_numpy()
    n_multi = min(len(multi), n // 2)
    chosen = np.concatenate([rng.choice(multi, n_multi, replace=False),
                             rng.choice(single, min(len(single), n - n_multi), replace=False)])
    # multi-photo first so the attention figures show fusion across photos
    sub = val[val["observation_id"].isin(chosen)]
    ds = ObservationDataset(sub, data["eval_tf"], data["species_map"], data["protocol"].genus_map,
                            max_photos=cfg.data.max_photos, train=False)
    items = sorted((ds[i] for i in range(len(ds))), key=lambda it: -it["n_images"])
    batch = collate_observations(items)
    batch["species"] = [it["species"] for it in items]
    return batch


def build_part_grids(probe: dict, cfg: ExperimentConfig, patch_size: int) -> dict:
    from fungai.data.masks import load_part_masks, mask_to_patch_grid

    names = [p.rsplit("/", 1)[-1] for paths in probe["image_paths"] for p in paths]
    masks = load_part_masks("val", names)
    return {fn: {part: mask_to_patch_grid(m, cfg.data.image_size, patch_size) for part, m in parts.items()}
            for fn, parts in masks.items()}


# --------------------------------------------------------------------------- memory

def measure_step_memory(model: ObservationClassifier, cfg: ExperimentConfig, device: torch.device) -> dict:
    """Measure one worst-case training step (batch_size observations x max_photos
    photos, backbone unfrozen) on the real device. Adds AdamW's two moment buffers,
    which only exist after the first optimizer step. Returns a dict in the shape
    `run_preflight` prints."""
    import psutil

    from fungai.safety.watchdog import gpu_memory_bytes

    b, p, s = cfg.data.batch_size, cfg.data.max_photos, cfg.data.image_size
    frozen = model.backbone_frozen
    model.freeze_backbone(False)
    model.train()
    param_bytes = sum(q.numel() * q.element_size() for q in model.parameters())
    rss0 = psutil.Process().memory_info().rss
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    images = torch.randn(b, p, 3, s, s, device=device)
    mask = torch.ones(b, p, dtype=torch.bool, device=device)
    label = torch.zeros(b, dtype=torch.long, device=device)
    out = model(images, mask)
    observation_loss(out, label, label, model.head_kind, model.fusion_kind,
                     cfg.model.photo_loss_weight)["total"].backward()
    if device.type == "mps":
        torch.mps.synchronize()
        gpu_peak = gpu_memory_bytes()
    elif device.type == "cuda":
        gpu_peak = torch.cuda.max_memory_allocated()
    else:
        gpu_peak = 0
    rss = psutil.Process().memory_info().rss
    for q in model.parameters():
        q.grad = None
    del images, mask, out
    if device.type == "mps":
        torch.mps.empty_cache()
    model.freeze_backbone(frozen)
    optimizer_bytes = 2 * param_bytes
    measured = gpu_peak if device.type != "cpu" else max(rss - rss0, 0) + 2 * param_bytes
    total = measured + optimizer_bytes
    return {
        "param_mb": param_bytes / 1024**2, "grad_mb": param_bytes / 1024**2,
        "optimizer_mb": optimizer_bytes / 1024**2,
        "batch_activation_mb": max(measured - 2 * param_bytes, 0) / 1024**2,
        "estimated_peak_mb": total / 1024**2, "estimated_peak_gb": total / 1024**3,
        "measured_worst_case_photos": b * p,
    }


# --------------------------------------------------------------------------- loop

def _batch_to(batch: dict, device) -> tuple:
    return (batch["images"].to(device, non_blocking=True), batch["image_mask"].to(device),
            batch["label"].to(device), batch["genus_label"].to(device))


def train_one_epoch(model, loader, optimizer, scheduler, cfg, device, step: int,
                    diagnostics=None, monitor=None) -> tuple[dict, int]:
    """One epoch with gradient accumulation. Returns (mean loss terms, global step).
    `monitor` (a RunMonitor) receives each batch's total loss for the live running loss."""
    t = cfg.training
    model.train()
    accum = t.accum_steps
    amp = t.amp and device.type == "cuda"
    trainable = [p for p in model.parameters() if p.requires_grad]
    sums: dict[str, float] = {}
    n = 0
    n_batches = len(loader)
    optimizer.zero_grad(set_to_none=True)
    for i, batch in enumerate(loader):
        images, mask, label, genus_label = _batch_to(batch, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            out = model(images, mask)
            parts = observation_loss(out, label, genus_label, model.head_kind, model.fusion_kind,
                                     cfg.model.photo_loss_weight)
        (parts["total"] / accum).backward()
        b = len(label)
        values = {k: float(v.detach()) for k, v in parts.items()}
        for k, v in values.items():
            sums[k] = sums.get(k, 0.0) + v * b
        n += b
        if monitor is not None:
            monitor.record_loss(values["total"])
        if (i + 1) % accum and (i + 1) != n_batches:
            continue
        log_now = diagnostics is not None and diagnostics.is_log_step(step)
        grad_norms = diagnostics.grad_norms() if log_now else {}
        pre_clip = float(nn.utils.clip_grad_norm_(
            trainable, t.grad_clip_norm if t.grad_clip_norm > 0 else float("inf")))
        if not math.isfinite(pre_clip):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        if diagnostics is not None:
            diagnostics.before_optimizer_step(step)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        if diagnostics is not None:
            diagnostics.after_optimizer_step(step, out, parts, optimizer, pre_clip, t.grad_clip_norm,
                                             grad_norms)
        step += 1
    return {k: v / max(n, 1) for k, v in sums.items()}, step


def evaluate(model, data: dict, cfg: ExperimentConfig, device, toxicity_map, species_to_genus,
             train_species_counts, presence=None, full: bool = False):
    """Observation-level val evaluation. full=True adds the observation sections
    (calibration, decision, intervals); otherwise only the per-epoch numbers."""
    from fungai.eval.metrics import macro_f1, top_k_accuracy
    from fungai.eval.observation_report import (
        build_observation_sections,
        collect_observation_predictions,
    )
    from fungai.eval.report import build_evaluation_report

    pred = collect_observation_predictions(model, data["val_loader"], device, cfg.model.photo_loss_weight)
    if not full:
        y = pred["labels"]
        return {
            "val_loss": pred["loss"]["total"],
            "val_macro_f1": macro_f1(y, pred["probs"].argmax(1)),
            "val_top_1_accuracy": top_k_accuracy(y, pred["probs"], 1),
            "val_photo_avg_macro_f1": macro_f1(y, pred["photo_avg_probs"].argmax(1)),
            "val_genus_head_accuracy": float((pred["genus_probs_head"].argmax(1)
                                              == pred["genus_labels"]).mean()),
        }, pred
    e = cfg.evaluation
    eval_config = {"aggregate_observations": e.aggregate_observations,
                   "failure_gallery_n": e.failure_gallery_n}
    report = build_evaluation_report(pred, data["species_map"], toxicity_map, species_to_genus,
                                     data["genus_names"], train_species_counts, eval_config)
    head = "flat" if (model.head_kind == "flat" or model.fusion_kind == "none") else "hierarchical"
    sections, fitted = build_observation_sections(
        pred, data["species_genus"].numpy(), data["genus_names"], species_to_genus, head, e, presence)
    report.update(sections)
    report["observation_shape"] = data["observation_shape"]
    return report, pred, fitted


def train(cfg: ExperimentConfig, limit: int | None = None) -> Path:
    import mlflow
    import yaml

    from fungai.config import config_to_flat_dict
    from fungai.data.masks import load_part_presence
    from fungai.diagnostics import Diagnostics
    from fungai.eval.observation_report import observation_predictions_table
    from fungai.eval.report import predictions_table
    from fungai.safety.preflight import run_preflight
    from fungai.tracking.mlflow_utils import (
        log_data_snapshot,
        log_epoch_metrics,
        log_evaluation_report,
        start_tracked_run,
    )
    from fungai.tracking.monitor import RunMonitor

    seed_everything(cfg.seed)
    device = pick_device()
    version = _next_version(cfg.experiment_name)
    run_dir = OUTPUTS_DIR / version
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.dump(config_to_flat_dict(cfg), default_flow_style=False))
    best_path, last_path = run_dir / "best.pth", run_dir / "last.pth"
    print(f"Run: {version}  ->  {run_dir}", flush=True)

    # The run starts before anything that can fail, so data, preflight and training
    # errors all end up in MLflow (see fungai.tracking.monitor).
    start_tracked_run(cfg, run_name=version)
    mlflow.set_tag("arch", "observation")
    mlflow.set_tag("device", device.type)
    if limit:
        mlflow.set_tag("limit_observations", str(limit))
        mlflow.set_tag("note", "pipeline check on a subset; not a result")

    watchdog = None
    try:
        with RunMonitor(run_dir, cfg.training.epochs, memory_limit_gb=cfg.safety.memory_limit_gb) as mon:
            mon.phase("loading data")
            data = build_data(cfg, limit)
            print(f"Data snapshot: {data['snapshot_dir'].name}  |  "
                  f"train obs {len(data['train_ds'])}  val obs {len(data['val_ds'])} "
                  f"(val truncated by cap: {data['val_ds'].n_truncated})", flush=True)
            log_data_snapshot(data["snapshot_dir"])
            (run_dir / "observation_shape.json").write_text(json.dumps(data["observation_shape"], indent=1))
            mlflow.log_artifact(str(run_dir / "observation_shape.json"), artifact_path="data")

            mon.phase("building model")
            model = build_observation_model(cfg, data["species_genus"], len(data["genus_names"])).to(device)
            steps_per_epoch = math.ceil(len(data["train_loader"]) / cfg.training.accum_steps)
            optimizer = build_optimizer(model, cfg)
            scheduler = build_observation_scheduler(optimizer, cfg, steps_per_epoch)

            toxicity_map = _build_toxicity_map(data["species_map"], data["train_df"])
            species_to_genus = _build_species_to_genus(data["species_map"], data["train_df"])
            train_species_counts = _build_train_species_counts(data["train_df"], data["species_map"])
            try:
                presence = load_part_presence("val")
            except FileNotFoundError:
                presence = None

            if cfg.safety.preflight_enabled:
                mon.phase("preflight")
                mem = measure_step_memory(model, cfg, device)

                def step_fn(batch):
                    images, mask, label, genus_label = _batch_to(batch, device)
                    out = model(images, mask)
                    return observation_loss(out, label, genus_label, model.head_kind, model.fusion_kind,
                                            cfg.model.photo_loss_weight)["total"]

                watchdog = run_preflight(cfg, model, data["train_loader"], optimizer, None, device,
                                         memory_estimate=mem, step_fn=step_fn,
                                         steps_per_epoch=len(data["train_loader"]))
                mon.attach_watchdog(watchdog)

            if cfg.training.freeze_epochs > 0:
                model.freeze_backbone(True)

            diagnostics = None
            if cfg.diagnostics.enabled:
                mon.phase("diagnostics setup")
                probe = build_probe_batch(cfg, data, presence)
                grids = build_part_grids(probe, cfg, model.backbone.patch_size) if (
                    cfg.diagnostics.part_attention and presence is not None) else {}
                diagnostics = Diagnostics(model, cfg.diagnostics, probe, run_dir, device, data["mean"],
                                          data["std"], part_grids=grids)
                diagnostics.capture_initial()

            early = EarlyStopping(cfg.training.early_stopping_patience, cfg.training.early_stopping_metric)
            step = 0
            for epoch in range(cfg.training.epochs):
                mon.start_epoch(epoch)
                if epoch == cfg.training.freeze_epochs and model.backbone_frozen:
                    model.freeze_backbone(False)
                    print(f"Epoch {epoch}: backbone unfrozen", flush=True)
                t0 = time.time()
                train_loss, step = train_one_epoch(model, mon.track(data["train_loader"], "train"),
                                                   optimizer, scheduler, cfg, device, step, diagnostics,
                                                   monitor=mon)
                val, _ = evaluate(model, {**data, "val_loader": mon.track(data["val_loader"], "val")},
                                  cfg, device, toxicity_map, species_to_genus, train_species_counts)
                metrics = {f"train_{k}": v for k, v in train_loss.items()} | val | {
                    "epoch_time_s": time.time() - t0, "backbone_frozen": float(model.backbone_frozen)}
                if watchdog is not None:
                    metrics["peak_memory_gb"] = watchdog.peak_gb
                log_epoch_metrics(epoch, metrics)
                if diagnostics is not None:
                    diagnostics.on_epoch_end(epoch, step)
                print(f"Epoch {epoch}: train_loss={train_loss['total']:.4f} val_loss={val['val_loss']:.4f} "
                      f"macro_f1={val['val_macro_f1']:.4f} (photo-avg {val['val_photo_avg_macro_f1']:.4f}) "
                      f"top1={val['val_top_1_accuracy']:.4f} genus={val['val_genus_head_accuracy']:.4f} "
                      f"[{metrics['epoch_time_s'] / 60:.1f} min]", flush=True)
                mon.phase(f"epoch {epoch + 1}/{cfg.training.epochs} · checkpointing")
                save_checkpoint(model, optimizer, epoch, metrics, last_path)
                stop = early.step(metrics, epoch)
                if early.improved:
                    save_checkpoint(model, optimizer, epoch, metrics, best_path)
                mon.end_epoch(metrics)
                if stop:
                    print(f"Early stopping at epoch {epoch}; best {early.metric_name}="
                          f"{early.best_value:.4f} at epoch {early.best_epoch}", flush=True)
                    break

            mon.phase("final evaluation (best checkpoint)")
            ckpt = torch.load(best_path, weights_only=True, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            report, pred, fitted = evaluate(
                model, {**data, "val_loader": mon.track(data["val_loader"], "final eval")}, cfg, device,
                toxicity_map, species_to_genus, train_species_counts, presence, full=True)
            report["best_epoch"] = early.best_epoch
            if diagnostics is not None:
                report["health_warnings"] = diagnostics.warnings
            mon.phase("logging results")
            (run_dir / "eval_report.json").write_text(json.dumps(report, indent=2, default=str))
            (run_dir / "calibration.json").write_text(json.dumps(fitted["calibration"], indent=2))
            log_evaluation_report(report)
            mlflow.log_artifact(str(run_dir / "calibration.json"), artifact_path="evaluation")
            mlflow.log_metrics({k: float(v) for k, v in {
                "final_val_macro_f1": report["primary_metrics"]["macro_f1"],
                "final_val_top_1_accuracy": report["primary_metrics"]["top_1_accuracy"],
                "final_val_photo_avg_macro_f1": report["fusion"]["photo_average"]["macro_f1"],
                "final_val_ece_after_temperature": report["calibration_fit"]["ece_after"],
                "best_epoch": early.best_epoch,
            }.items() if v is not None})
            table, probs = predictions_table(pred, data["species_map"], species_to_genus, data["genus_names"])
            inv = {v: k for k, v in data["species_map"].items()}
            table = observation_predictions_table(table, pred, fitted["per_observation"],
                                                  data["genus_names"], inv)
            table.to_parquet(run_dir / "val_predictions.parquet", index=False)
            np.save(run_dir / "val_probs.npy", probs)
            mlflow.log_artifact(str(run_dir / "val_predictions.parquet"), artifact_path="predictions")
            if diagnostics is not None:
                mlflow.log_artifacts(str(run_dir / "diagnostics"), artifact_path="diagnostics")
            _print_summary(report)
    finally:
        mlflow.end_run()  # no-op when the monitor already ended the run with its status
        if watchdog is not None:
            print(f"\nPeak memory during training: {watchdog.summary()}", flush=True)
            watchdog.stop()
    return run_dir


def _fmt_ci(ci: dict | None) -> str:
    if not ci or ci.get("value") is None:
        return "n/a"
    return f"{ci['value']:.4f} [{ci['lo']:.4f}, {ci['hi']:.4f}]"


def _print_summary(report: dict) -> None:
    iv = report["intervals"]
    print("\n=== Validation summary (best epoch, observation level) ===", flush=True)
    print(f"  macro F1 (fused):        {_fmt_ci(iv.get('macro_f1'))}")
    print(f"  macro F1 (photo-avg):    {_fmt_ci(iv.get('photo_average_macro_f1'))}")
    print(f"  fused - photo-avg:       {_fmt_ci(iv.get('fused_minus_photo_average_macro_f1'))}")
    print(f"  top-1:                   {_fmt_ci(iv.get('top_1_accuracy'))}")
    print(f"  held-out genus accuracy: {_fmt_ci(iv.get('held_out_genus_accuracy'))}")
    cal = report["calibration_fit"]
    print(f"  ECE before/after temp:   {cal['ece_before']:.4f} / {cal['ece_after']:.4f}  "
          f"(t_genus={cal['t_genus']:.2f}, t_species={cal['t_species']:.2f})")
    for group, row in report["decision"]["outcomes"].items():
        print(f"  decision ({group}, n={row['n']}): " + ", ".join(
            f"{k}={v:.3f}" for k, v in row.items() if k != "n" and v is not None))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the observation-level FungAI classifier")
    parser.add_argument("--config", type=str, default="configs/babys_first_model.yaml")
    parser.add_argument("--limit", type=int, default=None,
                        help="keep only this many train and val observations (pipeline check, not a result)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    if cfg.model.arch != "observation":
        raise SystemExit(f"{args.config} has model.arch={cfg.model.arch!r}; use fungai.model.train")
    train(cfg, args.limit)


if __name__ == "__main__":
    main()
