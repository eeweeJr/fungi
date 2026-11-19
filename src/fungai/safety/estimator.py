"""Pre-flight memory and training-time estimation."""

from __future__ import annotations

import time
from datetime import datetime, timedelta

import psutil
import torch


def estimate_model_memory_mb(
    backbone: str,
    image_size: int,
    batch_size: int,
    num_classes: int,
) -> dict:
    """Estimate peak training memory without allocating a full training run.

    Creates a temporary model (pretrained=False for speed) on CPU, measures
    parameter bytes, runs one forward+backward on batch_size=1 to gauge
    activation memory, then extrapolates to the requested batch_size.
    """
    from fungai.model.classifier import FungiClassifier

    model = FungiClassifier(
        num_classes=num_classes, backbone=backbone, pretrained=False, dropout=0.0
    )
    model.train()

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    grad_bytes = param_bytes
    optimizer_bytes = param_bytes * 2  # AdamW stores m and v

    # Use batch_size=2 minimum to avoid batchnorm single-sample errors
    probe_bs = max(batch_size, 2)
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    x = torch.randn(probe_bs, 3, image_size, image_size)
    out = model(x)
    loss = out.sum()
    loss.backward()
    rss_after = proc.memory_info().rss

    activation_total = max(rss_after - rss_before - param_bytes - grad_bytes, 0)
    activation_per_sample = activation_total // probe_bs
    batch_activation = activation_per_sample * batch_size

    del model, x, out, loss
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    total = param_bytes + grad_bytes + optimizer_bytes + batch_activation

    return {
        "param_mb": param_bytes / 1024**2,
        "grad_mb": grad_bytes / 1024**2,
        "optimizer_mb": optimizer_bytes / 1024**2,
        "activation_mb_per_sample": activation_per_sample / 1024**2,
        "batch_activation_mb": batch_activation / 1024**2,
        "estimated_peak_mb": total / 1024**2,
        "estimated_peak_gb": total / 1024**3,
    }


def estimate_training_time(
    model: torch.nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    num_epochs: int,
    step_fn=None,
    steps_per_epoch: int | None = None,
) -> dict:
    """Time a single forward+backward on one batch, extrapolate to full run.

    step_fn(batch) -> loss replaces the default image-classifier step. With a
    step_fn the optimizer is not stepped, so timing does not change the weights.
    steps_per_epoch defaults to len(loader) (batches, not optimizer steps).
    """
    model.train()
    batch = next(iter(loader))
    if step_fn is not None:
        return _time_custom_step(model, batch, step_fn, device, num_epochs,
                                 steps_per_epoch or len(loader))
    images = batch["image"].to(device)
    labels = batch["label"].to(device)

    # Warmup (MPS compiles kernels on first call)
    optimizer.zero_grad()
    out = model(images)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()

    if device.type == "mps":
        torch.mps.synchronize()

    # Timed pass
    start = time.perf_counter()
    optimizer.zero_grad()
    out = model(images)
    loss = criterion(out, labels)
    loss.backward()
    optimizer.step()
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - start

    batches_per_epoch = len(loader)
    epoch_time = elapsed * batches_per_epoch
    total_time = epoch_time * num_epochs
    completion = datetime.now() + timedelta(seconds=total_time)

    return {
        "batch_time_s": round(elapsed, 3),
        "batches_per_epoch": batches_per_epoch,
        "epoch_time_s": round(epoch_time, 1),
        "total_time_s": round(total_time, 1),
        "total_time_human": _format_duration(total_time),
        "estimated_completion": completion.strftime("%Y-%m-%d %H:%M"),
    }


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _time_custom_step(model, batch, step_fn, device, num_epochs, batches_per_epoch) -> dict:
    for p in model.parameters():
        p.grad = None
    step_fn(batch).backward()  # warmup: MPS compiles kernels on first call
    _sync(device)
    for p in model.parameters():
        p.grad = None
    start = time.perf_counter()
    step_fn(batch).backward()
    _sync(device)
    elapsed = time.perf_counter() - start
    for p in model.parameters():
        p.grad = None
    epoch_time = elapsed * batches_per_epoch
    total_time = epoch_time * num_epochs
    return {
        "batch_time_s": round(elapsed, 3),
        "batches_per_epoch": batches_per_epoch,
        "epoch_time_s": round(epoch_time, 1),
        "total_time_s": round(total_time, 1),
        "total_time_human": _format_duration(total_time),
        "estimated_completion": (datetime.now() + timedelta(seconds=total_time)).strftime("%Y-%m-%d %H:%M"),
    }


def _format_duration(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"
