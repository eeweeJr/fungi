"""Pre-flight check that prints a terminal report before every training cycle.

Estimates memory usage, estimates training time, and starts the watchdog.
Aborts with sys.exit(1) if estimated peak memory exceeds the budget.
"""

from __future__ import annotations

import sys

import psutil
import torch

from fungai.config import ExperimentConfig
from fungai.safety.estimator import estimate_model_memory_mb, estimate_training_time
from fungai.safety.watchdog import MemoryWatchdog


def run_preflight(
    config: ExperimentConfig,
    model: torch.nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
    memory_estimate: dict | None = None,
    step_fn=None,
    steps_per_epoch: int | None = None,
) -> MemoryWatchdog:
    """Run all pre-flight checks, print report, start watchdog.

    Returns the active MemoryWatchdog (caller must .stop() it after training).
    Calls sys.exit(1) if estimated memory exceeds the configured limit.

    memory_estimate: a dict shaped like `estimate_model_memory_mb`'s result, for
    models that function cannot build (it measures a FungiClassifier). step_fn:
    see `estimate_training_time`.
    """
    _p = lambda *a, **kw: print(*a, **kw, flush=True)
    _p("\n" + "=" * 70)
    _p("  FUNGAI PRE-FLIGHT CHECK")
    _p("=" * 70)

    # --- Config Summary ---
    _p("\n--- Config Summary ---")
    _p(f"  Backbone:    {config.model.backbone}")
    _p(f"  Image size:  {config.data.image_size}")
    _p(f"  Batch size:  {config.data.batch_size}")
    _p(f"  Epochs:      {config.training.epochs}")
    _p(f"  LR:          {config.training.lr}")
    _p(f"  Scheduler:   {config.training.scheduler}")
    _p(f"  Device:      {device}")
    _p(f"  Seed:        {config.seed}")

    # --- Dataset Stats ---
    _p("\n--- Dataset Stats ---")
    _p(f"  Train batches:  {len(train_loader)}")
    _p(f"  Train samples:  {len(train_loader.dataset)}")
    _p(f"  Num classes:    {config.model.num_classes}")

    # --- Memory Estimation ---
    _p("\n--- Memory Estimation ---")
    mem = memory_estimate or estimate_model_memory_mb(
        config.model.backbone,
        config.data.image_size,
        config.data.batch_size,
        config.model.num_classes,
    )
    _p(f"  Parameters:      {mem['param_mb']:>8.1f} MB")
    _p(f"  Gradients:       {mem['grad_mb']:>8.1f} MB")
    _p(f"  Optimizer state: {mem['optimizer_mb']:>8.1f} MB")
    _p(f"  Activations:     {mem['batch_activation_mb']:>8.1f} MB (batch)")
    _p(f"  ESTIMATED PEAK:  {mem['estimated_peak_gb']:>8.2f} GB")

    current_rss = psutil.Process().memory_info().rss / 1024**3
    _p(f"  Current RSS:     {current_rss:>8.2f} GB")
    _p(f"  Memory limit:    {config.safety.memory_limit_gb:>8.2f} GB")

    headroom = config.safety.memory_limit_gb - mem["estimated_peak_gb"] - current_rss
    if headroom < 0:
        _p(
            f"\n  *** ABORT: estimated peak ({mem['estimated_peak_gb']:.2f} GB) "
            f"+ current RSS ({current_rss:.2f} GB) "
            f"exceeds {config.safety.memory_limit_gb:.1f} GB limit ***"
        )
        _p("  Suggestion: reduce batch_size or image_size")
        sys.exit(1)
    else:
        _p(f"  Headroom:        {headroom:>8.2f} GB  [OK]")

    # --- Time Estimation ---
    _p("\n--- Time Estimation ---")
    time_est = estimate_training_time(
        model, train_loader, optimizer, criterion, device, config.training.epochs,
        step_fn=step_fn, steps_per_epoch=steps_per_epoch,
    )
    _p(f"  Batch time:      {time_est['batch_time_s']:>8.3f} s")
    _p(f"  Epoch time:      {time_est['epoch_time_s']:>8.1f} s  ({time_est['epoch_time_s']/60:.1f} min)")
    _p(f"  Total estimated: {time_est['total_time_human']}")
    _p(f"  ETA:             {time_est['estimated_completion']}")

    # --- Watchdog ---
    _p("\n--- Memory Watchdog ---")
    watchdog = MemoryWatchdog(
        limit_gb=config.safety.memory_limit_gb,
        poll_interval_s=config.safety.memory_poll_interval_s,
    )
    watchdog.start()
    _p(
        f"  Watchdog ACTIVE (limit={config.safety.memory_limit_gb}GB, "
        f"poll={config.safety.memory_poll_interval_s}s, counts process + GPU memory)"
    )

    _p("\n" + "=" * 70)
    _p("  PRE-FLIGHT COMPLETE — starting training")
    _p("=" * 70 + "\n")

    return watchdog
