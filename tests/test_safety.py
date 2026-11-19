"""Tests for fungai.safety — watchdog, estimator, preflight."""

import time
from unittest.mock import patch

import pytest
import torch
from torch import nn

from fungai.config import ExperimentConfig, SafetyConfig
from fungai.safety.estimator import (
    _format_duration,
    estimate_model_memory_mb,
    estimate_training_time,
)
from fungai.safety.watchdog import MemoryWatchdog


def test_watchdog_starts_and_stops():
    wd = MemoryWatchdog(limit_gb=64.0, poll_interval_s=0.1)
    wd.start()
    time.sleep(0.3)
    wd.stop()


def test_watchdog_tracks_peak():
    wd = MemoryWatchdog(limit_gb=64.0, poll_interval_s=0.1)
    wd.start()
    time.sleep(0.3)
    assert wd.peak_gb > 0
    wd.stop()


def test_watchdog_current_gb():
    wd = MemoryWatchdog(limit_gb=64.0)
    assert wd.current_gb > 0


def test_watchdog_counts_gpu_memory():
    """Regression: MPS memory is invisible to RSS, so the limit was not enforced for
    GPU training (10.8 GB on MPS read as 1.55 GB)."""
    wd = MemoryWatchdog(limit_gb=64.0)
    gpu = 5 * 1024**3
    with patch.object(MemoryWatchdog, "_get_gpu_bytes", return_value=gpu):
        rss, got = wd.sample()
        assert got == gpu
        assert wd.peak_gb == pytest.approx((rss + gpu) / 1024**3)
        assert wd.peak_gpu_gb == pytest.approx(5.0)
        assert wd.current_gb >= 5.0
        assert "GPU 5.00 GB" in wd.summary()


def test_watchdog_kills_when_gpu_pushes_total_over_limit():
    rss_only = MemoryWatchdog(limit_gb=64.0)._get_total_rss()
    wd = MemoryWatchdog(limit_gb=(rss_only + 1024**3) / 1024**3, poll_interval_s=0.05)
    with patch.object(MemoryWatchdog, "_get_gpu_bytes", return_value=4 * 1024**3), \
            patch("fungai.safety.watchdog.os.kill") as kill:
        wd.start()
        time.sleep(0.3)
        wd.stop()
    assert kill.called


def test_gpu_memory_bytes_is_safe_without_gpu():
    from fungai.safety.watchdog import gpu_memory_bytes
    assert gpu_memory_bytes() >= 0


def test_memory_estimation_keys():
    result = estimate_model_memory_mb(
        backbone="mobilenetv3_small_050", image_size=224, batch_size=2, num_classes=5
    )
    expected_keys = [
        "param_mb", "grad_mb", "optimizer_mb",
        "activation_mb_per_sample", "batch_activation_mb",
        "estimated_peak_mb", "estimated_peak_gb",
    ]
    for k in expected_keys:
        assert k in result, f"Missing key: {k}"


def test_memory_estimation_positive():
    result = estimate_model_memory_mb(
        backbone="mobilenetv3_small_050", image_size=224, batch_size=2, num_classes=5
    )
    assert result["param_mb"] > 0
    assert result["estimated_peak_gb"] > 0


def test_time_estimation_keys(tiny_model, dummy_dataloader):
    device = torch.device("cpu")
    model = tiny_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    result = estimate_training_time(model, dummy_dataloader, optimizer, criterion, device, 2)
    for k in ["batch_time_s", "batches_per_epoch", "epoch_time_s", "total_time_s",
              "total_time_human", "estimated_completion"]:
        assert k in result, f"Missing key: {k}"


def test_time_estimation_positive(tiny_model, dummy_dataloader):
    device = torch.device("cpu")
    model = tiny_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    result = estimate_training_time(model, dummy_dataloader, optimizer, criterion, device, 2)
    assert result["batch_time_s"] > 0
    assert result["total_time_s"] > 0


def test_preflight_prints_report(tiny_model, dummy_dataloader, capsys):
    device = torch.device("cpu")
    model = tiny_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    cfg = ExperimentConfig(
        safety=SafetyConfig(memory_limit_gb=64.0, memory_poll_interval_s=10.0),
    )
    cfg.model.backbone = "mobilenetv3_small_050"
    cfg.model.num_classes = 5
    from fungai.safety.preflight import run_preflight
    wd = run_preflight(cfg, model, dummy_dataloader, optimizer, criterion, device)
    wd.stop()
    out = capsys.readouterr().out
    assert "PRE-FLIGHT" in out
    assert "ESTIMATED PEAK" in out
    assert "Watchdog ACTIVE" in out


def test_preflight_aborts_on_tiny_limit(tiny_model, dummy_dataloader):
    device = torch.device("cpu")
    model = tiny_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    cfg = ExperimentConfig(
        safety=SafetyConfig(memory_limit_gb=0.001),
    )
    cfg.model.backbone = "mobilenetv3_small_050"
    cfg.model.num_classes = 5
    from fungai.safety.preflight import run_preflight
    with pytest.raises(SystemExit):
        run_preflight(cfg, model, dummy_dataloader, optimizer, criterion, device)


def test_format_duration():
    assert _format_duration(3661) == "1h 1m 1s"
    assert _format_duration(65) == "1m 5s"
    assert _format_duration(0) == "0m 0s"
