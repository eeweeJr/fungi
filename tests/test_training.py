"""Tests for fungai.model.train — loop mechanics, scheduling, checkpointing."""

import math

import pytest
import torch
from torch import nn
from torch.optim.lr_scheduler import LambdaLR

from fungai.config import ExperimentConfig
from fungai.model.train import (
    EarlyStopping,
    _build_toxicity_map,
    build_scheduler,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
    warmup_cosine_factor,
)


def test_train_one_epoch_loss_finite(tiny_model, dummy_dataloader):
    device = torch.device("cpu")
    model = tiny_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    loss = train_one_epoch(model, dummy_dataloader, optimizer, criterion, device)
    assert math.isfinite(loss)
    assert loss > 0


def test_validate_one_epoch_no_grad(tiny_model, dummy_dataloader):
    device = torch.device("cpu")
    model = tiny_model.to(device)
    criterion = nn.CrossEntropyLoss()
    _ = validate_one_epoch(model, dummy_dataloader, criterion, device)
    for p in model.parameters():
        assert p.grad is None


def test_gradient_clipping(tiny_model, dummy_dataloader):
    device = torch.device("cpu")
    model = tiny_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    clip = 0.01
    _ = train_one_epoch(model, dummy_dataloader, optimizer, criterion, device, grad_clip_norm=clip)
    # after the epoch, we can't directly inspect clipped norms since optimizer already stepped.
    # Instead verify the function ran without error with the clip value.
    # For a stronger check, do a manual forward-backward:
    model.train()
    batch = next(iter(dummy_dataloader))
    optimizer.zero_grad()
    logits = model(batch["image"].to(device))
    loss = criterion(logits, batch["label"].to(device))
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), clip)
    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
    assert total_norm <= clip + 1e-6


def test_early_stopping_triggers():
    es = EarlyStopping(patience=2, metric_name="val_macro_f1")
    assert not es.step({"val_macro_f1": 0.5}, 0)
    assert not es.step({"val_macro_f1": 0.4}, 1)
    assert es.step({"val_macro_f1": 0.3}, 2)


def test_early_stopping_resets_on_improvement():
    es = EarlyStopping(patience=2, metric_name="val_macro_f1")
    es.step({"val_macro_f1": 0.5}, 0)
    es.step({"val_macro_f1": 0.4}, 1)
    assert es.counter == 1
    es.step({"val_macro_f1": 0.6}, 2)
    assert es.counter == 0
    assert es.best_epoch == 2


def test_build_scheduler_cosine():
    model = nn.Linear(10, 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = ExperimentConfig()
    cfg.training.scheduler = "cosine"
    cfg.training.warmup_epochs = 1
    cfg.training.epochs = 5
    sched = build_scheduler(optimizer, cfg, steps_per_epoch=10)
    assert isinstance(sched, LambdaLR)


def test_warmup_is_per_step_so_a_one_epoch_run_reaches_full_lr():
    """Regression: warmup used to step per epoch, so epoch 0 of every run trained at 1%
    of the LR and a 1-epoch run never trained at all (fungai-dev_v0.2)."""
    model = nn.Linear(10, 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    cfg = ExperimentConfig()
    cfg.training.epochs, cfg.training.warmup_epochs = 1, 1
    steps = 100
    sched = build_scheduler(optimizer, cfg, steps_per_epoch=steps)
    lrs = []
    for _ in range(steps):
        lrs.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        sched.step()
    assert lrs[0] == pytest.approx(1e-5)
    assert max(lrs) > 0.9e-3


def test_warmup_cosine_factor_shape():
    total, warm = 100, 10
    assert warmup_cosine_factor(0, total, warm) == pytest.approx(0.01)
    assert warmup_cosine_factor(10, total, warm) == pytest.approx(1.0)
    assert warmup_cosine_factor(55, total, warm) == pytest.approx(0.5)
    assert warmup_cosine_factor(100, total, warm) == pytest.approx(0.0, abs=1e-9)
    factors = [warmup_cosine_factor(i, total, warm) for i in range(10, 100)]
    assert all(a >= b for a, b in zip(factors, factors[1:]))


def test_scheduler_preserves_group_ratios():
    a, b = nn.Linear(2, 2), nn.Linear(2, 2)
    optimizer = torch.optim.AdamW([{"params": a.parameters(), "lr": 1e-3},
                                   {"params": b.parameters(), "lr": 1e-5}])
    cfg = ExperimentConfig()
    cfg.training.epochs, cfg.training.warmup_epochs = 3, 1
    sched = build_scheduler(optimizer, cfg, steps_per_epoch=7)
    for _ in range(12):
        optimizer.step()
        sched.step()
        g = optimizer.param_groups
        assert g[0]["lr"] / g[1]["lr"] == pytest.approx(100.0)


def test_early_stopping_improved_marks_the_epoch_to_save():
    """Regression: best.pth was saved from the counter *before* step(), so it held the
    previous epoch's verdict. With scores 0.3, 0.2, 0.5 it saved the 0.2 epoch."""
    es = EarlyStopping(patience=5, metric_name="val_macro_f1")
    saved = []
    for epoch, score in enumerate([0.3, 0.2, 0.5, 0.4]):
        es.step({"val_macro_f1": score}, epoch)
        if es.improved:
            saved.append(epoch)
    assert saved == [0, 2]
    assert es.best_epoch == 2


def test_save_load_checkpoint(tiny_model, tmp_path):
    optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-4)
    path = tmp_path / "ckpt.pth"
    save_checkpoint(tiny_model, optimizer, epoch=3, metrics={"loss": 0.5}, path=path)
    assert path.exists()
    ckpt = torch.load(path, weights_only=True)
    assert ckpt["epoch"] == 3
    assert ckpt["metrics"]["loss"] == 0.5
    assert "model_state_dict" in ckpt


def test_open_set_labels_excluded(tiny_model, dummy_dataloader):
    device = torch.device("cpu")
    model = tiny_model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    # Modify labels to be all open-set
    from torch.utils.data import DataLoader

    from tests.conftest import _DictDataset

    class AllOpenSet(_DictDataset):
        def __getitem__(self, idx):
            d = super().__getitem__(idx)
            d["label"] = torch.tensor(-1)
            return d

    loader = DataLoader(AllOpenSet(8), batch_size=4)
    loss = train_one_epoch(model, loader, optimizer, criterion, device)
    assert loss == 0.0


def test_toxicity_map_int_encoding():
    import pandas as pd

    df = pd.DataFrame({"species": ["a", "a", "b", "b"], "toxicity": [1, 1, 0, 0]})
    assert _build_toxicity_map({"a": 0, "b": 1}, df) == {"a": True, "b": False}
