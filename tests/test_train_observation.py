"""Tests for the observation trainer: optimizer groups, schedule phases, accumulation,
one synthetic epoch, and the memory measurement."""

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from fungai.config import ExperimentConfig
from fungai.data.dataset import collate_observations
from fungai.model.backbone import TokenBackbone
from fungai.model.losses import observation_loss
from fungai.model.observation import ObservationClassifier
from fungai.model.train_observation import (
    build_observation_scheduler,
    build_optimizer,
    measure_step_memory,
    train_one_epoch,
)

SG = torch.tensor([0, 0, 1, 1, 2])
SIZE = 32


def tiny_model(seed=0, dropout=0.0):
    torch.manual_seed(seed)
    bb = TokenBackbone("vit_small_patch16_dinov3.lvd1689m", pretrained=False, image_size=SIZE)
    return ObservationClassifier(bb, SG, 3, fusion_dim=32, n_queries=4, fusion_blocks=1,
                                 fusion_heads=4, fusion_dropout=dropout, photo_dropout=dropout)


def cfg(**training):
    c = ExperimentConfig()
    c.data.image_size, c.data.batch_size, c.data.max_photos = SIZE, 2, 2
    c.training.epochs, c.training.freeze_epochs, c.training.warmup_epochs = 4, 1, 1
    c.training.lr, c.training.lr_backbone, c.training.layer_decay = 1e-3, 1e-4, 0.5
    c.training.weight_decay, c.training.accum_steps = 0.05, 1
    for k, v in training.items():
        setattr(c.training, k, v)
    return c


class SyntheticObs(Dataset):
    def __init__(self, n=8, photos=2, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.items = [{"images": list(torch.randn(photos, 3, SIZE, SIZE, generator=g)),
                       "label": i % 5, "genus_label": int(SG[i % 5]), "species": f"s{i % 5}",
                       "metadata": {}, "observation_id": str(i), "n_images": photos,
                       "image_paths": [f"/x/{j}-{i}.JPG" for j in range(photos)]} for i in range(n)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def test_optimizer_groups_follow_layer_decay_and_no_decay_rules():
    model, c = tiny_model(), cfg()
    opt = build_optimizer(model, c)
    by_name = {g["name"]: g for g in opt.param_groups}
    n = model.backbone.n_blocks
    assert by_name["new"]["lr"] == pytest.approx(1e-3)
    assert by_name[f"backbone_block_{n - 1}"]["lr"] == pytest.approx(1e-4)
    assert by_name[f"backbone_block_{n - 2}"]["lr"] == pytest.approx(5e-5)
    assert by_name["backbone_embed_no_decay"]["lr"] == pytest.approx(1e-4 * 0.5 ** n)
    assert all(g["weight_decay"] == 0 for g in opt.param_groups if g["name"].endswith("no_decay"))
    no_decay_ids = {id(p) for g in opt.param_groups if g["name"].endswith("no_decay") for p in g["params"]}
    assert id(model.fusion.queries) in no_decay_ids
    n_params = sum(len(g["params"]) for g in opt.param_groups)
    assert n_params == len(list(model.parameters()))


def test_schedule_holds_backbone_at_zero_while_frozen_then_ramps():
    model, c = tiny_model(), cfg()
    opt = build_optimizer(model, c)
    steps = 10
    sched = build_observation_scheduler(opt, c, steps)
    bb = [i for i, g in enumerate(opt.param_groups) if g["backbone"]]
    new = [i for i, g in enumerate(opt.param_groups) if not g["backbone"]]
    seen_bb, seen_new = [], []
    for i in range(4 * steps):
        seen_bb.append(opt.param_groups[bb[0]]["lr"])
        seen_new.append(opt.param_groups[new[0]]["lr"])
        if i == 2 * steps + 3:  # mid-run: backbone groups keep their layer-decay ratios
            for j in bb:
                g, top = opt.param_groups[j], opt.param_groups[bb[-1]]
                assert g["lr"] / top["lr"] == pytest.approx(g["base_lr"] / top["base_lr"])
        opt.step()
        sched.step()
    assert all(v == 0 for v in seen_bb[:steps])                  # frozen epoch
    assert seen_bb[steps] > 0 and seen_bb[steps] < seen_bb[2 * steps - 1]   # ramp
    assert max(seen_new) == pytest.approx(1e-3, rel=0.02)        # new layers reach full LR


def test_phase1_trains_no_backbone_parameter():
    model, c = tiny_model(), cfg()
    model.freeze_backbone(True)
    before = [p.detach().clone() for p in model.backbone.parameters()]
    q_before = model.fusion.queries.detach().clone()
    opt = build_optimizer(model, c)
    sched = build_observation_scheduler(opt, c, 4)
    loader = DataLoader(SyntheticObs(8), batch_size=2, collate_fn=collate_observations)
    losses, step = train_one_epoch(model, loader, opt, sched, c, torch.device("cpu"), 0)
    assert step == 4
    assert all(torch.isfinite(torch.tensor(v)) for v in losses.values())
    assert all(torch.equal(a, b) for a, b in zip(before, model.backbone.parameters()))
    assert not torch.equal(q_before, model.fusion.queries)


def test_accumulation_matches_one_large_batch():
    data = SyntheticObs(8)
    full = collate_observations([data[i] for i in range(8)])
    halves = [collate_observations([data[i] for i in range(j, j + 4)]) for j in (0, 4)]

    def grads(batches, accum):
        model = tiny_model(seed=1).train()
        for b in batches:
            out = model(b["images"], b["image_mask"])
            loss = observation_loss(out, b["label"], b["genus_label"], "hierarchical", "perceiver", 0.3)
            (loss["total"] / accum).backward()
        return [p.grad.clone() for p in model.parameters() if p.grad is not None]

    g_full, g_acc = grads([full], 1), grads(halves, 2)
    for a, b in zip(g_full, g_acc):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)


def test_accumulated_epoch_steps_once_per_group():
    model, c = tiny_model(), cfg(accum_steps=2)
    opt = build_optimizer(model, c)
    sched = build_observation_scheduler(opt, c, 2)
    loader = DataLoader(SyntheticObs(8), batch_size=2, collate_fn=collate_observations)
    _, step = train_one_epoch(model, loader, opt, sched, c, torch.device("cpu"), 0)
    assert step == 2


def test_measure_step_memory_on_cpu():
    model, c = tiny_model(), cfg()
    mem = measure_step_memory(model, c, torch.device("cpu"))
    assert mem["estimated_peak_gb"] > 0 and mem["measured_worst_case_photos"] == 4
    assert all(p.grad is None for p in model.parameters())
