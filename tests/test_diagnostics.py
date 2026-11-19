"""Tests for the diagnostics: statistic bounds, alarms, and a full probe pass."""

import json

import numpy as np
import pytest
import torch

from fungai.config import DiagnosticsConfig
from fungai.data.dataset import collate_observations
from fungai.diagnostics import Diagnostics
from fungai.diagnostics.attention import attention_stats, part_attention
from fungai.diagnostics.health import NonFiniteError, check_finite, health_warnings
from fungai.diagnostics.stats import effective_rank, mean_pairwise_cosine, normalised_entropy
from tests.test_train_observation import SyntheticObs, tiny_model


def test_normalised_entropy_bounds():
    uniform = torch.full((1, 10), 0.1)
    one_hot = torch.zeros(1, 10)
    one_hot[0, 3] = 1
    assert normalised_entropy(uniform, torch.tensor([10]))[0] == pytest.approx(1.0, abs=1e-5)
    assert normalised_entropy(one_hot, torch.tensor([10]))[0] == pytest.approx(0.0, abs=1e-5)


def test_collapse_metrics():
    same = torch.ones(5, 8)
    assert mean_pairwise_cosine(same) == pytest.approx(1.0)
    assert effective_rank(torch.randn(50, 8)) > 4
    rank_one = torch.randn(50, 1) * torch.randn(1, 8)
    assert effective_rank(rank_one) == pytest.approx(1.0, abs=0.05)


def test_attention_stats_padding_and_spread():
    b, h, k, p, t = 2, 2, 3, 2, 4
    key_mask = torch.tensor([[1] * 8, [1] * 4 + [0] * 4], dtype=torch.bool)
    image_mask = torch.tensor([[1, 1], [1, 0]], dtype=torch.bool)
    w = torch.rand(b, h, k, p * t).masked_fill(~key_mask[:, None, None, :], 0)
    w = w / w.sum(-1, keepdim=True)
    s = attention_stats(w, key_mask, image_mask, t)
    assert s["padding_mass"] == 0
    assert 0 < s["entropy"] <= 1
    assert "top_photo_share_multi" in s
    leaked = w.clone()
    leaked[1, :, :, -1] = 0.1
    assert attention_stats(leaked, key_mask, image_mask, t)["padding_mass"] > 0


def test_part_attention_lift():
    # one photo, 4 tokens; part covers token 0 (25% of the photo); all attention on token 0
    w = torch.zeros(1, 1, 2, 4)
    w[..., 0] = 1.0
    grids = {"a.JPG": {"fruiting_body": np.array([[1.0, 0], [0, 0]]),
                       "cap": np.array([[0, 1.0], [0, 0]])}}
    out = part_attention(w, torch.ones(1, 1, dtype=torch.bool), 4, [["/x/a.JPG"]], grids)
    assert out["lift"]["fruiting_body"] == pytest.approx(4.0)
    assert out["lift"]["cap"] == pytest.approx(0.0)


def test_alarms():
    cfg = DiagnosticsConfig()
    w = health_warnings({"query/pairwise_cos": 0.99, "attn/block0/entropy": 0.01,
                         "attn/block0/padding_mass": 1e-3, "grad/head_genus/norm": 0.0,
                         "check/permutation": 1e-2}, cfg)
    assert {"query_collapse", "attention_too_sharp", "padding_attended", "dead_gradient_head_genus",
            "invariance_permutation"} <= set(w)
    assert health_warnings({"query/pairwise_cos": 0.3, "attn/block0/entropy": 0.5}, cfg) == {}
    with pytest.raises(NonFiniteError):
        check_finite({"act/T1/nan": 3})
    with pytest.raises(NonFiniteError):
        check_finite({"loss/total": float("nan")})


def test_full_probe_pass_produces_every_metric_family(tmp_path):
    torch.manual_seed(0)
    model = tiny_model()
    data = SyntheticObs(6, photos=2)
    probe = collate_observations([data[i] for i in range(4)])
    probe["species"] = [data[i]["species"] for i in range(4)]
    grids = {f"{j}-{i}.JPG": {"fruiting_body": np.eye(2)} for i in range(4) for j in range(2)}
    logged, tags = [], {}
    cfg = DiagnosticsConfig(enabled=True, every_n_steps=1, backbone_tap_every=4,
                            save_attention_figures=True, n_figure_observations=2)
    diag = Diagnostics(model, cfg, probe, tmp_path, torch.device("cpu"), (0.5,) * 3, (0.5,) * 3,
                       part_grids=grids, log_fn=lambda m, step: logged.append(m),
                       tag_fn=lambda k, v: tags.__setitem__(k, v))
    diag.capture_initial()

    # one logged optimizer step
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = collate_observations([data[i] for i in range(2)])
    out = model(batch["images"], batch["image_mask"])
    from fungai.model.losses import observation_loss
    parts = observation_loss(out, batch["label"], batch["genus_label"], "hierarchical", "perceiver", 0.3)
    parts["total"].backward()
    gn = diag.grad_norms()
    diag.before_optimizer_step(0)
    opt.step()
    step_metrics = diag.after_optimizer_step(0, out, parts, opt, 1.0, 1.0, gn)
    for fam in ("loss/", "grad/", "upd/", "lr/", "head/"):
        assert any(k.startswith(fam) for k in step_metrics), fam

    m = diag.on_epoch_end(0, 1)
    for fam in ("act/T0_input", "act/T1_patch_embed", "act/T2_block", "act/T3_backbone_out",
                "act/T4_fusion_in", "attn/block0", "query/", "emb/", "head/", "drift/", "check/",
                "parts/fruiting_body"):
        assert any(k.startswith(fam) for k in m), fam
    assert m["check/permutation"] < 1e-4 and m["check/padding"] < 1e-4
    assert m["attn/block0/padding_mass"] == 0
    saved = json.loads((tmp_path / "diagnostics" / "epoch_0.json").read_text())
    assert "attn_block0" in saved["detail"]
    assert (tmp_path / "diagnostics" / "attention" / "epoch_0.png").exists()
