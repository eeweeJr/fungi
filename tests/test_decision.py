"""Tests for temperature scaling, the decision rule and the observation report sections."""

import numpy as np
import pytest
import torch

from fungai.config import EvalConfig
from fungai.eval.calibration import apply_temperatures, fit_temperatures
from fungai.eval.decision import decide, fit_thresholds, outcome_labels, outcome_table
from fungai.eval.observation_report import build_observation_sections, underside_flags

SG = np.array([0, 0, 1, 1, 2])


def test_decide_and_outcomes():
    sp = np.array([[0.9, 0.05, 0.03, 0.01, 0.01],   # confident, correct
                   [0.4, 0.35, 0.1, 0.1, 0.05],     # genus 0 sure, species not
                   [0.3, 0.2, 0.25, 0.2, 0.05],     # nothing sure
                   [0.05, 0.05, 0.8, 0.05, 0.05]])  # held-out, names a species
    g = np.stack([sp[:, :2].sum(1), sp[:, 2:4].sum(1), sp[:, 4]], 1)
    dec, sp_pred, g_pred = decide(sp, g, tau_species=0.6, tau_genus=0.7)
    assert dec.tolist() == ["species", "genus", "unknown", "species"]
    labels, genus = np.array([0, 1, 3, -1]), np.array([0, 0, 1, 1])
    oc = outcome_labels(dec, sp_pred, g_pred, labels, genus)
    assert oc.tolist() == ["species_correct", "genus_correct", "unknown", "species_named"]
    t = outcome_table(oc, labels)
    assert t["known"]["n"] == 3 and t["held_out"]["species_named"] == 1.0


def test_fit_thresholds_retention():
    rng = np.random.default_rng(0)
    sp = rng.dirichlet(np.ones(5), 200)
    g = np.stack([sp[:, :2].sum(1), sp[:, 2:4].sum(1), sp[:, 4]], 1)
    labels = rng.integers(0, 5, 200)
    th = fit_thresholds(sp, g, labels, 0.9)
    assert (sp.max(1) >= th["tau_species"]).mean() == pytest.approx(0.9, abs=0.01)


@pytest.mark.parametrize("head", ["hierarchical", "flat"])
def test_temperature_fit_does_not_increase_nll(head):
    rng = np.random.default_rng(0)
    n = 300
    labels = rng.integers(0, 5, n)
    genus = SG[labels]
    species_logits = rng.normal(size=(n, 5)) * 0.5
    species_logits[np.arange(n), labels] += 4.0      # overconfident
    genus_logits = rng.normal(size=(n, 3))
    genus_logits[np.arange(n), genus] += 6.0
    t = fit_temperatures(genus_logits, species_logits, genus, labels, SG, 3, head)

    def nll(tg, ts):
        p, _ = apply_temperatures(genus_logits, species_logits, SG, 3, tg, ts, head)
        return -np.log(p[np.arange(n), labels]).mean()

    assert nll(t["t_genus"], t["t_species"]) <= nll(1.0, 1.0) + 1e-9


def test_observation_sections_shape():
    rng = np.random.default_rng(0)
    n = 60
    labels = rng.integers(-1, 5, n)
    genus = np.where(labels >= 0, SG[np.clip(labels, 0, 4)], rng.integers(0, 3, n))
    sl, gl = rng.normal(size=(n, 5)), rng.normal(size=(n, 3))
    from fungai.model.hierarchy import hierarchical_logp
    out = hierarchical_logp(torch.tensor(gl), torch.tensor(sl), torch.tensor(SG), 3)
    probs = out["species_logp"].exp().numpy()
    pred = {"labels": labels, "genus_labels": genus, "probs": probs, "photo_avg_probs": probs[::-1].copy(),
            "genus_probs_head": out["genus_logp"].exp().numpy(), "genus_logits": gl, "species_logits": sl,
            "n_images": rng.integers(1, 5, n), "photo_paths": [[f"/x/{i}.JPG"] for i in range(n)],
            "loss": {"total": 1.0}}
    e = EvalConfig(bootstrap_samples=50)
    sections, fitted = build_observation_sections(pred, SG, ["A", "B", "C"], {i: "ABC"[g] for i, g in enumerate(SG)},
                                                  "hierarchical", e, presence=None)
    for key in ("fusion", "genus_head", "underside", "calibration_fit", "decision", "intervals"):
        assert key in sections
    assert set(sections["fusion"]["by_photo_count"]) <= {"1", "2", "3", "4"}
    assert sections["intervals"]["fused_minus_photo_average_macro_f1"]["lo"] is not None
    assert {"t_genus", "t_species", "tau_species", "tau_genus"} <= set(fitted["calibration"])


def test_underside_flags():
    import pandas as pd
    pres = pd.DataFrame({"filename": ["a.JPG", "b.JPG"], "has_underside": [True, False]})
    flags = underside_flags([["/p/a.JPG", "/p/b.JPG"], ["/p/b.JPG"], ["/p/c.JPG"]], pres)
    assert flags.tolist() == ["underside", "no_underside", "no_mask"]
