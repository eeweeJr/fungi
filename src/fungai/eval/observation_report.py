"""Evaluation for the observation model.

`collect_observation_predictions` returns one row per observation in the same shape
as `report.collect_predictions`, so `build_evaluation_report` works unchanged (each
observation id is unique, so its observation averaging is the identity). The extra
keys feed the sections added here:

    fusion        fused vs photo-averaged (the no-fusion baseline), by photo count
    genus_head    genus from the genus head vs from summed species probabilities
    underside     observations where any photo shows gills or pores vs none
    calibration   temperatures and ECE before / after
    decision      species / genus-only / unknown outcomes for known and held-out species
    intervals     95% bootstrap intervals over observations; paired for differences
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from fungai.eval.calibration import (
    apply_temperatures,
    expected_calibration_error,
    fit_temperatures,
)
from fungai.eval.decision import decide, fit_thresholds, outcome_labels, outcome_table
from fungai.eval.metrics import bootstrap_ci, genus_accuracy, genus_probs, macro_f1, top_k_accuracy
from fungai.model.losses import observation_loss


def _to(batch: dict, device) -> tuple:
    return (batch["images"].to(device), batch["image_mask"].to(device),
            batch["label"].to(device), batch["genus_label"].to(device))


@torch.no_grad()
def collect_observation_predictions(model, loader, device, photo_loss_weight: float = 0.3) -> dict:
    """Run the model over an observation loader. Also returns the mean loss terms."""
    model.eval()
    keep: dict[str, list] = {k: [] for k in (
        "probs", "labels", "genus_labels", "genus_probs_head", "photo_avg_probs",
        "genus_logits", "species_logits", "n_images", "revisited")}
    obs_ids, first_paths, photo_paths = [], [], []
    loss_sums: dict[str, float] = {}
    n_seen = 0
    for batch in loader:
        images, mask, label, genus_label = _to(batch, device)
        out = model(images, mask)
        parts = observation_loss(out, label, genus_label, model.head_kind, model.fusion_kind,
                                 photo_loss_weight)
        b = len(label)
        for k, v in parts.items():
            loss_sums[k] = loss_sums.get(k, 0.0) + float(v.detach()) * b
        n_seen += b
        keep["probs"].append(out["species_logp"].exp().float().cpu().numpy())
        keep["genus_probs_head"].append(out["genus_logp"].exp().float().cpu().numpy())
        keep["photo_avg_probs"].append(model.photo_average(out, b).exp().float().cpu().numpy())
        keep["genus_logits"].append(out["genus_logits"].float().cpu().numpy())
        keep["species_logits"].append(out["species_logits"].float().cpu().numpy())
        keep["labels"].append(batch["label"].numpy())
        keep["genus_labels"].append(batch["genus_label"].numpy())
        keep["n_images"].append(batch["n_images"].numpy())
        if "revisited_site" in batch:
            keep["revisited"].append(batch["revisited_site"].numpy())
        obs_ids.extend(batch["observation_id"])
        paths = batch.get("image_paths") or [[""] for _ in range(b)]
        first_paths.extend(p[0] for p in paths)
        photo_paths.extend(paths)
    pred = {k: np.concatenate(v) for k, v in keep.items() if v}
    pred["revisited"] = pred["revisited"].astype(bool) if "revisited" in pred else None
    if pred["revisited"] is None:
        del pred["revisited"]
    pred.update({"observation_ids": obs_ids, "image_paths": first_paths, "photo_paths": photo_paths,
                 "loss": {k: v / max(n_seen, 1) for k, v in loss_sums.items()}})
    return pred


def underside_flags(photo_paths: list[list[str]], presence: pd.DataFrame | None) -> np.ndarray:
    """Per observation: "underside" if any used photo shows gills or pores, "no_underside"
    if every used photo has masks but none shows them, "no_mask" if any photo lacks masks."""
    if presence is None:
        return np.full(len(photo_paths), "no_mask", dtype=object)
    has_under = dict(zip(presence["filename"], presence["has_underside"]))
    out = []
    for paths in photo_paths:
        names = [p.rsplit("/", 1)[-1] for p in paths]
        if any(has_under.get(n, False) for n in names):
            out.append("underside")
        elif all(n in has_under for n in names):
            out.append("no_underside")
        else:
            out.append("no_mask")
    return np.array(out, dtype=object)


def _acc_block(probs, labels, n_bins: int = 15) -> dict:
    known = labels >= 0
    n = int(known.sum())
    if n == 0:
        return {"n": 0, "macro_f1": None, "top_1_accuracy": None, "mean_confidence": None, "ece": None}
    return {
        "n": n,
        "macro_f1": macro_f1(labels, probs.argmax(1)),
        "top_1_accuracy": top_k_accuracy(labels, probs, 1),
        "mean_confidence": float(probs[known].max(1).mean()),
        "ece": expected_calibration_error(probs[known], labels[known], n_bins),
    }


def build_observation_sections(
    pred: dict,
    species_genus: np.ndarray,
    genus_names: list[str],
    species_to_genus: dict[int, str],
    head: str,
    eval_cfg,
    presence: pd.DataFrame | None = None,
    temperatures: dict | None = None,
) -> tuple[dict, dict]:
    """Report sections plus the fitted calibration (temperatures and thresholds).

    Pass `temperatures` to apply an existing calibration instead of fitting one.
    """
    labels, glabels = pred["labels"], pred["genus_labels"]
    known = labels >= 0
    n_genus = len(genus_names)
    n_bins = eval_cfg.ece_n_bins

    # ---- calibration
    if temperatures is None:
        temperatures = (fit_temperatures(pred["genus_logits"], pred["species_logits"], glabels,
                                         labels, species_genus, n_genus, head)
                        if eval_cfg.temperature_scaling else {"t_genus": 1.0, "t_species": 1.0})
    sp_cal, g_cal = apply_temperatures(pred["genus_logits"], pred["species_logits"], species_genus,
                                       n_genus, temperatures["t_genus"], temperatures["t_species"], head)
    calibration = {
        **temperatures,
        "ece_before": expected_calibration_error(pred["probs"][known], labels[known], n_bins)
        if known.any() else None,
        "ece_after": expected_calibration_error(sp_cal[known], labels[known], n_bins)
        if known.any() else None,
    }

    # ---- fusion vs photo average
    fused, avg = pred["probs"], pred["photo_avg_probs"]
    fusion = {"fused": _acc_block(fused, labels, n_bins), "photo_average": _acc_block(avg, labels, n_bins),
              "by_photo_count": {}}
    for k in sorted(set(pred["n_images"].tolist())):
        m = pred["n_images"] == k
        fusion["by_photo_count"][str(k)] = {"fused": _acc_block(fused[m], labels[m], n_bins),
                                            "photo_average": _acc_block(avg[m], labels[m], n_bins)}
    multi = pred["n_images"] >= 2
    fusion["multi_photo"] = {"fused": _acc_block(fused[multi], labels[multi], n_bins),
                             "photo_average": _acc_block(avg[multi], labels[multi], n_bins)}

    # ---- genus head vs summed species
    g_summed = genus_probs(fused, species_to_genus, genus_names)
    genus_head = {
        "head": {"known": genus_accuracy(glabels[known], pred["genus_probs_head"][known], genus_names),
                 "held_out": genus_accuracy(glabels[~known], pred["genus_probs_head"][~known], genus_names)},
        "summed_species": {"known": genus_accuracy(glabels[known], g_summed[known], genus_names),
                           "held_out": genus_accuracy(glabels[~known], g_summed[~known], genus_names)},
    }

    # ---- underside visibility
    flags = underside_flags(pred["photo_paths"], presence)
    underside = {name: _acc_block(sp_cal[flags == name], labels[flags == name], n_bins)
                 for name in ("underside", "no_underside", "no_mask")}

    # ---- decision rule (on calibrated probabilities)
    thresholds = fit_thresholds(sp_cal, g_cal, labels, eval_cfg.known_retention)
    decision, sp_pred, g_pred = decide(sp_cal, g_cal, thresholds["tau_species"], thresholds["tau_genus"])
    outcomes = outcome_labels(decision, sp_pred, g_pred, labels, glabels)
    decision_section = {**thresholds, "outcomes": outcome_table(outcomes, labels)}

    # ---- bootstrap intervals over observations
    n_s, seed = eval_cfg.bootstrap_samples, eval_cfg.bootstrap_seed
    ki = np.flatnonzero(known)
    hi = np.flatnonzero(~known)
    yk, fk, ak = labels[ki], fused[ki].argmax(1), avg[ki].argmax(1)
    intervals = {
        "macro_f1": bootstrap_ci(lambda i: macro_f1(yk[i], fk[i]), len(ki), n_s, seed),
        "top_1_accuracy": bootstrap_ci(lambda i: float((yk[i] == fk[i]).mean()), len(ki), n_s, seed),
        "photo_average_macro_f1": bootstrap_ci(lambda i: macro_f1(yk[i], ak[i]), len(ki), n_s, seed),
        "fused_minus_photo_average_macro_f1": bootstrap_ci(
            lambda i: macro_f1(yk[i], fk[i]) - macro_f1(yk[i], ak[i]), len(ki), n_s, seed),
    }
    if len(hi):
        gh = (pred["genus_probs_head"][hi].argmax(1) == glabels[hi]).astype(float)
        intervals["held_out_genus_accuracy"] = bootstrap_ci(lambda i: gh[i].mean(), len(hi), n_s, seed)
    for group, idx in (("known", ki), ("held_out", hi)):
        oc = outcomes[idx]
        for name in sorted(set(oc)):
            hit = (oc == name).astype(float)
            intervals[f"decision_{group}_{name}"] = bootstrap_ci(lambda i, h=hit: h[i].mean(),
                                                                  len(idx), n_s, seed)

    sections = {"fusion": fusion, "genus_head": genus_head, "underside": underside,
                "calibration_fit": calibration, "decision": decision_section, "intervals": intervals,
                "loss": pred.get("loss", {})}
    fitted = {**temperatures, **thresholds}
    extras = {"decision": decision, "outcome": outcomes, "underside": flags,
              "species_probs_calibrated": sp_cal, "genus_probs_calibrated": g_cal}
    return sections, {"calibration": fitted, "per_observation": extras}


def observation_predictions_table(table: pd.DataFrame, pred: dict, extras: dict,
                                  genus_names: list[str], inv_species: dict[int, str]) -> pd.DataFrame:
    """Extend `report.predictions_table` output (one row per observation, same order)."""
    out = table.copy()
    avg_pred = pred["photo_avg_probs"].argmax(1)
    out["photo_avg_pred_label"] = avg_pred
    out["photo_avg_pred_species"] = [inv_species.get(int(p), "") for p in avg_pred]
    out["photo_avg_confidence"] = pred["photo_avg_probs"].max(1)
    gh = pred["genus_probs_head"]
    out["genus_head_pred"] = [genus_names[i] for i in gh.argmax(1)]
    out["genus_head_confidence"] = gh.max(1)
    out["calibrated_confidence"] = extras["species_probs_calibrated"].max(1)
    out["decision"] = extras["decision"]
    out["outcome"] = extras["outcome"]
    out["n_images_used"] = pred["n_images"]
    out["underside"] = extras["underside"]
    return out
