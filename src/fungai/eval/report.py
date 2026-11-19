"""Evaluation report orchestrator: collect predictions → standardized report dict."""

from __future__ import annotations

import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fungai.eval.calibration import (
    accuracy_at_coverage,
    expected_calibration_error,
    open_set_detection_rate,
    open_set_metrics,
    reliability_diagram_data,
)
from fungai.eval.cost_matrix import toxic_edible_misclassifications
from fungai.eval.gallery import build_failure_gallery
from fungai.eval.metrics import (
    bootstrap_ci,
    error_decomposition,
    genus_accuracy,
    genus_confusion_matrix,
    genus_probs,
    macro_f1,
    rarity_buckets,
    stratified_breakdown,
    top_k_accuracy,
)


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """Run model on all batches. Returns probs, labels, image_paths, observation_ids."""
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[int] = []
    all_paths: list[str] = []
    all_obs_ids: list[str] = []
    all_genus: list[int] = []
    all_revisited: list[bool] = []

    for batch in loader:
        images = batch["image"].to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)
        all_labels.extend(batch["label"].numpy().tolist())
        if "image_path" in batch:
            all_paths.extend(batch["image_path"])
        if "observation_id" in batch:
            all_obs_ids.extend(batch["observation_id"])
        if "genus_label" in batch:
            all_genus.extend(batch["genus_label"].numpy().tolist())
        if "revisited_site" in batch:
            all_revisited.extend(batch["revisited_site"].numpy().tolist())

    predictions = {
        "probs": np.concatenate(all_probs, axis=0),
        "labels": np.array(all_labels),
        "image_paths": all_paths,
        "observation_ids": all_obs_ids,
    }
    if all_genus:
        predictions["genus_labels"] = np.array(all_genus)
    if all_revisited:
        predictions["revisited"] = np.array(all_revisited, dtype=bool)
    return predictions


def observation_first_indices(observation_ids: list[str]) -> np.ndarray:
    """Index of the first image of each observation, in aggregate_by_observation order."""
    first: dict[str, int] = {}
    for i, obs_id in enumerate(observation_ids):
        first.setdefault(obs_id, i)
    return np.array(list(first.values()), dtype=int)


def aggregate_by_observation(
    probs: np.ndarray,
    labels: np.ndarray,
    observation_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Average probabilities across images of the same observation."""
    from collections import defaultdict

    groups: dict[str, list[int]] = defaultdict(list)
    for i, obs_id in enumerate(observation_ids):
        groups[obs_id].append(i)

    obs_probs = []
    obs_labels = []
    for obs_id, indices in groups.items():
        obs_probs.append(probs[indices].mean(axis=0))
        obs_labels.append(labels[indices[0]])

    return np.array(obs_probs), np.array(obs_labels)


def predictions_table(
    predictions: dict,
    species_map: dict[str, int],
    species_to_genus: dict[int, str],
    genus_names: list[str],
    top_k: int = 5,
) -> tuple[pd.DataFrame, np.ndarray]:
    """One row per observation, ready to join to a data snapshot on observation_id.

    Returns the table and the matching [n_observations, n_classes] probabilities,
    so thresholds and calibration can be revisited without re-running the model.
    """
    probs, labels = predictions["probs"], predictions["labels"]
    obs_ids = predictions.get("observation_ids") or [str(i) for i in range(len(labels))]
    obs_probs, obs_labels = aggregate_by_observation(probs, labels, obs_ids)
    first = observation_first_indices(obs_ids)
    n_images = Counter(obs_ids)

    inv_map = {v: k for k, v in species_map.items()}
    order = np.argsort(-obs_probs, axis=1)
    pred = order[:, 0]
    true_rank = np.full(len(obs_labels), -1)
    known = obs_labels >= 0
    true_rank[known] = np.argmax(order[known] == obs_labels[known, None], axis=1) + 1

    g_probs = genus_probs(obs_probs, species_to_genus, genus_names)
    table = pd.DataFrame({
        "observation_id": [obs_ids[i] for i in first],
        "n_images": [n_images[obs_ids[i]] for i in first],
        "label": obs_labels,
        "is_unknown": ~known,
        "pred_label": pred,
        "pred_species": [inv_map.get(int(p), "") for p in pred],
        "pred_genus": [species_to_genus.get(int(p), "") for p in pred],
        "confidence": obs_probs.max(axis=1),
        "correct": known & (pred == obs_labels),
        "true_rank": true_rank,  # 1 = top prediction; -1 = unknown species
        "top_k_labels": [row[:top_k].tolist() for row in order],
        "genus_pred_summed": [genus_names[i] for i in g_probs.argmax(axis=1)],
        "genus_confidence_summed": g_probs.max(axis=1),
    })
    genus_true = predictions.get("genus_labels")
    if genus_true is not None and len(genus_true) == len(labels):
        genus_true = np.asarray(genus_true)[first]
        table["genus_label"] = genus_true
        table["genus_correct"] = g_probs.argmax(axis=1) == genus_true
    return table, obs_probs.astype(np.float32)


def build_evaluation_report(
    predictions: dict,
    species_map: dict[str, int],
    toxicity_map: dict[str, bool],
    species_to_genus: dict[int, str],
    genus_names: list[str],
    train_species_counts: dict[int, int],
    eval_config: dict | None = None,
) -> dict:
    """Assemble the full evaluation report from raw predictions."""
    eval_config = eval_config or {}
    probs = predictions["probs"]
    labels = predictions["labels"]

    if eval_config.get("aggregate_observations", True) and predictions.get("observation_ids"):
        obs_probs, obs_labels = aggregate_by_observation(
            probs, labels, predictions["observation_ids"]
        )
        first = observation_first_indices(predictions["observation_ids"])
    else:
        obs_probs, obs_labels = probs, labels
        first = np.arange(len(labels))

    def per_observation(key: str) -> np.ndarray | None:
        values = predictions.get(key)
        if values is None or len(values) != len(labels):
            return None
        return np.asarray(values)[first]

    closed_mask = obs_labels >= 0
    preds = obs_probs.argmax(axis=1)

    inv_map = {v: k for k, v in species_map.items()}
    species_names = [inv_map.get(i, f"class_{i}") for i in range(len(species_map))]

    # --- Stratified: by genus ---
    genus_labels = np.array([species_to_genus.get(int(l), "unknown") for l in obs_labels])
    by_genus = stratified_breakdown(obs_labels, preds, obs_probs, genus_labels, genus_names)

    # --- Stratified: by toxicity ---
    tox_labels = np.array([
        "unknown" if l < 0
        else "poisonous" if toxicity_map.get(inv_map.get(int(l), ""), False) else "edible"
        for l in obs_labels
    ])
    by_toxicity = stratified_breakdown(
        obs_labels, preds, obs_probs, tox_labels, ["poisonous", "edible"]
    )

    # --- Stratified: by rarity ---
    rbuckets = rarity_buckets(train_species_counts)
    rarity_labels = np.array([rbuckets.get(int(l), "unknown") for l in obs_labels])
    by_rarity = stratified_breakdown(
        obs_labels, preds, obs_probs, rarity_labels, ["rare", "few", "medium", "common"]
    )

    stratified = {"by_genus": by_genus, "by_toxicity": by_toxicity, "by_rarity": by_rarity}

    # --- Stratified: sites already photographed in a train year vs new sites ---
    revisited = per_observation("revisited")
    if revisited is not None:
        site_labels = np.where(revisited, "revisited", "new_site")
        stratified["by_site"] = stratified_breakdown(
            obs_labels, preds, obs_probs, site_labels, ["revisited", "new_site"]
        )

    # --- Genus: from summed species probabilities, for known and unknown species ---
    genus_report = None
    genus_true = per_observation("genus_labels")
    if genus_true is not None:
        g_probs = genus_probs(obs_probs, species_to_genus, genus_names)
        genus_report = {
            "known": genus_accuracy(genus_true[closed_mask], g_probs[closed_mask], genus_names),
            "unknown": genus_accuracy(genus_true[~closed_mask], g_probs[~closed_mask], genus_names),
        }

    # --- Bootstrap intervals over observations (only when asked: the final evaluation) ---
    intervals = None
    n_boot = eval_config.get("bootstrap_samples")
    if n_boot:
        seed = eval_config.get("bootstrap_seed", 0)
        ki = np.flatnonzero(closed_mask)
        yk, pk, probk = obs_labels[ki], preds[ki], obs_probs[ki]
        intervals = {
            "macro_f1": bootstrap_ci(lambda i: macro_f1(yk[i], pk[i]), len(ki), n_boot, seed,
                                     strata=yk),
            "top_1_accuracy": bootstrap_ci(lambda i: float((yk[i] == pk[i]).mean()), len(ki), n_boot,
                                           seed, strata=yk),
            "top_5_accuracy": bootstrap_ci(lambda i: top_k_accuracy(yk[i], probk[i], k=5), len(ki),
                                           n_boot, seed, strata=yk),
        }

    gallery_n = eval_config.get("failure_gallery_n", 50)
    image_paths = per_observation("image_paths")
    image_paths = [""] * len(obs_labels) if image_paths is None else image_paths.tolist()

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_samples": len(labels),
        "n_observations": len(obs_labels),
        "n_open_set": int((labels == -1).sum()),
        "n_open_set_observations": int((obs_labels == -1).sum()),
        "primary_metrics": {
            "macro_f1": macro_f1(obs_labels, preds),
            "top_1_accuracy": top_k_accuracy(obs_labels, obs_probs, k=1),
            "top_5_accuracy": top_k_accuracy(obs_labels, obs_probs, k=5),
        },
        "calibration": {
            "ece": expected_calibration_error(
                obs_probs[closed_mask], obs_labels[closed_mask]
            ) if closed_mask.sum() > 0 else 0.0,
            "reliability_diagram": reliability_diagram_data(
                obs_probs[closed_mask], obs_labels[closed_mask]
            ) if closed_mask.sum() > 0 else {},
        },
        "coverage": accuracy_at_coverage(
            obs_probs[closed_mask], obs_labels[closed_mask]
        ) if closed_mask.sum() > 0 else {"coverage": [], "accuracy": []},
        "open_set": {
            **open_set_detection_rate(obs_probs, obs_labels),
            **open_set_metrics(
                obs_probs, obs_labels, threshold=eval_config.get("open_set_threshold")
            ),
        },
        "error_decomposition": error_decomposition(obs_labels, preds, species_to_genus),
        "genus": genus_report,
        "cost_sensitive": toxic_edible_misclassifications(
            obs_labels, preds, species_map, toxicity_map
        ),
        "genus_confusion_matrix": genus_confusion_matrix(
            obs_labels, preds, species_to_genus, genus_names
        ).tolist(),
        "genus_names": genus_names,
        "stratified": stratified,
        "failure_gallery": build_failure_gallery(
            obs_probs, obs_labels, image_paths, species_names, n=gallery_n
        ),
    }
    if intervals is not None:
        report["intervals"] = intervals
    return report


def evaluate_model(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    species_map: dict[str, int],
    toxicity_map: dict[str, bool],
    species_to_genus: dict[int, str],
    genus_names: list[str],
    train_species_counts: dict[int, int],
    eval_config: dict | None = None,
    return_predictions: bool = False,
) -> dict | tuple[dict, dict]:
    """End-to-end: collect predictions → build report → return report dict
    (and the raw predictions when `return_predictions` is set)."""
    predictions = collect_predictions(model, val_loader, device)
    report = build_evaluation_report(
        predictions, species_map, toxicity_map,
        species_to_genus, genus_names, train_species_counts, eval_config,
    )
    return (report, predictions) if return_predictions else report
