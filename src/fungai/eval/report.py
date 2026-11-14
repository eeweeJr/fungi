"""Evaluation report orchestrator: collect predictions → standardized report dict."""

from __future__ import annotations

import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from fungai.eval.calibration import (
    accuracy_at_coverage,
    expected_calibration_error,
    open_set_detection_rate,
    reliability_diagram_data,
)
from fungai.eval.cost_matrix import toxic_edible_misclassifications
from fungai.eval.gallery import build_failure_gallery
from fungai.eval.metrics import (
    genus_confusion_matrix,
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

    return {
        "probs": np.concatenate(all_probs, axis=0),
        "labels": np.array(all_labels),
        "image_paths": all_paths,
        "observation_ids": all_obs_ids,
    }


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
    else:
        obs_probs, obs_labels = probs, labels

    closed_mask = obs_labels >= 0
    preds = obs_probs.argmax(axis=1)

    inv_map = {v: k for k, v in species_map.items()}
    species_names = [inv_map.get(i, f"class_{i}") for i in range(len(species_map))]

    # --- Stratified: by genus ---
    genus_labels = np.array([species_to_genus.get(int(l), "unknown") for l in obs_labels])
    by_genus = stratified_breakdown(obs_labels, preds, obs_probs, genus_labels, genus_names)

    # --- Stratified: by toxicity ---
    tox_labels = np.array([
        "poisonous" if toxicity_map.get(inv_map.get(int(l), ""), False) else "edible"
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

    gallery_n = eval_config.get("failure_gallery_n", 50)
    image_paths = predictions.get("image_paths", [])
    if len(image_paths) != len(labels):
        image_paths = [""] * len(obs_labels)

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_samples": len(labels),
        "n_observations": len(obs_labels),
        "n_open_set": int((labels == -1).sum()),
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
        "open_set": open_set_detection_rate(obs_probs, obs_labels),
        "cost_sensitive": toxic_edible_misclassifications(
            obs_labels, preds, species_map, toxicity_map
        ),
        "genus_confusion_matrix": genus_confusion_matrix(
            obs_labels, preds, species_to_genus, genus_names
        ).tolist(),
        "genus_names": genus_names,
        "stratified": {
            "by_genus": by_genus,
            "by_toxicity": by_toxicity,
            "by_rarity": by_rarity,
        },
        "failure_gallery": build_failure_gallery(
            obs_probs, obs_labels, image_paths, species_names, n=gallery_n
        ),
    }
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
) -> dict:
    """End-to-end: collect predictions → build report → return report dict."""
    predictions = collect_predictions(model, val_loader, device)
    return build_evaluation_report(
        predictions, species_map, toxicity_map,
        species_to_genus, genus_names, train_species_counts, eval_config,
    )
