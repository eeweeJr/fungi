"""Cost-sensitive evaluation for toxic/edible confusion pairs."""

from __future__ import annotations

import numpy as np
import torch

CONFUSION_PAIRS = [
    ("Amanita phalloides", "Macrolepiota procera"),
    ("Amanita phalloides", "Russula virescens"),
    ("Amanita phalloides", "Tricholoma equestre"),
    ("Amanita virosa", "Agaricus campestris"),
    ("Amanita virosa", "Agaricus arvensis"),
    ("Galerina marginata", "Kuehneromyces mutabilis"),
    ("Cortinarius rubellus", "Craterellus tubaeformis"),
]

DEFAULT_MISCLASS_COST = 1.0
TOXIC_EDIBLE_COST = 10.0
CONFUSION_PAIR_COST = 20.0


def build_cost_matrix(
    species_map: dict[str, int],
    toxicity_map: dict[str, bool],
) -> np.ndarray:
    """N×N cost matrix. Default 1.0 for misclassifications,
    10.0 for toxic→edible, 20.0 for documented confusion pairs."""
    n = len(species_map)
    costs = np.full((n, n), DEFAULT_MISCLASS_COST)
    np.fill_diagonal(costs, 0.0)

    inv_map = {v: k for k, v in species_map.items()}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            sp_true = inv_map.get(i, "")
            sp_pred = inv_map.get(j, "")
            true_toxic = toxicity_map.get(sp_true, False)
            pred_toxic = toxicity_map.get(sp_pred, False)
            if true_toxic and not pred_toxic:
                costs[i, j] = TOXIC_EDIBLE_COST

    for toxic_sp, edible_sp in CONFUSION_PAIRS:
        ti = species_map.get(toxic_sp)
        ei = species_map.get(edible_sp)
        if ti is not None and ei is not None:
            costs[ti, ei] = max(costs[ti, ei], CONFUSION_PAIR_COST)

    return costs


def toxic_edible_misclassifications(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    species_map: dict[str, int],
    toxicity_map: dict[str, bool],
) -> dict:
    """Count and rate of toxic species predicted as edible, and per-pair details."""
    inv_map = {v: k for k, v in species_map.items()}
    mask = y_true >= 0
    yt, yp = y_true[mask], y_pred[mask]

    toxic_as_edible = 0
    edible_as_toxic = 0
    total_toxic = 0
    total_edible = 0

    for t, p in zip(yt, yp):
        sp_true = inv_map.get(int(t), "")
        sp_pred = inv_map.get(int(p), "")
        true_toxic = toxicity_map.get(sp_true, False)
        pred_toxic = toxicity_map.get(sp_pred, False)
        if true_toxic:
            total_toxic += 1
            if t != p and not pred_toxic:
                toxic_as_edible += 1
        else:
            total_edible += 1
            if t != p and pred_toxic:
                edible_as_toxic += 1

    pair_details = []
    for toxic_sp, edible_sp in CONFUSION_PAIRS:
        ti = species_map.get(toxic_sp)
        ei = species_map.get(edible_sp)
        in_dataset = ti is not None and ei is not None
        t_as_e = 0
        e_as_t = 0
        if in_dataset:
            t_as_e = int(((yt == ti) & (yp == ei)).sum())
            e_as_t = int(((yt == ei) & (yp == ti)).sum())
        pair_details.append({
            "toxic": toxic_sp,
            "edible": edible_sp,
            "in_dataset": in_dataset,
            "toxic_as_edible": t_as_e,
            "edible_as_toxic": e_as_t,
        })

    return {
        "toxic_predicted_edible_count": toxic_as_edible,
        "toxic_predicted_edible_rate": toxic_as_edible / max(total_toxic, 1),
        "edible_predicted_toxic_count": edible_as_toxic,
        "edible_predicted_toxic_rate": edible_as_toxic / max(total_edible, 1),
        "confusion_pair_details": pair_details,
    }


def cost_sensitive_weights(
    species_map: dict[str, int],
    toxicity_map: dict[str, bool],
) -> torch.Tensor:
    """Per-class weights for CrossEntropyLoss — higher for toxic species."""
    n = len(species_map)
    weights = torch.ones(n)
    inv_map = {v: k for k, v in species_map.items()}
    for i in range(n):
        sp = inv_map.get(i, "")
        if toxicity_map.get(sp, False):
            weights[i] = 3.0
    return weights
