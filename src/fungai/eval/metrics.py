"""Core metric computations. All functions are pure: numpy arrays in, scalars/dicts out."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true >= 0
    if mask.sum() == 0:
        return 0.0
    return float(f1_score(y_true[mask], y_pred[mask], average="macro", zero_division=0))


def top_k_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int) -> float:
    mask = y_true >= 0
    if mask.sum() == 0:
        return 0.0
    top_k_preds = np.argsort(probs[mask], axis=1)[:, -k:]
    correct = np.any(top_k_preds == y_true[mask, None], axis=1)
    return float(correct.mean())


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict[str, dict]:
    from sklearn.metrics import precision_recall_fscore_support

    mask = y_true >= 0
    labels = list(range(len(class_names)))
    p, r, f, s = precision_recall_fscore_support(
        y_true[mask], y_pred[mask], labels=labels, zero_division=0
    )
    return {
        class_names[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
        for i in range(len(class_names))
    }


def genus_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    species_to_genus: dict[int, str],
    genus_names: list[str],
) -> np.ndarray:
    """Map species-level predictions to genus, return genus×genus confusion matrix."""
    genus_idx = {g: i for i, g in enumerate(genus_names)}
    n = len(genus_names)
    cm = np.zeros((n, n), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        if yt < 0:
            continue
        gt_genus = species_to_genus.get(int(yt))
        pd_genus = species_to_genus.get(int(yp))
        if gt_genus in genus_idx and pd_genus in genus_idx:
            cm[genus_idx[gt_genus], genus_idx[pd_genus]] += 1
    return cm


def stratified_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probs: np.ndarray,
    group_labels: np.ndarray,
    group_names: list[str],
) -> dict[str, dict]:
    """Compute metrics broken down by an arbitrary grouping (genus, toxicity, rarity)."""
    result = {}
    for gname in group_names:
        mask = group_labels == gname
        if mask.sum() == 0:
            result[gname] = {"macro_f1": 0.0, "top_1_accuracy": 0.0, "n_samples": 0}
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        p = probs[mask]
        result[gname] = {
            "macro_f1": macro_f1(yt, yp),
            "top_1_accuracy": top_k_accuracy(yt, p, k=1),
            "n_samples": int(mask.sum()),
        }
    return result


def rarity_buckets(species_counts: dict[int, int]) -> dict[int, str]:
    """Assign species (by label index) to rarity buckets based on training image count."""
    buckets = {}
    for species_id, count in species_counts.items():
        if count < 10:
            buckets[species_id] = "rare"
        elif count < 50:
            buckets[species_id] = "few"
        elif count < 200:
            buckets[species_id] = "medium"
        else:
            buckets[species_id] = "common"
    return buckets
