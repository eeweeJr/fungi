"""Failure gallery: the N most confident wrong predictions."""

from __future__ import annotations

import numpy as np


def build_failure_gallery(
    probs: np.ndarray,
    y_true: np.ndarray,
    image_paths: list[str],
    species_names: list[str],
    n: int = 50,
) -> list[dict]:
    """Select the N worst predictions: highest confidence on wrong class."""
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)
    wrong = preds != y_true

    wrong_indices = np.where(wrong & (y_true >= 0))[0]
    if len(wrong_indices) == 0:
        return []

    sorted_wrong = wrong_indices[np.argsort(-confidences[wrong_indices])]
    top_n = sorted_wrong[:n]

    gallery = []
    for rank, idx in enumerate(top_n):
        gallery.append({
            "rank": rank + 1,
            "image_path": image_paths[idx] if idx < len(image_paths) else "",
            "true_species": species_names[y_true[idx]] if 0 <= y_true[idx] < len(species_names) else "UNKNOWN",
            "predicted_species": species_names[preds[idx]] if preds[idx] < len(species_names) else "UNKNOWN",
            "true_label": int(y_true[idx]),
            "predicted_label": int(preds[idx]),
            "confidence": float(confidences[idx]),
        })
    return gallery
