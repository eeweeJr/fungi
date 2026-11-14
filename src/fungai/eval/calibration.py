"""Calibration metrics: ECE, reliability diagrams, coverage curves, open-set detection."""

from __future__ import annotations

import numpy as np


def expected_calibration_error(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> float:
    """ECE: weighted average of |accuracy - confidence| per confidence bin."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == y_true

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            avg_confidence = float(confidences[in_bin].mean())
            avg_accuracy = float(accuracies[in_bin].mean())
            ece += abs(avg_accuracy - avg_confidence) * prop_in_bin
    return float(ece)


def reliability_diagram_data(
    probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15
) -> dict:
    """Bin edges, accuracies, confidences, counts for a reliability diagram."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = predictions == y_true

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        count = int(in_bin.sum())
        bin_counts.append(count)
        if count > 0:
            bin_accs.append(float(accuracies[in_bin].mean()))
            bin_confs.append(float(confidences[in_bin].mean()))
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    return {
        "bin_edges": bin_boundaries.tolist(),
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
    }


def accuracy_at_coverage(
    probs: np.ndarray, y_true: np.ndarray, n_steps: int = 20
) -> dict:
    """Accuracy when the model abstains on its least-confident predictions.

    Returns parallel arrays: coverage fraction and accuracy at that coverage.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = predictions == y_true

    sorted_indices = np.argsort(-confidences)
    sorted_correct = correct[sorted_indices]

    n = len(y_true)
    coverages, accuracies = [], []
    for step in range(1, n_steps + 1):
        k = max(1, int(n * step / n_steps))
        coverages.append(float(k / n))
        accuracies.append(float(sorted_correct[:k].mean()))

    return {"coverage": coverages, "accuracy": accuracies}


def open_set_detection_rate(
    probs: np.ndarray, y_true: np.ndarray, confidence_threshold: float = 0.5
) -> dict:
    """Measure false acceptance rate on open-set samples (label == -1)."""
    open_mask = y_true == -1
    n_open = int(open_mask.sum())
    if n_open == 0:
        return {"n_open_set": 0, "false_acceptance_rate": None, "mean_open_set_confidence": None}

    open_confidences = probs[open_mask].max(axis=1)
    return {
        "n_open_set": n_open,
        "false_acceptance_rate": float((open_confidences > confidence_threshold).mean()),
        "mean_open_set_confidence": float(open_confidences.mean()),
    }
