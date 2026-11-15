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


def open_set_metrics(
    probs: np.ndarray,
    y_true: np.ndarray,
    known_retention: float = 0.95,
    threshold: float | None = None,
) -> dict:
    """How well top-1 confidence separates known species from unknown ones (label -1).

    auroc: chance a known sample is more confident than an unknown one.
    The false acceptance rate is measured at `threshold`; when none is given it is
    set so that `known_retention` of the known samples are kept. Tune it on val and
    pass it in for test.
    """
    open_mask = y_true == -1
    confidences = probs.max(axis=1)
    known, unknown = confidences[~open_mask], confidences[open_mask]
    if len(known) == 0 or len(unknown) == 0:
        return {"auroc": None, "threshold": threshold, "known_retained": None,
                "unknown_false_acceptance_rate": None}

    from sklearn.metrics import roc_auc_score

    if threshold is None:
        threshold = float(np.quantile(known, 1.0 - known_retention))
    return {
        "auroc": float(roc_auc_score(~open_mask, confidences)),
        "threshold": float(threshold),
        "known_retained": float((known >= threshold).mean()),
        "unknown_false_acceptance_rate": float((unknown >= threshold).mean()),
    }
