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


def _fit_temperature(logits, labels, logp_fn, max_iter: int = 100) -> float:
    """Temperature minimising mean NLL of logp_fn(logits / T) on labels. Never returns
    a temperature that is worse than T = 1."""
    import torch

    logits = torch.as_tensor(logits, dtype=torch.float64)
    labels = torch.as_tensor(labels, dtype=torch.long)
    if len(labels) == 0:
        return 1.0

    def nll(t):
        return torch.nn.functional.nll_loss(logp_fn(logits / t), labels)

    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = nll(log_t.exp())
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.detach().exp().clamp(0.05, 20.0))
    with torch.no_grad():
        if not np.isfinite(t) or nll(torch.tensor(t)) > nll(torch.tensor(1.0)):
            return 1.0
    return t


def fit_temperatures(genus_logits, species_logits, genus_labels, labels, species_genus,
                     n_genus: int, head: str = "hierarchical") -> dict:
    """Temperatures fitted on validation observations by minimising NLL.

    hierarchical: t_genus on p(genus) over all observations with a genus label, and
    t_species on p(species | genus) over known-species observations. The two are
    independent because log p(species) = log p(genus) + log p(species | genus).
    flat: one t_species on the species softmax; t_genus = 1.
    """
    import torch

    from fungai.model.hierarchy import membership

    sg = torch.as_tensor(species_genus, dtype=torch.long)
    labels, genus_labels = np.asarray(labels), np.asarray(genus_labels)
    known = labels >= 0
    if head == "flat":
        t_s = _fit_temperature(np.asarray(species_logits)[known], labels[known],
                               lambda x: x.log_softmax(-1))
        return {"t_genus": 1.0, "t_species": t_s}
    g_ok = genus_labels >= 0
    t_g = _fit_temperature(np.asarray(genus_logits)[g_ok], genus_labels[g_ok],
                           lambda x: x.log_softmax(-1))
    member = membership(sg, n_genus)

    def cond_logp(x):
        masked = x.unsqueeze(1).masked_fill(~member.unsqueeze(0), float("-inf"))
        return x - torch.logsumexp(masked, dim=-1)[:, sg]

    t_s = _fit_temperature(np.asarray(species_logits)[known], labels[known], cond_logp)
    return {"t_genus": t_g, "t_species": t_s}


def apply_temperatures(genus_logits, species_logits, species_genus, n_genus: int,
                       t_genus: float, t_species: float, head: str = "hierarchical"):
    """(species_probs, genus_probs) as numpy after temperature scaling."""
    import torch

    from fungai.model.hierarchy import flat_logp, hierarchical_logp

    g = torch.as_tensor(np.asarray(genus_logits), dtype=torch.float64)
    s = torch.as_tensor(np.asarray(species_logits), dtype=torch.float64)
    sg = torch.as_tensor(species_genus, dtype=torch.long)
    if head == "flat":
        out = flat_logp(s, sg, n_genus, t_species=t_species)
    else:
        out = hierarchical_logp(g, s, sg, n_genus, t_genus=t_genus, t_species=t_species)
    return out["species_logp"].exp().numpy(), out["genus_logp"].exp().numpy()
