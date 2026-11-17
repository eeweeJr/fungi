"""Core metric computations. All functions are pure: numpy arrays in, scalars/dicts out."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Macro F1 over the known species present in y_true (open-set rows, label < 0, are
    dropped). Averaging only over true species is the standard convention: sklearn's
    default would also average over species that were only *predicted*, adding a 0 for
    each and making the score depend on which wrong species the model happened to name
    (0.553 instead of 0.570 for E0, and far worse inside small strata)."""
    mask = y_true >= 0
    if mask.sum() == 0:
        return 0.0
    yt, yp = y_true[mask], y_pred[mask]
    return float(f1_score(yt, yp, labels=np.unique(yt), average="macro", zero_division=0))


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
            "macro_f1": macro_f1(yt, yp),  # over the species in this group only
            "top_1_accuracy": top_k_accuracy(yt, p, k=1),
            "n_samples": int(mask.sum()),
            "n_species": int(np.unique(yt[yt >= 0]).size),
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


def genus_probs(
    probs: np.ndarray,
    species_to_genus: dict[int, str],
    genus_names: list[str],
) -> np.ndarray:
    """Sum species probabilities within each genus. Shape [n, n_genera]."""
    genus_idx = {g: i for i, g in enumerate(genus_names)}
    out = np.zeros((probs.shape[0], len(genus_names)), dtype=probs.dtype)
    for species_id in range(probs.shape[1]):
        g = genus_idx.get(species_to_genus.get(species_id))
        if g is not None:
            out[:, g] += probs[:, species_id]
    return out


def genus_accuracy(
    genus_true: np.ndarray,
    g_probs: np.ndarray,
    genus_names: list[str],
) -> dict:
    """Genus accuracy, plus the mean of per-genus recall so big genera don't dominate."""
    mask = genus_true >= 0
    if mask.sum() == 0:
        return {"accuracy": None, "macro_accuracy": None, "per_genus_recall": {}, "n_samples": 0}
    yt = genus_true[mask]
    correct = g_probs[mask].argmax(axis=1) == yt
    recall = {
        name: float(correct[yt == i].mean())
        for i, name in enumerate(genus_names) if (yt == i).any()
    }
    return {
        "accuracy": float(correct.mean()),
        "macro_accuracy": float(np.mean(list(recall.values()))),
        "per_genus_recall": recall,
        "n_samples": int(mask.sum()),
    }


def error_decomposition(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    species_to_genus: dict[int, str],
) -> dict:
    """Split known-species predictions into: right, sibling species, wrong genus."""
    mask = y_true >= 0
    n = int(mask.sum())
    if n == 0:
        return {"correct": None, "right_genus_wrong_species": None, "wrong_genus": None, "n_samples": 0}
    yt, yp = y_true[mask], y_pred[mask]
    correct = yt == yp
    same_genus = np.array([
        species_to_genus.get(int(t)) == species_to_genus.get(int(p)) for t, p in zip(yt, yp)
    ])
    return {
        "correct": float(correct.mean()),
        "right_genus_wrong_species": float((~correct & same_genus).mean()),
        "wrong_genus": float((~same_genus).mean()),
        "n_samples": n,
    }


def bootstrap_ci(
    stat_fn,
    n_rows: int,
    n_samples: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    strata: np.ndarray | None = None,
) -> dict:
    """Percentile bootstrap interval for a statistic over rows (observations).

    `strata` (one value per row, e.g. the true species) resamples within each stratum,
    so every stratum keeps its size in every draw. Use it for macro F1: a plain
    resample often drops species with one or two val observations, and since those are
    the hardest, the draws come out too high (E0: plain [0.555, 0.617] around 0.570,
    stratified by species [0.545, 0.588]).

    `stat_fn(idx)` computes the statistic on the rows at `idx` (an int array, with
    repeats). For a paired comparison of two predictors on the same observations,
    pass a function that returns their difference: every resample then uses the same
    rows for both, which is what makes the interval paired.

    Returns {"value", "lo", "hi", "n", "n_samples"}; value is stat_fn on all rows.
    """
    if n_rows == 0:
        return {"value": None, "lo": None, "hi": None, "n": 0, "n_samples": n_samples}
    rng = np.random.default_rng(seed)
    full = float(stat_fn(np.arange(n_rows)))
    groups = None
    if strata is not None:
        strata = np.asarray(strata)
        if len(strata) != n_rows:
            raise ValueError(f"strata has {len(strata)} rows, expected {n_rows}")
        groups = [np.flatnonzero(strata == s) for s in np.unique(strata)]
    draws = np.empty(n_samples)
    for i in range(n_samples):
        if groups is None:
            idx = rng.integers(0, n_rows, size=n_rows)
        else:
            idx = np.concatenate([g[rng.integers(0, len(g), size=len(g))] for g in groups])
        draws[i] = stat_fn(idx)
    lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
    return {"value": full, "lo": float(lo), "hi": float(hi), "n": int(n_rows),
            "n_samples": int(n_samples)}
