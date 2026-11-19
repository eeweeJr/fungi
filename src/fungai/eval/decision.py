"""Decision rule: name a species, name only a genus, or say "unknown".

    species  if p(top species) >= tau_species
    genus    else if p(top genus) >= tau_genus     (the genus that wins on its own)
    unknown  otherwise

Both thresholds are set on val so that `known_retention` of known-species
observations clear each one. Fit them after temperature scaling.
"""

from __future__ import annotations

import numpy as np

OUTCOMES_KNOWN = ("species_correct", "species_wrong", "genus_correct", "genus_wrong", "unknown")
OUTCOMES_HELD_OUT = ("species_named", "genus_correct", "genus_wrong", "unknown")


def fit_thresholds(species_probs: np.ndarray, genus_probs: np.ndarray, labels: np.ndarray,
                   known_retention: float = 0.95) -> dict:
    known = labels >= 0
    if not known.any():
        return {"tau_species": 1.0, "tau_genus": 1.0, "known_retention": known_retention}
    q = 1.0 - known_retention
    return {
        "tau_species": float(np.quantile(species_probs[known].max(axis=1), q)),
        "tau_genus": float(np.quantile(genus_probs[known].max(axis=1), q)),
        "known_retention": known_retention,
    }


def decide(species_probs: np.ndarray, genus_probs: np.ndarray, tau_species: float,
           tau_genus: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (decision ["species"|"genus"|"unknown"], species prediction, genus prediction)."""
    sp_pred, g_pred = species_probs.argmax(axis=1), genus_probs.argmax(axis=1)
    decision = np.where(species_probs.max(axis=1) >= tau_species, "species",
                        np.where(genus_probs.max(axis=1) >= tau_genus, "genus", "unknown"))
    return decision, sp_pred, g_pred


def outcome_labels(decision: np.ndarray, sp_pred: np.ndarray, g_pred: np.ndarray,
                   labels: np.ndarray, genus_labels: np.ndarray) -> np.ndarray:
    """Per-observation outcome. Held-out species (label -1) can never be named
    correctly, so naming any species is a failure for them."""
    out = np.empty(len(labels), dtype=object)
    known = labels >= 0
    is_sp, is_g = decision == "species", decision == "genus"
    g_ok = g_pred == genus_labels
    out[known & is_sp & (sp_pred == labels)] = "species_correct"
    out[known & is_sp & (sp_pred != labels)] = "species_wrong"
    out[~known & is_sp] = "species_named"
    out[is_g & g_ok] = "genus_correct"
    out[is_g & ~g_ok] = "genus_wrong"
    out[decision == "unknown"] = "unknown"
    return out


def outcome_table(outcomes: np.ndarray, labels: np.ndarray) -> dict:
    known = labels >= 0
    table = {}
    for name, mask, keys in (("known", known, OUTCOMES_KNOWN), ("held_out", ~known, OUTCOMES_HELD_OUT)):
        n = int(mask.sum())
        table[name] = {"n": n, **{k: (float((outcomes[mask] == k).mean()) if n else None) for k in keys}}
    return table
