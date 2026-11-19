"""Losses for the observation model. Every term is returned so it can be logged."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def observation_loss(out: dict, label: torch.Tensor, genus_label: torch.Tensor,
                     head_kind: str, fusion_kind: str, photo_loss_weight: float) -> dict:
    """total and its parts.

    hierarchical: genus NLL + species-within-genus NLL. For a known species this sum
    equals the NLL of the joint p(species), so it trains the same target as a flat
    head while giving each level its own signal.
    flat: species NLL (genus NLL reported, not trained on).
    fusion none: only the per-photo loss trains; observation terms are reported.
    photo: cross-entropy of each real photo against its observation's label.
    Rows with label -1 (held-out species) contribute only to the genus term.
    """
    known = label >= 0
    zero = out["species_logp"].sum() * 0.0
    genus_ok = genus_label >= 0
    loss_genus = F.nll_loss(out["genus_logp"][genus_ok], genus_label[genus_ok]) if genus_ok.any() else zero
    if known.any():
        loss_cond = F.nll_loss(out["species_cond_logp"][known], label[known])
        loss_species = F.nll_loss(out["species_logp"][known], label[known])
    else:
        loss_cond = loss_species = zero

    photo_labels = label[out["photo_obs_index"]]
    photo_known = photo_labels >= 0
    loss_photo = (F.cross_entropy(out["photo_logits"][photo_known], photo_labels[photo_known])
                  if photo_known.any() else zero)

    if fusion_kind == "none":
        obs_term = zero
    elif head_kind == "hierarchical":
        obs_term = loss_genus + loss_cond
    else:
        obs_term = loss_species
    total = obs_term + photo_loss_weight * loss_photo
    return {"total": total, "genus": loss_genus.detach(), "species_cond": loss_cond.detach(),
            "species": loss_species.detach(), "photo": loss_photo.detach()}
