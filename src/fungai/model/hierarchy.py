"""Prediction heads: hierarchical genus x species-within-genus, and a flat ablation.

Both return the same dict, so the rest of the system does not care which is used:
    genus_logits [B, G], species_logits [B, S]     raw, for temperature scaling
    genus_logp [B, G]                              log p(genus)
    species_cond_logp [B, S]                       log p(species | its genus)
    species_logp [B, S]                            log p(species), sums to 1 over S
"""

from __future__ import annotations

import torch
from torch import nn


def build_species_genus_index(species_map: dict[str, int], species_to_genus: dict[str, str],
                              genus_map: dict[str, int]) -> torch.Tensor:
    """Long tensor [S]: genus index of each species label. Every genus must have at
    least one known species, or its within-genus softmax would be empty."""
    idx = torch.full((len(species_map),), -1, dtype=torch.long)
    for species, label in species_map.items():
        genus = species_to_genus[species]
        if genus not in genus_map:
            raise KeyError(f"genus {genus!r} of species {species!r} not in the genus map")
        idx[label] = genus_map[genus]
    if (idx < 0).any():
        raise ValueError("species labels are not contiguous")
    empty = set(range(len(genus_map))) - set(idx.tolist())
    if empty:
        raise ValueError(f"genera with no known species: {sorted(empty)}")
    return idx


def membership(species_genus: torch.Tensor, n_genus: int) -> torch.Tensor:
    """Bool [G, S]: species s belongs to genus g."""
    return torch.arange(n_genus, device=species_genus.device)[:, None] == species_genus[None, :]


def hierarchical_logp(genus_logits: torch.Tensor, species_logits: torch.Tensor,
                      species_genus: torch.Tensor, n_genus: int,
                      t_genus: float = 1.0, t_species: float = 1.0) -> dict:
    """p(species) = p(genus) * p(species | genus), each softmax at its own temperature."""
    genus_logp = torch.log_softmax(genus_logits / t_genus, dim=-1)
    s = species_logits / t_species
    member = membership(species_genus, n_genus)
    masked = s.unsqueeze(1).masked_fill(~member.unsqueeze(0), float("-inf"))
    lse = torch.logsumexp(masked, dim=-1)                      # [B, G]
    cond = s - lse[:, species_genus]
    return {
        "genus_logits": genus_logits, "species_logits": species_logits,
        "genus_logp": genus_logp, "species_cond_logp": cond,
        "species_logp": genus_logp[:, species_genus] + cond,
    }


def flat_logp(species_logits: torch.Tensor, species_genus: torch.Tensor, n_genus: int,
              t_species: float = 1.0) -> dict:
    """One softmax over species; genus by summing its species."""
    species_logp = torch.log_softmax(species_logits / t_species, dim=-1)
    member = membership(species_genus, n_genus)
    masked = species_logp.unsqueeze(1).masked_fill(~member.unsqueeze(0), float("-inf"))
    genus_logp = torch.logsumexp(masked, dim=-1)
    return {
        "genus_logits": genus_logp, "species_logits": species_logits,
        "genus_logp": genus_logp, "species_cond_logp": species_logp - genus_logp[:, species_genus],
        "species_logp": species_logp,
    }


class HierarchicalHead(nn.Module):
    kind = "hierarchical"

    def __init__(self, dim: int, species_genus: torch.Tensor, n_genus: int):
        super().__init__()
        self.register_buffer("species_genus", species_genus.clone(), persistent=False)
        self.n_genus = n_genus
        self.genus = nn.Linear(dim, n_genus)
        self.species = nn.Linear(dim, len(species_genus))

    def forward(self, x: torch.Tensor) -> dict:
        return hierarchical_logp(self.genus(x), self.species(x), self.species_genus, self.n_genus)


class FlatHead(nn.Module):
    kind = "flat"

    def __init__(self, dim: int, species_genus: torch.Tensor, n_genus: int):
        super().__init__()
        self.register_buffer("species_genus", species_genus.clone(), persistent=False)
        self.n_genus = n_genus
        self.species = nn.Linear(dim, len(species_genus))

    def forward(self, x: torch.Tensor) -> dict:
        return flat_logp(self.species(x), self.species_genus, self.n_genus)


def build_head(kind: str, dim: int, species_genus: torch.Tensor, n_genus: int) -> nn.Module:
    if kind == "hierarchical":
        return HierarchicalHead(dim, species_genus, n_genus)
    if kind == "flat":
        return FlatHead(dim, species_genus, n_genus)
    raise ValueError(f"unknown head {kind!r}")
