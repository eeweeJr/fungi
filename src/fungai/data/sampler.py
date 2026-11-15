"""Training sampler that draws observations, not images.

An observation is one mushroom photographed one or more times. Drawing by image
lets a 16-photo observation count 16 times; drawing by observation and then taking
one random photo of it counts every mushroom once, and still cycles through all of
its photos over the epochs.

Class balance is tempered with `alpha`: an observation of a species with n training
observations is drawn with weight n ** -alpha.

  alpha = 0    natural distribution (a plain shuffle of observations)
  alpha = 0.5  species drawn in proportion to sqrt(n): lifts the tail without
               repeating a handful of rare observations hundreds of times
  alpha = 1    every species equally likely
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Sampler

SAMPLER_ALPHAS = {"observation": 0.0, "sqrt": 0.5, "balanced": 1.0}


class ObservationSampler(Sampler[int]):
    """Yields image positions: one random image from each drawn observation."""

    def __init__(
        self,
        observation_ids: Sequence,
        class_labels: Sequence,
        alpha: float = 0.5,
        num_samples: int | None = None,
        seed: int = 0,
    ):
        if len(observation_ids) != len(class_labels):
            raise ValueError("observation_ids and class_labels must be the same length")
        obs_codes, _ = pd.factorize(np.asarray(observation_ids))
        self._order = np.argsort(obs_codes, kind="stable")  # image positions grouped by observation
        self._counts = np.bincount(obs_codes)
        self._starts = np.concatenate([[0], np.cumsum(self._counts)[:-1]])

        self.obs_classes = np.asarray(class_labels)[self._order[self._starts]]
        _, class_codes = np.unique(self.obs_classes, return_inverse=True)
        class_sizes = np.bincount(class_codes)
        self.alpha = alpha
        self.weights = torch.as_tensor(class_sizes[class_codes].astype(np.float64) ** -alpha)
        self.num_samples = num_samples or len(self._counts)
        self._generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        n_obs = len(self._counts)
        if self.alpha == 0 and self.num_samples == n_obs:
            obs = torch.randperm(n_obs, generator=self._generator)
        else:
            obs = torch.multinomial(self.weights, self.num_samples, replacement=True,
                                    generator=self._generator)
        obs = obs.numpy()
        counts = self._counts[obs]
        offsets = (torch.rand(len(obs), generator=self._generator).numpy() * counts).astype(np.int64)
        offsets = np.minimum(offsets, counts - 1)
        yield from self._order[self._starts[obs] + offsets].tolist()

    def expected_class_share(self) -> dict:
        """Expected fraction of draws per class, for logging and tests."""
        w = self.weights.numpy()
        shares = pd.Series(w).groupby(self.obs_classes).sum() / w.sum()
        return shares.to_dict()


def build_train_sampler(train_df: pd.DataFrame, mode: str, seed: int = 0) -> ObservationSampler | None:
    """Sampler for a config mode; None means the plain per-image shuffle."""
    if mode == "none":
        return None
    if mode not in SAMPLER_ALPHAS:
        raise ValueError(f"unknown sampler {mode!r}, expected 'none' or one of {list(SAMPLER_ALPHAS)}")
    return ObservationSampler(
        train_df["observation_id"].to_numpy(),
        train_df["species"].astype(object).to_numpy(),
        alpha=SAMPLER_ALPHAS[mode],
        seed=seed,
    )


class ObservationLevelSampler(Sampler[int]):
    """Yields observation indices (one item = one observation) for an
    ObservationDataset, weighted like ObservationSampler: an observation of a species
    with n training observations is drawn with weight n ** -alpha."""

    def __init__(self, observation_species: Sequence, alpha: float = 0.5,
                 num_samples: int | None = None, seed: int = 0):
        self.obs_classes = np.asarray(observation_species, dtype=object)
        _, class_codes = np.unique(self.obs_classes.astype(str), return_inverse=True)
        class_sizes = np.bincount(class_codes)
        self.alpha = alpha
        self.weights = torch.as_tensor(class_sizes[class_codes].astype(np.float64) ** -alpha)
        self.num_samples = num_samples or len(self.obs_classes)
        self._generator = torch.Generator().manual_seed(seed)

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.obs_classes)
        if self.alpha == 0 and self.num_samples == n:
            yield from torch.randperm(n, generator=self._generator).tolist()
            return
        yield from torch.multinomial(self.weights, self.num_samples, replacement=True,
                                     generator=self._generator).tolist()

    def expected_class_share(self) -> dict:
        w = self.weights.numpy()
        shares = pd.Series(w).groupby(self.obs_classes.astype(str)).sum() / w.sum()
        return shares.to_dict()


def build_observation_sampler(dataset, mode: str, seed: int = 0) -> ObservationLevelSampler | None:
    """Observation-level sampler for an ObservationDataset; None means a plain shuffle."""
    if mode == "none":
        return None
    if mode not in SAMPLER_ALPHAS:
        raise ValueError(f"unknown sampler {mode!r}, expected 'none' or one of {list(SAMPLER_ALPHAS)}")
    return ObservationLevelSampler(dataset.observation_species, alpha=SAMPLER_ALPHAS[mode], seed=seed)
