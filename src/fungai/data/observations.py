"""Observation-level views of a split: which photos are used, and how the split is shaped.

An observation is one mushroom photographed one or more times. The observation model
predicts once per observation from at most `max_photos` photos. `select_eval_photos`
is the single rule for which photos those are at evaluation time. It is used by the
observation dataset *and* by the per-image baseline's evaluation, so both are scored
on exactly the same photos.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_eval_photos(df: pd.DataFrame, max_photos: int) -> pd.DataFrame:
    """Deterministically keep the first `max_photos` photos of each observation,
    ordered by filename. Row order of the result is (observation_id, filename)."""
    if max_photos < 1:
        raise ValueError(f"max_photos must be >= 1, got {max_photos}")
    ordered = df.sort_values(["observation_id", "filename"], kind="stable")
    rank = ordered.groupby("observation_id", sort=False).cumcount()
    return ordered[rank < max_photos].reset_index(drop=True)


def photos_per_observation(df: pd.DataFrame) -> pd.Series:
    return df.groupby("observation_id").size()


def observation_summary(df: pd.DataFrame, max_photos: int | None = None) -> dict:
    """Shape of a split at the observation level: counts, photo-count histogram,
    known vs held-out observations, and how many the photo cap truncates."""
    counts = photos_per_observation(df)
    hist = counts.clip(upper=10).value_counts().sort_index()
    out = {
        "observations": len(counts),
        "photos": len(df),
        "photos_per_observation_mean": round(float(counts.mean()), 4) if len(counts) else 0.0,
        "multi_photo_fraction": round(float((counts >= 2).mean()), 4) if len(counts) else 0.0,
        # key "10" means 10 or more
        "photos_per_observation_hist": {str(int(k)): int(v) for k, v in hist.items()},
    }
    if "is_unknown" in df.columns:
        unk = df.groupby("observation_id")["is_unknown"].first()
        out["known_observations"] = int((~unk).sum())
        out["unknown_observations"] = int(unk.sum())
    if max_photos is not None:
        over = (counts - max_photos).clip(lower=0)
        out["max_photos"] = int(max_photos)
        out["observations_truncated"] = int((counts > max_photos).sum())
        out["photos_dropped_by_cap"] = int(over.sum())
    return out


class ObservationIndex:
    """Precomputed grouping of a frame's rows by observation, for O(1) lookups.

    `rows(i)` gives the row positions of the i-th observation, sorted by filename.
    Observations are ordered by observation_id.
    """

    def __init__(self, df: pd.DataFrame):
        order = np.lexsort((df["filename"].astype(str).to_numpy(), df["observation_id"].to_numpy()))
        obs_sorted = df["observation_id"].to_numpy()[order]
        boundaries = np.flatnonzero(np.r_[True, obs_sorted[1:] != obs_sorted[:-1]])
        self.order = order
        self.starts = boundaries
        self.counts = np.diff(np.r_[boundaries, len(order)])
        self.observation_ids = obs_sorted[boundaries]

    def __len__(self) -> int:
        return len(self.starts)

    def rows(self, i: int) -> np.ndarray:
        s = self.starts[i]
        return self.order[s: s + self.counts[i]]
