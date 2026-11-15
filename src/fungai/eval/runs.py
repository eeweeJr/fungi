"""Load a finished run for error analysis.

    from fungai.eval.runs import load_run
    df = load_run("outputs/fungai-dev_v0.4")
    df[~df.correct & ~df.is_unknown].groupby("genus").size()

Each row is one validation observation: the snapshot's metadata (species, genus,
habitat, location, revisited_site, ...) joined with what the model predicted.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fungai.data.snapshot import SNAPSHOT_DIR, load_snapshot


def load_run(run_dir: str | Path, split: str = "val", root: Path = SNAPSHOT_DIR,
             with_probs: bool = False) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    run_dir = Path(run_dir)
    pointer = run_dir / "data_snapshot.json"
    if not pointer.exists():
        raise FileNotFoundError(f"{pointer} missing: this run predates data snapshots")
    snapshot_id = json.loads(pointer.read_text())["snapshot"]
    preds = pd.read_parquet(run_dir / f"{split}_predictions.parquet")

    meta = load_snapshot(snapshot_id, split, root=root)
    meta["observation_id"] = meta["observation_id"].astype(str)
    images = meta.groupby("observation_id")["image_path"].agg(list) if "image_path" in meta else None
    meta = meta.drop_duplicates("observation_id").drop(columns=["image_path"], errors="ignore")
    meta = meta.drop(columns=["label", "genus_label", "is_unknown"])  # the predictions carry these

    preds["observation_id"] = preds["observation_id"].astype(str)
    df = preds.merge(meta, on="observation_id", how="left", validate="one_to_one")
    missing = df["species"].isna().sum()
    if missing:
        raise ValueError(f"{missing} predicted observations are not in snapshot {snapshot_id}")
    if images is not None:
        df["image_paths"] = df["observation_id"].map(images)
    if with_probs:
        return df, np.load(run_dir / f"{split}_probs.npy")
    return df
