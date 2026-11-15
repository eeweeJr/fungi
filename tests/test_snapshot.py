"""Tests for split snapshots, the predictions table, run loading and MLflow logging."""

import json

import mlflow
import numpy as np
import pandas as pd
import pytest

from fungai.data.snapshot import (
    frame_digest,
    list_snapshots,
    load_snapshot,
    prepare_splits,
    read_manifest,
    resolve_snapshot,
    verify_snapshot,
    write_snapshot,
)
from fungai.data.splits import build_protocol
from fungai.eval.report import predictions_table
from fungai.eval.runs import load_run
from tests.test_splits import _make_split


def _raw(seed=0):
    rng = np.random.default_rng(seed)
    raw = {"train": _make_split(rng, 6, 0), "val": _make_split(rng, 3, 100_000),
           "test": _make_split(rng, 3, 200_000)}
    for name, df in raw.items():
        df["filename"] = [f"{i % 2}-{o}.jpg" for i, o in enumerate(df["observation_id"])]
        df["date"] = pd.Timestamp({"train": 2021, "val": 2022, "test": 2023}[name], 9, 1)
        df["genus"] = df["genus"].astype("category")
        df["image_path"] = "/somewhere/" + df["filename"]
        # like land_cover: a categorical whose values are floats, which parquet
        # reads back as plain float64
        df["land_cover"] = pd.Categorical(rng.choice([10.0, 12.0, np.nan], len(df)))
    return raw


@pytest.fixture
def built(tmp_path):
    raw = _raw()
    protocol = build_protocol("open", raw["train"], raw["val"], raw["test"], n_holdout=8, split_seed=1)
    frames = prepare_splits(protocol, raw)
    path = write_snapshot(frames, protocol, root=tmp_path)
    return path, frames, protocol, raw


def test_prepare_splits_derived_columns(built):
    _, frames, protocol, _ = built
    for split, df in frames.items():
        assert (df["split"] == split).all()
        assert ((df["label"] == -1) == df["is_unknown"]).all()
        assert (df["genus_label"] >= 0).all()
    assert not frames["train"]["revisited_site"].any()
    assert frames["train"]["label"].nunique() == len(protocol.species_map)


def test_snapshot_round_trips_exactly(built):
    path, frames, _, _ = built
    for split, df in frames.items():
        loaded = load_snapshot(path, split, root=path.parent, image_paths=False)
        assert frame_digest(loaded) == frame_digest(df)
        assert len(loaded) == len(df)
        assert "image_path" not in loaded.columns  # machine specific: never stored
    assert verify_snapshot(path, root=path.parent) == []


def test_snapshot_id_is_content_addressed(built, tmp_path):
    path, _, protocol, raw = built
    mtime = (path / "train.parquet").stat().st_mtime_ns
    again = write_snapshot(prepare_splits(protocol, raw), protocol, root=tmp_path)
    assert again == path
    assert (path / "train.parquet").stat().st_mtime_ns == mtime  # reused, not rewritten

    other = build_protocol("open", raw["train"], raw["val"], raw["test"], n_holdout=8, split_seed=2)
    assert write_snapshot(prepare_splits(other, raw), other, root=tmp_path) != path


def test_digest_ignores_machine_paths_and_column_order(built):
    _, frames, _, _ = built
    df = frames["val"]
    moved = df.assign(image_path="/another/machine/" + df["filename"])
    assert frame_digest(moved) == frame_digest(df)
    assert frame_digest(df[df.columns[::-1]]) == frame_digest(df)
    changed = df.copy()
    changed.loc[0, "species"] = "Something else"
    assert frame_digest(changed) != frame_digest(df)


def test_verify_detects_tampering(built):
    path, _, _, _ = built
    df = pd.read_parquet(path / "val.parquet")
    df.loc[0, "label"] = 999
    df.to_parquet(path / "val.parquet", index=False)
    assert verify_snapshot(path, root=path.parent) == ["val: content does not match the manifest digest"]
    (path / "test.parquet").unlink()
    assert "test: file missing" in verify_snapshot(path, root=path.parent)


def test_manifest_records_protocol_and_counts(built):
    path, frames, protocol, _ = built
    m = read_manifest(path, root=path.parent)
    assert m["id"] == path.name
    assert m["protocol"]["unknown_val_species"] == protocol.unknown_val_species
    assert m["splits"]["val"]["rows"] == len(frames["val"])
    assert m["splits"]["val"]["unknown_observations"] == frames["val"].drop_duplicates(
        "observation_id")["is_unknown"].sum()
    assert set(m["git"]) == {"commit", "dirty"}


def test_list_resolve_and_load_all(built):
    path, frames, _, _ = built
    listing = list_snapshots(path.parent)
    assert listing["id"].tolist() == [path.name]
    assert resolve_snapshot("latest", path.parent) == path
    assert resolve_snapshot(path.name, path.parent) == path
    everything = load_snapshot("latest", root=path.parent, image_paths=False)
    assert everything.groupby("split").size().to_dict() == {s: len(f) for s, f in frames.items()}
    with pytest.raises(FileNotFoundError):
        resolve_snapshot("nope", path.parent)
    with pytest.raises(FileNotFoundError):
        resolve_snapshot("latest", path.parent / "empty")


# --- predictions and runs ---


def _predictions():
    probs = np.array([
        [0.7, 0.2, 0.1], [0.5, 0.4, 0.1],   # obs a: label 0, correct
        [0.1, 0.6, 0.3],                    # obs b: label 2, predicted 1, true at rank 2
        [0.4, 0.3, 0.3],                    # obs c: unknown
    ])
    return {
        "probs": probs, "labels": np.array([0, 0, 2, -1]),
        "observation_ids": ["a", "a", "b", "c"], "genus_labels": np.array([0, 0, 1, 1]),
        "image_paths": ["", "", "", ""],
    }


def test_predictions_table():
    table, probs = predictions_table(_predictions(), {"s0": 0, "s1": 1, "s2": 2},
                                     {0: "g0", 1: "g1", 2: "g1"}, ["g0", "g1"], top_k=2)
    assert table["observation_id"].tolist() == ["a", "b", "c"]
    assert table["n_images"].tolist() == [2, 1, 1]
    assert table["correct"].tolist() == [True, False, False]
    assert table["true_rank"].tolist() == [1, 2, -1]
    assert table["is_unknown"].tolist() == [False, False, True]
    assert table["pred_species"].tolist() == ["s0", "s1", "s0"]
    assert table["top_k_labels"].iloc[1] == [1, 2]
    # c is an unknown species, but its genus (g1) still wins once species are summed: 0.6 vs 0.4
    assert table["genus_correct"].tolist() == [True, True, True]
    assert table["genus_confidence_summed"].iloc[2] == pytest.approx(0.6)
    assert table["confidence"].iloc[0] == pytest.approx(0.6)
    assert probs.shape == (3, 3)


def test_load_run_joins_predictions_to_snapshot(built, tmp_path):
    path, frames, protocol, _ = built
    val = frames["val"]
    obs = val.drop_duplicates("observation_id")
    n_cls = len(protocol.species_map)
    probs = np.full((len(val), n_cls), 1.0 / n_cls)
    predictions = {"probs": probs, "labels": val["label"].to_numpy(),
                   "observation_ids": val["observation_id"].astype(str).tolist(),
                   "genus_labels": val["genus_label"].to_numpy()}
    table, obs_probs = predictions_table(predictions, protocol.species_map,
                                         {i: protocol.species_to_genus[s] for s, i in protocol.species_map.items()},
                                         protocol.genus_names)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    table.to_parquet(run_dir / "val_predictions.parquet", index=False)
    np.save(run_dir / "val_probs.npy", obs_probs)
    (run_dir / "data_snapshot.json").write_text(json.dumps({"snapshot": path.name}))

    df, p = load_run(run_dir, root=path.parent, with_probs=True)
    assert len(df) == len(obs) == len(p)
    assert (df["species"].isin(protocol.unknown_val_species) == df["is_unknown"]).all()
    assert {"species", "genus", "latitude", "revisited_site", "pred_species", "confidence"} <= set(df.columns)


def test_load_run_rejects_foreign_observations(built, tmp_path):
    path, _, _, _ = built
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pd.DataFrame({"observation_id": ["not-in-snapshot"], "label": [0], "is_unknown": [False]}).to_parquet(
        run_dir / "val_predictions.parquet")
    (run_dir / "data_snapshot.json").write_text(json.dumps({"snapshot": path.name}))
    with pytest.raises(ValueError, match="not in snapshot"):
        load_run(run_dir, root=path.parent)


def test_log_data_snapshot_records_inputs(built, tmp_path):
    from fungai.tracking.mlflow_utils import log_data_snapshot, setup_experiment

    path, _, _, _ = built
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    setup_experiment("test-snapshot")
    with mlflow.start_run() as run:
        log_data_snapshot(path)
    logged = mlflow.get_run(run.info.run_id)
    assert logged.data.tags["data.snapshot"] == path.name
    names = sorted(d.dataset.name for d in logged.inputs.dataset_inputs)
    assert names == sorted(f"{path.name}/{s}" for s in ("train", "val", "test"))
    contexts = {t.value for d in logged.inputs.dataset_inputs for t in d.tags}
    assert {"train", "val", "test"} <= contexts
