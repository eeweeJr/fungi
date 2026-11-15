"""Tests for observation-level split handling: photo selection, summary, split gate,
bootstrap intervals."""

import numpy as np
import pandas as pd
import pytest

from fungai.data.leakage import mixed_known_unknown_observations
from fungai.data.observations import ObservationIndex, observation_summary, select_eval_photos
from fungai.data.split_gate import SplitGateError, run_split_gate
from fungai.data.splits import SplitProtocol
from fungai.eval.metrics import bootstrap_ci, macro_f1


def _frame(obs_photos: dict, species="A a", genus="A", unknown=False):
    rows = []
    for obs, n in obs_photos.items():
        for i in range(n):
            rows.append({"observation_id": obs, "filename": f"{i}-{obs}.JPG",
                         "species": species, "genus": genus, "is_unknown": unknown})
    return pd.DataFrame(rows)


def test_select_eval_photos_caps_deterministically():
    df = _frame({1: 6, 2: 1, 3: 3}).sample(frac=1, random_state=0)
    out = select_eval_photos(df, 4)
    assert out.groupby("observation_id").size().to_dict() == {1: 4, 2: 1, 3: 3}
    assert out[out.observation_id == 1].filename.tolist() == [f"{i}-1.JPG" for i in range(4)]
    again = select_eval_photos(df.sample(frac=1, random_state=1), 4)
    pd.testing.assert_frame_equal(out, again)


def test_observation_summary_counts_cap():
    s = observation_summary(_frame({1: 6, 2: 1, 3: 3}), max_photos=4)
    assert s["observations"] == 3 and s["photos"] == 10
    assert s["observations_truncated"] == 1 and s["photos_dropped_by_cap"] == 2
    assert s["photos_per_observation_hist"] == {"1": 1, "3": 1, "6": 1}
    assert s["known_observations"] == 3


def test_observation_index_groups_rows():
    df = _frame({7: 2, 3: 3}).sample(frac=1, random_state=0).reset_index(drop=True)
    idx = ObservationIndex(df)
    assert len(idx) == 2
    assert list(idx.observation_ids) == [3, 7]
    assert df.loc[idx.rows(0), "filename"].tolist() == ["0-3.JPG", "1-3.JPG", "2-3.JPG"]


def _protocol():
    return SplitProtocol(name="open", subset="mini", known_species=["A a"],
                         unknown_val_species=["A b"], unknown_test_species=["A c"],
                         species_to_genus={"A a": "A", "A b": "A", "A c": "A"})


def _clean_frames():
    return {
        "train": _frame({1: 2, 2: 1}),
        "val": pd.concat([_frame({10: 2}), _frame({11: 1}, species="A b", unknown=True)]),
        "test": pd.concat([_frame({20: 1}), _frame({21: 2}, species="A c", unknown=True)]),
    }


def test_split_gate_passes_clean_frames():
    run_split_gate(_protocol(), _clean_frames())


@pytest.mark.parametrize("breakage", ["shared_obs", "two_species", "mixed_unknown",
                                      "unknown_genus", "excluded_present"])
def test_split_gate_fails_each_invariant(breakage):
    frames, proto = _clean_frames(), _protocol()
    if breakage == "shared_obs":
        frames["val"] = pd.concat([frames["val"], _frame({1: 1})])
    elif breakage == "two_species":
        bad = _frame({12: 2})
        bad.loc[1, "species"] = "A z"
        frames["val"] = pd.concat([frames["val"], bad])
    elif breakage == "mixed_unknown":
        bad = _frame({13: 2})
        bad.loc[1, ["species", "is_unknown"]] = ["A b", True]
        frames["val"] = pd.concat([frames["val"], bad])
    elif breakage == "unknown_genus":
        frames["val"] = pd.concat([frames["val"], _frame({14: 1}, species="A b", genus="Z", unknown=True)])
    elif breakage == "excluded_present":
        proto.excluded_files = ["0-1.JPG"]
    with pytest.raises(SplitGateError):
        run_split_gate(proto, frames)


def test_mixed_known_unknown_detects():
    df = _frame({1: 2})
    df.loc[1, "is_unknown"] = True
    assert mixed_known_unknown_observations(df) == [1]


def test_bootstrap_ci_covers_value_and_paired_difference():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 400)
    good = np.where(rng.random(400) < 0.8, y, (y + 1) % 5)
    bad = np.where(rng.random(400) < 0.5, y, (y + 1) % 5)
    ci = bootstrap_ci(lambda i: macro_f1(y[i], good[i]), 400, n_samples=200, seed=1)
    assert ci["lo"] <= ci["value"] <= ci["hi"]
    diff = bootstrap_ci(lambda i: macro_f1(y[i], good[i]) - macro_f1(y[i], bad[i]), 400,
                        n_samples=200, seed=1)
    assert diff["lo"] > 0
    assert bootstrap_ci(lambda i: 0.0, 0)["value"] is None


@pytest.mark.data
def test_real_open_snapshot_passes_split_gate():
    """The real open-protocol splits are safe per observation (checked during planning)."""
    from pathlib import Path

    from fungai.data.snapshot import load_snapshot

    path = Path("data/processed/snapshots")
    if not path.exists() or not any(path.iterdir()):
        pytest.skip("no snapshot on disk")
    proto_path = Path("data/processed/splits/mini_open_seed42_h30_t2.json")
    if not proto_path.exists():
        pytest.skip("no open protocol on disk")
    df = load_snapshot("latest", image_paths=False)
    frames = {s: df[df["split"] == s] for s in ("train", "val", "test")}
    run_split_gate(SplitProtocol.load(proto_path), frames)
