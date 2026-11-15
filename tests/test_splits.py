"""Tests for fungai.data.splits and fungai.data.sampler. Synthetic metadata only."""

from collections import Counter

import numpy as np
import pandas as pd
import pytest

from fungai.data.sampler import ObservationSampler, build_train_sampler
from fungai.data.splits import (
    SplitProtocol,
    build_protocol,
    flag_revisited_sites,
    select_holdout_species,
)

GENERA = {"Big": 20, "Mid": 10, "Tiny": 2}  # Tiny is below the hold-out minimum
TOXIC = {"Big_00", "Big_07", "Mid_03"}


def _make_split(rng, obs_per_species, start_id, imgs_per_obs=2):
    rows = []
    obs_id = start_id
    for genus, n_species in GENERA.items():
        for i in range(n_species):
            species = f"{genus}_{i:02d}"
            for _ in range(obs_per_species + i):  # frequency grows with the index
                lat, lon = rng.uniform(55, 57), rng.uniform(9, 12)
                for _ in range(imgs_per_obs):
                    rows.append({
                        "observation_id": obs_id, "species": species, "genus": genus,
                        "toxicity": int(species in TOXIC), "latitude": lat, "longitude": lon,
                    })
                obs_id += 1
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def splits():
    rng = np.random.default_rng(0)
    return (_make_split(rng, 6, 0), _make_split(rng, 3, 100_000), _make_split(rng, 3, 200_000))


@pytest.fixture(scope="module")
def protocol(splits):
    return build_protocol("open", *splits, n_holdout=8, n_toxic=2, split_seed=1)


def test_holdout_counts_and_disjoint_groups(protocol):
    val, test = set(protocol.unknown_val_species), set(protocol.unknown_test_species)
    assert len(val) == len(test) == 4
    assert not val & test
    assert not (val | test) & set(protocol.known_species)
    assert len(protocol.known_species) == 32 - 8


def test_holdout_respects_toxic_quota_and_small_genera(protocol):
    held = protocol.unknown_val_species + protocol.unknown_test_species
    assert sum(s in TOXIC for s in held) == 2
    assert sum(s in TOXIC for s in protocol.unknown_val_species) == 1
    assert not any(s.startswith("Tiny") for s in held)
    by_genus = Counter(s.split("_")[0] for s in held)
    assert by_genus["Big"] > by_genus["Mid"] > 0  # proportional to genus size


def test_holdout_is_deterministic_per_seed(splits):
    a = select_holdout_species(*splits, n_holdout=8, n_toxic=2, seed=1)[:2]
    b = select_holdout_species(*splits, n_holdout=8, n_toxic=2, seed=1)[:2]
    c = select_holdout_species(*splits, n_holdout=8, n_toxic=2, seed=2)[:2]
    assert a == b
    assert a != c


def test_holdout_skips_species_missing_from_eval(splits):
    train, val, test = splits
    val_missing = val[val["species"] != "Big_05"]
    for seed in range(5):
        uv, ut, _ = select_holdout_species(train, val_missing, test, n_holdout=8, n_toxic=2, seed=seed)
        assert "Big_05" not in uv + ut


def test_apply_keeps_unknowns_out_of_train(protocol, splits):
    train, val, test = splits
    held = set(protocol.unknown_val_species) | set(protocol.unknown_test_species)
    tr = protocol.apply(train, "train")
    assert not set(tr["species"]) & held
    assert not tr["is_unknown"].any()

    va = protocol.apply(val, "val")
    assert set(va.loc[va["is_unknown"], "species"]) == set(protocol.unknown_val_species)
    assert not set(va["species"]) & set(protocol.unknown_test_species)

    te = protocol.apply(test, "test")
    assert set(te.loc[te["is_unknown"], "species"]) == set(protocol.unknown_test_species)
    assert not set(te["species"]) & set(protocol.unknown_val_species)

    # no observation is shared between any two splits
    ids = [set(d["observation_id"]) for d in (tr, va, te)]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]


def test_label_and_genus_maps(protocol):
    assert sorted(protocol.species_map.values()) == list(range(len(protocol.known_species)))
    assert protocol.genus_names == ["Big", "Mid", "Tiny"]
    # a held-out species still resolves to a known genus
    assert protocol.species_to_genus[protocol.unknown_val_species[0]] in protocol.genus_map


def test_closed_protocol_keeps_everything(splits):
    closed = build_protocol("closed", splits[0])
    assert len(closed.known_species) == 32
    assert closed.unknown_val_species == [] and closed.unknown_test_species == []
    assert len(closed.apply(splits[1], "val")) == len(splits[1])


def test_protocol_save_load_roundtrip(protocol, tmp_path):
    path = tmp_path / "p.json"
    protocol.save(path)
    assert SplitProtocol.load(path) == protocol


def test_flag_revisited_sites():
    train = pd.DataFrame({"species": ["a", "b"], "latitude": [55.1234, 56.0], "longitude": [10.0, 11.0]})
    ev = pd.DataFrame({
        "species": ["a", "a", "b", "a"],
        "latitude": [55.1231, 55.2, 55.1234, np.nan],   # same cell / elsewhere / other species / no coords
        "longitude": [10.0002, 10.0, 10.0, 10.0],
    })
    assert flag_revisited_sites(ev, train).tolist() == [True, False, False, False]


# --- sampler ---


def _sampler_inputs():
    # class "big": 90 observations, class "small": 10; every observation has 3 images
    obs = np.repeat(np.arange(100), 3)
    cls = np.where(obs < 90, "big", "small")
    return obs, cls


def test_sampler_yields_one_image_per_observation_when_natural():
    obs, cls = _sampler_inputs()
    sampler = ObservationSampler(obs, cls, alpha=0.0, seed=0)
    idx = list(sampler)
    assert len(idx) == len(sampler) == 100
    assert sorted(obs[idx].tolist()) == list(range(100))  # each observation exactly once


def test_sampler_cycles_through_images_of_an_observation():
    obs, cls = _sampler_inputs()
    sampler = ObservationSampler(obs, cls, alpha=0.0, seed=0)
    seen = {i for _ in range(30) for i in sampler if obs[i] == 0}
    assert seen == {0, 1, 2}


@pytest.mark.parametrize("alpha, expected_small", [(0.0, 0.10), (0.5, 0.25), (1.0, 0.50)])
def test_sampler_class_share(alpha, expected_small):
    obs, cls = _sampler_inputs()
    sampler = ObservationSampler(obs, cls, alpha=alpha, num_samples=20_000, seed=0)
    assert sampler.expected_class_share()["small"] == pytest.approx(expected_small)
    drawn = cls[list(sampler)]
    assert (drawn == "small").mean() == pytest.approx(expected_small, abs=0.02)


def test_sampler_handles_unsorted_observation_ids():
    obs = np.array([7, 3, 7, 3, 9])
    cls = np.array(["x", "y", "x", "y", "x"])
    sampler = ObservationSampler(obs, cls, alpha=0.0, seed=0)
    assert sorted(obs[list(sampler)].tolist()) == [3, 7, 9]


def test_build_train_sampler_modes():
    df = pd.DataFrame({"observation_id": [1, 1, 2, 3], "species": ["a", "a", "a", "b"]})
    assert build_train_sampler(df, "none") is None
    assert len(build_train_sampler(df, "sqrt")) == 3
    with pytest.raises(ValueError):
        build_train_sampler(df, "nope")
