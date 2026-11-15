"""Leakage tests.

Two halves:
  - synthetic: plant each kind of leak and check the detector catches it, so a
    passing real-data check actually means something
  - real data (marked `data`, skipped when the dataset is not downloaded): the
    splits the training script actually builds must be clean

Skip the real-data half with:  pytest -m "not data"
"""

import shutil

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fungai.data import leakage
from fungai.data.sampler import build_train_sampler
from fungai.data.splits import SplitProtocol, build_protocol

# --------------------------------------------------------------------------- #
# Synthetic: each planted leak is caught
# --------------------------------------------------------------------------- #


def _noise_image(path, seed, size=64):
    rng = np.random.default_rng(seed)
    # smooth random blobs, so resizing and re-encoding keep the structure
    small = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
    Image.fromarray(small).resize((size, size), Image.BICUBIC).save(path, quality=95)
    return str(path)


def _frame(tmp_path, split, year, n_obs, start, species=("a", "b")):
    rows = []
    for i in range(n_obs):
        obs = start + i
        for k in range(2):
            name = f"{k}-{obs}.jpg"
            rows.append({
                "observation_id": obs, "filename": name,
                "image_path": _noise_image(tmp_path / name, seed=obs * 10 + k),
                "species": species[i % len(species)], "genus": "g",
                "year": year, "date": pd.Timestamp(year, 9, 1 + i),
                "latitude": 55.0 + obs * 0.01, "longitude": 10.0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def clean(tmp_path):
    return {
        "train": _frame(tmp_path, "train", 2021, 4, 100),
        "val": _frame(tmp_path, "val", 2022, 3, 200),
        "test": _frame(tmp_path, "test", 2023, 3, 300),
    }


def _hashed(dfs):
    return leakage.with_hashes(dfs, cache_path=None)


def test_clean_synthetic_splits_pass_every_check(clean):
    assert leakage.shared_values(clean, "observation_id") == {}
    assert leakage.shared_values(clean, "filename") == {}
    assert leakage.shared_events(clean) == {}
    assert leakage.temporal_violations(clean) == []
    hashed = _hashed(clean)
    assert leakage.duplicate_groups(hashed).empty
    assert leakage.cross_split_near_duplicates(hashed).empty


def test_detects_observation_split_across_train_and_val(clean):
    moved = clean["train"].iloc[[0]]  # one of an observation's two photos lands in val
    clean["val"] = pd.concat([clean["val"], moved], ignore_index=True)
    assert leakage.shared_values(clean, "observation_id") == {"train&val": [100]}
    assert leakage.shared_values(clean, "filename") == {"train&val": ["0-100.jpg"]}


def test_detects_same_outing_filed_as_two_observations(clean):
    # a different observation id, but the same day and the same spot as a train row
    row = clean["test"].iloc[[0]].copy()
    first = clean["train"].iloc[0]
    row["date"], row["longitude"] = first["date"], first["longitude"]
    row["latitude"] = first["latitude"] + 0.00002  # about 2 m away
    clean["test"] = pd.concat([clean["test"].iloc[1:], row], ignore_index=True)
    assert list(leakage.shared_events(clean)) == ["train&test"]
    assert leakage.shared_values(clean, "observation_id") == {}  # invisible to the id check


def test_detects_year_overlap(clean):
    clean["val"].loc[0, "year"] = 2021
    assert leakage.temporal_violations(clean) == ["train runs to 2021 but val starts at 2021"]


def test_detects_byte_identical_image_in_another_split(clean, tmp_path):
    src = clean["train"]["image_path"].iloc[0]
    copy = tmp_path / "renamed-copy.jpg"
    shutil.copyfile(src, copy)
    clean["test"].loc[0, "image_path"] = str(copy)
    groups = leakage.duplicate_groups(_hashed(clean))
    assert len(groups) == 1
    assert groups.iloc[0]["splits"] == ["test", "train"]
    assert groups.iloc[0]["n_splits"] == 2


def test_detects_resized_and_reencoded_copy(clean, tmp_path):
    src = clean["train"]["image_path"].iloc[0]
    copy = tmp_path / "reencoded.jpg"
    Image.open(src).resize((40, 40)).save(copy, quality=60)
    clean["val"].loc[0, "image_path"] = str(copy)
    hashed = _hashed(clean)
    assert leakage.duplicate_groups(hashed).empty  # the bytes differ, so md5 misses it
    pairs = leakage.cross_split_near_duplicates(hashed)
    assert len(pairs) == 1
    assert {pairs.iloc[0]["image_a"], pairs.iloc[0]["image_b"]} == {src, str(copy)}


def test_identical_image_with_two_species_is_reported(clean, tmp_path):
    src = clean["train"]["image_path"].iloc[0]  # species "a"
    copy = tmp_path / "relabelled.jpg"
    shutil.copyfile(src, copy)
    clean["train"].loc[2, "image_path"] = str(copy)  # an observation of species "b"
    groups = leakage.duplicate_groups(_hashed(clean))
    assert groups.iloc[0]["n_species"] == 2
    assert groups.iloc[0]["species"] == ["a", "b"]
    hashed = _hashed(clean)
    flagged = leakage.conflicting_label_files(hashed)
    assert flagged == sorted([clean["train"]["filename"].iloc[0], clean["train"]["filename"].iloc[2]])
    # a protocol built with them excluded drops both copies
    protocol = build_protocol("closed", clean["train"])
    protocol.excluded_files = flagged
    assert not set(protocol.apply(clean["train"], "train")["filename"]) & set(flagged)


def test_detects_observation_with_two_species(clean):
    clean["train"].loc[0, "species"] = "b"  # observation 100 is otherwise species "a"
    assert leakage.inconsistent_observations(clean["train"]) == [100]
    assert leakage.inconsistent_observations(clean["val"]) == []


def test_hash_cache_reuses_and_invalidates(tmp_path, monkeypatch):
    path = _noise_image(tmp_path / "x.jpg", seed=1)
    cache = tmp_path / "hashes.csv"
    first = leakage.hash_images([path], cache)
    calls = []
    real = leakage._hash_file
    monkeypatch.setattr(leakage, "_hash_file", lambda p: calls.append(p) or real(p))
    assert leakage.hash_images([path], cache).equals(first)
    assert calls == []  # served from the cache
    Image.new("RGB", (30, 30), "red").save(path)  # the file changed on disk
    assert leakage.hash_images([path], cache)["md5"][0] != first["md5"][0]
    assert calls == [path]


# --- protocol ---


def _protocol_frames():
    def frame(species, start):
        return pd.DataFrame({"observation_id": range(start, start + len(species)), "species": species})

    protocol = SplitProtocol("open", "x", known_species=["k1", "k2"],
                             unknown_val_species=["uv"], unknown_test_species=["ut"])
    raw = {"train": frame(["k1", "k2", "uv", "ut"], 0), "val": frame(["k1", "uv", "ut"], 10),
           "test": frame(["k2", "uv", "ut"], 20)}
    return protocol, raw


def test_protocol_applied_splits_are_clean():
    protocol, raw = _protocol_frames()
    applied = {name: protocol.apply(df, name) for name, df in raw.items()}
    assert leakage.protocol_violations(protocol, applied) == []


def test_detects_unfiltered_splits():
    protocol, raw = _protocol_frames()
    problems = leakage.protocol_violations(protocol, raw)
    assert any("held-out species in train" in p for p in problems)
    assert any("test-only unknown species in val" in p for p in problems)
    assert any("val-only unknown species in test" in p for p in problems)


def test_detects_broken_protocol_definitions():
    protocol, _ = _protocol_frames()
    protocol.unknown_test_species = ["uv"]
    assert any("both unknown groups" in p for p in leakage.protocol_violations(protocol, {}))
    protocol, _ = _protocol_frames()
    protocol.known_species.append("uv")
    assert any("also listed as known" in p for p in leakage.protocol_violations(protocol, {}))


def test_detects_wrong_unknown_flags():
    protocol, raw = _protocol_frames()
    val = protocol.apply(raw["val"], "val")
    val["is_unknown"] = ~val["is_unknown"]
    problems = leakage.protocol_violations(protocol, {"val": val})
    assert any("known species flagged unknown" in p for p in problems)
    assert any("unknown species not flagged" in p for p in problems)


# --------------------------------------------------------------------------- #
# Real data: the splits the training script builds
# --------------------------------------------------------------------------- #

SUBSET, SIZE = "mini", "300"


@pytest.fixture(scope="module")
def raw_splits():
    from fungai.data.ingest import load_metadata

    try:
        dfs = {s: load_metadata(SUBSET, s, SIZE) for s in ("train", "val", "test")}
    except FileNotFoundError:
        pytest.skip("FungiTastic metadata not downloaded")
    if any("image_path" not in df.columns for df in dfs.values()):
        pytest.skip("FungiTastic images not downloaded")
    return dfs


@pytest.fixture(scope="module", params=["open", "closed"])
def protocol_splits(request, raw_splits, hashed):
    """Built fresh from the metadata, not read from the cached manifest."""
    excluded = leakage.conflicting_label_files(hashed) if request.param == "open" else []
    protocol = build_protocol(request.param, raw_splits["train"], raw_splits["val"], raw_splits["test"],
                              excluded_files=excluded)
    return protocol, {name: protocol.apply(df, name) for name, df in raw_splits.items()}


@pytest.fixture(scope="module")
def hashed(raw_splits):
    return leakage.with_hashes(raw_splits)


@pytest.mark.data
def test_real_no_observation_or_file_in_two_splits(raw_splits):
    assert leakage.shared_values(raw_splits, "observation_id") == {}
    assert leakage.shared_values(raw_splits, "filename") == {}
    assert leakage.shared_values(raw_splits, "image_path") == {}


@pytest.mark.data
def test_real_splits_are_separated_in_time_and_place(raw_splits):
    assert leakage.temporal_violations(raw_splits) == []
    assert leakage.shared_events(raw_splits) == {}


@pytest.mark.data
def test_real_every_observation_has_one_species(raw_splits):
    for name, df in raw_splits.items():
        assert leakage.inconsistent_observations(df) == [], name


@pytest.mark.data
def test_real_no_identical_image_in_two_splits(hashed):
    groups = leakage.duplicate_groups(hashed)
    crossing = groups[groups["n_splits"] > 1]
    assert crossing.empty, crossing[["splits", "files"]].to_string()


@pytest.mark.data
def test_real_no_near_identical_image_in_two_splits(hashed):
    pairs = leakage.cross_split_near_duplicates(hashed)
    assert pairs.empty, pairs.head(20).to_string()


@pytest.mark.data
def test_real_protocol_has_no_violations(protocol_splits):
    protocol, dfs = protocol_splits
    assert leakage.protocol_violations(protocol, dfs) == []
    assert leakage.shared_values(dfs, "observation_id") == {}


@pytest.mark.data
def test_real_no_image_of_a_heldout_species_reaches_train(protocol_splits, hashed):
    """Even via a duplicate: a photo of a held-out species must not be in train
    under a known species' label."""
    protocol, dfs = protocol_splits
    held = set(protocol.unknown_val_species) | set(protocol.unknown_test_species)
    held_md5 = set(hashed.loc[hashed["species"].astype(str).isin(held), "md5"])
    train_md5 = set(hashed.loc[hashed["image_path"].isin(dfs["train"]["image_path"]), "md5"])
    assert not held_md5 & train_md5


@pytest.mark.data
def test_real_open_protocol_has_no_identical_image_with_two_species(raw_splits, hashed):
    excluded = leakage.conflicting_label_files(hashed)
    protocol = build_protocol("open", raw_splits["train"], raw_splits["val"], raw_splits["test"],
                              excluded_files=excluded)
    kept = pd.concat(protocol.apply(df, name)["filename"] for name, df in raw_splits.items())
    assert leakage.conflicting_label_files(hashed[hashed["filename"].isin(kept)]) == []


@pytest.mark.data
def test_real_cached_manifest_matches_a_fresh_build(raw_splits, hashed):
    """The manifest on disk is what training reads; it must not have drifted."""
    from fungai.data.splits import manifest_path

    path = manifest_path(SUBSET, "open")
    if not path.exists():
        pytest.skip("no cached open manifest yet")
    fresh = build_protocol("open", raw_splits["train"], raw_splits["val"], raw_splits["test"],
                           excluded_files=leakage.conflicting_label_files(hashed))
    assert SplitProtocol.load(path) == fresh


@pytest.mark.data
def test_real_labels_match_the_protocol(protocol_splits):
    protocol, dfs = protocol_splits
    labels = {name: df["species"].map(protocol.species_map) for name, df in dfs.items()}
    assert labels["train"].notna().all()  # every train row has a class
    assert set(labels["train"].astype(int)) == set(range(len(protocol.species_map)))  # no empty class
    for name in ("val", "test"):
        assert (labels[name].isna() == dfs[name]["is_unknown"]).all()  # unknown <=> label -1
        genus = dfs[name]["genus"].astype(str).map(protocol.genus_map)
        assert genus.notna().all()  # every eval row, unknowns included, has a known genus


@pytest.mark.data
def test_real_sampler_only_draws_train_rows(protocol_splits):
    protocol, dfs = protocol_splits
    train = dfs["train"]
    sampler = build_train_sampler(train, "sqrt", seed=0)
    idx = np.fromiter(sampler, dtype=int)
    assert idx.min() >= 0 and idx.max() < len(train)
    drawn = train.iloc[idx]
    assert drawn["species"].isin(protocol.known_species).all()
    eval_obs = set(dfs["val"]["observation_id"]) | set(dfs["test"]["observation_id"])
    assert not set(drawn["observation_id"]) & eval_obs
