"""Leakage checks between train, val and test.

Every check takes plain dataframes and returns the offending items, so an empty
result means clean. They are used by the test suite (on planted leaks and on the
real data) and by `python -m fungai.data.validate`.

Image identity is checked two ways:
  md5     byte-identical files
  phash   a 256-bit difference hash, which also matches a photo that was resized
          or re-encoded. Unrelated photos differ in roughly half the bits; the
          default threshold of 8 bits only matches copies of the same picture.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations, pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from fungai.data.ingest import DATA_DIR

HASH_CACHE = DATA_DIR / "processed" / "image_hashes.csv"
NEAR_DUPLICATE_BITS = 8


# --------------------------------------------------------------------------- #
# Metadata-level checks
# --------------------------------------------------------------------------- #

def shared_values(dfs: dict[str, pd.DataFrame], column: str) -> dict[str, list]:
    """Values of `column` (observation_id, filename, ...) present in more than one split."""
    out: dict[str, list] = {}
    for a, b in combinations(dfs, 2):
        shared = set(dfs[a][column].dropna()) & set(dfs[b][column].dropna())
        if shared:
            out[f"{a}&{b}"] = sorted(shared, key=str)
    return out


def shared_events(dfs: dict[str, pd.DataFrame], decimals: int = 4) -> dict[str, list]:
    """Same day and same spot (about 10 m) in more than one split.

    Catches one outing being divided across splits even when the photos were
    filed as separate observations.
    """
    def keys(df: pd.DataFrame) -> set[str]:
        ok = df["date"].notna() & df["latitude"].notna() & df["longitude"].notna()
        d = df[ok]
        return set(
            pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d") + "|" + d["latitude"].round(decimals).astype(str)
            + "|" + d["longitude"].round(decimals).astype(str)
        )

    out: dict[str, list] = {}
    for a, b in combinations(dfs, 2):
        shared = keys(dfs[a]) & keys(dfs[b])
        if shared:
            out[f"{a}&{b}"] = sorted(shared)
    return out


def temporal_violations(dfs: dict[str, pd.DataFrame], order=("train", "val", "test")) -> list[str]:
    """The official split is by year: each split must end before the next begins."""
    out = []
    present = [s for s in order if s in dfs and "year" in dfs[s].columns]
    for earlier, later in pairwise(present):
        last, first = dfs[earlier]["year"].max(), dfs[later]["year"].min()
        if last >= first:
            out.append(f"{earlier} runs to {int(last)} but {later} starts at {int(first)}")
    return out


def inconsistent_observations(df: pd.DataFrame) -> list:
    """Observations whose images disagree on species: one mushroom, two labels."""
    n_species = df.groupby("observation_id")["species"].agg(lambda s: s.astype(str).nunique())
    return n_species[n_species > 1].index.tolist()


def mixed_known_unknown_observations(df: pd.DataFrame) -> list:
    """Observations whose photos are partly known and partly held-out species."""
    if "is_unknown" not in df.columns:
        return []
    n = df.groupby("observation_id")["is_unknown"].nunique()
    return n[n > 1].index.tolist()


def inconsistent_genus_observations(df: pd.DataFrame) -> list:
    """Observations whose images disagree on genus."""
    n = df.groupby("observation_id")["genus"].agg(lambda s: s.astype(str).nunique())
    return n[n > 1].index.tolist()


def observation_split_violations(protocol, dfs: dict[str, pd.DataFrame]) -> list[str]:
    """Invariants an observation-level model relies on, over protocol-applied splits.

    - no observation in more than one split
    - every observation has one species and one genus
    - no observation mixes known and held-out species
    - every held-out species belongs to a genus the model knows
    - no excluded file is still present
    - the protocol's own checks (`protocol_violations`)
    Empty list means the splits are safe to use per observation.
    """
    out: list[str] = []
    for pair, ids in shared_values(dfs, "observation_id").items():
        out.append(f"observations in more than one split ({pair}): {len(ids)}, e.g. {ids[:3]}")
    known_genera = set(protocol.genus_names)
    excluded = set(getattr(protocol, "excluded_files", []) or [])
    for name, df in dfs.items():
        for label, bad in (
            ("more than one species", inconsistent_observations(df)),
            ("more than one genus", inconsistent_genus_observations(df)),
            ("mixed known and held-out species", mixed_known_unknown_observations(df)),
        ):
            if bad:
                out.append(f"{name}: {len(bad)} observations with {label}, e.g. {bad[:3]}")
        if "is_unknown" in df.columns:
            unknown_genera = set(df.loc[df["is_unknown"], "genus"].astype(str)) - known_genera
            if unknown_genera:
                out.append(f"{name}: held-out species in genera the model does not know: "
                           f"{sorted(unknown_genera)}")
        if excluded and "filename" in df.columns:
            still = set(df["filename"]) & excluded
            if still:
                out.append(f"{name}: {len(still)} excluded files still present")
    out.extend(protocol_violations(protocol, dfs))
    return out


def protocol_violations(protocol, dfs: dict[str, pd.DataFrame]) -> list[str]:
    """Check protocol-applied splits: held-out species must never reach train, and
    each unknown group must appear only in its own eval split."""
    known = set(protocol.known_species)
    unknown_val, unknown_test = set(protocol.unknown_val_species), set(protocol.unknown_test_species)
    species = {name: set(df["species"].astype(str)) for name, df in dfs.items()}
    out = []

    def check(bad: set, message: str) -> None:
        if bad:
            out.append(f"{message}: {sorted(bad)[:5]}{' ...' if len(bad) > 5 else ''}")

    check(unknown_val & unknown_test, "species in both unknown groups")
    check((unknown_val | unknown_test) & known, "held-out species also listed as known")
    check(set(protocol.species_map) - known, "label map contains a species that is not known")
    if "train" in dfs:
        check(species["train"] & (unknown_val | unknown_test), "held-out species in train")
        check(species["train"] - known, "train species missing from the label map")
        if "is_unknown" in dfs["train"].columns and dfs["train"]["is_unknown"].any():
            out.append("train contains rows flagged is_unknown")
    if "val" in dfs:
        check(species["val"] & unknown_test, "test-only unknown species in val")
    if "test" in dfs:
        check(species["test"] & unknown_val, "val-only unknown species in test")
    for name in ("val", "test"):
        if name in dfs and "is_unknown" in dfs[name].columns:
            flagged = set(dfs[name].loc[dfs[name]["is_unknown"], "species"].astype(str))
            check(flagged & known, f"known species flagged unknown in {name}")
            check(species[name] - flagged - known, f"unknown species not flagged in {name}")
    return out


# --------------------------------------------------------------------------- #
# Image hashing
# --------------------------------------------------------------------------- #

def _hash_file(path: str) -> tuple[str, str]:
    md5 = hashlib.md5(Path(path).read_bytes()).hexdigest()
    with Image.open(path) as img:
        px = np.asarray(img.convert("L").resize((17, 16), Image.BILINEAR), dtype=np.int16)
    phash = np.packbits((px[:, 1:] > px[:, :-1]).flatten()).tobytes().hex()
    return md5, phash


def hash_images(paths: list[str], cache_path: Path | None = HASH_CACHE, workers: int = 8) -> pd.DataFrame:
    """md5 and phash for each path, in order. Results are cached by path and file size."""
    sizes = [Path(p).stat().st_size for p in paths]
    cache = pd.DataFrame(columns=["image_path", "size", "md5", "phash"])
    if cache_path is not None and Path(cache_path).exists():
        cache = pd.read_csv(cache_path)
    known = {(p, s): (m, h) for p, s, m, h in cache.itertuples(index=False)}

    todo = [(p, s) for p, s in dict.fromkeys(zip(paths, sizes)) if (p, s) not in known]
    if todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for key, result in zip(todo, ex.map(_hash_file, [p for p, _ in todo])):
                known[key] = result
        if cache_path is not None:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                [(p, s, m, h) for (p, s), (m, h) in known.items()], columns=cache.columns
            ).to_csv(cache_path, index=False)

    values = [known[key] for key in zip(paths, sizes)]
    return pd.DataFrame(values, columns=["md5", "phash"])


def with_hashes(dfs: dict[str, pd.DataFrame], cache_path: Path | None = HASH_CACHE) -> pd.DataFrame:
    """Stack the splits into one frame with `split`, `md5` and `phash` columns."""
    frames = []
    for name, df in dfs.items():
        hashes = hash_images(df["image_path"].tolist(), cache_path)
        frames.append(df.reset_index(drop=True).assign(split=name, md5=hashes["md5"], phash=hashes["phash"]))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# Image-level checks
# --------------------------------------------------------------------------- #

def duplicate_groups(hashed: pd.DataFrame) -> pd.DataFrame:
    """One row per set of byte-identical images, with how far each set is spread."""
    grouped = hashed.assign(species=hashed["species"].astype(str)).groupby("md5")
    groups = grouped.agg(
        n_images=("md5", "size"),
        n_splits=("split", "nunique"),
        n_observations=("observation_id", "nunique"),
        n_species=("species", "nunique"),
        splits=("split", lambda s: sorted(set(s))),
        species=("species", lambda s: sorted(set(s))),
        files=("image_path", list),
    )
    return groups[groups["n_images"] > 1].reset_index()


def conflicting_label_files(hashed: pd.DataFrame) -> list[str]:
    """Filenames of byte-identical images that carry more than one species label.

    The same photo attached to observations of two species: at least one label is
    wrong and there is no telling which, so neither copy can be trusted.
    """
    species_per_image = hashed.groupby("md5")["species"].transform(lambda s: s.astype(str).nunique())
    return sorted(hashed.loc[species_per_image > 1, "filename"])


def near_duplicate_pairs(
    a: pd.DataFrame, b: pd.DataFrame, max_bits: int = NEAR_DUPLICATE_BITS, chunk: int = 100,
) -> pd.DataFrame:
    """Pairs (one image from `a`, one from `b`) whose phashes differ by at most `max_bits`."""
    def words(df: pd.DataFrame) -> np.ndarray:
        raw = np.frombuffer(bytes.fromhex("".join(df["phash"])), dtype=np.uint64)
        return raw.reshape(len(df), -1)

    rows = []
    if len(a) and len(b):
        ha, hb = words(a), words(b)
        for start in range(0, len(ha), chunk):
            dist = np.bitwise_count(ha[start:start + chunk, None, :] ^ hb[None, :, :]).sum(axis=-1)
            for i, j in zip(*np.where(dist <= max_bits)):
                rows.append((a["image_path"].iloc[start + i], b["image_path"].iloc[j], int(dist[i, j])))
    return pd.DataFrame(rows, columns=["image_a", "image_b", "bits_different"])


def cross_split_near_duplicates(hashed: pd.DataFrame, max_bits: int = NEAR_DUPLICATE_BITS) -> pd.DataFrame:
    """Near-identical images that sit in different splits."""
    frames = []
    for a, b in combinations(hashed["split"].unique(), 2):
        pairs = near_duplicate_pairs(hashed[hashed["split"] == a], hashed[hashed["split"] == b], max_bits)
        frames.append(pairs.assign(split_a=a, split_b=b))
    return pd.concat(frames, ignore_index=True)
