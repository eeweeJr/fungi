"""Split protocols: which species are known, and which are held out as unknowns.

The official FungiTastic splits are by year (train <= 2021, val 2022, test 2023) and
never share an observation, so the row-level split is kept as is. A protocol decides
the label space on top of it:

  closed  every train species is a class (the published benchmark setting)
  open    some species are removed from train entirely and only appear at eval
          time as unknowns (label -1). They are split into two disjoint groups:
          one seen in val (to tune a rejection threshold) and one seen only in
          test (to report on), so the threshold is never tuned on the species it
          is scored on.

The held-out choice is frozen in a manifest under data/processed/splits/ and is
keyed by its own seed, so changing the experiment seed never moves the label space.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fungai.data.ingest import DATA_DIR, DEFAULT_DEST, load_metadata

SPLITS_DIR = DATA_DIR / "processed" / "splits"
PROTOCOLS = ("open", "closed")
_TOXIC_VALUES = ("1", "1.0", "poisonous", "toxic", "true")


def is_toxic_value(value) -> bool:
    return str(value).lower() in _TOXIC_VALUES


@dataclass
class SplitProtocol:
    name: str
    subset: str
    known_species: list[str]
    unknown_val_species: list[str] = field(default_factory=list)
    unknown_test_species: list[str] = field(default_factory=list)
    species_to_genus: dict[str, str] = field(default_factory=dict)  # includes held-out species
    toxic_species: list[str] = field(default_factory=list)
    split_seed: int | None = None
    holdout_details: list[dict] = field(default_factory=list)
    # Images dropped from every split: identical photos filed under two species.
    excluded_files: list[str] = field(default_factory=list)

    @property
    def species_map(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(sorted(self.known_species))}

    @property
    def genus_names(self) -> list[str]:
        return sorted({self.species_to_genus[s] for s in self.known_species})

    @property
    def genus_map(self) -> dict[str, int]:
        return {g: i for i, g in enumerate(self.genus_names)}

    def apply(self, df: pd.DataFrame, split: str) -> pd.DataFrame:
        """Filter one official split down to the rows this protocol allows.

        train: known species only. val/test: everything except the unknown group
        reserved for the other split. Adds an `is_unknown` column.
        """
        if split not in ("train", "val", "test"):
            raise ValueError(f"unknown split {split!r}")
        species = df["species"].astype(object)
        known = species.isin(self.known_species)
        if split == "train":
            keep = known
        else:
            reserved = self.unknown_test_species if split == "val" else self.unknown_val_species
            keep = ~species.isin(reserved)
        if self.excluded_files:
            keep &= ~df["filename"].isin(self.excluded_files)
        out = df[keep].copy()
        out["species"] = species[keep]
        out["is_unknown"] = ~known[keep]
        return out.reset_index(drop=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1))

    @classmethod
    def load(cls, path: Path) -> SplitProtocol:
        return cls(**json.loads(path.read_text()))


# --------------------------------------------------------------------------- #
# Held-out species selection
# --------------------------------------------------------------------------- #

def _species_table(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """One row per train species: genus, toxicity, observation counts per split."""
    def obs_counts(df: pd.DataFrame) -> pd.Series:
        return df.groupby(df["species"].astype(object))["observation_id"].nunique()

    sp = train_df["species"].astype(object)
    table = pd.DataFrame({
        "genus": train_df["genus"].astype(object).groupby(sp).first(),
        "toxic": train_df["toxicity"].map(is_toxic_value).groupby(sp).any()
        if "toxicity" in train_df.columns else False,
        "train_obs": obs_counts(train_df),
    })
    table["val_obs"] = obs_counts(val_df).reindex(table.index).fillna(0).astype(int)
    table["test_obs"] = obs_counts(test_df).reindex(table.index).fillna(0).astype(int)
    return table.sort_index()


def _genus_quotas(n_species: pd.Series, capacity: pd.Series, total: int) -> dict[str, int]:
    """Share `total` across genera in proportion to species count, capped by capacity."""
    if capacity.sum() < total:
        raise ValueError(f"cannot hold out {total} species: only {int(capacity.sum())} are eligible")
    ideal = n_species / n_species.sum() * total
    quota = np.minimum(np.floor(ideal), capacity).astype(int)
    while quota.sum() < total:
        room = quota < capacity
        quota[(ideal - quota).where(room, -np.inf).idxmax()] += 1
    return quota.to_dict()


def select_holdout_species(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_holdout: int = 30,
    n_toxic: int = 2,
    seed: int = 42,
    min_genus_species: int = 5,
    min_eval_obs: int = 3,
) -> tuple[list[str], list[str], pd.DataFrame]:
    """Pick species to hold out, returned as (unknown_val, unknown_test, details).

    - spread across genera in proportion to their species count; genera with fewer
      than `min_genus_species` species are left intact
    - spread across the frequency range within each genus (one pick per quantile bin)
    - exactly `n_toxic` toxic species, split between the two groups
    - only species with at least `min_eval_obs` observations in both val and test,
      so every held-out species is actually measurable whichever group it lands in
    """
    rng = np.random.default_rng(seed)
    table = _species_table(train_df, val_df, test_df)
    genus_size = table.groupby("genus").size()
    eligible = table[
        table["genus"].map(genus_size).ge(min_genus_species)
        & table["val_obs"].ge(min_eval_obs)
        & table["test_obs"].ge(min_eval_obs)
    ]
    genera = sorted(eligible["genus"].unique())
    capacity = eligible.groupby("genus").size().reindex(genera)
    quotas = _genus_quotas(genus_size.reindex(genera), capacity, n_holdout)

    toxic_picks: list[str] = []
    for sp in rng.permutation(eligible.index[eligible["toxic"]].to_numpy()):
        if len(toxic_picks) == n_toxic:
            break
        genus = eligible.at[sp, "genus"]
        if quotas[genus] > 0:
            quotas[genus] -= 1
            toxic_picks.append(sp)
    if len(toxic_picks) < n_toxic:
        raise ValueError(f"wanted {n_toxic} toxic hold-outs, only {len(toxic_picks)} eligible")

    picks: list[str] = []
    for genus in genera:
        pool = eligible[(eligible["genus"] == genus) & ~eligible["toxic"]]
        pool = pool.sort_values(["train_obs"], kind="stable")
        if quotas[genus] > len(pool):
            raise ValueError(f"not enough non-toxic eligible species in {genus}")
        for chunk in np.array_split(pool.index.to_numpy(), quotas[genus]) if quotas[genus] else []:
            picks.append(rng.choice(chunk))

    # Picks are ordered by genus then frequency; a coin flip inside each consecutive
    # pair gives both groups the same genus and frequency spread.
    groups: tuple[list[str], list[str]] = ([], [])
    for ordered in (toxic_picks, picks):
        for i in range(0, len(ordered), 2):
            pair = ordered[i:i + 2]
            first = int(rng.integers(2)) if len(pair) == 2 else len(groups[0]) > len(groups[1])
            for j, sp in enumerate(pair):
                groups[(first + j) % 2].append(str(sp))

    unknown_val, unknown_test = sorted(groups[0]), sorted(groups[1])
    details = table.loc[unknown_val + unknown_test].reset_index(names="species")
    details.insert(1, "group", ["val"] * len(unknown_val) + ["test"] * len(unknown_test))
    return unknown_val, unknown_test, details


# --------------------------------------------------------------------------- #
# Building and caching
# --------------------------------------------------------------------------- #

def build_protocol(
    name: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
    subset: str = "mini",
    split_seed: int = 42,
    n_holdout: int = 30,
    n_toxic: int = 2,
    excluded_files: list[str] | None = None,
) -> SplitProtocol:
    if name not in PROTOCOLS:
        raise ValueError(f"unknown protocol {name!r}, expected one of {PROTOCOLS}")
    sp = train_df["species"].astype(object)
    all_species = sorted(sp.dropna().unique().tolist())
    species_to_genus = train_df["genus"].astype(object).groupby(sp).first().to_dict()
    toxic = (
        sorted(sp[train_df["toxicity"].map(is_toxic_value)].unique().tolist())
        if "toxicity" in train_df.columns else []
    )
    if name == "closed":
        return SplitProtocol(name, subset, all_species, species_to_genus=species_to_genus,
                             toxic_species=toxic)
    if val_df is None or test_df is None:
        raise ValueError("the open protocol needs val and test metadata to pick hold-outs")
    unknown_val, unknown_test, details = select_holdout_species(
        train_df, val_df, test_df, n_holdout=n_holdout, n_toxic=n_toxic, seed=split_seed,
    )
    held_out = set(unknown_val) | set(unknown_test)
    return SplitProtocol(
        name, subset,
        known_species=[s for s in all_species if s not in held_out],
        unknown_val_species=unknown_val,
        unknown_test_species=unknown_test,
        species_to_genus=species_to_genus,
        toxic_species=toxic,
        split_seed=split_seed,
        holdout_details=json.loads(details.to_json(orient="records")),
        excluded_files=list(excluded_files or []),
    )


def find_conflicting_label_files(dfs: dict[str, pd.DataFrame]) -> list[str]:
    """Hash every image (cached) and return the files to exclude. Needs the images."""
    from fungai.data import leakage

    if any("image_path" not in df.columns for df in dfs.values()):
        raise FileNotFoundError(
            "images are needed to build the open protocol (duplicate check); run ingest first"
        )
    return leakage.conflicting_label_files(leakage.with_hashes(dfs))


def manifest_path(subset: str, name: str, split_seed: int = 42,
                  n_holdout: int = 30, n_toxic: int = 2) -> Path:
    if name == "closed":
        return SPLITS_DIR / f"{subset}_closed.json"
    return SPLITS_DIR / f"{subset}_open_seed{split_seed}_h{n_holdout}_t{n_toxic}.json"


def load_or_build_protocol(
    subset: str = "mini",
    name: str = "open",
    split_seed: int = 42,
    n_holdout: int = 30,
    n_toxic: int = 2,
    dest: Path = DEFAULT_DEST,
    size: str = "300",
) -> SplitProtocol:
    """Load the frozen manifest for these settings, building it on first use.

    The closed protocol is the official benchmark and is left untouched. The open
    protocol also drops identical photos that are filed under two species, since one
    of them could put a held-out species into train under a known label.
    """
    path = manifest_path(subset, name, split_seed, n_holdout, n_toxic)
    if path.exists():
        return SplitProtocol.load(path)
    train_df = load_metadata(subset, "train", size, dest=dest)
    val_df = test_df = None
    excluded: list[str] = []
    if name == "open":
        val_df = load_metadata(subset, "val", size, dest=dest)
        test_df = load_metadata(subset, "test", size, dest=dest)
        excluded = find_conflicting_label_files({"train": train_df, "val": val_df, "test": test_df})
    protocol = build_protocol(name, train_df, val_df, test_df, subset=subset,
                              split_seed=split_seed, n_holdout=n_holdout, n_toxic=n_toxic,
                              excluded_files=excluded)
    protocol.save(path)
    return protocol


# --------------------------------------------------------------------------- #
# Revisited sites
# --------------------------------------------------------------------------- #

def flag_revisited_sites(eval_df: pd.DataFrame, train_df: pd.DataFrame, decimals: int = 3) -> pd.Series:
    """True where an eval row is the same species within one grid cell of a train row.

    Three decimals of latitude is about 100 m: a patch that was photographed in a
    train year and again in an eval year. These are legitimate test cases, but
    likely easier, so they are reported as their own slice.
    """
    def keys(df: pd.DataFrame) -> pd.Series:
        lat = df["latitude"].round(decimals).astype(str)
        lon = df["longitude"].round(decimals).astype(str)
        return df["species"].astype(str) + "|" + lat + "|" + lon

    has_coords = eval_df["latitude"].notna() & eval_df["longitude"].notna()
    train_ok = train_df["latitude"].notna() & train_df["longitude"].notna()
    return keys(eval_df).isin(set(keys(train_df[train_ok]))) & has_coords


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Build and summarise a split protocol")
    parser.add_argument("--subset", default="mini")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="open")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--n-holdout", type=int, default=30)
    parser.add_argument("--n-toxic", type=int, default=2)
    args = parser.parse_args()

    protocol = load_or_build_protocol(args.subset, args.protocol, args.split_seed,
                                      args.n_holdout, args.n_toxic)
    print(f"manifest: {manifest_path(args.subset, args.protocol, args.split_seed, args.n_holdout, args.n_toxic)}")
    print(f"known species: {len(protocol.known_species)}  genera: {protocol.genus_names}")
    print(f"excluded images (identical photo, two species): {len(protocol.excluded_files)}")
    train_df = protocol.apply(load_metadata(args.subset, "train"), "train")
    print(f"{'split':<6} {'images':>7} {'obs':>7} {'unknown obs':>12} {'revisited obs':>14}")
    print(f"{'train':<6} {len(train_df):>7} {train_df['observation_id'].nunique():>7} {'-':>12} {'-':>14}")
    for split in ("val", "test"):
        df = protocol.apply(load_metadata(args.subset, split), split)
        df["revisited_site"] = flag_revisited_sites(df, train_df)
        obs = df.drop_duplicates("observation_id")
        print(f"{split:<6} {len(df):>7} {len(obs):>7} {int(obs['is_unknown'].sum()):>12} "
              f"{int(obs['revisited_site'].sum()):>14}")
    if protocol.holdout_details:
        print("\nheld-out species:")
        print(pd.DataFrame(protocol.holdout_details).to_string(index=False))


if __name__ == "__main__":
    main()
