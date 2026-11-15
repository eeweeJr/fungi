"""Versioned, immutable snapshots of the train/val/test splits.

A snapshot holds the exact rows each split contained once the protocol was applied,
plus the derived columns training uses (label, genus_label, is_unknown,
revisited_site):

    data/processed/snapshots/<subset>_<protocol>_<digest>/
        train.parquet  val.parquet  test.parquet  manifest.json

The digest is computed from the rows, so the same splits always get the same id and
a different selection always gets a new one. A snapshot is never overwritten.
Training runs record the id; EDA and error analysis load by id:

    from fungai.data.snapshot import load_snapshot
    df = load_snapshot("latest")            # all splits, with a `split` column

CLI:
    python -m fungai.data.snapshot          # build (or reuse) the snapshot for the defaults
    python -m fungai.data.snapshot --list
    python -m fungai.data.snapshot --verify <id>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from fungai.data.ingest import CATEGORICAL, DATA_DIR, DEFAULT_DEST, find_image_dir, load_metadata
from fungai.data.splits import SplitProtocol, flag_revisited_sites, load_or_build_protocol

SNAPSHOT_DIR = DATA_DIR / "processed" / "snapshots"
SPLITS = ("train", "val", "test")
MACHINE_COLUMNS = ("image_path",)  # differs per machine: not stored, re-attached on load


# --------------------------------------------------------------------------- #
# Building
# --------------------------------------------------------------------------- #

def prepare_splits(protocol: SplitProtocol, raw: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Apply the protocol to the official splits and add the derived columns."""
    frames: dict[str, pd.DataFrame] = {}
    for split in SPLITS:
        df = protocol.apply(raw[split], split)
        df["split"] = split
        df["label"] = df["species"].map(protocol.species_map).fillna(-1).astype(int)
        df["genus_label"] = (
            df["genus"].astype(object).map(protocol.genus_map).fillna(-1).astype(int)
        )
        df["revisited_site"] = (
            False if split == "train" else flag_revisited_sites(df, frames["train"])
        )
        frames[split] = df
    return frames


def frame_digest(df: pd.DataFrame) -> str:
    """sha256 of a frame's content: independent of column order, dtype storage units
    and categorical encoding, so a frame read back from parquet hashes the same."""
    canon = df.drop(columns=list(MACHINE_COLUMNS), errors="ignore")
    cols = sorted(canon.columns)
    canon = canon[cols].copy()
    for col in cols:
        if pd.api.types.is_datetime64_any_dtype(canon[col]):
            canon[col] = canon[col].astype("datetime64[ns]")
        elif isinstance(canon[col].dtype, pd.CategoricalDtype):
            # parquet may hand a categorical back as its value type (e.g. float64)
            try:
                canon[col] = canon[col].astype(canon[col].cat.categories.dtype)
            except (TypeError, ValueError):  # e.g. integer categories with gaps
                canon[col] = canon[col].astype(object)
    h = hashlib.sha256("|".join(cols).encode())
    h.update(pd.util.hash_pandas_object(canon, index=False).to_numpy().tobytes())
    return h.hexdigest()


def _git_state() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL).decode().strip()
        except Exception:  # noqa: BLE001 - not a git checkout
            return ""

    return {"commit": run("rev-parse", "HEAD") or "unknown", "dirty": bool(run("status", "--porcelain"))}


def _split_summary(df: pd.DataFrame) -> dict:
    obs = df.drop_duplicates("observation_id")
    return {
        "rows": len(df),
        "observations": len(obs),
        "species": int(df["species"].nunique()),
        "unknown_observations": int(obs["is_unknown"].sum()),
        "revisited_observations": int(obs["revisited_site"].sum()),
        "digest": frame_digest(df),
    }


def write_snapshot(
    frames: dict[str, pd.DataFrame],
    protocol: SplitProtocol,
    size: str = "300",
    root: Path = SNAPSHOT_DIR,
) -> Path:
    """Write the frames as a snapshot and return its folder. An existing snapshot with
    the same content is reused untouched."""
    summaries = {split: _split_summary(frames[split]) for split in SPLITS}
    combined = hashlib.sha256("".join(summaries[s]["digest"] for s in SPLITS).encode()).hexdigest()
    snap_id = f"{protocol.subset}_{protocol.name}_{combined[:12]}"
    path = Path(root) / snap_id
    if (path / "manifest.json").exists():
        return path

    tmp = Path(root) / f".{snap_id}.partial"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for split in SPLITS:
        frames[split].drop(columns=list(MACHINE_COLUMNS), errors="ignore").to_parquet(
            tmp / f"{split}.parquet", index=False
        )
    manifest = {
        "id": snap_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "subset": protocol.subset,
        "image_size": size,
        "git": _git_state(),
        "splits": summaries,
        "protocol": asdict(protocol),
    }
    (tmp / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    tmp.rename(path)  # atomic: a half-written snapshot never gets the real name
    return path


def build_snapshot(
    subset: str = "mini",
    protocol_name: str = "open",
    split_seed: int = 42,
    n_holdout: int = 30,
    n_toxic: int = 2,
    size: str = "300",
    dest: Path = DEFAULT_DEST,
    root: Path = SNAPSHOT_DIR,
) -> tuple[Path, dict[str, pd.DataFrame], SplitProtocol]:
    """Build the splits for these settings, snapshot them, and return
    (snapshot folder, frames with image paths, protocol) for training."""
    protocol = load_or_build_protocol(subset, protocol_name, split_seed, n_holdout, n_toxic,
                                      dest=dest, size=size)
    raw = {split: load_metadata(subset, split, size, dest=dest) for split in SPLITS}
    frames = prepare_splits(protocol, raw)
    return write_snapshot(frames, protocol, size=size, root=root), frames, protocol


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def list_snapshots(root: Path = SNAPSHOT_DIR) -> pd.DataFrame:
    rows = []
    for manifest_file in sorted(Path(root).glob("*/manifest.json")):
        m = json.loads(manifest_file.read_text())
        rows.append({
            "id": m["id"], "created_at": m["created_at"], "protocol": m["protocol"]["name"],
            "known_species": len(m["protocol"]["known_species"]),
            **{f"{s}_rows": m["splits"][s]["rows"] for s in SPLITS},
            "git_commit": m["git"]["commit"][:8], "git_dirty": m["git"]["dirty"],
        })
    columns = ["id", "created_at", "protocol", "known_species",
               *(f"{s}_rows" for s in SPLITS), "git_commit", "git_dirty"]
    return pd.DataFrame(rows, columns=columns).sort_values("created_at", ignore_index=True)


def resolve_snapshot(snapshot: str | Path, root: Path = SNAPSHOT_DIR) -> Path:
    """Accept a snapshot id, a folder path, or "latest"."""
    if snapshot == "latest":
        listing = list_snapshots(root)
        if listing.empty:
            raise FileNotFoundError(f"no snapshots in {root}; run `python -m fungai.data.snapshot`")
        return Path(root) / listing["id"].iloc[-1]
    path = Path(snapshot)
    if not (path / "manifest.json").exists():
        path = Path(root) / str(snapshot)
    if not (path / "manifest.json").exists():
        raise FileNotFoundError(f"no snapshot {snapshot!r} in {root}")
    return path


def read_manifest(snapshot: str | Path, root: Path = SNAPSHOT_DIR) -> dict:
    return json.loads((resolve_snapshot(snapshot, root) / "manifest.json").read_text())


def load_snapshot(
    snapshot: str | Path = "latest",
    splits: tuple[str, ...] | str = SPLITS,
    root: Path = SNAPSHOT_DIR,
    image_paths: bool = True,
    dest: Path = DEFAULT_DEST,
) -> pd.DataFrame:
    """Load one or more splits of a snapshot as one frame with a `split` column.

    image_path is re-attached from the local image folders when they exist.
    """
    path = resolve_snapshot(snapshot, root)
    manifest = json.loads((path / "manifest.json").read_text())
    splits = (splits,) if isinstance(splits, str) else tuple(splits)
    frames = []
    for split in splits:
        df = pd.read_parquet(path / f"{split}.parquet")
        if image_paths:
            img_dir = find_image_dir(manifest["subset"], split, manifest["image_size"], dest)
            if img_dir is not None:
                df["image_path"] = df["filename"].map(lambda f, d=img_dir: str(d / str(f)))
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for col in CATEGORICAL:
        if col in out.columns:
            out[col] = out[col].astype("category")
    return out


def verify_snapshot(snapshot: str | Path, root: Path = SNAPSHOT_DIR) -> list[str]:
    """Re-hash every split and compare with the manifest. Empty list means intact."""
    path = resolve_snapshot(snapshot, root)
    manifest = json.loads((path / "manifest.json").read_text())
    problems = []
    for split in SPLITS:
        file = path / f"{split}.parquet"
        if not file.exists():
            problems.append(f"{split}: file missing")
            continue
        df = pd.read_parquet(file)
        expected = manifest["splits"][split]
        if len(df) != expected["rows"]:
            problems.append(f"{split}: {len(df)} rows, manifest says {expected['rows']}")
        if frame_digest(df) != expected["digest"]:
            problems.append(f"{split}: content does not match the manifest digest")
    return problems


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Build, list or verify split snapshots")
    parser.add_argument("--subset", default="mini")
    parser.add_argument("--protocol", choices=("open", "closed"), default="open")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--n-holdout", type=int, default=30)
    parser.add_argument("--n-toxic", type=int, default=2)
    parser.add_argument("--size", default="300")
    parser.add_argument("--list", action="store_true", help="list existing snapshots")
    parser.add_argument("--verify", metavar="ID", help="check a snapshot against its manifest")
    args = parser.parse_args()

    pd.set_option("display.width", 200)
    if args.list:
        print(list_snapshots().to_string(index=False))
        return
    if args.verify:
        problems = verify_snapshot(args.verify)
        print("intact" if not problems else "\n".join(problems))
        raise SystemExit(1 if problems else 0)

    path, _, _ = build_snapshot(args.subset, args.protocol, args.split_seed, args.n_holdout,
                                args.n_toxic, args.size)
    manifest = json.loads((path / "manifest.json").read_text())
    print(f"snapshot: {manifest['id']}\npath:     {path}")
    print(pd.DataFrame(manifest["splits"]).T.drop(columns="digest").to_string())


if __name__ == "__main__":
    main()
