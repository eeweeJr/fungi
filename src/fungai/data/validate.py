"""Standalone data-validation suite for ingested FungiTastic data.

Run:  python -m fungai.data.validate --subset mini --size 300
Prints a sectioned PASS/FAIL report, writes data/validation_report.json,
exits non-zero if any check fails. Checks that can't run (missing columns
or files) are reported as SKIP, not silently dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from PIL import Image

from fungai.data.ingest import DATA_DIR, DEFAULT_DEST, load_metadata

Image.MAX_IMAGE_PIXELS = None  # dataset photos can be large; we check size ourselves

REPORT_PATH = DATA_DIR / "validation_report.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass
class CheckResult:
    name: str
    status: str  # PASS | FAIL | SKIP
    summary: str
    details: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Checks
# --------------------------------------------------------------------------- #

def check_files_vs_metadata(df: pd.DataFrame, split: str) -> CheckResult:
    name = f"file-metadata integrity [{split}]"
    if "image_path" not in df.columns:
        return CheckResult(name, "SKIP", "no image_path column (images not downloaded?)")
    paths = df["image_path"].map(Path)
    missing = [str(p) for p in paths if not p.exists()]
    img_dir = paths.iloc[0].parent if len(paths) else None
    orphans: list[str] = []
    if img_dir and img_dir.exists():
        on_disk = {p for p in img_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
        referenced = set(paths)
        orphans = [str(p) for p in on_disk - referenced]
    status = "FAIL" if missing else "PASS"
    return CheckResult(
        name, status,
        f"{len(df)} rows; {len(missing)} missing files; {len(orphans)} orphan files on disk",
        {"missing_sample": missing[:10], "orphans_sample": orphans[:10],
         "n_missing": len(missing), "n_orphans": len(orphans)},
    )


def _image_health(path_str: str) -> str | None:
    p = Path(path_str)
    try:
        if p.stat().st_size == 0:
            return "zero-byte"
        with Image.open(p) as im:
            im.verify()
        with Image.open(p) as im:
            im.load()
            w, h = im.size
        if min(w, h) < 100:
            return f"small ({w}x{h})"
        if max(w, h) / max(min(w, h), 1) > 5:
            return f"extreme aspect ({w}x{h})"
    except Exception as exc:  # noqa: BLE001 - any decode failure = bad image
        return f"corrupt: {type(exc).__name__}"
    return None


def check_image_health(df: pd.DataFrame, split: str, sample: int | None,
                       workers: int = 8) -> CheckResult:
    name = f"image health [{split}]"
    if "image_path" not in df.columns:
        return CheckResult(name, "SKIP", "no image_path column")
    paths = [p for p in df["image_path"] if Path(p).exists()]
    if sample and len(paths) > sample:
        step = len(paths) // sample
        paths = paths[::step][:sample]
    problems: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for path, issue in zip(paths, ex.map(_image_health, paths)):
            if issue:
                problems[path] = issue
    corrupt = {p: i for p, i in problems.items() if i.startswith(("corrupt", "zero"))}
    status = "FAIL" if corrupt else "PASS"
    return CheckResult(
        name, status,
        f"checked {len(paths)} images; {len(corrupt)} corrupt/zero-byte; "
        f"{len(problems) - len(corrupt)} quality flags (small/aspect)",
        {"problems_sample": dict(list(problems.items())[:15]), "n_problems": len(problems)},
    )


def check_metadata_sanity(df: pd.DataFrame, split: str) -> CheckResult:
    name = f"metadata sanity [{split}]"
    issues: dict[str, int] = {}
    null_rates = {c: float(df[c].isna().mean()) for c in df.columns}
    high_null = {c: round(r, 4) for c, r in null_rates.items() if r > 0.01}

    if {"latitude", "longitude"} <= set(df.columns):
        lat, lon = df["latitude"], df["longitude"]
        issues["bad_latitude"] = int(((lat < -90) | (lat > 90)).sum())
        issues["bad_longitude"] = int(((lon < -180) | (lon > 180)).sum())
        issues["zero_coords"] = int(((lat == 0) & (lon == 0)).sum())
    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
        issues["unparseable_dates"] = int(dates.isna().sum() - df["date"].isna().sum())
        valid = dates.dropna()
        # pre-1990 records exist (historical observations back to the 1960s)
        issues["dates_out_of_range"] = int(
            ((valid < "1960-01-01") | (valid > pd.Timestamp.now())).sum()
        )
        if "month" in df.columns:
            both = pd.DataFrame({"d": dates, "m": df["month"]}).dropna()
            issues["month_mismatch"] = int((both["d"].dt.month != both["m"]).sum())
    if "elevation" in df.columns:
        elev = pd.to_numeric(df["elevation"], errors="coerce")
        issues["bad_elevation"] = int(((elev < -100) | (elev > 5000)).sum())

    hard_fail = sum(v for k, v in issues.items() if k != "zero_coords") > 0
    return CheckResult(
        name, "FAIL" if hard_fail else "PASS",
        f"{len(df)} rows; issues: " + (", ".join(f"{k}={v}" for k, v in issues.items() if v)
                                       or "none"),
        {"issues": issues, "columns_with_>1%_nulls": high_null},
    )


def check_taxonomy(df: pd.DataFrame, split: str) -> CheckResult:
    name = f"taxonomic consistency [{split}]"
    if "species" not in df.columns:
        return CheckResult(name, "SKIP", "no species column")
    problems: dict[str, list] = {}
    for level in ("genus", "family", "order"):
        if level in df.columns:
            per_species = df.dropna(subset=["species"]).groupby("species", observed=True)[level].nunique()
            bad = per_species[per_species > 1]
            if len(bad):
                problems[f"species_with_multiple_{level}"] = bad.index.tolist()[:10]
    if "toxicity" in df.columns:
        tox_n = df.dropna(subset=["species"]).groupby("species", observed=True)["toxicity"].nunique()
        incons = tox_n[tox_n > 1]
        if len(incons):
            problems["species_with_inconsistent_toxicity"] = incons.index.tolist()[:10]
        missing_tox = df["toxicity"].isna().mean()
    else:
        missing_tox = None
    status = "FAIL" if problems else "PASS"
    return CheckResult(
        name, status,
        f"{df['species'].nunique()} species; hierarchy conflicts: {len(problems)}; "
        f"toxicity null rate: {missing_tox if missing_tox is None else round(missing_tox, 4)}",
        problems,
    )


def check_splits(dfs: dict[str, pd.DataFrame]) -> CheckResult:
    name = "split hygiene"
    if "observation_id" not in next(iter(dfs.values())).columns:
        return CheckResult(name, "SKIP", "no observation_id column")
    overlap: dict[str, int] = {}
    class_gap: dict[str, int] = {}
    splits = list(dfs)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            shared = set(dfs[a]["observation_id"]) & set(dfs[b]["observation_id"])
            if shared:
                overlap[f"{a}∩{b}"] = len(shared)
    if "species" in dfs.get("train", pd.DataFrame()).columns:
        train_species = set(dfs["train"]["species"].dropna())
        for sp in splits:
            if sp == "train":
                continue
            unseen = set(dfs[sp]["species"].dropna()) - train_species
            class_gap[f"{sp}_species_not_in_train"] = len(unseen)
    status = "FAIL" if overlap else "PASS"
    return CheckResult(
        name, status,
        f"observation overlap: {overlap or 'none'}; unseen-class counts (open-set, "
        f"informational): {class_gap or 'n/a'}",
        {"overlap": overlap, "class_gap": class_gap},
    )


def _md5(path_str: str) -> tuple[str, str]:
    h = hashlib.md5()
    with open(path_str, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return path_str, h.hexdigest()


def check_duplicates(dfs: dict[str, pd.DataFrame], sample: int | None,
                     workers: int = 8) -> CheckResult:
    name = "duplicate images"
    split_of: dict[str, str] = {}
    all_paths: list[str] = []
    for split, df in dfs.items():
        if "image_path" not in df.columns:
            return CheckResult(name, "SKIP", "no image_path column")
        paths = [p for p in df["image_path"] if Path(p).exists()]
        if sample and len(paths) > sample:
            step = len(paths) // sample
            paths = paths[::step][:sample]
        for p in paths:
            split_of[p] = split
            all_paths.append(p)
    hashes: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for path, digest in ex.map(_md5, all_paths):
            hashes.setdefault(digest, []).append(path)
    dupe_groups = {d: ps for d, ps in hashes.items() if len(ps) > 1}
    cross_split = {
        d: ps for d, ps in dupe_groups.items()
        if len({split_of[p] for p in ps}) > 1
    }
    status = "FAIL" if cross_split else "PASS"
    return CheckResult(
        name, status,
        f"hashed {len(all_paths)} images; {len(dupe_groups)} exact-duplicate groups; "
        f"{len(cross_split)} CROSS-SPLIT (leakage)",
        {"cross_split_sample": {d: ps for d, ps in list(cross_split.items())[:5]},
         "n_within_split_groups": len(dupe_groups) - len(cross_split)},
    )


def check_leakage(dfs: dict[str, pd.DataFrame], subset: str) -> CheckResult:
    """Cross-split leakage beyond shared ids: time, place, near-identical images,
    and the open protocol's held-out species."""
    from fungai.data import leakage
    from fungai.data.splits import build_protocol

    name = "leakage"
    if set(dfs) != {"train", "val", "test"} or any("image_path" not in d.columns for d in dfs.values()):
        return CheckResult(name, "SKIP", "needs all three splits with images")
    hashed = leakage.with_hashes(dfs)
    near = leakage.cross_split_near_duplicates(hashed)
    events = leakage.shared_events(dfs)
    temporal = leakage.temporal_violations(dfs)
    conflicts = leakage.conflicting_label_files(hashed)
    protocol = build_protocol("open", dfs["train"], dfs["val"], dfs["test"], subset=subset,
                              excluded_files=conflicts)
    applied = {s: protocol.apply(df, s) for s, df in dfs.items()}
    violations = leakage.protocol_violations(protocol, applied)
    failed = len(near) or events or temporal or violations
    return CheckResult(
        name, "FAIL" if failed else "PASS",
        f"near-identical images across splits: {len(near)}; same day and spot across splits: "
        f"{sum(map(len, events.values()))}; year overlap: {temporal or 'none'}; open-protocol "
        f"violations: {len(violations)}; identical photos with two species (excluded from the "
        f"open protocol, informational): {len(conflicts)}",
        {"near_duplicates_sample": near.head(10).to_dict("records"), "shared_events": events,
         "temporal": temporal, "protocol_violations": violations, "conflicting_label_files": conflicts},
    )


def check_masks(dfs: dict[str, pd.DataFrame], dest: Path) -> CheckResult:
    """Masks ship as parquet: one row per (image, body-part) RLE segmentation.

    Two mask filename conventions exist:
      - "{idx}-{observationID}.JPG"      -> matches the metadata `filename` column
      - "{observationID}-{photoID}.JPG"  -> joinable on observation id prefix
    """
    name = "mask alignment"
    masks_root = dest / "masks"
    if not masks_root.exists():
        return CheckResult(name, "SKIP", "masks/ not downloaded")
    parquets = list(masks_root.glob("*.parquet"))
    if not parquets:
        return CheckResult(name, "FAIL", "masks/ exists but contains no parquet files")

    known_filenames: set[str] = set()
    known_obs_ids: set[str] = set()
    for df in dfs.values():
        if "filename" in df.columns:
            known_filenames |= set(df["filename"].astype(str))
        if "observation_id" in df.columns:
            known_obs_ids |= set(df["observation_id"].astype(str))

    details: dict = {}
    unmatched_total = 0
    rows_total = 0
    for pq_path in parquets:
        masks = pd.read_parquet(pq_path, columns=["label", "file_name"])
        rows_total += len(masks)
        names = masks["file_name"].astype(str)
        direct = names.isin(known_filenames)
        obs_prefix = names.str.split("-").str[0].isin(known_obs_ids)
        unmatched = int((~direct & ~obs_prefix).sum())
        unmatched_total += unmatched
        details[pq_path.name] = {
            "rows": len(masks),
            "matched_by_filename": int(direct.sum()),
            "matched_by_observation_id": int((~direct & obs_prefix).sum()),
            "unmatched": unmatched,
            "part_labels": masks["label"].value_counts().to_dict(),
        }
    # tolerate a small unmatched remainder (masks for splits not downloaded, e.g. dna-test)
    frac = unmatched_total / max(rows_total, 1)
    status = "PASS" if frac < 0.10 else "FAIL"
    return CheckResult(
        name, status,
        f"{len(parquets)} parquet files, {rows_total} part-masks; "
        f"{unmatched_total} ({frac:.1%}) not joinable to loaded metadata",
        details,
    )


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #

def run(subset: str, size: str, dest: Path, sample: int | None) -> int:
    splits = ("train", "val", "test")
    dfs: dict[str, pd.DataFrame] = {}
    results: list[CheckResult] = []

    for split in splits:
        try:
            dfs[split] = load_metadata(subset, split, size, dest)
        except FileNotFoundError as exc:
            results.append(CheckResult(f"load metadata [{split}]", "FAIL", str(exc)))
    for split, df in dfs.items():
        results.append(check_files_vs_metadata(df, split))
        results.append(check_image_health(df, split, sample))
        results.append(check_metadata_sanity(df, split))
        results.append(check_taxonomy(df, split))
    if dfs:
        results.append(check_splits(dfs))
        results.append(check_duplicates(dfs, sample))
        results.append(check_leakage(dfs, subset))
    results.append(check_masks(dfs, dest))

    width = max(len(r.name) for r in results) + 2
    print("\n" + "=" * 78)
    print(f"FungiTastic validation report  (subset={subset}, size={size})")
    print("=" * 78)
    for r in results:
        print(f"[{r.status:^4}] {r.name:<{width}} {r.summary}")
    n_fail = sum(r.status == "FAIL" for r in results)
    n_skip = sum(r.status == "SKIP" for r in results)
    print("-" * 78)
    print(f"{len(results)} checks: {len(results) - n_fail - n_skip} passed, "
          f"{n_fail} failed, {n_skip} skipped")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({
        "subset": subset, "size": size, "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": [asdict(r) for r in results],
    }, indent=2, default=str))
    print(f"report written to {REPORT_PATH}")
    return 1 if n_fail else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate ingested FungiTastic data")
    parser.add_argument("--subset", default="mini")
    parser.add_argument("--size", default="300")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--quick", action="store_true",
                        help="sample 2000 images per split for health/duplicate checks")
    args = parser.parse_args()
    sys.exit(run(args.subset, args.size, args.dest, sample=2000 if args.quick else None))


if __name__ == "__main__":
    main()
