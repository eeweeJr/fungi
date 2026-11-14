"""FungiTastic download, extraction, and metadata loading.

Archive naming (host: cmp.felk.cvut.cz):
  FungiTastic-{Mini|FewShot|}-{train|val|test|dna-test}-{300p|500p|720p|fullsize}.zip
  plus fixed-name extras: metadata.zip, masks.zip, climatic.zip, satellite_*.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger("fungai.ingest")

DOWNLOAD_ROOT = "https://cmp.felk.cvut.cz/datagrid/FungiTastic/shared/download"
DATA_DIR = Path(__file__).resolve().parents[3] / "data"
DEFAULT_DEST = DATA_DIR / "raw" / "FungiTastic"

SUBSET_TOKENS = {"mini": "Mini", "fs": "FewShot", "full": None}
SIZE_TOKENS = {"300": "300p", "500": "500p", "720": "720p", "fullsize": "fullsize"}
IMAGE_SPLITS = ("train", "val", "test", "dna-test")
EXTRA_ARCHIVES = {
    "metadata": ("metadata.zip", "metadata"),
    "masks": ("masks.zip", "masks"),
    "climatic": ("climatic.zip", "climate"),
    "satellite_rgb": ("satellite_RGB.zip", "satellite"),
    "satellite_nir": ("satellite_NIR.zip", "satellite"),
}


@dataclass(frozen=True)
class Archive:
    name: str  # zip filename on the server
    url: str
    kind: str  # images | metadata | masks | climate | satellite
    subset: str | None = None
    split: str | None = None
    size: str | None = None

    @property
    def extract_dir(self) -> str:
        return self.kind


def image_archive(subset: str, split: str, size: str) -> Archive:
    if subset not in SUBSET_TOKENS:
        raise ValueError(f"unknown subset {subset!r}, expected one of {list(SUBSET_TOKENS)}")
    if split not in IMAGE_SPLITS:
        raise ValueError(f"unknown split {split!r}, expected one of {IMAGE_SPLITS}")
    if size not in SIZE_TOKENS:
        raise ValueError(f"unknown size {size!r}, expected one of {list(SIZE_TOKENS)}")
    if subset == "fs" and split == "dna-test":
        raise ValueError("FewShot subset has no dna-test split")
    token = SUBSET_TOKENS[subset]
    middle = f"{token}-" if token else ""
    name = f"FungiTastic-{middle}{split}-{SIZE_TOKENS[size]}.zip"
    return Archive(
        name=name, url=f"{DOWNLOAD_ROOT}/{name}", kind="images",
        subset=subset, split=split, size=size,
    )


def extra_archive(extra: str) -> Archive:
    if extra not in EXTRA_ARCHIVES:
        raise ValueError(f"unknown extra {extra!r}, expected one of {list(EXTRA_ARCHIVES)}")
    name, kind = EXTRA_ARCHIVES[extra]
    return Archive(name=name, url=f"{DOWNLOAD_ROOT}/{name}", kind=kind)


def build_archives(
    subset: str = "mini",
    size: str = "300",
    splits: tuple[str, ...] = ("train", "val", "test"),
    extras: tuple[str, ...] = ("metadata", "masks", "climatic"),
) -> list[Archive]:
    archives = [extra_archive(e) for e in extras]
    archives += [image_archive(subset, sp, size) for sp in splits]
    return archives


# --------------------------------------------------------------------------- #
# Download machinery
# --------------------------------------------------------------------------- #

def _manifest_path(archive: Archive, dest: Path) -> Path:
    return dest / ".manifest" / f"{archive.name}.json"


def is_ingested(archive: Archive, dest: Path) -> bool:
    mpath = _manifest_path(archive, dest)
    if not mpath.exists():
        return False
    target = dest / archive.extract_dir
    if not target.exists():
        logger.warning("manifest exists for %s but %s is missing", archive.name, target)
        return False
    return True


def head_size(url: str, session=None) -> int | None:
    import requests

    session = session or requests
    try:
        r = session.head(url, allow_redirects=True, timeout=30)
        if r.ok and "Content-Length" in r.headers:
            return int(r.headers["Content-Length"])
    except Exception as exc:  # noqa: BLE001 - preflight is best-effort
        logger.warning("HEAD failed for %s: %s", url, exc)
    return None


def preflight(archives: list[Archive], dest: Path) -> list[dict]:
    """HEAD every archive, print a size table, and check free disk space."""
    import requests

    rows = []
    with requests.Session() as session:
        for ar in archives:
            if is_ingested(ar, dest):
                rows.append({"archive": ar, "bytes": None, "status": "cached"})
                continue
            size = head_size(ar.url, session)
            status = "to-download" if size is not None else "unknown-size"
            rows.append({"archive": ar, "bytes": size, "status": status})

    print(f"{'file':<45} {'size':>10}  status")
    print("-" * 70)
    total = 0
    for row in rows:
        gb = f"{row['bytes'] / 1e9:.2f} GB" if row["bytes"] else "?"
        print(f"{row['archive'].name:<45} {gb:>10}  {row['status']}")
        total += row["bytes"] or 0
    free = shutil.disk_usage(dest.anchor or "/").free
    print("-" * 70)
    print(f"{'total to download':<45} {total / 1e9:>7.2f} GB  (free disk: {free / 1e9:.1f} GB)")

    # 2x headroom: zip + extracted copy coexist during extraction
    needed = total * 2
    if needed > free:
        raise RuntimeError(
            f"not enough disk space: need ~{needed / 1e9:.1f} GB "
            f"(download + extraction headroom), have {free / 1e9:.1f} GB free"
        )
    return rows


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stream_download(url: str, part_path: Path, session) -> None:
    from tqdm import tqdm

    existing = part_path.stat().st_size if part_path.exists() else 0
    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with session.get(url, stream=True, headers=headers, timeout=60) as r:
        if r.status_code == 206:
            mode = "ab"
        elif r.status_code == 200:
            mode, existing = "wb", 0
        else:
            r.raise_for_status()
            mode, existing = "wb", 0
        total = int(r.headers.get("Content-Length", 0)) + existing
        with open(part_path, mode) as f, tqdm(
            total=total or None, initial=existing, unit="B", unit_scale=True,
            desc=part_path.name, leave=False,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))


def download_archive(
    archive: Archive, dest: Path, keep_zip: bool = False, force: bool = False,
    max_retries: int = 3,
) -> None:
    import requests

    if is_ingested(archive, dest) and not force:
        logger.info("skip %s (already ingested)", archive.name)
        return

    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / archive.name
    part_path = dest / f"{archive.name}.part"

    if not zip_path.exists():
        for attempt in range(1, max_retries + 1):
            try:
                with requests.Session() as session:
                    _stream_download(archive.url, part_path, session)
                break
            except (requests.ConnectionError, requests.Timeout) as exc:
                logger.warning("attempt %d/%d failed for %s: %s",
                               attempt, max_retries, archive.name, exc)
                if attempt == max_retries:
                    raise
                time.sleep(2 ** attempt)
        part_path.rename(zip_path)  # atomic: only complete downloads get the real name

    logger.info("verifying zip integrity: %s", archive.name)
    try:
        with zipfile.ZipFile(zip_path) as zf:
            bad = zf.testzip()
            if bad is not None:
                raise zipfile.BadZipFile(f"corrupt member: {bad}")
            names = zf.namelist()
    except zipfile.BadZipFile:
        logger.error("corrupt zip %s — deleting; re-run to retry", zip_path)
        zip_path.unlink()
        raise

    extract_to = dest / archive.extract_dir
    extract_to.mkdir(parents=True, exist_ok=True)
    logger.info("extracting %s -> %s (%d entries)", archive.name, extract_to, len(names))
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)

    manifest = {
        "name": archive.name,
        "url": archive.url,
        "kind": archive.kind,
        "bytes": zip_path.stat().st_size,
        "sha256": _sha256(zip_path),
        "entries": len(names),
        "extracted_to": str(extract_to),
        "ingested_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **{k: v for k, v in asdict(archive).items() if k in ("subset", "split", "size") and v},
    }
    mpath = _manifest_path(archive, dest)
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=2))

    if not keep_zip:
        zip_path.unlink()
    logger.info("done: %s", archive.name)


def download_dataset(
    subset: str = "mini",
    size: str = "300",
    splits: tuple[str, ...] = ("train", "val", "test"),
    extras: tuple[str, ...] = ("metadata", "masks", "climatic"),
    dest: Path = DEFAULT_DEST,
    keep_zip: bool = False,
    force: bool = False,
    dry_run: bool = False,
) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    archives = build_archives(subset, size, splits, extras)
    preflight(archives, dest)
    if dry_run:
        return dest
    for ar in archives:
        download_archive(ar, dest, keep_zip=keep_zip, force=force)
    return dest


# --------------------------------------------------------------------------- #
# Metadata loading
# --------------------------------------------------------------------------- #

# Canonical name -> candidate column names seen across FungiTastic/DF20 releases.
COLUMNS: dict[str, list[str]] = {
    "observation_id": ["observationID", "observation_id", "obs_id"],
    "filename": ["filename", "image_path", "imageFilename", "image_name"],
    "species": ["species", "scientificName", "taxon_name"],
    "genus": ["genus"],
    "family": ["family"],
    "order": ["order"],
    "class": ["class"],
    "phylum": ["phylum"],
    "category_id": ["category_id", "class_id", "label"],
    "date": ["eventDate", "date", "observation_date"],
    "year": ["year"],
    "month": ["month"],
    "day": ["day"],
    "latitude": ["latitude", "Latitude", "lat"],
    "longitude": ["longitude", "Longitude", "lon", "lng"],
    "elevation": ["elevation", "Elevation", "altitude"],
    "habitat": ["habitat", "Habitat"],
    "substrate": ["substrate", "Substrate"],
    "country": ["countryCode", "country", "Country"],
    "region": ["region", "Region"],
    "district": ["district", "District"],
    "biogeo_zone": ["biogeographicalRegion", "bioGeoRegion", "biogeographical_zone"],
    "land_cover": ["landcover", "landCover", "land_cover", "MODIS_landcover"],
    "toxicity": ["poisonous", "toxic", "edibility"],
}

CATEGORICAL = ("species", "genus", "family", "order", "class", "phylum",
               "habitat", "substrate", "country", "region", "biogeo_zone", "land_cover")


def resolve_column(df: pd.DataFrame, canonical: str) -> str | None:
    """Return the dataframe's actual column name for a canonical concept."""
    for cand in COLUMNS.get(canonical, []):
        if cand in df.columns:
            return cand
    return None


def normalize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Rename resolvable columns to canonical names, parse dates, set categoricals."""
    rename = {}
    for canonical in COLUMNS:
        actual = resolve_column(df, canonical)
        if actual and actual != canonical:
            rename[actual] = canonical
    df = df.rename(columns=rename)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for col in CATEGORICAL:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def metadata_dir(dest: Path = DEFAULT_DEST) -> Path:
    return Path(dest) / "metadata"


# Extracted layout: metadata/FungiTastic[-Mini|-FewShot]/FungiTastic[-Mini|-FewShot]-<Split>.csv
# The full set names its val/test by task (ClosedSet/OpenSet); Mini and FewShot do not.
_SPLIT_FILES = {
    ("mini", "train"): "FungiTastic-Mini/FungiTastic-Mini-Train.csv",
    ("mini", "val"): "FungiTastic-Mini/FungiTastic-Mini-Val.csv",
    ("mini", "test"): "FungiTastic-Mini/FungiTastic-Mini-Test.csv",
    ("mini", "dna-test"): "FungiTastic-Mini/FungiTastic-Mini-DNA-Test.csv",
    ("fs", "train"): "FungiTastic-FewShot/FungiTastic-FewShot-Train.csv",
    ("fs", "val"): "FungiTastic-FewShot/FungiTastic-FewShot-Val.csv",
    ("fs", "test"): "FungiTastic-FewShot/FungiTastic-FewShot-Test.csv",
    ("full", "train"): "FungiTastic/FungiTastic-Train.csv",
    ("full", "val"): "FungiTastic/FungiTastic-ClosedSet-Val.csv",
    ("full", "test"): "FungiTastic/FungiTastic-ClosedSet-Test.csv",
    ("full", "val-open"): "FungiTastic/FungiTastic-OpenSet-Val.csv",
    ("full", "test-open"): "FungiTastic/FungiTastic-OpenSet-Test.csv",
    ("full", "dna-test"): "FungiTastic/FungiTastic-DNA-Test.csv",
}


def find_metadata_csv(subset: str, split: str, dest: Path = DEFAULT_DEST) -> Path:
    """Locate the extracted metadata CSV for a subset/split."""
    rel = _SPLIT_FILES.get((subset, split))
    if rel is None:
        raise ValueError(f"no metadata file defined for subset={subset!r} split={split!r}; "
                         f"known: {sorted(_SPLIT_FILES)}")
    path = metadata_dir(dest) / rel
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found — run `python -m fungai.data.ingest` first"
        )
    return path


def find_image_dir(subset: str, split: str, size: str, dest: Path = DEFAULT_DEST) -> Path | None:
    """Locate the extracted image directory for subset/split/size, if present."""
    images_root = Path(dest) / "images"
    if not images_root.exists():
        return None
    size_token = SIZE_TOKENS.get(size, size)
    matches = [
        p for p in images_root.rglob("*")
        if p.is_dir() and split.lower() in str(p).lower() and size_token in str(p)
    ]
    if not matches:
        return None
    # deepest match = most specific
    return max(matches, key=lambda p: len(p.parts))


def load_metadata(subset: str, split: str, size: str = "300",
                  dest: Path = DEFAULT_DEST, include_captions: bool = False) -> pd.DataFrame:
    """Load + normalize metadata; attach resolved local image paths when images exist.

    The MOLMO captions column is dropped by default: it is by far the largest
    column in RAM (long free text per row) and most workflows don't need it.
    """
    csv_path = find_metadata_csv(subset, split, dest)
    df = pd.read_csv(csv_path, low_memory=False)
    if not include_captions and "captions" in df.columns:
        df = df.drop(columns=["captions"])
    df = normalize_metadata(df)
    img_dir = find_image_dir(subset, split, size, dest)
    if img_dir is not None and "filename" in df.columns:
        df["image_path"] = df["filename"].map(lambda f: str(img_dir / str(f)))
    return df


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser(description="Download FungiTastic data")
    parser.add_argument("--subset", choices=list(SUBSET_TOKENS), default="mini")
    parser.add_argument("--size", choices=list(SIZE_TOKENS), default="300")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"],
                        choices=list(IMAGE_SPLITS))
    parser.add_argument("--extras", nargs="+", default=["metadata", "masks", "climatic"],
                        choices=list(EXTRA_ARCHIVES))
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--keep-zip", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the size table and exit without downloading")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    download_dataset(
        subset=args.subset, size=args.size, splits=tuple(args.splits),
        extras=tuple(args.extras), dest=args.dest, keep_zip=args.keep_zip,
        force=args.force, dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
