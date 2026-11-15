"""FungiTastic part masks: presence table, RLE decoding and patch-grid projection.

Masks are used only to *measure* the model (evaluation slices, attention-on-part
diagnostics). They never feed training or inference.

Format, established empirically on the validation masks and checked by overlay:
columns `label, file_name, width, height, rle`, where width/height are the original
photo size and `rle` is alternating run lengths starting with background,
row-major. The runs overshoot width x height by a small, variable amount past the
end of the image; decoding clips at width x height, which aligns exactly with the
photos.

Train mask filenames mostly do not join to the train metadata (an older naming
scheme), so only val and test are supported. Never load the `rle` column of a whole
file at once: the train file is ~700 MB on disk.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

from fungai.data.ingest import DEFAULT_DEST

MASK_FILES = {
    "train": "FungiTastic-Mini-TrainMasks.parquet",
    "val": "FungiTastic-Mini-ValidationMasks.parquet",
    "test": "FungiTastic-Mini-TestMasks.parquet",
}
PRESENCE_PARTS = ("cap", "gills", "stem", "ring", "pores", "fruiting_body")
BATCH_ROWS = 2000


def mask_file(split: str, dest: Path = DEFAULT_DEST) -> Path:
    if split not in MASK_FILES:
        raise ValueError(f"unknown split {split!r}")
    path = Path(dest) / "masks" / MASK_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run the ingest with --extras masks")
    return path


def load_part_presence(split: str = "val", dest: Path = DEFAULT_DEST) -> pd.DataFrame:
    """One row per photo that has any mask: has_<part> flags, has_underside
    (gills or pores) and has_any_mask. Reads only `label` and `file_name`."""
    if split == "train":
        raise ValueError("train masks do not join to the train metadata; use val or test")
    table = pq.read_table(mask_file(split, dest), columns=["label", "file_name"]).to_pandas()
    flags = pd.crosstab(table["file_name"], table["label"]).gt(0)
    out = pd.DataFrame(index=flags.index)
    for part in PRESENCE_PARTS:
        out[f"has_{part}"] = flags[part] if part in flags.columns else False
    out["has_underside"] = out["has_gills"] | out["has_pores"]
    out["has_any_mask"] = True
    return out.reset_index().rename(columns={"file_name": "filename"})


def decode_mask(rle: Iterable[int], width: int, height: int) -> np.ndarray:
    """Decode one mask to a bool array [height, width]."""
    counts = np.asarray(rle, dtype=np.int64)
    n = int(width) * int(height)
    ends = np.cumsum(counts)
    starts = ends - counts
    fg_starts, fg_ends = starts[1::2], ends[1::2]
    keep = fg_starts < n
    flat = np.zeros(n + 1, dtype=np.int32)
    np.add.at(flat, fg_starts[keep], 1)
    np.add.at(flat, np.minimum(fg_ends[keep], n), -1)
    return (np.cumsum(flat[:n]) > 0).reshape(int(height), int(width))


def load_part_masks(
    split: str,
    filenames: Iterable[str],
    parts: Iterable[str] = ("fruiting_body", "cap", "gills", "stem"),
    dest: Path = DEFAULT_DEST,
) -> dict[str, dict[str, np.ndarray]]:
    """Decoded masks for a small set of photos: {filename: {part: bool [h, w]}}.

    Several instances of one part in a photo are merged. Streams the file in
    batches so memory stays bounded.
    """
    wanted, parts = set(filenames), set(parts)
    out: dict[str, dict[str, np.ndarray]] = {}
    pf = pq.ParquetFile(mask_file(split, dest))
    for batch in pf.iter_batches(batch_size=BATCH_ROWS):
        d = batch.to_pydict()
        for label, fn, w, h, rle in zip(d["label"], d["file_name"], d["width"], d["height"], d["rle"]):
            if fn not in wanted or label not in parts:
                continue
            m = decode_mask(rle, w, h)
            slot = out.setdefault(fn, {})
            slot[label] = slot[label] | m if label in slot else m
    return out


def mask_to_patch_grid(mask: np.ndarray, image_size: int, patch_size: int) -> np.ndarray:
    """Project a full-resolution mask through the eval transform's geometry (resize
    the shorter side, centre-crop to image_size) and area-average it onto the patch
    grid. Returns the covered fraction of each patch, float32 [g, g]."""
    from fungai.data.dataset import eval_resize_size

    if image_size % patch_size:
        raise ValueError(f"image_size {image_size} is not a multiple of patch_size {patch_size}")
    h, w = mask.shape
    short = eval_resize_size(image_size)
    if w <= h:
        new_w, new_h = short, int(short * h / w)
    else:
        new_w, new_h = int(short * w / h), short
    img = Image.fromarray(mask.astype(np.uint8) * 255).resize((new_w, new_h), Image.BILINEAR)
    left, top = (new_w - image_size) // 2, (new_h - image_size) // 2
    crop = np.asarray(img.crop((left, top, left + image_size, top + image_size)), dtype=np.float32) / 255
    g = image_size // patch_size
    return crop.reshape(g, patch_size, g, patch_size).mean(axis=(1, 3))
