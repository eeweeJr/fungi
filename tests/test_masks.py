"""Tests for fungai.data.masks: RLE decoding, patch-grid projection, presence table."""

import numpy as np
import pytest

from fungai.data.masks import decode_mask, load_part_masks, load_part_presence, mask_to_patch_grid


def test_decode_mask_alternating_runs_row_major():
    # 4x3 image: background 2, foreground 3, background 7 (runs overshoot by 5)
    m = decode_mask([2, 3, 12], width=4, height=3)
    assert m.shape == (3, 4)
    assert m.ravel().tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]


def test_decode_mask_clips_runs_past_the_image():
    m = decode_mask([10, 50], width=4, height=3)  # foreground starts at 10 and overruns
    assert m.ravel().tolist() == [0] * 10 + [1, 1]


def test_mask_to_patch_grid_follows_eval_geometry():
    # 200 wide x 100 tall, left half foreground. Eval: shorter side -> 36, crop 32 centred.
    mask = np.zeros((100, 200), bool)
    mask[:, :100] = True
    grid = mask_to_patch_grid(mask, image_size=32, patch_size=8)
    assert grid.shape == (4, 4)
    assert grid[:, :2].mean() == pytest.approx(1.0, abs=0.05)
    assert grid[:, 2:].mean() == pytest.approx(0.0, abs=0.05)
    with pytest.raises(ValueError):
        mask_to_patch_grid(mask, image_size=30, patch_size=8)


@pytest.mark.data
def test_real_mask_pinned_image():
    """5-3925712883.JPG (val): a bolete-shaped cap on a stem, checked by overlay.
    Cap and stem lie inside the fruiting body; gills are present."""
    fn = "5-3925712883.JPG"
    try:
        masks = load_part_masks("val", [fn], parts=("fruiting_body", "cap", "stem", "gills"))
    except FileNotFoundError:
        pytest.skip("masks not downloaded")
    m = masks[fn]
    assert m["fruiting_body"].shape == (1200, 1600)
    for part in ("cap", "stem", "gills"):
        inside = (m[part] & m["fruiting_body"]).sum() / m[part].sum()
        assert inside > 0.99, part
    assert 0.1 < m["fruiting_body"].mean() < 0.6
    pres = load_part_presence("val").set_index("filename").loc[fn]
    assert pres.has_gills and pres.has_underside and pres.has_cap and pres.has_stem
