"""Network-free unit tests for the ingest layer."""

import json
import zipfile

import pandas as pd
import pytest

from fungai.data.ingest import (
    Archive,
    build_archives,
    download_archive,
    extra_archive,
    image_archive,
    is_ingested,
    normalize_metadata,
    resolve_column,
)

DOWNLOAD_ROOT = "https://cmp.felk.cvut.cz/datagrid/FungiTastic/shared/download"


class TestUrlRegistry:
    def test_mini_image_url(self):
        ar = image_archive("mini", "train", "300")
        assert ar.name == "FungiTastic-Mini-train-300p.zip"
        assert ar.url == f"{DOWNLOAD_ROOT}/FungiTastic-Mini-train-300p.zip"
        assert ar.kind == "images"

    def test_full_subset_has_no_token(self):
        ar = image_archive("full", "val", "500")
        assert ar.name == "FungiTastic-val-500p.zip"

    def test_fewshot_naming(self):
        ar = image_archive("fs", "test", "300")
        assert ar.name == "FungiTastic-FewShot-test-300p.zip"

    def test_fullsize_has_no_p_suffix(self):
        ar = image_archive("mini", "train", "fullsize")
        assert ar.name == "FungiTastic-Mini-train-fullsize.zip"

    def test_dna_split(self):
        ar = image_archive("mini", "dna-test", "300")
        assert ar.name == "FungiTastic-Mini-dna-test-300p.zip"

    def test_fewshot_has_no_dna_split(self):
        with pytest.raises(ValueError):
            image_archive("fs", "dna-test", "300")

    def test_invalid_inputs_rejected(self):
        with pytest.raises(ValueError):
            image_archive("mega", "train", "300")
        with pytest.raises(ValueError):
            image_archive("mini", "holdout", "300")
        with pytest.raises(ValueError):
            image_archive("mini", "train", "1080")

    def test_extras(self):
        assert extra_archive("metadata").url == f"{DOWNLOAD_ROOT}/metadata.zip"
        assert extra_archive("climatic").kind == "climate"
        assert extra_archive("satellite_rgb").name == "satellite_RGB.zip"

    def test_build_archives_composition(self):
        archives = build_archives("mini", "300", ("train", "val"), ("metadata",))
        names = [a.name for a in archives]
        assert names == [
            "metadata.zip",
            "FungiTastic-Mini-train-300p.zip",
            "FungiTastic-Mini-val-300p.zip",
        ]


class TestManifestSkip:
    def _fake_zip_archive(self, tmp_path, content=b"hello"):
        ar = Archive(name="test.zip", url="file://unused", kind="metadata")
        zip_path = tmp_path / ar.name
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("inner.csv", content)
        return ar, zip_path

    def test_extract_writes_manifest_and_removes_zip(self, tmp_path):
        ar, zip_path = self._fake_zip_archive(tmp_path)
        download_archive(ar, tmp_path)  # zip already present -> no network touched
        assert not zip_path.exists()
        assert (tmp_path / "metadata" / "inner.csv").read_bytes() == b"hello"
        manifest = json.loads((tmp_path / ".manifest" / "test.zip.json").read_text())
        assert manifest["entries"] == 1
        assert manifest["sha256"]

    def test_is_ingested_requires_manifest_and_dir(self, tmp_path):
        ar, _ = self._fake_zip_archive(tmp_path)
        assert not is_ingested(ar, tmp_path)
        download_archive(ar, tmp_path)
        assert is_ingested(ar, tmp_path)

    def test_second_run_skips(self, tmp_path):
        ar, _ = self._fake_zip_archive(tmp_path)
        download_archive(ar, tmp_path)
        # would need network if it did not skip -> passing means it skipped
        download_archive(ar, tmp_path)

    def test_keep_zip(self, tmp_path):
        ar, zip_path = self._fake_zip_archive(tmp_path)
        download_archive(ar, tmp_path, keep_zip=True)
        assert zip_path.exists()

    def test_corrupt_zip_deleted_and_raises(self, tmp_path):
        ar = Archive(name="bad.zip", url="file://unused", kind="metadata")
        bad = tmp_path / ar.name
        bad.write_bytes(b"this is not a zip file")
        with pytest.raises(zipfile.BadZipFile):
            download_archive(ar, tmp_path)
        assert not bad.exists()


class TestMetadataNormalization:
    def test_renames_and_dtypes(self):
        df = pd.DataFrame({
            "scientificName": ["Amanita muscaria", "Boletus edulis"],
            "Latitude": [55.6, 56.1],
            "Longitude": [12.5, 10.2],
            "eventDate": ["2021-09-15", "2021-10-01"],
            "Habitat": ["Mixed woodland", "Deciduous woodland"],
            "poisonous": [True, False],
            "month": [9, 10],
        })
        out = normalize_metadata(df)
        assert {"species", "latitude", "longitude", "date", "habitat",
                    "toxicity", "month"} <= set(out.columns)
        assert out["date"].dtype.kind == "M"
        assert out["species"].dtype == "category"
        assert out["habitat"].dtype == "category"

    def test_unparseable_dates_become_nat(self):
        out = normalize_metadata(pd.DataFrame({"date": ["2021-09-15", "not-a-date"]}))
        assert out["date"].isna().sum() == 1

    def test_resolve_column_miss_returns_none(self):
        assert resolve_column(pd.DataFrame({"x": [1]}), "species") is None

    def test_already_canonical_names_untouched(self):
        df = pd.DataFrame({"species": ["a"], "genus": ["b"]})
        out = normalize_metadata(df)
        assert list(out.columns) == ["species", "genus"]
