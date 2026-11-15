"""Tests for fungai.tracking.hub against a fake Hub backed by a local folder."""

import fnmatch
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import mlflow
import pytest
import torch
import yaml
from huggingface_hub.hf_api import RepoFile, RepoFolder

from fungai.config import load_config
from fungai.tracking import hub


class FakeHub:
    """Implements the HfApi calls hub.py makes, storing repos under a local folder."""

    def __init__(self, root: Path, private: bool = True):
        self.root, self.private, self.commits = root, private, 0

    def create_repo(self, repo_id, repo_type, private, exist_ok):
        repo = self.root / repo_id
        if repo.exists() and not exist_ok:
            raise FileExistsError(repo_id)
        if not repo.exists():
            repo.mkdir(parents=True)
            self.private = private

    def repo_info(self, repo_id, repo_type):
        return SimpleNamespace(private=self.private)

    def upload_folder(self, repo_id, repo_type, folder_path, path_in_repo, allow_patterns, commit_message):
        dest = self.root / repo_id / path_in_repo
        dest.mkdir(parents=True, exist_ok=True)
        for f in Path(folder_path).iterdir():
            if any(fnmatch.fnmatch(f.name, pat) for pat in allow_patterns):
                shutil.copy(f, dest / f.name)
        self.commits += 1
        oid = f"{self.commits:040d}"
        return SimpleNamespace(oid=oid, commit_url=f"https://huggingface.co/{repo_id}/commit/{oid}")

    def snapshot_download(self, repo_id, repo_type, revision, local_dir, allow_patterns):
        repo = self.root / repo_id
        for f in repo.rglob("*"):
            rel = f.relative_to(repo).as_posix()
            if f.is_file() and any(fnmatch.fnmatch(rel, pat) for pat in allow_patterns):
                (Path(local_dir) / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, Path(local_dir) / rel)
        return local_dir

    def list_repo_tree(self, repo_id, path_in_repo, repo_type):
        base = self.root / repo_id / path_in_repo
        for p in sorted(base.iterdir()) if base.exists() else []:
            rel = f"{path_in_repo}/{p.name}"
            if p.is_dir():
                yield RepoFolder(path=rel, oid="0")
            else:
                yield RepoFile(path=rel, size=p.stat().st_size, oid="0")


@pytest.fixture
def run_dir(tmp_path):
    d = tmp_path / "outputs" / "fungai-dev_v0.7"
    d.mkdir(parents=True)
    torch.save({"model_state_dict": {"w": torch.arange(6.0)}, "epoch": 3}, d / "best.pth")
    torch.save({"model_state_dict": {"w": torch.zeros(6)}}, d / "last.pth")
    (d / "config.yaml").write_text("seed: 42\n")
    (d / "eval_report.json").write_text("{}")
    (d / "data_snapshot.json").write_text(json.dumps({"snapshot": "mini_open_abc"}))
    (d / "val_probs.npy").write_bytes(b"x" * 10)
    return d


@pytest.fixture
def fake(tmp_path):
    return FakeHub(tmp_path / "remote")


INFO = hub.build_run_info("fungai-dev_v0.7", "mlflow-id-1", "fungai-dev", "abc123", "mini_open_abc",
                          best_epoch=3)


def test_push_uploads_only_best_checkpoint_and_metadata(run_dir, fake):
    record = hub.push_best_checkpoint(run_dir, "me/ckpts", INFO, api=fake)
    remote = fake.root / "me/ckpts/runs/fungai-dev_v0.7"
    assert sorted(p.name for p in remote.iterdir()) == sorted(
        ["best.pth", "run_info.json", "config.yaml", "eval_report.json", "data_snapshot.json"])
    assert (remote / "best.pth").read_bytes() == (run_dir / "best.pth").read_bytes()  # unchanged
    assert record["path"] == "runs/fungai-dev_v0.7"
    assert record["uri"] == f"hf://me/ckpts/runs/fungai-dev_v0.7/best.pth@{record['commit']}"
    assert json.loads((run_dir / "hub.json").read_text()) == record


def test_checkpoint_maps_back_to_its_run(run_dir, fake):
    hub.push_best_checkpoint(run_dir, "me/ckpts", INFO, api=fake)
    info = json.loads((fake.root / "me/ckpts/runs/fungai-dev_v0.7/run_info.json").read_text())
    assert info["mlflow_run_id"] == "mlflow-id-1"
    assert info["git_commit"] == "abc123"
    assert info["data_snapshot"] == "mini_open_abc"
    assert info["best_epoch"] == 3


def test_push_refuses_a_public_repo(run_dir, fake):
    (fake.root / "me/public").mkdir(parents=True)
    fake.private = False
    with pytest.raises(RuntimeError, match="public"):
        hub.push_best_checkpoint(run_dir, "me/public", INFO, api=fake)
    assert not (fake.root / "me/public/runs").exists()


def test_push_needs_best_checkpoint(run_dir, fake):
    (run_dir / "best.pth").unlink()
    with pytest.raises(FileNotFoundError):
        hub.push_best_checkpoint(run_dir, "me/ckpts", INFO, api=fake)


def test_pull_restores_the_checkpoint(run_dir, fake, tmp_path):
    hub.push_best_checkpoint(run_dir, "me/ckpts", INFO, api=fake)
    restored = hub.pull_run("fungai-dev_v0.7", "me/ckpts", dest=tmp_path / "elsewhere", api=fake)
    assert restored == tmp_path / "elsewhere" / "fungai-dev_v0.7"
    assert (restored / "best.pth").read_bytes() == (run_dir / "best.pth").read_bytes()
    assert torch.load(restored / "best.pth", weights_only=True)["epoch"] == 3
    with pytest.raises(FileNotFoundError, match="no run"):
        hub.pull_run("missing", "me/ckpts", dest=tmp_path, api=fake)


def test_remote_runs_lists_run_folders(run_dir, fake):
    assert hub.remote_runs("me/ckpts", api=fake) == []
    hub.push_best_checkpoint(run_dir, "me/ckpts", INFO, api=fake)
    other = run_dir.parent / "fungai-dev_v0.8"
    shutil.copytree(run_dir, other)
    hub.push_best_checkpoint(other, "me/ckpts", INFO, api=fake)
    assert hub.remote_runs("me/ckpts", api=fake) == ["fungai-dev_v0.7", "fungai-dev_v0.8"]


def test_location_is_tagged_on_the_mlflow_run(run_dir, fake, tmp_path):
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow.db'}")
    mlflow.set_experiment("hub-test")
    record = hub.push_best_checkpoint(run_dir, "me/ckpts", INFO, api=fake)
    with mlflow.start_run() as run:
        hub.log_checkpoint_location(record)
    tags = mlflow.get_run(run.info.run_id).data.tags
    assert tags["checkpoint.path"] == "runs/fungai-dev_v0.7"
    assert tags["checkpoint.commit"] == record["commit"]


def test_hub_config_defaults_off_and_requires_repo(tmp_path):
    cfg = load_config("configs/default.yaml")
    assert cfg.hub.push_best_checkpoint is False
    p = tmp_path / "c.yaml"
    p.write_text(yaml.dump({"hub": {"push_best_checkpoint": True}}))
    with pytest.raises(ValueError, match="repo_id"):
        load_config(p)
    p.write_text(yaml.dump({"hub": {"push_best_checkpoint": True, "repo_id": "me/ckpts"}}))
    assert load_config(p).hub.repo_id == "me/ckpts"
