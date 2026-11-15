"""Store each run's best checkpoint in a private Hugging Face model repo.

Layout in the repo:

    runs/<run name>/
        best.pth             the checkpoint, unchanged
        run_info.json        how to map it back: MLflow run id, git commit, data snapshot, ...
        config.yaml          the run's config
        eval_report.json     the run's final validation report
        data_snapshot.json   the split snapshot it was trained on

The MLflow run gets `checkpoint.*` tags pointing at the exact Hub commit, and the
local run folder gets a hub.json with the same record, so you can go either way:
run -> checkpoint (tags) or checkpoint -> run (run_info.json).

Login once with `hf auth login` (or set HF_TOKEN); nothing here handles tokens.

CLI:
    python -m fungai.tracking.hub push outputs/<run name> [--repo user/fungai-checkpoints]
    python -m fungai.tracking.hub pull <run name>
    python -m fungai.tracking.hub pull --mlflow-run-id <id>
    python -m fungai.tracking.hub list
The repo defaults to $FUNGAI_HUB_REPO, then to hub.repo_id in configs/default.yaml.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

UPLOAD_FILES = ("best.pth", "run_info.json", "config.yaml", "eval_report.json", "data_snapshot.json")
RUNS_PREFIX = "runs"


def _api(api=None):
    if api is not None:
        return api
    from huggingface_hub import HfApi

    return HfApi()


def build_run_info(run_name: str, mlflow_run_id: str | None, experiment: str | None,
                   git_commit: str | None, data_snapshot: str | None, **extra) -> dict:
    return {
        "run_name": run_name,
        "mlflow_run_id": mlflow_run_id,
        "experiment": experiment,
        "git_commit": git_commit,
        "data_snapshot": data_snapshot,
        "pushed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        **extra,
    }


def ensure_private_repo(repo_id: str, api=None) -> None:
    """Create the repo as private if it does not exist; refuse to use a public one."""
    api = _api(api)
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    if not api.repo_info(repo_id, repo_type="model").private:
        raise RuntimeError(
            f"{repo_id} is public. Checkpoints are only pushed to private repos; make it "
            "private on huggingface.co or choose another repo."
        )


def push_best_checkpoint(run_dir: str | Path, repo_id: str, run_info: dict, api=None) -> dict:
    """Upload a run's best checkpoint and its metadata. Returns the location record."""
    run_dir = Path(run_dir)
    if not (run_dir / "best.pth").exists():
        raise FileNotFoundError(f"{run_dir / 'best.pth'} not found")
    api = _api(api)
    ensure_private_repo(repo_id, api)
    (run_dir / "run_info.json").write_text(json.dumps(run_info, indent=2, default=str))

    path_in_repo = f"{RUNS_PREFIX}/{run_dir.name}"
    commit = api.upload_folder(
        repo_id=repo_id, repo_type="model", folder_path=str(run_dir), path_in_repo=path_in_repo,
        allow_patterns=list(UPLOAD_FILES),
        commit_message=f"Add best checkpoint for {run_dir.name}",
    )
    record = {
        "repo_id": repo_id,
        "path": path_in_repo,
        "commit": commit.oid,
        "url": commit.commit_url,
        "uri": f"hf://{repo_id}/{path_in_repo}/best.pth@{commit.oid}",
    }
    (run_dir / "hub.json").write_text(json.dumps(record, indent=2))
    return record


def log_checkpoint_location(record: dict) -> None:
    """Tag the active MLflow run with where its checkpoint lives."""
    import mlflow

    for key in ("repo_id", "path", "commit", "url", "uri"):
        mlflow.set_tag(f"checkpoint.{key}", record[key])


def pull_run(run_name: str, repo_id: str, dest: str | Path = "outputs",
             revision: str | None = None, api=None) -> Path:
    """Download a run folder from the repo into dest/<run name>. Returns that folder."""
    api = _api(api)
    target = Path(dest) / run_name
    with tempfile.TemporaryDirectory() as staging:
        api.snapshot_download(
            repo_id, repo_type="model", revision=revision, local_dir=staging,
            allow_patterns=[f"{RUNS_PREFIX}/{run_name}/*"],
        )
        source = Path(staging) / RUNS_PREFIX / run_name
        if not source.exists():
            raise FileNotFoundError(f"no run {run_name!r} in {repo_id}")
        target.mkdir(parents=True, exist_ok=True)
        for file in source.iterdir():
            if file.is_file():
                shutil.move(str(file), target / file.name)
    return target


def remote_runs(repo_id: str, api=None) -> list[str]:
    from huggingface_hub.hf_api import RepoFolder

    api = _api(api)
    entries = api.list_repo_tree(repo_id, path_in_repo=RUNS_PREFIX, repo_type="model")
    return sorted(Path(e.path).name for e in entries if isinstance(e, RepoFolder))


# --------------------------------------------------------------------------- #
# MLflow lookups (for runs pushed after the fact and for pulling by run id)
# --------------------------------------------------------------------------- #

def _mlflow_client():
    from mlflow import MlflowClient

    from fungai.tracking.mlflow_utils import _ensure_tracking_uri

    _ensure_tracking_uri()
    return MlflowClient()


def find_mlflow_run(run_name: str):
    """The MLflow run whose `version` param is this run name, or None."""
    client = _mlflow_client()
    experiments = [e.experiment_id for e in client.search_experiments()]
    runs = client.search_runs(experiments, filter_string=f"params.version = '{run_name}'")
    return runs[0] if runs else None


def run_info_from_mlflow(run_dir: Path) -> dict:
    """Rebuild run_info for a run folder that was not pushed during training."""
    run = find_mlflow_run(run_dir.name)
    snapshot = None
    if (run_dir / "data_snapshot.json").exists():
        snapshot = json.loads((run_dir / "data_snapshot.json").read_text())["snapshot"]
    if run is None:
        return build_run_info(run_dir.name, None, None, None, snapshot)
    experiment = _mlflow_client().get_experiment(run.info.experiment_id).name
    return build_run_info(run_dir.name, run.info.run_id, experiment,
                          run.data.params.get("git_hash"), snapshot)


def default_repo() -> str | None:
    if os.environ.get("FUNGAI_HUB_REPO"):
        return os.environ["FUNGAI_HUB_REPO"]
    try:
        from fungai.config import load_config

        return load_config("configs/default.yaml").hub.repo_id
    except Exception:  # noqa: BLE001 - no config in this directory
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Push and pull best checkpoints on the Hugging Face Hub")
    parser.add_argument("--repo", default=None, help="e.g. your-username/fungai-checkpoints")
    sub = parser.add_subparsers(dest="command", required=True)
    push = sub.add_parser("push", help="upload a finished run's best checkpoint")
    push.add_argument("run_dir", type=Path)
    pull = sub.add_parser("pull", help="download a run's checkpoint and metadata into outputs/")
    pull.add_argument("run_name", nargs="?")
    pull.add_argument("--mlflow-run-id")
    pull.add_argument("--dest", type=Path, default=Path("outputs"))
    sub.add_parser("list", help="list runs stored in the repo")
    args = parser.parse_args()

    repo = args.repo or default_repo()
    if not repo:
        parser.error("no repo: pass --repo, set FUNGAI_HUB_REPO, or set hub.repo_id in the config")

    if args.command == "push":
        record = push_best_checkpoint(args.run_dir, repo, run_info_from_mlflow(args.run_dir))
        run = find_mlflow_run(args.run_dir.name)
        if run is not None:
            client = _mlflow_client()
            for key in ("repo_id", "path", "commit", "url", "uri"):
                client.set_tag(run.info.run_id, f"checkpoint.{key}", record[key])
        print(f"pushed {args.run_dir.name} -> {record['url']}")
    elif args.command == "pull":
        name, revision = args.run_name, None
        if args.mlflow_run_id:
            tags = _mlflow_client().get_run(args.mlflow_run_id).data.tags
            if "checkpoint.path" not in tags:
                raise SystemExit(f"MLflow run {args.mlflow_run_id} has no checkpoint tags")
            name, revision = Path(tags["checkpoint.path"]).name, tags["checkpoint.commit"]
        if not name:
            parser.error("pull needs a run name or --mlflow-run-id")
        print(f"pulled into {pull_run(name, repo, args.dest, revision)}")
    else:
        print("\n".join(remote_runs(repo)) or "(no runs yet)")


if __name__ == "__main__":
    main()
