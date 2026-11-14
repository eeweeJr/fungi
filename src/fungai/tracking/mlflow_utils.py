"""MLflow experiment setup, run context management, and artifact logging helpers."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

import mlflow
import yaml

from fungai.config import ExperimentConfig, config_to_flat_dict

MLFLOW_DB = Path("mlflow.db")


def _ensure_tracking_uri() -> None:
    """Set MLflow to use a local SQLite database if no URI has been configured."""
    current = mlflow.get_tracking_uri()
    if current and not current.startswith("file:"):
        return
    db_path = MLFLOW_DB.resolve()
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")


def setup_experiment(experiment_name: str) -> str:
    """Set or create an MLflow experiment. Returns experiment_id."""
    _ensure_tracking_uri()
    mlflow.set_experiment(experiment_name)
    exp = mlflow.get_experiment_by_name(experiment_name)
    return exp.experiment_id


def get_git_hash() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def start_tracked_run(config: ExperimentConfig, run_name: str | None = None):
    """Set up experiment, start MLflow run, log flattened params + git hash + config artifact."""
    setup_experiment(config.experiment_name)

    run_name = run_name or config.run_name or (
        f"{config.model.backbone}_bs{config.data.batch_size}"
    )
    run = mlflow.start_run(run_name=run_name)

    flat = config_to_flat_dict(config)
    for k, v in flat.items():
        mlflow.log_param(k, str(v)[:500])

    mlflow.log_param("git_hash", get_git_hash())

    log_config_artifact(flat)

    return run


def log_config_artifact(config_dict: dict, filename: str = "config.yaml") -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f, default_flow_style=False)
        f.flush()
        mlflow.log_artifact(f.name, artifact_path="config")


def log_evaluation_report(report: dict, filename: str = "eval_report.json") -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(report, f, indent=2, default=str)
        f.flush()
        mlflow.log_artifact(f.name, artifact_path="evaluation")


def log_model_checkpoint(model_path: Path) -> None:
    mlflow.log_artifact(str(model_path), artifact_path="checkpoints")


def log_epoch_metrics(epoch: int, metrics: dict) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            mlflow.log_metric(key, value, step=epoch)
