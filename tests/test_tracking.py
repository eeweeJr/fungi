"""Tests for fungai.tracking.mlflow_utils — experiment setup, logging."""

import mlflow
import pytest

from fungai.tracking.mlflow_utils import get_git_hash, log_config_artifact, log_epoch_metrics, setup_experiment


@pytest.fixture(autouse=True)
def _isolated_mlflow(tmp_path):
    db_path = tmp_path / "mlflow.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    yield
    if mlflow.active_run():
        mlflow.end_run()


def test_get_git_hash():
    h = get_git_hash()
    assert isinstance(h, str)
    assert len(h) == 40 or h == "unknown"


def test_setup_experiment():
    eid = setup_experiment("test-experiment")
    assert isinstance(eid, str)
    exp = mlflow.get_experiment_by_name("test-experiment")
    assert exp is not None


def test_log_epoch_metrics():
    setup_experiment("test-metrics")
    with mlflow.start_run():
        log_epoch_metrics(0, {"train_loss": 1.5, "val_loss": 1.8, "lr": 0.001})
        log_epoch_metrics(1, {"train_loss": 1.2, "val_loss": 1.5, "lr": 0.0008})


def test_log_config_artifact():
    setup_experiment("test-artifact")
    with mlflow.start_run():
        log_config_artifact({"model.backbone": "resnet18", "training.lr": 0.001})
