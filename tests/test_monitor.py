"""RunMonitor: progress, memory, console capture, failure status and the stale-run reaper,
all against a throwaway SQLite MLflow store."""

from __future__ import annotations

import logging
import time

import mlflow
import pytest
from mlflow.tracking import MlflowClient

from fungai.tracking import monitor as monitor_mod
from fungai.tracking.monitor import HEARTBEAT_TAG, RunMonitor, reap


@pytest.fixture
def tracking(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("monitor-test")
    yield MlflowClient(uri)
    if mlflow.active_run():
        mlflow.end_run()


class _Loader(list):
    batch_size = 4


def _artifact_text(client, run_id, path, tmp_path) -> str:
    local = client.download_artifacts(run_id, path, str(tmp_path / "dl"))
    with open(local) as f:
        return f.read()


def _start(**kw):
    return mlflow.start_run(run_name="t", log_system_metrics=False, **kw)


def test_success_logs_progress_memory_console_and_finishes(tracking, tmp_path):
    run = _start()
    with RunMonitor(tmp_path / "run", total_epochs=2, progress_every_s=0,
                    memory_limit_gb=16) as mon:
        mon.phase("loading data")
        print("hello from print")
        logging.getLogger("fungai.test").warning("something odd")
        for epoch in range(2):
            mon.start_epoch(epoch)
            for _ in mon.track(_Loader(range(5)), "train"):
                mon.record_loss(1.5)
            for _ in mon.track(_Loader(range(3)), "val"):
                pass
            mon.end_epoch({"val_macro_f1": 0.5, "train_loss": 1.5})
    r = tracking.get_run(run.info.run_id)
    assert r.info.status == "FINISHED"
    assert mlflow.active_run() is None
    m = r.data.metrics
    assert m["progress/train_pct"] == 100.0
    assert m["progress/val_pct"] == 100.0
    assert m["train/loss_running"] == pytest.approx(1.5)
    assert m["progress/run_pct"] == 100.0
    assert "memory/total_gb" in m and m["memory/limit_gb"] == 16
    assert "progress/train_items_per_s" in m
    hist = tracking.get_metric_history(run.info.run_id, "progress/train_pct")
    assert max(h.step for h in hist) == 10  # global train batch counter
    tags = r.data.tags
    assert tags["phase"] == "finished"
    assert "Warnings / errors logged:** 1 / 0" in tags["mlflow.note.content"]
    assert HEARTBEAT_TAG in tags
    console = _artifact_text(tracking, run.info.run_id, "logs/console.log", tmp_path)
    assert "hello from print" in console
    assert "WARNING fungai.test: something odd" in console


def test_exception_marks_run_failed_with_traceback(tracking, tmp_path):
    run = _start()
    with pytest.raises(ValueError, match="boom"), RunMonitor(tmp_path / "run", total_epochs=1):
        raise ValueError("boom")
    r = tracking.get_run(run.info.run_id)
    assert r.info.status == "FAILED"
    assert r.data.tags["error.type"] == "ValueError"
    assert r.data.tags["error.message"] == "boom"
    assert "test_monitor.py" in r.data.tags["error.where"]
    tb = _artifact_text(tracking, run.info.run_id, "logs/traceback.txt", tmp_path)
    assert "ValueError: boom" in tb


def test_ctrl_c_marks_run_killed(tracking, tmp_path):
    run = _start()
    with pytest.raises(KeyboardInterrupt), RunMonitor(tmp_path / "run", total_epochs=1):
        raise KeyboardInterrupt
    assert tracking.get_run(run.info.run_id).info.status == "KILLED"


def test_preflight_exit_marks_run_failed(tracking, tmp_path):
    run = _start()
    with pytest.raises(SystemExit), RunMonitor(tmp_path / "run", total_epochs=1):
        raise SystemExit(1)
    assert tracking.get_run(run.info.run_id).info.status == "FAILED"


def test_streams_and_logging_restored(tracking, tmp_path):
    import sys

    out, err, handlers = sys.stdout, sys.stderr, list(logging.getLogger().handlers)
    _start()
    with RunMonitor(tmp_path / "run", total_epochs=1):
        assert sys.stdout is not out
    assert sys.stdout is out and sys.stderr is err
    assert logging.getLogger().handlers == handlers


def test_watchdog_kill_is_recorded_before_the_kill(tracking, tmp_path, monkeypatch):
    from fungai.safety import watchdog as wd_mod

    kills = []
    monkeypatch.setattr(wd_mod.os, "kill", lambda pid, sig: kills.append(sig))
    run = _start()
    wd = wd_mod.MemoryWatchdog(limit_gb=0.0001, poll_interval_s=0.05)
    with RunMonitor(tmp_path / "run", total_epochs=1) as mon:
        mon.attach_watchdog(wd)
        wd.start()
        deadline = time.time() + 5
        while not kills and time.time() < deadline:
            time.sleep(0.05)
        wd.stop()
        r = tracking.get_run(run.info.run_id)
        assert kills, "watchdog never fired"
        assert r.data.tags["error.type"] == "MemoryLimitExceeded"
        assert r.info.status == "KILLED"


def test_reap_marks_dead_stale_runs_failed(tracking, tmp_path):
    exp = mlflow.get_experiment_by_name("monitor-test")
    dead = tracking.create_run(exp.experiment_id, tags={
        HEARTBEAT_TAG: str(int(time.time()) - 3600), "monitor.pid": "999999",
        "monitor.host": monitor_mod.socket.gethostname()})
    fresh = tracking.create_run(exp.experiment_id, tags={HEARTBEAT_TAG: str(int(time.time()))})
    assert reap(dry_run=True) == [dead.info.run_id]
    assert tracking.get_run(dead.info.run_id).info.status == "RUNNING"
    assert reap() == [dead.info.run_id]
    assert tracking.get_run(dead.info.run_id).info.status == "FAILED"
    assert tracking.get_run(dead.info.run_id).data.tags["error.type"] == "ProcessLost"
    assert tracking.get_run(fresh.info.run_id).info.status == "RUNNING"
