"""Live run monitoring: what you would otherwise watch in the terminal, in MLflow.

One RunMonitor wraps one MLflow run. While it is open it:

  progress   logs progress inside every epoch and validation pass (fraction done,
             batches/s, ETA for the stage and the whole run) under `progress/*`, plus
             a running train loss under `train/loss_running` when the loop reports it;
  status     keeps a live status in the run's description (Overview tab) and in the
             `status` and `phase` tags, so the runs table shows where each run is;
  memory     logs this process's memory as the watchdog sees it (RSS + MPS/CUDA) and
             the headroom to the limit under `memory/*`. MLflow's own system metrics
             (`system/*`: CPU, RAM, disk, network) are turned on by start_tracked_run;
             MLflow has no Apple GPU monitor, so MPS memory comes from here;
  console    tees stdout, stderr and logging into <run_dir>/console.log and uploads it
             to Artifacts > logs/console.log every minute; warnings and errors logged
             anywhere are counted (`log/warnings`, `log/errors`) and the latest shown;
  failures   on an exception records the traceback (tags `error.*`, artifact
             logs/traceback.txt) and ends the run FAILED; Ctrl-C ends it KILLED, and so
             does the memory watchdog, which reports before it kills the process.

A process that dies without warning (power loss, kill -9, a segfault) cannot end its
run. Its `monitor.heartbeat` tag stops moving, a segfault's stack goes to console.log,
and `python -m fungai.tracking.monitor reap` marks such runs FAILED and uploads their
console log.
"""

from __future__ import annotations

import argparse
import faulthandler
import logging
import math
import os
import socket
import sys
import threading
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Self

import psutil

logger = logging.getLogger("fungai.monitor")

HEARTBEAT_TAG = "monitor.heartbeat"
STALE_AFTER_S = 600
_NOTE_TAG = "mlflow.note.content"  # the run description shown on the Overview tab


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None or math.isnan(seconds) or seconds < 0:
        return "?"
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


class _Sink:
    """Thread-safe, line-timestamped writer for console.log."""

    def __init__(self, path: Path):
        self.path = path
        self._f = open(path, "a", buffering=1, encoding="utf-8")  # noqa: SIM115 - closed in close()
        self._lock = threading.Lock()
        self._line_start = True

    def write(self, s: str) -> int:
        if not s:
            return 0
        with self._lock:
            if self._f.closed:
                return len(s)
            out = []
            for piece in s.splitlines(keepends=True):
                if self._line_start:
                    out.append(time.strftime("%H:%M:%S "))
                out.append(piece)
                self._line_start = piece.endswith(("\n", "\r"))
            self._f.write("".join(out))
        return len(s)

    def flush(self) -> None:
        with self._lock:
            if not self._f.closed:
                self._f.flush()

    def fileno(self) -> int:
        return self._f.fileno()

    def close(self) -> None:
        with self._lock:
            self._f.close()


class _Tee:
    """Stands in for sys.stdout / sys.stderr: writes go to the terminal and the sink."""

    def __init__(self, stream, sink: _Sink):
        self._stream, self._sink = stream, sink

    def write(self, s: str) -> int:
        self._sink.write(s)
        return self._stream.write(s)

    def flush(self) -> None:
        self._sink.flush()
        self._stream.flush()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _LogTap(logging.Handler):
    """Copies log records into console.log and remembers warnings and errors."""

    def __init__(self, sink: _Sink, monitor: RunMonitor):
        super().__init__(logging.INFO)
        self.sink, self.monitor = sink, monitor
        self.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.sink.write(self.format(record) + "\n")
            if record.levelno >= logging.WARNING:
                self.monitor._note_problem(record)
        except Exception:  # noqa: BLE001, S110 - logging must never raise
            pass


class _Tracked:
    """Iterable wrapper that reports each batch of a loader to the monitor."""

    def __init__(self, loader, monitor: RunMonitor, stage: str):
        self._loader, self._monitor, self._stage = loader, monitor, stage

    def __len__(self) -> int:
        return len(self._loader)

    def __iter__(self):
        n = len(self._loader)
        self._monitor._begin_stage(self._stage, n, getattr(self._loader, "batch_size", None))
        try:
            for batch in self._loader:
                yield batch
                self._monitor._tick()
        finally:
            self._monitor._end_stage()

    def __getattr__(self, name):
        return getattr(self._loader, name)


class RunMonitor:
    """Context manager around an already-started MLflow run. See the module docstring.

    with RunMonitor(run_dir, total_epochs=cfg.training.epochs) as mon:
        for epoch in range(...):
            mon.start_epoch(epoch)
            for batch in mon.track(train_loader, "train"): ...; mon.record_loss(loss)
            evaluate(mon.track(val_loader, "val"))
            mon.end_epoch(metrics)
    """

    def __init__(self, run_dir: Path, total_epochs: int, watchdog=None, run_id: str | None = None,
                 progress_every_s: float = 30.0, heartbeat_every_s: float = 30.0,
                 upload_every_s: float = 60.0, memory_limit_gb: float | None = None,
                 client=None):
        import mlflow
        from mlflow.tracking import MlflowClient

        active = mlflow.active_run()
        if run_id is None and active is None:
            raise RuntimeError("RunMonitor needs an active MLflow run (or run_id)")
        self.run_id = run_id or active.info.run_id
        self.client = client or MlflowClient()
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.total_epochs = max(int(total_epochs), 1)
        self.watchdog = watchdog
        self.memory_limit_gb = memory_limit_gb or (
            watchdog.limit_bytes / 1024**3 if watchdog is not None else None)
        self.progress_every_s = progress_every_s
        self.heartbeat_every_s = heartbeat_every_s
        self.upload_every_s = upload_every_s
        self.console_path = self.run_dir / "console.log"

        self.t_start = time.time()
        self.phase_name = "starting"
        self.epoch: int | None = None
        self.epoch_label = ""
        self.epoch_t0: float | None = None
        self.epoch_durations: list[float] = []
        self.stage_durations: dict[str, float] = {}
        self.last_metrics: dict[str, float] = {}
        self.global_batch = 0
        self.n_warnings = 0
        self.n_errors = 0
        self.recent_problems: deque[str] = deque(maxlen=3)
        self._loss_sum = 0.0
        self._loss_n = 0
        self._stage: dict | None = None
        self._last_progress_log = 0.0
        self._last_status = ""
        self._mem_sample = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._finished = False
        self._proc = psutil.Process(os.getpid())

    # ------------------------------------------------------------------ public API

    def track(self, loader, stage: str) -> _Tracked:
        """Wrap a loader so iterating it reports progress for `stage` ("train", "val", ...)."""
        return _Tracked(loader, self, stage)

    def phase(self, name: str) -> None:
        """Name what the run is doing outside a tracked loop (loading data, final eval...)."""
        self.phase_name = name
        self._set_tag("phase", name)
        self._push_status(force=True)

    def start_epoch(self, epoch: int, label: str | None = None) -> None:
        self.epoch = epoch
        self.epoch_label = label or f"epoch {epoch + 1}/{self.total_epochs}"
        self.epoch_t0 = time.time()
        self.phase(f"{self.epoch_label}")

    def end_epoch(self, metrics: dict | None = None) -> None:
        if self.epoch_t0 is not None:
            self.epoch_durations.append(time.time() - self.epoch_t0)
        if metrics:
            self.last_metrics = {k: float(v) for k, v in metrics.items()
                                 if isinstance(v, (int, float)) and not isinstance(v, bool)}
        self._log_metrics({"progress/run_pct": 100.0 * len(self.epoch_durations) / self.total_epochs,
                           "progress/epochs_done": float(len(self.epoch_durations))},
                          step=self.global_batch)
        self._push_status(force=True)

    def record_loss(self, value: float) -> None:
        """Feed a per-batch training loss; its mean since the last log is `train/loss_running`."""
        self._loss_sum += float(value)
        self._loss_n += 1

    def attach_watchdog(self, watchdog) -> None:
        """Hook up a watchdog started after the monitor (e.g. by the preflight)."""
        if watchdog is None or watchdog is self.watchdog:
            return
        self.watchdog = watchdog
        self.memory_limit_gb = self.memory_limit_gb or watchdog.limit_bytes / 1024**3
        watchdog.add_on_limit(self._on_memory_kill)

    # ------------------------------------------------------------------ context manager

    def __enter__(self) -> Self:
        self._sink = _Sink(self.console_path)
        self._stdout, self._stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(self._stdout, self._sink), _Tee(self._stderr, self._sink)
        self._tap = _LogTap(self._sink, self)
        logging.getLogger().addHandler(self._tap)
        logging.captureWarnings(True)
        self._faulthandler_was_enabled = faulthandler.is_enabled()
        try:
            faulthandler.enable(file=self._sink._f, all_threads=True)
        except Exception:  # noqa: BLE001, S110 - nice to have
            pass
        self._set_tags({
            "monitor.pid": str(os.getpid()),
            "monitor.host": socket.gethostname(),
            "monitor.console_log": str(self.console_path.resolve()),
            "monitor.run_dir": str(self.run_dir.resolve()),
            HEARTBEAT_TAG: str(int(time.time())),
        })
        if self.watchdog is not None:
            self.watchdog.add_on_limit(self._on_memory_kill)
        self._thread = threading.Thread(target=self._background, daemon=True, name="run-monitor")
        self._thread.start()
        self.phase(self.phase_name)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        import mlflow

        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if exc_type is None:
            status, self.phase_name = "FINISHED", "finished"
        elif issubclass(exc_type, KeyboardInterrupt):
            status, self.phase_name = "KILLED", "interrupted (Ctrl-C)"
        elif issubclass(exc_type, SystemExit) and exc.code in (0, None):
            status, self.phase_name = "FINISHED", "finished"
        else:
            status, self.phase_name = "FAILED", "failed"
        if exc_type is not None and status != "FINISHED":
            self._record_exception(exc_type, exc, tb)
        self._log_memory()
        if self.watchdog is not None:
            self._log_metrics({"memory/peak_total_gb": self.watchdog.peak_gb,
                               "memory/peak_process_rss_gb": self.watchdog.peak_rss_gb,
                               "memory/peak_gpu_gb": self.watchdog.peak_gpu_gb})
        self._set_tag("phase", self.phase_name)
        self._push_status(force=True)
        self._restore_streams()
        self._upload_console()
        self._finished = True
        active = mlflow.active_run()
        if active is not None and active.info.run_id == self.run_id:
            mlflow.end_run(status=status)  # also stops MLflow's system-metrics thread
        else:
            self._safe(self.client.set_terminated, self.run_id, status)
        return False

    # ------------------------------------------------------------------ progress internals

    def _begin_stage(self, stage: str, n_batches: int, batch_size: int | None) -> None:
        now = time.time()
        self._stage = {"name": stage.replace(" ", "_"), "label": stage, "n": max(n_batches, 1),
                       "i": 0, "t0": now, "batch_size": batch_size}
        label = f"{self.epoch_label} · {stage}" if self.epoch_label else stage
        self.phase_name = label
        self._set_tag("phase", label)
        self._last_progress_log = now
        self._push_status(force=True)

    def _tick(self) -> None:
        st = self._stage
        if st is None:
            return
        st["i"] += 1
        if st["name"] == "train":
            self.global_batch += 1
        now = time.time()
        if now - self._last_progress_log >= self.progress_every_s or st["i"] == st["n"]:
            self._log_progress(now)

    def _end_stage(self) -> None:
        st = self._stage
        if st is not None:
            self.stage_durations[st["name"]] = time.time() - st["t0"]
            self._log_metrics({f"time/{st['name']}_s": self.stage_durations[st["name"]]},
                              step=self.global_batch)
        self._stage = None

    def _rates(self, now: float) -> dict:
        st = self._stage
        elapsed = now - st["t0"]
        rate = st["i"] / elapsed if elapsed > 0 else 0.0
        eta_stage = (st["n"] - st["i"]) / rate if rate > 0 else None
        return {"elapsed": elapsed, "rate": rate, "eta_stage": eta_stage}

    def _eta_run(self, eta_stage: float | None) -> float | None:
        """Seconds to the end of the last epoch, assuming no early stop."""
        if eta_stage is None or self.epoch is None:
            return None
        st = self._stage
        epochs_after = max(self.total_epochs - (self.epoch + 1), 0)
        rest_of_epoch = eta_stage
        if st is not None and st["name"] == "train":
            rest_of_epoch += sum(v for k, v in self.stage_durations.items() if k != "train")
        if self.epoch_durations:
            per_epoch = sum(self.epoch_durations[-3:]) / len(self.epoch_durations[-3:])
        elif st is not None and st["name"] == "train":
            per_epoch = st["n"] / max(st["i"], 1) * (time.time() - st["t0"])
        else:
            return None
        return rest_of_epoch + epochs_after * per_epoch

    def _log_progress(self, now: float) -> None:
        st = self._stage
        r = self._rates(now)
        pct = 100.0 * st["i"] / st["n"]
        name = st["name"]
        m = {f"progress/{name}_pct": pct, f"progress/{name}_batches_per_s": r["rate"]}
        if st["batch_size"]:
            m[f"progress/{name}_items_per_s"] = r["rate"] * st["batch_size"]
        if r["eta_stage"] is not None:
            m[f"progress/{name}_eta_min"] = r["eta_stage"] / 60
        eta_run = self._eta_run(r["eta_stage"])
        if eta_run is not None:
            m["progress/run_eta_min"] = eta_run / 60
        if self.epoch is not None and name == "train":
            m["progress/epoch"] = self.epoch + st["i"] / st["n"]
        if self._loss_n and name == "train":
            m["train/loss_running"] = self._loss_sum / self._loss_n
            self._loss_sum, self._loss_n = 0.0, 0
        self._log_metrics(m, step=self.global_batch)
        self._last_progress_log = now
        self._push_status()

    # ------------------------------------------------------------------ status

    def _status_line(self) -> str:
        st = self._stage
        if st is None:
            return self.phase_name
        r = self._rates(time.time())
        eta_run = self._eta_run(r["eta_stage"])
        s = f"{self.phase_name} {st['i']}/{st['n']} ({100 * st['i'] / st['n']:.0f}%)"
        s += f" · stage ETA {_fmt_duration(r['eta_stage'])}"
        if eta_run is not None:
            s += f" · run ETA {_fmt_duration(eta_run)}"
        return s

    def _status_markdown(self) -> str:
        st = self._stage
        lines = [f"### {self._status_line()}", ""]
        lines.append(f"- **Running for:** {_fmt_duration(time.time() - self.t_start)}"
                     f" · updated {time.strftime('%H:%M:%S')}")
        if st is not None:
            r = self._rates(time.time())
            speed = f"{r['rate']:.2f} batches/s"
            if st["batch_size"]:
                speed += f" ({r['rate'] * st['batch_size']:.1f} items/s)"
            lines.append(f"- **Speed:** {speed}")
        if self.epoch_durations:
            lines.append(f"- **Epochs done:** {len(self.epoch_durations)}/{self.total_epochs}"
                         f" · last took {_fmt_duration(self.epoch_durations[-1])}")
        mem = self._memory()
        if mem:
            limit = f" / {self.memory_limit_gb:.0f} GB limit" if self.memory_limit_gb else ""
            lines.append(f"- **Memory:** {mem['memory/total_gb']:.1f} GB{limit}"
                         f" (process {mem['memory/process_rss_gb']:.1f}, GPU {mem['memory/gpu_gb']:.1f})")
        if self.last_metrics:
            keys = [k for k in self.last_metrics if "val" in k][:8] or list(self.last_metrics)[:8]
            lines.append("- **Last epoch:** " + ", ".join(
                f"{k}={self.last_metrics[k]:.4g}" for k in keys))
        lines.append(f"- **Warnings / errors logged:** {self.n_warnings} / {self.n_errors}")
        for p in self.recent_problems:
            lines.append(f"    - `{p}`")
        lines.append("")
        lines.append("Full terminal output: Artifacts → `logs/console.log` (uploaded every minute).")
        return "\n".join(lines)

    def _push_status(self, force: bool = False) -> None:
        line = self._status_line()
        if not force and line == self._last_status:
            return
        self._last_status = line
        self._set_tags({"status": line[:5000], _NOTE_TAG: self._status_markdown()})

    def _note_problem(self, record: logging.LogRecord) -> None:
        if record.name.startswith("mlflow") and record.levelno < logging.ERROR:
            return  # MLflow's own chatter is not a training problem
        if record.levelno >= logging.ERROR:
            self.n_errors += 1
        else:
            self.n_warnings += 1
        msg = record.getMessage().strip().splitlines()[0][:200] if record.getMessage() else ""
        self.recent_problems.append(f"{record.levelname} {record.name}: {msg}")

    # ------------------------------------------------------------------ background thread

    def _background(self) -> None:
        last_beat = last_upload = 0.0
        while not self._stop.wait(1.0):
            now = time.time()
            if now - last_beat >= self.heartbeat_every_s:
                last_beat = now
                self._log_memory()
                self._safe(self.client.set_tag, self.run_id, HEARTBEAT_TAG, str(int(now)))
                self._log_metrics({"log/warnings": float(self.n_warnings),
                                   "log/errors": float(self.n_errors)})
            if now - last_upload >= self.upload_every_s:
                last_upload = now
                self._upload_console()

    def _memory(self) -> dict[str, float]:
        from fungai.safety.watchdog import gpu_memory_bytes

        try:
            rss = self._proc.memory_info().rss
            for child in self._proc.children(recursive=True):
                try:
                    rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return {}
        gpu = gpu_memory_bytes()
        m = {"memory/process_rss_gb": rss / 1024**3, "memory/gpu_gb": gpu / 1024**3,
             "memory/total_gb": (rss + gpu) / 1024**3}
        if self.memory_limit_gb:
            m["memory/limit_gb"] = self.memory_limit_gb
            m["memory/headroom_gb"] = self.memory_limit_gb - m["memory/total_gb"]
        return m

    def _log_memory(self) -> None:
        m = self._memory()
        try:
            m["process/cpu_percent"] = self._proc.cpu_percent(interval=None)
            m["process/threads"] = float(self._proc.num_threads())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        self._mem_sample += 1
        self._log_metrics(m, step=self._mem_sample)

    def _on_memory_kill(self, message: str) -> None:
        """Called by the watchdog on its own thread just before it SIGKILLs the process."""
        self._sink.write(f"CRITICAL fungai.safety: {message}\n")
        self._set_tags({"error.type": "MemoryLimitExceeded", "error.message": message[:5000],
                        "phase": "killed by memory watchdog",
                        "status": f"killed by memory watchdog: {message}"[:5000]})
        self._upload_console()
        self._safe(self.client.set_terminated, self.run_id, "KILLED")

    # ------------------------------------------------------------------ failure

    def _record_exception(self, exc_type, exc, tb) -> None:
        text = "".join(traceback.format_exception(exc_type, exc, tb))
        self._sink.write(text)
        path = self.run_dir / "traceback.txt"
        path.write_text(text)
        frames = traceback.extract_tb(tb)
        where = f"{frames[-1].filename}:{frames[-1].lineno} in {frames[-1].name}" if frames else ""
        self.n_errors += 1
        self.recent_problems.append(f"{exc_type.__name__}: {str(exc)[:200]}")
        self._set_tags({"error.type": exc_type.__name__, "error.message": str(exc)[:5000],
                        "error.where": where[:5000]})
        self._safe(self.client.log_artifact, self.run_id, str(path), "logs")

    # ------------------------------------------------------------------ plumbing

    def _restore_streams(self) -> None:
        if sys.stdout is not self._stdout and isinstance(sys.stdout, _Tee):
            sys.stdout = self._stdout
        if sys.stderr is not self._stderr and isinstance(sys.stderr, _Tee):
            sys.stderr = self._stderr
        logging.getLogger().removeHandler(self._tap)
        logging.captureWarnings(False)
        try:
            if self._faulthandler_was_enabled:
                faulthandler.enable(file=sys.stderr, all_threads=True)
            else:
                faulthandler.disable()
        except Exception:  # noqa: BLE001, S110
            pass

    def _upload_console(self) -> None:
        self._sink.flush()
        self._safe(self.client.log_artifact, self.run_id, str(self.console_path), "logs")

    def _log_metrics(self, metrics: dict, step: int | None = None) -> None:
        from mlflow.entities import Metric

        ts = int(time.time() * 1000)
        step = self._mem_sample if step is None else step
        batch = [Metric(k, float(v), ts, int(step)) for k, v in metrics.items()
                 if v is not None and not math.isnan(float(v))]
        if batch:
            self._safe(self.client.log_batch, self.run_id, metrics=batch)

    def _set_tag(self, key: str, value: str) -> None:
        self._safe(self.client.set_tag, self.run_id, key, value)

    def _set_tags(self, tags: dict[str, str]) -> None:
        from mlflow.entities import RunTag

        self._safe(self.client.log_batch, self.run_id,
                   tags=[RunTag(k, str(v)) for k, v in tags.items()])

    def _safe(self, fn, *args, **kwargs) -> None:
        """Monitoring must never stop training: a failed write is dropped."""
        with self._lock:
            try:
                fn(*args, **kwargs)
            except Exception as exc:  # noqa: BLE001
                try:
                    self._stderr.write(f"[monitor] MLflow write failed: {exc}\n")
                except Exception:  # noqa: BLE001, S110
                    pass


# ---------------------------------------------------------------------- stale-run reaper

def _alive(tags: dict) -> bool:
    if tags.get("monitor.host") != socket.gethostname():
        return False
    try:
        pid = int(tags.get("monitor.pid", "-1"))
        return pid > 0 and psutil.pid_exists(pid) and "python" in " ".join(
            psutil.Process(pid).cmdline()).lower()
    except (ValueError, psutil.Error):
        return False


def reap(stale_after_s: float = STALE_AFTER_S, dry_run: bool = False) -> list[str]:
    """Mark RUNNING runs whose process is gone and whose heartbeat is stale as FAILED."""
    from mlflow.tracking import MlflowClient

    from fungai.tracking.mlflow_utils import _ensure_tracking_uri

    _ensure_tracking_uri()
    client = MlflowClient()
    exp_ids = [e.experiment_id for e in client.search_experiments()]
    reaped = []
    now = time.time()
    for run in client.search_runs(exp_ids, "attributes.status = 'RUNNING'", max_results=1000):
        tags = run.data.tags
        beat = float(tags.get(HEARTBEAT_TAG, run.info.start_time / 1000))
        if now - beat < stale_after_s or _alive(tags):
            continue
        name = run.info.run_name or run.info.run_id
        print(f"{'would reap' if dry_run else 'reaping'} {name}: last heartbeat "
              f"{_fmt_duration(now - beat)} ago")
        reaped.append(run.info.run_id)
        if dry_run:
            continue
        console = tags.get("monitor.console_log")
        if console and Path(console).exists():
            client.log_artifact(run.info.run_id, console, "logs")
        msg = ("process stopped reporting without ending the run (crash, kill -9, power loss); "
               "see logs/console.log")
        client.set_tag(run.info.run_id, "error.type", "ProcessLost")
        client.set_tag(run.info.run_id, "error.message", msg)
        client.set_tag(run.info.run_id, "status", f"lost: last heartbeat {_fmt_duration(now - beat)} ago")
        client.set_tag(run.info.run_id, "phase", "lost")
        client.set_terminated(run.info.run_id, "FAILED", end_time=int(beat * 1000))
    return reaped


def main() -> None:
    parser = argparse.ArgumentParser(description="Run-monitor utilities")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("reap", help="mark dead RUNNING runs as FAILED")
    p.add_argument("--stale-min", type=float, default=STALE_AFTER_S / 60)
    p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.cmd == "reap":
        n = len(reap(args.stale_min * 60, args.dry_run))
        print(f"{n} run(s) {'to reap' if args.dry_run else 'reaped'}")


if __name__ == "__main__":
    main()
