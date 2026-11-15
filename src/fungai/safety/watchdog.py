"""Background memory watchdog that kills the process if memory exceeds a limit.

macOS rejects setrlimit for RSS/AS/DATA on Apple Silicon (EINVAL), so we
poll via psutil and send SIGKILL when the threshold is crossed.

The total is process RSS (this process and its children) plus GPU memory held by
torch. On Apple Silicon, MPS allocations do not show up in RSS at all: a ViT-L
fine-tune held 10.8 GB on MPS while RSS read 1.55 GB. Counting only RSS would
leave the limit unenforced for any GPU training.
"""

from __future__ import annotations

import logging
import os
import signal
import threading

import psutil

logger = logging.getLogger("fungai.safety")


def gpu_memory_bytes() -> int:
    """Bytes held by torch on the GPU: MPS driver allocation or CUDA reserved memory.

    Returns 0 when torch is not imported, no GPU backend is in use, or the query
    fails. torch is only read from sys.modules so the watchdog never imports it.
    """
    import sys

    torch = sys.modules.get("torch")
    if torch is None:
        return 0
    try:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return int(torch.mps.driver_allocated_memory())
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            return int(sum(torch.cuda.memory_reserved(i) for i in range(torch.cuda.device_count())))
    except Exception:  # noqa: BLE001 - a failed query must never stop the watchdog
        return 0
    return 0


class MemoryWatchdog:
    """Daemon thread monitoring process-tree RSS plus torch GPU memory."""

    def __init__(self, limit_gb: float = 16.0, poll_interval_s: float = 2.0):
        self.limit_bytes = int(limit_gb * 1024**3)
        self.poll_interval = poll_interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_bytes: int = 0
        self._peak_rss_bytes: int = 0
        self._peak_gpu_bytes: int = 0
        self._proc = psutil.Process(os.getpid())

    def _get_total_rss(self) -> int:
        try:
            total = self._proc.memory_info().rss
            for child in self._proc.children(recursive=True):
                try:
                    total += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            return total
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return 0

    def _get_gpu_bytes(self) -> int:
        return gpu_memory_bytes()

    def sample(self) -> tuple[int, int]:
        """(rss_bytes, gpu_bytes) right now, and update the peaks."""
        rss, gpu = self._get_total_rss(), self._get_gpu_bytes()
        self._peak_rss_bytes = max(self._peak_rss_bytes, rss)
        self._peak_gpu_bytes = max(self._peak_gpu_bytes, gpu)
        self._peak_bytes = max(self._peak_bytes, rss + gpu)
        return rss, gpu

    def _monitor(self) -> None:
        while not self._stop_event.is_set():
            rss, gpu = self.sample()
            if rss + gpu > self.limit_bytes:
                logger.critical(
                    "MEMORY LIMIT EXCEEDED: %.2f GB (process %.2f GB + GPU %.2f GB) > %.2f GB "
                    "limit. Killing process.",
                    (rss + gpu) / 1024**3, rss / 1024**3, gpu / 1024**3,
                    self.limit_bytes / 1024**3,
                )
                os.kill(os.getpid(), signal.SIGKILL)
            self._stop_event.wait(self.poll_interval)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._monitor, daemon=True, name="memory-watchdog"
        )
        self._thread.start()
        logger.info(
            "Memory watchdog started (limit=%.1f GB, poll=%.1fs)",
            self.limit_bytes / 1024**3,
            self.poll_interval,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def peak_gb(self) -> float:
        """Peak of process RSS + GPU memory, the figure the limit is checked against."""
        return self._peak_bytes / 1024**3

    @property
    def peak_rss_gb(self) -> float:
        return self._peak_rss_bytes / 1024**3

    @property
    def peak_gpu_gb(self) -> float:
        return self._peak_gpu_bytes / 1024**3

    @property
    def current_gb(self) -> float:
        return (self._get_total_rss() + self._get_gpu_bytes()) / 1024**3

    def summary(self) -> str:
        return (f"{self.peak_gb:.2f} GB (process {self.peak_rss_gb:.2f} GB, "
                f"GPU {self.peak_gpu_gb:.2f} GB)")
