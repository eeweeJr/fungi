"""Background memory watchdog that kills the process if RSS exceeds a limit.

macOS rejects setrlimit for RSS/AS/DATA on Apple Silicon (EINVAL), so we
poll via psutil and send SIGKILL when the threshold is crossed.
"""

from __future__ import annotations

import logging
import os
import signal
import threading

import psutil

logger = logging.getLogger("fungai.safety")


class MemoryWatchdog:
    """Daemon thread monitoring RSS of the current process tree."""

    def __init__(self, limit_gb: float = 16.0, poll_interval_s: float = 2.0):
        self.limit_bytes = int(limit_gb * 1024**3)
        self.poll_interval = poll_interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._peak_bytes: int = 0
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

    def _monitor(self) -> None:
        while not self._stop_event.is_set():
            rss = self._get_total_rss()
            self._peak_bytes = max(self._peak_bytes, rss)
            if rss > self.limit_bytes:
                logger.critical(
                    "MEMORY LIMIT EXCEEDED: %.2f GB > %.2f GB limit. Killing process.",
                    rss / 1024**3,
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
        return self._peak_bytes / 1024**3

    @property
    def current_gb(self) -> float:
        return self._get_total_rss() / 1024**3
