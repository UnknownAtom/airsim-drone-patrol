"""Small thread-safe rolling performance meters used by capture, YOLO and GUI."""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class StatsSnapshot:
    # 注意：total_count 是自统计开始以来的累计样本数，
    # 而 average_ms / maximum_ms 只反映最近窗口内的样本。
    total_count: int = 0
    average_ms: float = 0.0
    maximum_ms: float = 0.0


class RollingStats:
    """Keep recent timing samples while retaining a cheap count for diagnostics."""

    def __init__(self, max_samples: int = 120) -> None:
        self.max_samples = max(1, int(max_samples))
        self._samples: deque[float] = deque(maxlen=self.max_samples)
        self._recent_total_ms = 0.0
        self._total_count = 0
        self._maximum_ms = 0.0
        self._lock = threading.Lock()

    def add(self, value_ms: float) -> None:
        value = max(0.0, float(value_ms))
        with self._lock:
            if len(self._samples) == self.max_samples:
                self._recent_total_ms -= self._samples[0]
            self._samples.append(value)
            self._recent_total_ms += value
            self._total_count += 1
            self._maximum_ms = max(self._maximum_ms, value)

    def snapshot(self) -> StatsSnapshot:
        with self._lock:
            if not self._samples:
                return StatsSnapshot(total_count=self._total_count)
            return StatsSnapshot(
                total_count=self._total_count,
                average_ms=self._recent_total_ms / len(self._samples),
                maximum_ms=self._maximum_ms,
            )


class RateWindow:
    """Calculate a recent event rate without being distorted by old idle time."""

    def __init__(self, seconds: float = 3.0) -> None:
        self.seconds = max(0.5, float(seconds))
        self._events: deque[float] = deque()
        self._lock = threading.Lock()

    def mark(self, now: float | None = None) -> None:
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            self._events.append(timestamp)
            self._prune(timestamp)

    def rate(self, now: float | None = None) -> float:
        timestamp = time.monotonic() if now is None else float(now)
        with self._lock:
            self._prune(timestamp)
            if len(self._events) < 2:
                return 0.0
            elapsed = max(0.001, timestamp - self._events[0])
            # N timestamps describe N - 1 completed intervals.  Counting the
            # initial event as a full interval overstates low rates and makes
            # the FPS panel misleading while a stream is warming up.
            return (len(self._events) - 1) / min(self.seconds, elapsed)

    def _prune(self, now: float) -> None:
        cutoff = now - self.seconds
        while self._events and self._events[0] < cutoff:
            self._events.popleft()
