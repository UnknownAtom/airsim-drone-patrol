"""Neutral latest-frame transport shared by capture, inference and display.

The stream intentionally has capacity one: for this live application a new
frame is more useful than preserving an unbounded backlog of old frames.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Generic, TypeVar

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    """One raw RGB frame and the time at which capture made it available."""

    frame: np.ndarray
    frame_id: int
    captured_monotonic: float = field(default_factory=time.monotonic)


FrameType = TypeVar("FrameType")


class LatestValueQueue(Generic[FrameType]):
    """Thread-safe single-slot queue that overwrites stale values."""

    def __init__(self) -> None:
        self._queue: queue.Queue[FrameType] = queue.Queue(maxsize=1)
        self._stats_lock = threading.Lock()
        self._dropped = 0
        self._published = 0

    def put_latest(self, value: FrameType) -> bool:
        """Publish ``value`` and discard an older unread value if necessary."""
        dropped = False
        try:
            self._queue.get_nowait()
            dropped = True
        except queue.Empty:
            pass
        try:
            self._queue.put_nowait(value)
        except queue.Full:
            # A consumer may race the previous get.  Never block the producer
            # or turn a live stream into a backlog in this edge case.
            dropped = True
        with self._stats_lock:
            self._published += 1
            if dropped:
                self._dropped += 1
        return dropped

    def get_nowait(self) -> FrameType:
        return self._queue.get_nowait()

    def get(self, timeout: float | None = None) -> FrameType:
        return self._queue.get(timeout=timeout)

    def empty(self) -> bool:
        return self._queue.empty()

    @property
    def dropped(self) -> int:
        with self._stats_lock:
            return self._dropped

    @property
    def published(self) -> int:
        with self._stats_lock:
            return self._published
