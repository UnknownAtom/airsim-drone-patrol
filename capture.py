"""取图模块：AirSim 相机连接、帧采集与相机线程。

- ``get_scene_frame``：读取 Scene 相机原始帧，失败时返回错误描述；
- ``CaptureWorker``：独立相机线程，连接失败无限重试，连续取图失败自动重连；
- ``put_latest``：只保留最新帧的覆盖式入队。

依赖 ``detector``（把帧提交给检测线程），不依赖项目内其他模块。
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import airsim
import cv2
import numpy as np

from detector import DetectionWorker


_last_image_warning_at = 0.0


def get_scene_frame(client: airsim.MultirotorClient, camera_name: str) -> tuple[np.ndarray | None, str | None]:
    """Return (frame, None) on success or (None, error_description) on failure.

    Failures are reported to the caller instead of being silently swallowed,
    so the capture worker can keep a useful error for the final summary.
    """
    global _last_image_warning_at
    try:
        responses = client.simGetImages(
            [airsim.ImageRequest(camera_name, airsim.ImageType.Scene, False, False)]
        )
        if not responses:
            return None, "simGetImages 返回空列表"
        response = responses[0]
        payload = bytes(response.image_data_uint8 or b"")
        if response.height <= 0 or response.width <= 0 or not payload:
            return None, f"无效图像: h={response.height}, w={response.width}, len={len(payload)}"

        expected_size = response.height * response.width * 3
        image = np.frombuffer(payload, dtype=np.uint8)
        if image.size < expected_size:
            return None, f"图像数据过短: {image.size} < {expected_size}"
        image = image[:expected_size].reshape(response.height, response.width, 3)

        # Keep the raw AirSim pixel order here. This matches the user's known-
        # working loop, which sends the reshaped Scene buffer directly to YOLO.
        # Display resizing is handled later and does not alter the inference input.
        return image, None
    except Exception as exc:
        message = f"simGetImages 异常: {type(exc).__name__}: {exc}"
        now = time.monotonic()
        if now - _last_image_warning_at >= 2.0:
            _last_image_warning_at = now
            print(f"[CAMERA] {message}")
        return None, message


@dataclass(frozen=True)
class FramePacket:
    frame: np.ndarray
    frame_id: int


class CaptureWorker:
    """Continuously acquire raw AirSim frames independently of flight control.

    Connection failures are retried forever instead of killing the run, and a
    run of image failures triggers a full reconnect. This worker never touches
    the flight's stop event, so the patrol cannot be aborted by a camera issue.
    """

    def __init__(
        self,
        detector: DetectionWorker,
        frame_queue: queue.Queue[FramePacket],
        args: argparse.Namespace,
        stop_event: threading.Event,
        ui: dict[str, Any],
        state_lock: threading.Lock,
    ) -> None:
        self.detector = detector
        self.frame_queue = frame_queue
        self.args = args
        self.stop_event = stop_event
        self.ui = ui
        self.state_lock = state_lock
        self.error: Exception | None = None
        self.last_image_error: str | None = None
        self.last_capture_rpc_ms = 0.0
        self.frames_captured = 0
        self.frames_dropped = 0
        self.connect_attempts = 0
        self._fps_started = 0.0
        self._save_warned: str | None = None
        self.thread = threading.Thread(target=self._run, name="airsim-camera", daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _connect(self) -> airsim.MultirotorClient | None:
        delay = 1.0
        while not self.stop_event.is_set():
            self.connect_attempts += 1
            camera_client = None
            try:
                camera_client = airsim.MultirotorClient(
                    ip=self.args.airsim_ip,
                    port=self.args.airsim_port,
                    timeout_value=self.args.airsim_timeout,
                )
                camera_client.confirmConnection()
                with self.state_lock:
                    self.ui["camera_error"] = ""
                print(
                    f"[CAMERA] 相机线程已连接："
                    f"{self.args.airsim_ip}:{self.args.airsim_port}"
                )
                return camera_client
            except Exception as exc:
                if camera_client is not None:
                    try:
                        camera_client.client.close()
                    except Exception:
                        pass
                self.error = exc
                error_text = f"{type(exc).__name__}: {exc}"
                with self.state_lock:
                    self.ui["camera_error"] = error_text
                print(
                    f"[CAMERA] 连接 AirSim 失败 "
                    f"({self.args.airsim_ip}:{self.args.airsim_port})，"
                    f"{delay:g}s 后重试: {error_text}"
                )
                if self.stop_event.wait(delay):
                    break
                delay = min(delay * 2, 10.0)
        return None

    def _run(self) -> None:
        camera_client = self._connect()
        if camera_client is None:
            return
        self._fps_started = time.monotonic()
        capture_interval = 1.0 / self.args.capture_fps
        next_capture_at = time.monotonic()
        consecutive_failures = 0
        try:
            while not self.stop_event.is_set():
                wait_seconds = next_capture_at - time.monotonic()
                if wait_seconds > 0 and self.stop_event.wait(wait_seconds):
                    break
                rpc_started = time.perf_counter()
                frame, error = get_scene_frame(camera_client, self.args.camera)
                self.last_capture_rpc_ms = (time.perf_counter() - rpc_started) * 1000.0
                if frame is None:
                    consecutive_failures += 1
                    with self.state_lock:
                        self.ui["camera_ok"] = False
                    if error:
                        self.last_image_error = error
                    if consecutive_failures >= 50:
                        print("[CAMERA] 连续取图失败，尝试重新连接...")
                        try:
                            camera_client.close()
                        except Exception:
                            pass
                        camera_client = self._connect()
                        if camera_client is None:
                            break
                        consecutive_failures = 0
                    if self.stop_event.wait(max(0.02, capture_interval)):
                        break
                    next_capture_at = time.monotonic()
                    continue

                consecutive_failures = 0
                self.frames_captured += 1
                with self.state_lock:
                    self.ui["frames_captured"] = self.frames_captured
                    self.ui["camera_ok"] = True
                    self.ui["camera_error"] = ""
                    self.ui["source_size"] = (int(frame.shape[1]), int(frame.shape[0]))

                frame_id = self.detector.submit(frame)
                if put_latest(self.frame_queue, FramePacket(frame, frame_id)):
                    self.frames_dropped += 1
                self._maybe_save_frame(frame, frame_id)
                if self.args.debug and frame_id % 10 == 0:
                    print(f"[DEBUG] capture frame_id={frame_id}")
                next_capture_at += capture_interval
                now = time.monotonic()
                if next_capture_at < now:
                    # RPC or downstream processing exceeded one frame period;
                    # skip the missed slot instead of building a backlog.
                    next_capture_at = now
        except Exception as exc:
            self.error = exc
            print(f"[CAMERA] 相机线程异常退出: {type(exc).__name__}: {exc}")
        finally:
            # 统一释放客户端（正常退出/异常退出/重连失败退出均覆盖）
            if camera_client is not None:
                try:
                    camera_client.close()
                except Exception:
                    pass

    def _maybe_save_frame(self, frame: np.ndarray, frame_id: int) -> None:
        if self.args.save_every <= 0:
            return
        if frame_id % self.args.save_every != 0:
            return
        try:
            save_dir = Path(self.args.capture_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            # AirSim Scene 是 RGB 顺序，cv2.imwrite 按 BGR 保存：仅落盘时转换，
            # 内存帧与 YOLO 推理输入保持原始顺序不变。
            save_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_dir / f"frame_{frame_id:06d}.png"), save_frame)
        except Exception as exc:
            message = str(exc)
            if message != self._save_warned:
                self._save_warned = message
                print(f"[CAMERA] 保存帧失败: {exc}")

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=10.0)

    @property
    def capture_fps(self) -> float:
        if self._fps_started <= 0:
            return 0.0
        elapsed = time.monotonic() - self._fps_started
        if elapsed < 0.1:
            return 0.0
        return self.frames_captured / elapsed


def put_latest(target: queue.Queue[FramePacket], packet: FramePacket) -> bool:
    """Put a packet while keeping only the newest one.

    Returns True when an older packet had to be discarded. This is intentional
    for a live view, but the counter is useful for diagnosing a slow GUI.
    """
    dropped = False
    try:
        target.get_nowait()
        dropped = True
    except queue.Empty:
        pass
    try:
        target.put_nowait(packet)
    except queue.Full:
        dropped = True
    return dropped
