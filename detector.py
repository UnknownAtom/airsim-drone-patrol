"""检测模块：YOLO 模型加载、推理与检测线程。

从 simu.py 拆出的独立模块（逻辑与拆分前完全一致）：

- ``DetectorBackend``：统一加载旧版 YOLOv5（Torch Hub）与新版 Ultralytics 权重；
- ``DetectionWorker``：独立推理线程，只处理最新帧，推理失败自动降级为无检测。

本模块不依赖项目内其他模块。
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

from performance import RateWindow, RollingStats, StatsSnapshot


def class_name(names: Any, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, class_id))
    if 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def normalize_torch_device(value: str | None) -> str:
    """Convert CLI device syntax (0/cuda:0/cpu) to valid torch syntax."""
    if value is None or value.strip().lower() == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    normalized = value.strip().lower()
    if normalized.isdigit():
        return f"cuda:{normalized}"
    if normalized == "cuda":
        return "cuda:0"
    return normalized


@dataclass(frozen=True)
class ModelBox:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    class_id: int
    confidence: float
    class_name: str


class DetectorBackend:
    """Load old YOLOv5 or current Ultralytics weights behind one interface."""

    def __init__(self, model_path: str | Path, args: argparse.Namespace) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        self.args = args
        self.is_yolov5 = "yolov5" in self.model_path.name.lower()
        self.device = normalize_torch_device(args.device)
        self.use_half = self._resolve_half()
        self.model_load_ms = 0.0
        self.warmup_ms = 0.0
        # ultralytics 的 half 参数在 8.x 会发 deprecation 警告；只在加载时设置一次，
        # 避免每次推理都创建 catch_warnings 上下文。
        warnings.filterwarnings("ignore", message=r".*'half' is deprecated.*")
        load_started = time.perf_counter()

        if self.is_yolov5:
            self.model = self._load_yolov5()
            if self.use_half:
                self.model.half()
            self.names = getattr(self.model, "names", {})
            print(f"已加载 YOLOv5 VisDrone 模型：{self.model_path.name}")
            print(f"类别：{self.names}")
        else:
            self.model = YOLO(str(self.model_path))
            self.names = getattr(self.model, "names", {})
            print(f"已加载 Ultralytics 模型：{self.model_path.name}")
        self.model_load_ms = (time.perf_counter() - load_started) * 1000.0
        print(
            f"推理设备：{self.device}；精度：{'FP16' if self.use_half else 'FP32'}；"
            f"模型加载：{self.model_load_ms:.1f} ms"
        )
        self.warmup_ms = self._warmup()
        if self.warmup_ms > 0:
            print(f"模型预热完成：{self.warmup_ms:.1f} ms")

    def _load_yolov5(self) -> Any:
        """Load the legacy checkpoint through the official YOLOv5 hub code."""
        cached_repo = Path.home() / ".cache" / "torch" / "hub" / "ultralytics_yolov5_master"
        if cached_repo.exists():
            source = "local"
            repo = str(cached_repo)
        else:
            source = "github"
            repo = "ultralytics/yolov5"

        model = torch.hub.load(
            repo,
            "custom",
            path=str(self.model_path),
            source=source,
            trust_repo=True,
            verbose=False,
        )
        model.to(self.device)
        model.conf = self.args.conf
        model.iou = self.args.iou
        return model

    def _resolve_half(self) -> bool:
        requested = bool(getattr(self.args, "half", False))
        cuda_available = torch.cuda.is_available() and self.device.startswith("cuda")
        if requested and not cuda_available:
            print("[DETECTOR] --half 需要 CUDA，当前设备改用 FP32")
        # RTX/CUDA uses FP16 by default; --half makes the intent explicit.
        return cuda_available

    def synchronize(self) -> None:
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()

    def _predict(self, frame: np.ndarray) -> Any:
        if self.is_yolov5:
            return self.model(frame, size=self.args.imgsz)
        options: dict[str, Any] = {
            "source": frame,
            "conf": self.args.conf,
            "iou": self.args.iou,
            "imgsz": self.args.imgsz,
            "device": self.device,
            "half": self.use_half,
            "verbose": False,
        }
        return self.model.predict(**options)

    def _warmup(self) -> float:
        """Run one representative inference before the first live frame."""
        dummy = np.zeros((720, 1280, 3), dtype=np.uint8)
        try:
            self.synchronize()
            started = time.perf_counter()
            with torch.inference_mode():
                self._predict(dummy)
            self.synchronize()
            return (time.perf_counter() - started) * 1000.0
        except Exception as exc:
            print(f"[DETECTOR] 模型预热失败，将继续使用实时首帧初始化：{type(exc).__name__}: {exc}")
            return 0.0

    def infer(self, frame: np.ndarray) -> list[ModelBox]:
        with torch.inference_mode():
            return self._infer(frame)

    def _infer(self, frame: np.ndarray) -> list[ModelBox]:
        if self.is_yolov5:
            results = self._predict(frame)
            predictions = results.xyxy[0].detach().cpu().numpy()
            names = getattr(results, "names", self.names)
            return [
                ModelBox(
                    xmin=float(row[0]),
                    ymin=float(row[1]),
                    xmax=float(row[2]),
                    ymax=float(row[3]),
                    class_id=int(row[5]),
                    confidence=float(row[4]),
                    class_name=class_name(names, int(row[5])),
                )
                for row in predictions
            ]

        result = self._predict(frame)[0]
        boxes: list[ModelBox] = []
        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            bbox = box.xyxy[0].detach().cpu().numpy().tolist()
            boxes.append(
                ModelBox(
                    xmin=float(bbox[0]),
                    ymin=float(bbox[1]),
                    xmax=float(bbox[2]),
                    ymax=float(bbox[3]),
                    class_id=class_id,
                    confidence=confidence,
                    class_name=class_name(self.names, class_id),
                )
            )
        return boxes


@dataclass(frozen=True)
class FrameJob:
    frame: np.ndarray
    frame_id: int
    submitted_monotonic: float


@dataclass(frozen=True)
class DetectionSnapshot:
    frame_id: int
    boxes: tuple[tuple[float, float, float, float, int, float, str], ...]
    latency_ms: float = 0.0


class DetectionWorker:
    """Run YOLO away from the OpenCV GUI thread so the window stays responsive.

    Inference failures never abort the program: they are printed immediately,
    and after several consecutive failures detection disables itself while the
    camera display keeps running.
    """

    def __init__(self, backend: DetectorBackend) -> None:
        self.backend = backend
        self.frame_count = 0
        self.jobs: queue.Queue[FrameJob] = queue.Queue(maxsize=1)
        self.outputs: queue.Queue[DetectionSnapshot] = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.error: Exception | None = None
        self.disabled = False
        self._last_error_message: str | None = None
        self.jobs_dropped = 0
        self.outputs_dropped = 0
        self.inferences_completed = 0
        self.last_inference_ms = 0.0
        self.first_inference_ms = 0.0
        self.last_latency_ms = 0.0
        self.inference_stats = RollingStats()
        self.latency_stats = RollingStats()
        self._inference_rate = RateWindow(seconds=3.0)
        self.thread = threading.Thread(target=self._run, name="yolo-detection", daemon=True)
        self.thread.start()

    def submit(self, frame: np.ndarray) -> int:
        self.frame_count += 1
        frame_id = self.frame_count
        submitted_monotonic = time.monotonic()
        if self.disabled:
            return frame_id
        job = FrameJob(
            frame=frame,
            frame_id=frame_id,
            submitted_monotonic=submitted_monotonic,
        )
        # Keep only the newest frame. A backlog would increase latency and make
        # the display appear frozen even though the program is still running.
        try:
            self.jobs.get_nowait()
            self.jobs_dropped += 1
        except queue.Empty:
            pass
        try:
            self.jobs.put_nowait(job)
        except queue.Full:
            self.jobs_dropped += 1
        return frame_id

    def _run(self) -> None:
        consecutive_errors = 0
        while not self.stop_event.is_set() or not self.jobs.empty():
            try:
                job = self.jobs.get(timeout=0.05)
            except queue.Empty:
                continue
            try:
                self.backend.synchronize()
                started = time.perf_counter()
                model_boxes = self.backend.infer(job.frame)
                self.backend.synchronize()
                inference_ms = (time.perf_counter() - started) * 1000.0
                completed_monotonic = time.monotonic()
                latency_ms = max(0.0, (completed_monotonic - job.submitted_monotonic) * 1000.0)
                snapshot_boxes: list[tuple[float, float, float, float, int, float, str]] = []
                for box in model_boxes:
                    snapshot_boxes.append(
                        (
                            box.xmin,
                            box.ymin,
                            box.xmax,
                            box.ymax,
                            box.class_id,
                            box.confidence,
                            box.class_name,
                        )
                    )
                try:
                    self.outputs.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self.outputs.put_nowait(
                        DetectionSnapshot(
                            job.frame_id,
                            tuple(snapshot_boxes),
                            latency_ms,
                        )
                    )
                except queue.Full:
                    self.outputs_dropped += 1
                self.inferences_completed += 1
                self.last_inference_ms = inference_ms
                self.last_latency_ms = latency_ms
                if self.inferences_completed == 1:
                    self.first_inference_ms = inference_ms
                self.inference_stats.add(inference_ms)
                self.latency_stats.add(latency_ms)
                self._inference_rate.mark(completed_monotonic)
                consecutive_errors = 0
            except Exception as exc:
                consecutive_errors += 1
                message = f"{type(exc).__name__}: {exc}"
                if message != self._last_error_message:
                    self._last_error_message = message
                    print(f"[DETECTOR] 推理失败（跳过本帧）: {message}")
                if self.error is None:
                    self.error = exc
                if consecutive_errors >= 5:
                    self.disabled = True
                    print("[DETECTOR] 连续推理失败，已停止检测；画面显示继续。")
                    break

    def poll_snapshot(self) -> DetectionSnapshot | None:
        latest = None
        while True:
            try:
                latest = self.outputs.get_nowait()
            except queue.Empty:
                return latest

    def close(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=15.0)

    @property
    def inference_fps(self) -> float:
        return self._inference_rate.rate()

    @property
    def performance_snapshot(self) -> dict[str, StatsSnapshot]:
        return {
            "inference": self.inference_stats.snapshot(),
            "latency": self.latency_stats.snapshot(),
        }
