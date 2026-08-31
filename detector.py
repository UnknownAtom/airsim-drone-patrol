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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from ultralytics import YOLO

from frame_stream import FramePacket, LatestValueQueue
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
        if not self.model_path.is_file():
            raise FileNotFoundError(f"模型文件不存在：{self.model_path}")
        self.args = args
        self.is_yolov5 = self._resolve_backend(getattr(args, "backend", "auto"))
        self.device = normalize_torch_device(args.device)
        self.use_half = self._resolve_half()
        self.model_load_ms = 0.0
        self.warmup_ms = 0.0
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

    def _resolve_backend(self, backend: str | None) -> bool:
        """Decide between legacy Torch Hub YOLOv5 and current Ultralytics.

        ``--backend`` 显式指定最可靠；``auto`` 保留按文件名推断的旧行为
        （仅作兼容，文件名含 "yolov5" 即走 Torch Hub）。
        """
        backend = (backend or "auto").strip().lower()
        if backend == "yolov5":
            return True
        if backend == "ultralytics":
            return False
        return "yolov5" in self.model_path.name.lower()

    def _load_yolov5(self) -> Any:
        """Load the legacy checkpoint through the official YOLOv5 hub code."""
        cached_repo = Path.home() / ".cache" / "torch" / "hub" / "ultralytics_yolov5_master"
        if cached_repo.exists():
            source = "local"
            repo = str(cached_repo)
        else:
            source = "github"
            repo = "ultralytics/yolov5"
            print(
                "[DETECTOR] 未找到本地 YOLOv5 hub 缓存，将从 GitHub 下载并执行"
                " ultralytics/yolov5 代码（首次运行需联网；建议预下载到 "
                "~/.cache/torch/hub/ultralytics_yolov5_master 以便离线使用）"
            )

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
        cuda_available = torch.cuda.is_available() and self.device.startswith("cuda")
        if not cuda_available:
            if getattr(self.args, "half", False):
                print("[DETECTOR] --half 需要 CUDA，当前设备改用 FP32")
            return False
        if getattr(self.args, "no_half", False):
            print("[DETECTOR] --no-half 已指定，推理使用 FP32")
            return False
        # CUDA 默认 FP16（性能模式）；--half 仅为兼容旧命令，--no-half 可强制 FP32。
        return True

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
            # Ultralytics 8.4+ renamed the precision option from ``half`` to
            # ``quantize``.  16 means FP16; None keeps the normal FP32 path.
            # Using the new key avoids a deprecation log on every frame.
            "quantize": 16 if self.use_half else None,
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
        result_boxes = result.boxes
        if result_boxes is None or len(result_boxes) == 0:
            return []

        # Transfer the three result tensors only once.  The previous per-box
        # ``.item()``/``.cpu()`` calls each force CUDA work to be visible to
        # Python, which is especially expensive in crowded scenes.
        coordinates = result_boxes.xyxy.detach().cpu().numpy()
        class_ids = result_boxes.cls.detach().cpu().numpy().astype(np.intp, copy=False)
        confidences = result_boxes.conf.detach().cpu().numpy()
        names = getattr(result, "names", self.names)
        return [
            ModelBox(
                xmin=float(coordinates[index, 0]),
                ymin=float(coordinates[index, 1]),
                xmax=float(coordinates[index, 2]),
                ymax=float(coordinates[index, 3]),
                class_id=int(class_ids[index]),
                confidence=float(confidences[index]),
                class_name=class_name(names, int(class_ids[index])),
            )
            for index in range(len(coordinates))
        ]


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

    def __init__(
        self,
        backend: DetectorBackend,
        frames: LatestValueQueue[FramePacket],
    ) -> None:
        self.backend = backend
        self.frames = frames
        self.outputs: LatestValueQueue[DetectionSnapshot] = LatestValueQueue()
        self.stop_event = threading.Event()
        self.error: Exception | None = None
        self.disabled = False
        self._last_error_message: str | None = None
        self.inferences_completed = 0
        self.last_inference_ms = 0.0
        self.first_inference_ms = 0.0
        self.last_latency_ms = 0.0
        self.inference_stats = RollingStats()
        self.latency_stats = RollingStats()
        self._inference_rate = RateWindow(seconds=3.0)
        self.retry_seconds = 10.0
        self._retry_at = 0.0
        self.thread = threading.Thread(target=self._run, name="yolo-detection", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        consecutive_errors = 0
        while not self.stop_event.is_set() or not self.frames.empty():
            if self.disabled:
                if self.stop_event.is_set():
                    break
                if time.monotonic() >= self._retry_at:
                    self.disabled = False
                    consecutive_errors = 0
                    self._last_error_message = None
                    print("[DETECTOR] 检测自动重试：恢复推理")
                else:
                    time.sleep(0.05)
                    continue
            try:
                packet = self.frames.get(timeout=0.05)
            except queue.Empty:
                continue
            job = FrameJob(
                frame=packet.frame,
                frame_id=packet.frame_id,
                submitted_monotonic=packet.captured_monotonic,
            )
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
                self.outputs.put_latest(
                    DetectionSnapshot(
                        job.frame_id,
                        tuple(snapshot_boxes),
                        latency_ms,
                    )
                )
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
                    self._retry_at = time.monotonic() + self.retry_seconds
                    print(
                        "[DETECTOR] 连续推理失败，暂停检测"
                        f"（{self.retry_seconds:g} 秒后自动重试）；画面显示继续。"
                    )

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
    def frame_count(self) -> int:
        """Number of frames accepted by the detection input channel."""
        return self.frames.published

    @property
    def jobs_dropped(self) -> int:
        return self.frames.dropped

    @property
    def outputs_dropped(self) -> int:
        return self.outputs.dropped

    @property
    def performance_snapshot(self) -> dict[str, StatsSnapshot]:
        return {
            "inference": self.inference_stats.snapshot(),
            "latency": self.latency_stats.snapshot(),
        }
