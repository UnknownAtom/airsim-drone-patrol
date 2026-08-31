"""AirSim multi-waypoint patrol with real-time YOLO detection.

Coordinate system: AirSim uses NED coordinates, so altitude is a negative Z
value.

Module layout (split from a single script; behavior unchanged):

- ``simu.py``: argument parsing, assembly and the GUI main loop;
- ``flight.py``: waypoints, collision monitor and the flight thread;
- ``capture.py``: camera connection/retry and the capture thread;
- ``detector.py``: YOLO model loading, inference and the detection thread;
- ``ui_qt.py``: PyQt6 前端（左侧 PIL 视频区 + 右侧 Qt Widgets 面板，自包含）。

Robustness design (differences from earlier versions that "took off then
landed immediately and showed no camera images"):

- Each worker owns its own stop condition. A camera or YOLO failure no longer
  aborts the flight: capture keeps retrying and reconnecting, and detection
  degrades to no-overlay instead of killing the patrol.
- All worker errors are printed immediately with a [CAMERA]/[DETECTOR]/
  [FLIGHT] prefix instead of being silently swallowed until program exit.
- Collision detection ignores stale `has_collided` flags left over from
  spawn/reset: only collisions whose timestamp is newer than the baseline
  taken at start-up (and after a grace period) abort the patrol.
- `--save-every N` writes every Nth raw camera frame to ./captures so the
  vision pipeline can be verified without looking at the GUI.

Frontend: the PyQt6 console follows the reference SCD application's light
analysis layout: a large pale-blue camera surface, a white real-time analysis
panel, blue task actions, and compact flight metrics.
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path
from typing import Any

import cv2

from capture import CaptureWorker
from detector import DetectorBackend, DetectionWorker
from frame_stream import FramePacket, LatestValueQueue
from flight import flight_worker, load_waypoints, normalize_waypoints
from ui_qt import DetectionDisplay

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "visdrone-yolov26l.pt"
DEFAULT_WAYPOINT_FILE = BASE_DIR / "waypoints.json"
DEFAULT_CAPTURE_DIR = BASE_DIR / "captures"


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Fail early with a readable CLI error instead of failing mid-flight."""
    errors: list[str] = []
    if not 1 <= args.airsim_port <= 65535:
        errors.append("--airsim-port 必须在 1 到 65535 之间")
    if args.airsim_timeout <= 0:
        errors.append("--airsim-timeout 必须大于 0")
    if args.rpc_retry_limit <= 0:
        errors.append("--rpc-retry-limit 必须大于 0")
    if args.conf < 0 or args.conf > 1:
        errors.append("--conf 必须在 0 到 1 之间")
    if args.iou < 0 or args.iou > 1:
        errors.append("--iou 必须在 0 到 1 之间")
    if args.imgsz <= 0:
        errors.append("--imgsz 必须大于 0")
    if args.loops < 0:
        errors.append("--loops 不能为负数")
    if args.poll_interval < 0:
        errors.append("--poll-interval 不能为负数")
    if args.capture_fps <= 0 or args.capture_fps > 120:
        errors.append("--capture-fps 必须大于 0 且不超过 120")
    if args.waypoint_timeout <= 0:
        errors.append("--waypoint-timeout 必须大于 0")
    if args.dwell_seconds < 0:
        errors.append("--dwell-seconds 不能为负数")
    if args.cruise_z >= 0:
        errors.append("--cruise-z 必须是负数（AirSim NED 坐标）")
    if args.takeoff_z is not None and args.takeoff_z >= 0:
        errors.append("--takeoff-z 必须是负数（AirSim NED 坐标）")
    if args.max_speed <= 0:
        errors.append("--max-speed 必须大于 0")
    if args.collision_grace < 0:
        errors.append("--collision-grace 不能为负数")
    if args.display_width <= 0 or args.display_height <= 0:
        errors.append("--display-width 和 --display-height 必须大于 0")
    if args.display_fps <= 0 or args.display_fps > 60:
        errors.append("--display-fps 必须大于 0 且不超过 60")
    if args.save_every < 0 or args.save_ui_every < 0:
        errors.append("--save-every 和 --save-ui-every 不能为负数")
    if errors:
        parser.error("；".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AirSim + YOLO UAV patrol demo")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO .pt model path")
    parser.add_argument(
        "--backend",
        choices=("auto", "yolov5", "ultralytics"),
        default="auto",
        help="模型后端：auto 按文件名推断；yolov5 走旧版 Torch Hub；ultralytics 走新版",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size; camera can remain HD")
    parser.add_argument(
        "--half",
        action="store_true",
        help="兼容参数：CUDA 上默认即为 FP16，该开关仅为保留旧命令；用 --no-half 关闭",
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="强制 FP32 推理（即使 CUDA 可用也不使用 FP16）",
    )
    parser.add_argument("--camera", default="0", help="AirSim camera name")
    parser.add_argument(
        "--airsim-ip",
        "--ip",
        dest="airsim_ip",
        default="127.0.0.1",
        help="AirSim RPC server address",
    )
    parser.add_argument(
        "--airsim-port",
        "--port",
        dest="airsim_port",
        type=int,
        default=41451,
        help="AirSim RPC server port",
    )
    parser.add_argument(
        "--airsim-timeout",
        type=float,
        default=5.0,
        help="AirSim RPC call timeout in seconds",
    )
    parser.add_argument(
        "--rpc-retry-limit",
        type=int,
        default=5,
        help="Consecutive telemetry RPC failures allowed before aborting",
    )
    parser.add_argument(
        "--waypoints-file",
        default=str(DEFAULT_WAYPOINT_FILE),
        help="JSON file containing waypoint objects",
    )
    parser.add_argument("--loops", type=int, default=1, help="Patrol rounds; 0 means repeat until Q")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.10,
        help="Seconds between flight telemetry polls",
    )
    parser.add_argument(
        "--capture-fps",
        type=float,
        default=25.0,
        help="Target camera capture rate; actual rate depends on AirSim RPC latency",
    )
    parser.add_argument(
        "--waypoint-timeout",
        type=float,
        default=180.0,
        help="Maximum seconds allowed for one waypoint",
    )
    parser.add_argument("--dwell-seconds", type=float, default=0.0, help="Extra capture time at each waypoint")
    parser.add_argument(
        "--cruise-z",
        type=float,
        default=-15.0,
        help="Minimum patrol altitude in NED coordinates; -15 means about 15 m high",
    )
    parser.add_argument(
        "--max-speed",
        type=float,
        default=2.0,
        help="Maximum horizontal speed in m/s",
    )
    parser.add_argument(
        "--no-axis-split",
        action="store_true",
        help="Do not split each horizontal leg into X then Y movement",
    )
    parser.add_argument(
        "--continue-after-collision",
        action="store_true",
        help="Keep running after a collision; default is to stop immediately",
    )
    parser.add_argument(
        "--collision-grace",
        type=float,
        default=5.0,
        help="Seconds after flight start during which collisions are ignored (spawn/reset artifacts)",
    )
    parser.add_argument("--no-display", action="store_true", help="Disable the PyQt6 visualization window")
    parser.add_argument("--display-width", type=int, default=1600)
    parser.add_argument("--display-height", type=int, default=900)
    parser.add_argument(
        "--display-fps",
        type=float,
        default=18.0,
        help="GUI 最大渲染帧率；采集线程仍可按 --capture-fps 独立运行",
    )
    parser.add_argument(
        "--theme",
        choices=("dark", "light"),
        default="light",
        help="保留兼容参数；当前前端固定使用 SCD 参考风格的浅色界面",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=0,
        help="Save every Nth captured frame as PNG under --capture-dir (0 = off)",
    )
    parser.add_argument(
        "--save-ui-every",
        type=int,
        default=0,
        help="Save every Nth rendered UI frame as PNG under --capture-dir (0 = off)",
    )
    parser.add_argument("--capture-dir", default=str(DEFAULT_CAPTURE_DIR), help="Directory for saved frames")
    parser.add_argument(
        "--log-file",
        default=None,
        help="将终端输出同时追加写入该文件（调试排查用）",
    )
    parser.add_argument("--debug", action="store_true", help="Print frame and waypoint diagnostics")
    parser.add_argument(
        "--takeoff-z",
        type=float,
        default=None,
        help="Takeoff altitude in NED coordinates; defaults to --cruise-z",
    )
    args = parser.parse_args()
    _validate_args(parser, args)
    return args


def _install_log_tee(path: str) -> None:
    """Mirror stdout/stderr into a log file so GUI runs keep a full trace."""
    import sys

    log_stream = open(path, "a", encoding="utf-8", buffering=1)

    class _Tee:
        def __init__(self, primary, mirror):
            self.primary = primary
            self.mirror = mirror

        def write(self, text):
            self.primary.write(text)
            self.mirror.write(text)

        def flush(self):
            self.primary.flush()
            self.mirror.flush()

        def fileno(self):
            return self.primary.fileno()

        def isatty(self):
            return self.primary.isatty()

    sys.stdout = _Tee(sys.stdout, log_stream)
    sys.stderr = _Tee(sys.stderr, log_stream)
    print(f"[MAIN] 日志已写入：{path}")


def print_summary(
    *,
    capture_worker: CaptureWorker,
    detector: DetectionWorker,
    worker: threading.Thread,
    worker_result: dict[str, Any],
    ui: dict[str, Any],
    display: DetectionDisplay | None = None,
) -> None:
    capture_perf = capture_worker.performance_snapshot()
    detector_perf = detector.performance_snapshot
    print("=" * 60)
    print("运行结束汇总：")
    print(f"  采集帧数        : {capture_worker.frames_captured}")
    print(
        f"  采集帧率        : {capture_worker.capture_fps:.2f} FPS "
        f"（总平均 {capture_worker.capture_fps_total:.2f} FPS）"
    )
    rpc = capture_perf["rpc"]
    parse = capture_perf["parse"]
    capture_time = capture_perf["capture"]
    print(f"  simGetImages     : 平均 {rpc.average_ms:.2f} / 最大 {rpc.maximum_ms:.2f} ms")
    print(f"  图像解析         : 平均 {parse.average_ms:.2f} / 最大 {parse.maximum_ms:.2f} ms")
    print(f"  取图总耗时       : 平均 {capture_time.average_ms:.2f} / 最大 {capture_time.maximum_ms:.2f} ms")
    print(f"  相机丢帧数      : {capture_worker.frames_dropped}")
    print(f"  提交检测帧数    : {detector.frame_count}")
    print(f"  检测完成帧数    : {detector.inferences_completed}")
    inference = detector_perf["inference"]
    latency = detector_perf["latency"]
    print(
        f"  YOLO 推理       : 平均 {inference.average_ms:.2f} / 最大 {inference.maximum_ms:.2f} ms "
        f"（首帧 {detector.first_inference_ms:.2f} ms）"
    )
    print(f"  检测帧率        : {detector.inference_fps:.2f} FPS")
    print(f"  检测结果延迟     : 平均 {latency.average_ms:.2f} / 最大 {latency.maximum_ms:.2f} ms")
    print(f"  检测队列丢帧    : {detector.jobs_dropped}")
    print(f"  检测输出丢帧    : {detector.outputs_dropped}")
    print(f"  模型加载/预热   : {detector.backend.model_load_ms:.2f} / {detector.backend.warmup_ms:.2f} ms")
    print(f"  推理设备/精度   : {detector.backend.device} / {'FP16' if detector.backend.use_half else 'FP32'}")
    if display is not None:
        render = display.render_performance
        print(
            f"  GUI 渲染        : {display.render_fps:.2f} FPS；"
            f"平均 {render.average_ms:.2f} / 最大 {render.maximum_ms:.2f} ms"
        )
    print(f"  最近推理耗时    : {detector.last_inference_ms:.2f} ms")
    print(f"  飞行 RPC 失败   : {ui.get('rpc_failures', 0)}")
    print(f"  相机连接尝试    : {capture_worker.connect_attempts}")
    if detector.disabled:
        print("  [DETECTOR] 检测失败已暂停，将在重试窗口后自动恢复（画面显示不受影响）")
    if detector.error is not None:
        print(f"  [DETECTOR] 检测线程错误: {type(detector.error).__name__}: {detector.error}")
    if capture_worker.error is not None:
        print(f"  [CAMERA] 相机线程曾出错: {type(capture_worker.error).__name__}: {capture_worker.error}")
    if capture_worker.last_image_error is not None:
        print(f"  [CAMERA] 最近一次取图错误: {capture_worker.last_image_error}")
    if worker_result["error"] is not None:
        print(f"  [FLIGHT] 巡航线程错误: {type(worker_result['error']).__name__}: {worker_result['error']}")
    if worker.is_alive():
        print("  警告：AirSim 飞行线程仍在等待 RPC 返回，已不再阻塞 GUI 退出。")
    print("=" * 60)


def main() -> None:
    args = parse_args()
    if args.log_file:
        _install_log_tee(args.log_file)
    waypoints = normalize_waypoints(load_waypoints(args.waypoints_file), args.cruise_z, args.max_speed)
    args.takeoff_z = args.cruise_z if args.takeoff_z is None else min(args.takeoff_z, args.cruise_z)
    # 每次运行使用独立的保存子目录，避免多次运行互相覆盖同名帧。
    args.capture_run_dir = str(
        Path(args.capture_dir) / time.strftime("run_%Y%m%d_%H%M%S")
    )
    backend = DetectorBackend(args.model, args)
    display_frames: LatestValueQueue[FramePacket] = LatestValueQueue()
    detection_frames: LatestValueQueue[FramePacket] = LatestValueQueue()
    detector = DetectionWorker(backend, detection_frames)
    # A headless patrol should not construct QApplication, widgets, fonts or
    # PIL rendering state.  Besides lowering start-up cost, this allows the
    # inference-only path to run on machines without a usable display server.
    display = None if args.no_display else DetectionDisplay(args.theme)
    stop_event = threading.Event()
    done_event = threading.Event()
    state_lock = threading.Lock()
    ui: dict[str, Any] = {
        "patrol_round": 0,
        "waypoint_index": 0,
        "waypoints_total": len(waypoints),
        "camera_ok": False,
        "camera_error": "",
        "airsim_connected": False,
        "airsim_ready": False,
        "airsim_error": "",
        "flight_state": "DISCONNECTED",
        "cruise_started": False,
        "stop_cruise": threading.Event(),
        "capture_fps": 0.0,
        "detection_fps": 0.0,
        "capture_rpc_avg_ms": 0.0,
        "capture_rpc_max_ms": 0.0,
        "detection_avg_ms": 0.0,
        "detection_max_ms": 0.0,
        "render_fps": 0.0,
        "camera_drops": 0,
        "detection_drops": 0,
        "rpc_failures": 0,
    }
    worker_result: dict[str, Any] = {
        "waypoints": waypoints,
        "error": None,
    }
    worker = threading.Thread(
        target=flight_worker,
        kwargs={
            "args": args,
            "stop_event": stop_event,
            "done_event": done_event,
            "ui": ui,
            "state_lock": state_lock,
            "result": worker_result,
        },
        name="airsim-flight",
        daemon=True,
    )

    if display is not None:
        # Qt 窗口在首次 show() 时创建并显示；先渲染一次空状态画面，
        # 避免窗口是空白灰框。
        display.show(None, None, args, ui, state_lock)

    capture_worker = CaptureWorker(
        display_frames=display_frames,
        detection_frames=detection_frames,
        args=args,
        stop_event=stop_event,
        ui=ui,
        state_lock=state_lock,
    )
    worker.start()
    capture_worker.start()

    last_saved_render_count = 0
    next_metrics_refresh_at = 0.0
    try:
        while not done_event.is_set():
            raw_frame = None
            try:
                raw_frame = display_frames.get_nowait()
            except queue.Empty:
                pass

            snapshot = detector.poll_snapshot()
            if args.debug and snapshot is not None:
                print(f"[DEBUG] detect frame_id={snapshot.frame_id}")

            # These counters are shown only by the GUI.  Sampling them once
            # per display period avoids lock acquisition and statistics work
            # on every 10 ms coordinator-loop pass.
            now = time.monotonic()
            if display is not None and now >= next_metrics_refresh_at:
                ui["capture_fps"] = capture_worker.capture_fps
                ui["detection_fps"] = detector.inference_fps
                ui["camera_drops"] = capture_worker.frames_dropped
                ui["detection_drops"] = detector.jobs_dropped
                capture_rpc = capture_worker.performance_snapshot()["rpc"]
                ui["capture_rpc_avg_ms"] = capture_rpc.average_ms
                ui["capture_rpc_max_ms"] = capture_rpc.maximum_ms
                inference = detector.performance_snapshot["inference"]
                ui["detection_avg_ms"] = inference.average_ms
                ui["detection_max_ms"] = inference.maximum_ms
                ui["render_fps"] = display.render_fps
                next_metrics_refresh_at = now + 1.0 / max(1.0, args.display_fps)

            if display is None:
                should_quit = False
            elif raw_frame is not None or snapshot is not None:
                should_quit = display.show(raw_frame, snapshot, args, ui, state_lock)
            else:
                should_quit = display.process_events()

            if (
                display is not None
                and args.save_ui_every > 0
                and display.last_render is not None
                and display.render_count != last_saved_render_count
            ):
                last_saved_render_count = display.render_count
                if display.render_count % args.save_ui_every == 0:
                    save_dir = Path(args.capture_run_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(
                        str(save_dir / f"ui_{display.render_count:05d}.png"),
                        display.last_render,
                    )

            if should_quit:
                print("收到 Q 键，正在停止飞行线程...")
                stop_event.set()
                break
            time.sleep(0.01)
    except KeyboardInterrupt:
        stop_event.set()
        print("收到停止指令，正在安全退出...")
    finally:
        stop_event.set()
        # Keep pumping window messages while waiting for the flight thread so
        # the console is never flagged "Not Responding" (which leaves a ghost
        # window behind if the process is then force-killed).
        shutdown_deadline = time.monotonic() + 30.0
        while worker.is_alive() and time.monotonic() < shutdown_deadline:
            worker.join(timeout=0.1)
            if display is not None:
                display.process_events()
        capture_worker.close()

        detector.close()
        print_summary(
            capture_worker=capture_worker,
            detector=detector,
            worker=worker,
            worker_result=worker_result,
            ui=ui,
            display=display,
        )


if __name__ == "__main__":
    main()
