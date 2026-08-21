"""AirSim multi-waypoint patrol with real-time YOLO detection.

Coordinate system: AirSim uses NED coordinates, so altitude is a negative Z
value.

Module layout (split from a single script; behavior unchanged):

- ``simu.py``: argument parsing, assembly and the GUI main loop;
- ``flight.py``: waypoints, collision monitor and the flight thread;
- ``capture.py``: camera connection/retry and the capture thread;
- ``detector.py``: YOLO model loading, inference and the detection thread;
- ``ui.py``: SCD-style visual frontend (unchanged).

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

Frontend: the OpenCV console window follows the reference SCD application's
light analysis layout: a large pale-blue camera surface, a white real-time
analysis panel, blue task actions, and compact flight metrics.
"""

from __future__ import annotations

import argparse
import queue
import threading
import time
from pathlib import Path
from typing import Any

import cv2

from capture import CaptureWorker, FramePacket
from detector import DetectorBackend, DetectionWorker
from flight import flight_worker, load_waypoints, normalize_waypoints
from ui import DetectionDisplay, UIMessages, WINDOW_NAME, fix_window_title

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL = BASE_DIR / "visdrone-yolov26l.pt"
DEFAULT_WAYPOINT_FILE = BASE_DIR / "waypoints.json"
DEFAULT_CAPTURE_DIR = BASE_DIR / "captures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AirSim + YOLO UAV patrol demo")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="YOLO .pt model path")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0 or cpu")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size; camera can remain HD")
    parser.add_argument("--camera", default="0", help="AirSim camera name")
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
        help="Seconds between image/control polls; 0.10 is about 10 FPS",
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
    parser.add_argument("--no-display", action="store_true", help="Disable the OpenCV visualization window")
    parser.add_argument("--display-width", type=int, default=1600)
    parser.add_argument("--display-height", type=int, default=900)
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
    parser.add_argument("--debug", action="store_true", help="Print frame and waypoint diagnostics")
    parser.add_argument(
        "--takeoff-z",
        type=float,
        default=None,
        help="Takeoff altitude in NED coordinates; defaults to --cruise-z",
    )
    return parser.parse_args()


def print_summary(
    *,
    capture_worker: CaptureWorker,
    detector: DetectionWorker,
    worker: threading.Thread,
    worker_result: dict[str, Any],
) -> None:
    print("=" * 60)
    print("运行结束汇总：")
    print(f"  采集帧数        : {capture_worker.frames_captured}")
    print(f"  提交检测帧数    : {detector.frame_count}")
    if detector.disabled:
        print("  [DETECTOR] 检测因连续失败已禁用（画面显示不受影响）")
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
    waypoints = normalize_waypoints(load_waypoints(args.waypoints_file), args.cruise_z, args.max_speed)
    args.takeoff_z = args.cruise_z if args.takeoff_z is None else min(args.takeoff_z, args.cruise_z)
    backend = DetectorBackend(args.model, args)
    detector = DetectionWorker(backend)
    display = DetectionDisplay(args.theme)
    started_monotonic = time.monotonic()
    frame_queue: queue.Queue[FramePacket] = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    done_event = threading.Event()
    state_lock = threading.Lock()
    ui: dict[str, Any] = {
        "position": (0.0, 0.0, 0.0),
        "altitude": 0.0,
        "speed": 0.0,
        "patrol_round": 0,
        "waypoint_index": 0,
        "waypoints_total": len(waypoints),
        "waypoints": tuple((waypoint.x, waypoint.y) for waypoint in waypoints),
        "frames_captured": 0,
        "detections": 0,
        "camera_ok": False,
        "source_size": (0, 0),
        "fps": 0.0,
        "messages": UIMessages(),
    }
    worker_result: dict[str, Any] = {
        "waypoints": waypoints,
        "started_monotonic": started_monotonic,
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

    if not args.no_display:
        # Register the window with a pure-ASCII name: HighGUI mishandles
        # non-ASCII names on zh-CN Windows and can spawn extra windows with
        # mojibake / "\\uXXXX" titles.  Paint the empty-state console first so
        # the window is never a blank gray box, then apply the Chinese title
        # through the Win32 API (imshow keeps using the ASCII name, so all
        # frames go into this single window).
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, args.display_width, args.display_height)
        display.show(None, None, args, ui)
        fix_window_title()

    capture_worker = CaptureWorker(
        detector=detector,
        frame_queue=frame_queue,
        args=args,
        stop_event=stop_event,
        ui=ui,
        state_lock=state_lock,
        started_monotonic=started_monotonic,
    )
    worker.start()
    capture_worker.start()

    notified: set[str] = set()
    fps_window_frames = 0
    fps_window_started = time.monotonic()
    ui_saved_frames = 0
    try:
        while not done_event.is_set():
            raw_frame = None
            try:
                raw_frame = frame_queue.get_nowait()
            except queue.Empty:
                pass

            snapshot = detector.poll_snapshot()
            if args.debug and snapshot is not None:
                print(f"[DEBUG] detect frame_id={snapshot.frame_id}")

            # Refresh UI counters (cheap, once per main-loop tick)
            if raw_frame is not None:
                fps_window_frames += 1
            now = time.monotonic()
            if now - fps_window_started >= 1.0:
                ui["fps"] = fps_window_frames / (now - fps_window_started)
                fps_window_frames = 0
                fps_window_started = now
            ui["frames_captured"] = capture_worker.frames_captured
            if snapshot is not None:
                ui["detections"] = len(snapshot.boxes)

            # Surface worker errors in the UI once
            if capture_worker.error is not None and "camera" not in notified:
                notified.add("camera")
                ui["messages"].push("error", f"相机线程错误：{capture_worker.error}")
            if detector.error is not None and "detector" not in notified:
                notified.add("detector")
                ui["messages"].push("error", f"检测线程错误：{detector.error}")
            if worker_result["error"] is not None and "flight" not in notified:
                notified.add("flight")
                ui["messages"].push("error", f"巡航线程错误：{worker_result['error']}")

            if raw_frame is not None or snapshot is not None:
                should_quit = display.show(raw_frame, snapshot, args, ui)
            else:
                should_quit = False if args.no_display else display.process_events()

            if args.save_ui_every > 0 and display.last_render is not None:
                ui_saved_frames += 1
                if ui_saved_frames % args.save_ui_every == 0:
                    save_dir = Path(args.capture_dir)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_dir / f"ui_{ui_saved_frames:05d}.png"), display.last_render)

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
            if not args.no_display:
                cv2.waitKey(1)
        capture_worker.close()
        if not args.no_display:
            cv2.destroyAllWindows()

        detector.close()
        print_summary(
            capture_worker=capture_worker,
            detector=detector,
            worker=worker,
            worker_result=worker_result,
        )


if __name__ == "__main__":
    main()
