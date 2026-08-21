"""飞行模块：航点、碰撞监控与飞行线程。

从 simu.py 拆出的独立模块（逻辑与拆分前完全一致）：

- 航点加载/钳制（NED 负 Z、速度上限）；
- ``CollisionMonitor``：只报告起飞基线之后的新碰撞；
- ``flight_worker``：独占 AirSim RPC 的飞行线程（起飞、巡航、降落）；
- ``fly_to_leg`` / ``fly_to_waypoint``：按航点飞行，支持 X/Y 分轴移动。

本模块不依赖项目内其他模块（只与 ``ui`` 字典、``args`` 交互）。
"""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import airsim

BASE_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Waypoint:
    x: float
    y: float
    z: float
    speed: float
    tolerance: float = 1.5


DEFAULT_WAYPOINTS = (
    Waypoint(8, 0, -15, 1.5),
    Waypoint(32, 0, -15, 1.5),
    Waypoint(32, 6, -15, 1.5),
    Waypoint(8, 6, -15, 1.5),
    Waypoint(8, 12, -15, 1.5),
    Waypoint(32, 12, -15, 1.5),
    Waypoint(32, 18, -15, 1.5),
    Waypoint(8, 18, -15, 1.5),
    Waypoint(8, 24, -15, 1.5),
    Waypoint(32, 24, -15, 1.5),
)


def normalize_waypoints(waypoints: list[Waypoint], cruise_z: float, max_speed: float) -> list[Waypoint]:
    """Prevent a waypoint file from accidentally commanding low altitude or high speed."""
    if cruise_z >= 0:
        raise ValueError("--cruise-z must be negative in AirSim NED coordinates")
    if max_speed <= 0:
        raise ValueError("--max-speed must be positive")
    return [
        Waypoint(
            x=waypoint.x,
            y=waypoint.y,
            z=min(waypoint.z, cruise_z),
            speed=min(waypoint.speed, max_speed),
            tolerance=waypoint.tolerance,
        )
        for waypoint in waypoints
    ]


def load_waypoints(path: str | Path) -> list[Waypoint]:
    waypoint_path = Path(path)
    if not waypoint_path.is_absolute():
        waypoint_path = BASE_DIR / waypoint_path
    if not waypoint_path.exists():
        return list(DEFAULT_WAYPOINTS)

    with waypoint_path.open("r", encoding="utf-8") as file:
        raw_waypoints = json.load(file)

    waypoints: list[Waypoint] = []
    for index, item in enumerate(raw_waypoints, start=1):
        if isinstance(item, dict):
            values = item
        elif isinstance(item, list) and len(item) >= 4:
            values = dict(zip(("x", "y", "z", "speed", "tolerance"), item))
        else:
            raise ValueError(f"Invalid waypoint #{index}: {item!r}")
        try:
            waypoint = Waypoint(
                x=float(values["x"]),
                y=float(values["y"]),
                z=float(values["z"]),
                speed=float(values["speed"]),
                tolerance=float(values.get("tolerance", 1.5)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid waypoint #{index}: {item!r}") from exc
        if waypoint.speed <= 0 or waypoint.tolerance <= 0:
            raise ValueError(f"Waypoint #{index} speed and tolerance must be positive")
        waypoints.append(waypoint)

    if not waypoints:
        raise ValueError("At least one waypoint is required")
    return waypoints


def get_position(client: airsim.MultirotorClient) -> tuple[float, float, float]:
    state = client.getMultirotorState()
    position = state.kinematics_estimated.position
    return float(position.x_val), float(position.y_val), float(position.z_val)


def poll_flight(
    client: airsim.MultirotorClient,
    target: Waypoint,
    ui: dict[str, Any],
    state_lock: threading.Lock,
) -> float:
    """Fetch telemetry, refresh the display state, and return target distance."""
    state = client.getMultirotorState()
    position = state.kinematics_estimated.position
    velocity = state.kinematics_estimated.linear_velocity
    x, y, z = float(position.x_val), float(position.y_val), float(position.z_val)
    with state_lock:
        ui["position"] = (x, y, z)
        ui["altitude"] = z
        ui["speed"] = math.hypot(float(velocity.x_val), float(velocity.y_val))
    return math.sqrt((x - target.x) ** 2 + (y - target.y) ** 2 + (z - target.z) ** 2)


class CollisionMonitor:
    """Report only *new* collisions.

    AirSim keeps `has_collided` set for the whole session, including across
    `client.reset()` and previous crashed runs. Comparing the collision
    timestamp against a baseline taken at start-up (plus a grace period for
    spawn/takeoff artifacts) avoids aborting the patrol on stale flags.
    """

    def __init__(self, client: airsim.MultirotorClient, grace_seconds: float = 5.0) -> None:
        self.client = client
        self.grace_seconds = grace_seconds
        self.started = time.monotonic()
        self.baseline_ts = 0.0
        try:
            info = client.simGetCollisionInfo()
            self.baseline_ts = float(getattr(info, "time_stamp", 0) or 0)
        except Exception:
            pass
        self.last_ts = self.baseline_ts
        self.last_message: str | None = None

    def check(self) -> str | None:
        """Return a description when a *new* collision happened, else None."""
        try:
            info = self.client.simGetCollisionInfo()
        except Exception:
            return None
        if not info.has_collided:
            return None
        ts = float(getattr(info, "time_stamp", 0) or 0)
        if ts <= self.last_ts:
            return None
        self.last_ts = ts
        if time.monotonic() - self.started < self.grace_seconds:
            return None
        object_name = getattr(info, "object_name", "unknown") or "unknown"
        object_id = getattr(info, "object_id", -1)
        self.last_message = f"碰撞对象：{object_name}（ObjID {object_id}）"
        return self.last_message


def fly_to_leg(
    *,
    client: airsim.MultirotorClient,
    target: Waypoint,
    stop_event: threading.Event,
    args: argparse.Namespace,
    patrol_round: int,
    waypoint_index: int,
    started_monotonic: float,
    monitor: CollisionMonitor | None,
    ui: dict[str, Any],
    state_lock: threading.Lock,
) -> None:
    client.moveToPositionAsync(target.x, target.y, target.z, target.speed)
    deadline = time.monotonic() + args.waypoint_timeout

    while True:
        if stop_event.is_set():
            client.cancelLastTask()
            return
        if ui["stop_cruise"].is_set():
            # “停止任务”按钮：提前结束当前航段
            client.cancelLastTask()
            return
        if monitor is not None:
            message = monitor.check()
            if message:
                ui["messages"].push("warn", message)
                if args.continue_after_collision:
                    print(f"[FLIGHT] {message}（继续飞行）")
                else:
                    client.cancelLastTask()
                    raise RuntimeError(message)

        if poll_flight(client, target, ui, state_lock) <= target.tolerance:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Waypoint {waypoint_index} was not reached within {args.waypoint_timeout:.1f}s")
        time.sleep(max(0.0, args.poll_interval))

    # The position tolerance can be reached before AirSim's async task has
    # completed. Cancel it before hovering, otherwise join() may wait on a
    # stale movement task and stop all subsequent camera polling.
    client.cancelLastTask()
    time.sleep(0.2)
    if args.debug:
        print(f"[DEBUG] waypoint {waypoint_index} reached; movement task cancelled")
    ui["messages"].push("info", f"到达航点 {waypoint_index}")
    client.hoverAsync()
    if args.dwell_seconds > 0:
        time.sleep(args.dwell_seconds)


def fly_to_waypoint(
    *,
    client: airsim.MultirotorClient,
    waypoint: Waypoint,
    stop_event: threading.Event,
    args: argparse.Namespace,
    patrol_round: int,
    waypoint_index: int,
    started_monotonic: float,
    monitor: CollisionMonitor | None,
    ui: dict[str, Any],
    state_lock: threading.Lock,
) -> None:
    """Fly at the enforced altitude, optionally using orthogonal X/Y legs."""
    if args.no_axis_split:
        legs = [waypoint]
    else:
        current_x, current_y, _ = get_position(client)
        legs = []
        if abs(current_x - waypoint.x) > waypoint.tolerance:
            legs.append(Waypoint(waypoint.x, current_y, waypoint.z, waypoint.speed, waypoint.tolerance))
        if abs(current_y - waypoint.y) > waypoint.tolerance or not legs:
            legs.append(waypoint)

    for leg in legs:
        fly_to_leg(
            client=client,
            target=leg,
            stop_event=stop_event,
            args=args,
            patrol_round=patrol_round,
            waypoint_index=waypoint_index,
            started_monotonic=started_monotonic,
            monitor=monitor,
            ui=ui,
            state_lock=state_lock,
        )


def safe_cancel(client: airsim.MultirotorClient) -> None:
    try:
        client.cancelLastTask()
    except Exception:
        pass


def _wait_for_connection(
    client: airsim.MultirotorClient,
    stop_event: threading.Event,
    ui: dict[str, Any],
    state_lock: threading.Lock,
) -> bool:
    """等待 AirSim 就绪；未启动时保持运行，不自动退出。

    连接期间界面显示“等待 AirSim 信号”；按 Q 可退出。连接成功返回 True，
    被停止返回 False。
    """
    delay = 1.0
    print("[FLIGHT] 等待 AirSim 连接…（未检测到 AirSim 信号）")
    ui["messages"].push("info", "等待 AirSim 信号，请启动模拟器…")
    with state_lock:
        ui["airsim_connected"] = False
    while not stop_event.is_set():
        try:
            client.ping()
            return True
        except Exception:
            with state_lock:
                ui["airsim_connected"] = False
            time.sleep(delay)
            delay = min(delay * 2, 10.0)
    return False


def _wait_for_ready(
    client: airsim.MultirotorClient,
    stop_event: threading.Event,
    ui: dict[str, Any],
    state_lock: threading.Lock,
) -> bool:
    """等待 AirSim 场景加载完成（RPC 通了但场景可能还在加载）。

    ``getMultirotorState`` 能返回有效位置即视为就绪；期间界面显示
    “等待场景就绪”，按 Q 可退出。就绪返回 True，被停止返回 False。
    """
    print("[FLIGHT] AirSim 已连接，等待场景就绪…")
    ui["messages"].push("info", "AirSim 已连接，等待场景加载…")
    while not stop_event.is_set():
        try:
            state = client.getMultirotorState()
            position = state.kinematics_estimated.position
            if position is not None:
                with state_lock:
                    ui["airsim_ready"] = True
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


def _takeoff(
    client: airsim.MultirotorClient,
    args: argparse.Namespace,
    ui: dict[str, Any],
    state_lock: threading.Lock,
) -> None:
    """起飞并爬升至巡航高度（首次与“重新开始”复用）。"""
    client.armDisarm(True)
    print("正在起飞无人机...")
    ui["messages"].push("info", "正在起飞…")
    try:
        client.takeoffAsync(timeout_sec=30).join()
    except Exception as exc:
        raise RuntimeError(f"起飞失败（30 秒超时或 RPC 错误）: {exc}") from exc
    try:
        client.moveToZAsync(args.takeoff_z, 2.0).join()
    except Exception as exc:
        raise RuntimeError(f"爬升至巡航高度失败: {exc}") from exc
    with state_lock:
        ui["cruise_started"] = True


def flight_worker(
    *,
    args: argparse.Namespace,
    stop_event: threading.Event,
    done_event: threading.Event,
    ui: dict[str, Any],
    state_lock: threading.Lock,
    result: dict[str, Any],
) -> None:
    """Own all AirSim RPC calls so the GUI thread can never block on AirSim.

    任务循环：首次连接/就绪后自动起飞巡航；巡航可被“停止任务”按钮
    中断（降落并回到待命），再点“多航点巡航”可重新开始；Q 键完全退出。
    """
    client = airsim.MultirotorClient(timeout_value=5.0)
    # timeout_value=5.0：默认 3600s，未连接时 ping() 会阻塞 1 小时导致
    # “等待 AirSim 信号”永远等不到；短超时让连接检测快速失败并重试。
    api_control = False
    try:
        if not _wait_for_connection(client, stop_event, ui, state_lock):
            return  # 用户在连接等待期间按 Q 退出
        with state_lock:
            ui["airsim_connected"] = True
        if not _wait_for_ready(client, stop_event, ui, state_lock):
            return  # 用户在场景加载等待期间按 Q 退出
        client.reset()
        time.sleep(1.0)
        client.enableApiControl(True)
        api_control = True

        monitor = CollisionMonitor(client, grace_seconds=args.collision_grace)
        first_run = True
        while not stop_event.is_set():
            # 等待开始指令：首次自动开始；停止后等待“多航点巡航”按钮
            if not first_run:
                print("[FLIGHT] 任务待命：点击“多航点巡航”开始巡航")
                ui["messages"].push("info", "任务待命：点击“多航点巡航”开始巡航")
                while not ui["start_cruise"].wait(0.2) and not stop_event.is_set():
                    pass
                ui["start_cruise"].clear()
                if stop_event.is_set():
                    break
            first_run = False

            _takeoff(client, args, ui, state_lock)
            print(f"巡航高度：{-args.takeoff_z:g} m；最大速度：{args.max_speed:g} m/s")
            print(f"开始巡航：{len(result['waypoints'])} 个航点（“停止任务”按钮或 Q 可停止）")
            ui["messages"].push("info", f"起飞完成，开始巡航（{len(result['waypoints'])} 个航点）")

            patrol_round = 0
            cruise_aborted = False
            while (args.loops == 0 or patrol_round < args.loops) and not stop_event.is_set():
                patrol_round += 1
                for waypoint_index, waypoint in enumerate(result["waypoints"], start=1):
                    if stop_event.is_set():
                        cruise_aborted = True
                        break
                    if ui["stop_cruise"].is_set():
                        ui["stop_cruise"].clear()
                        cruise_aborted = True
                        print("[FLIGHT] 收到停止指令，正在降落…")
                        ui["messages"].push("info", "收到停止指令，正在降落…")
                        break
                    with state_lock:
                        ui["patrol_round"] = patrol_round
                        ui["waypoint_index"] = waypoint_index
                    print(
                        f"[FLIGHT] 第 {patrol_round} 圈 / 航点 {waypoint_index}: "
                        f"({waypoint.x:g}, {waypoint.y:g}, {waypoint.z:g})"
                    )
                    fly_to_waypoint(
                        client=client,
                        waypoint=waypoint,
                        stop_event=stop_event,
                        args=args,
                        patrol_round=patrol_round,
                        waypoint_index=waypoint_index,
                        started_monotonic=result["started_monotonic"],
                        monitor=monitor,
                        ui=ui,
                        state_lock=state_lock,
                    )
                if cruise_aborted:
                    break

            # 本次巡航结束：降落，回到待命
            safe_cancel(client)
            try:
                client.landAsync().join()
            except Exception as exc:
                print(f"降落过程中出现问题：{exc}")
            try:
                client.armDisarm(False)
            except Exception:
                pass
            with state_lock:
                ui["cruise_started"] = False
            if not stop_event.is_set():
                print("[FLIGHT] 巡航结束，已降落；等待“多航点巡航”重新开始")
                ui["messages"].push("info", "巡航结束，已降落")
    except Exception as exc:
        result["error"] = exc
        print(f"[FLIGHT] 巡航线程出错，准备降落: {type(exc).__name__}: {exc}")
        ui["messages"].push("error", f"巡航线程出错：{type(exc).__name__}")
    finally:
        safe_cancel(client)
        if api_control:
            try:
                client.landAsync().join()
            except Exception as exc:
                print(f"降落过程中出现问题：{exc}")
            try:
                client.armDisarm(False)
                client.enableApiControl(False)
            except Exception:
                pass
        ui["messages"].push("info", "任务结束，已降落")
        done_event.set()
