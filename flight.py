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


FLIGHT_STATES = {
    "DISCONNECTED",
    "CONNECTING",
    "READY",
    "TAKING_OFF",
    "CRUISING",
    "STOPPING",
    "LANDING",
    "ERROR",
    "STOPPED",
}


def set_flight_state(ui: dict[str, Any], state_lock: threading.Lock, state: str) -> None:
    """Publish one consistent flight state for the GUI and diagnostics."""
    if state not in FLIGHT_STATES:
        raise ValueError(f"Unknown flight state: {state}")
    with state_lock:
        ui["flight_state"] = state


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
    waypoint_index: int,
    monitor: CollisionMonitor | None,
    ui: dict[str, Any],
    state_lock: threading.Lock,
) -> bool:
    """飞向一个航段。正常到达返回 True；被停止（Q/停止按钮）返回 False。

    遥测读取（``getMultirotorState``）允许瞬态失败并重试
    （``--rpc-retry-limit`` 次），避免单次超时终止整个任务。
    """
    client.moveToPositionAsync(target.x, target.y, target.z, target.speed)
    deadline = time.monotonic() + args.waypoint_timeout
    consecutive_rpc_errors = 0

    while True:
        if stop_event.is_set():
            set_flight_state(ui, state_lock, "STOPPING")
            client.cancelLastTask()
            return False
        if ui["stop_cruise"].is_set():
            # “停止任务”按钮：提前结束当前航段
            set_flight_state(ui, state_lock, "STOPPING")
            client.cancelLastTask()
            return False
        if monitor is not None:
            message = monitor.check()
            if message:
                ui["messages"].push("warn", message)
                if args.continue_after_collision:
                    print(f"[FLIGHT] {message}（继续飞行）")
                else:
                    client.cancelLastTask()
                    raise RuntimeError(message)

        try:
            distance = poll_flight(client, target, ui, state_lock)
        except Exception as exc:
            consecutive_rpc_errors += 1
            with state_lock:
                ui["rpc_failures"] = int(ui.get("rpc_failures", 0)) + 1
                ui["rpc_consecutive_failures"] = consecutive_rpc_errors
            if consecutive_rpc_errors >= args.rpc_retry_limit:
                with state_lock:
                    ui["airsim_connected"] = False
                    ui["airsim_ready"] = False
                    ui["airsim_error"] = f"遥测连续失败：{type(exc).__name__}: {exc}"
                set_flight_state(ui, state_lock, "ERROR")
                raise
            print(
                f"[FLIGHT] 遥测读取失败 "
                f"({consecutive_rpc_errors}/{args.rpc_retry_limit})："
                f"{type(exc).__name__}: {exc}"
            )
            time.sleep(max(0.0, args.poll_interval))
            continue
        consecutive_rpc_errors = 0
        with state_lock:
            ui["rpc_consecutive_failures"] = 0
        if distance <= target.tolerance:
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
        # dwell 期间持续刷新 UI，并响应停止/退出
        dwell_deadline = time.monotonic() + args.dwell_seconds
        while time.monotonic() < dwell_deadline:
            if stop_event.is_set() or ui["stop_cruise"].is_set():
                break
            try:
                poll_flight(client, target, ui, state_lock)
            except Exception:
                pass  # 悬停期间遥测失败不打断 dwell
            time.sleep(max(0.0, args.poll_interval))
    return True


def fly_to_waypoint(
    *,
    client: airsim.MultirotorClient,
    waypoint: Waypoint,
    stop_event: threading.Event,
    args: argparse.Namespace,
    waypoint_index: int,
    monitor: CollisionMonitor | None,
    ui: dict[str, Any],
    state_lock: threading.Lock,
) -> None:
    """Fly at the enforced altitude, optionally using orthogonal X/Y legs."""
    if args.no_axis_split:
        legs = [waypoint]
    else:
        try:
            current_x, current_y, _ = get_position(client)
        except Exception as exc:
            # 定位遥测失败：降级为单腿直飞，不中断任务
            print(f"[FLIGHT] 分轴定位失败（{type(exc).__name__}），本次航点直飞")
            ui["messages"].push("warn", "分轴定位失败，本次航点直飞")
            legs = [waypoint]
        else:
            legs = []
            if abs(current_x - waypoint.x) > waypoint.tolerance:
                legs.append(Waypoint(waypoint.x, current_y, waypoint.z, waypoint.speed, waypoint.tolerance))
            if abs(current_y - waypoint.y) > waypoint.tolerance or not legs:
                legs.append(waypoint)

    for leg in legs:
        if not fly_to_leg(
            client=client,
            target=leg,
            stop_event=stop_event,
            args=args,
            waypoint_index=waypoint_index,
            monitor=monitor,
            ui=ui,
            state_lock=state_lock,
        ):
            return  # 被停止：不再执行后续航段


def _close_client(client: airsim.MultirotorClient | None) -> None:
    """Close the underlying msgpack session after a failed RPC attempt."""
    if client is None:
        return
    try:
        client.client.close()
    except Exception:
        pass


def safe_cancel(client: airsim.MultirotorClient | None) -> None:
    if client is None:
        return
    try:
        client.cancelLastTask()
    except Exception:
        pass


def _new_client(args: argparse.Namespace) -> airsim.MultirotorClient:
    return airsim.MultirotorClient(
        ip=args.airsim_ip,
        port=args.airsim_port,
        timeout_value=args.airsim_timeout,
    )


def _wait_for_connection(
    args: argparse.Namespace,
    stop_event: threading.Event,
    ui: dict[str, Any],
    state_lock: threading.Lock,
) -> airsim.MultirotorClient | None:
    """等待 AirSim 就绪；未启动时保持运行，不自动退出。

    连接期间界面显示“等待 AirSim 信号”；按 Q 可退出。连接成功返回新的
    AirSim 客户端，被停止返回 None。
    """
    delay = 1.0
    endpoint = f"{args.airsim_ip}:{args.airsim_port}"
    last_error = ""
    print(f"[FLIGHT] 等待 AirSim 连接…（{endpoint}）")
    ui["messages"].push("info", "等待 AirSim 信号，请启动模拟器…")
    with state_lock:
        ui["airsim_connected"] = False
        ui["airsim_ready"] = False
        ui["airsim_error"] = ""
    set_flight_state(ui, state_lock, "CONNECTING")
    while not stop_event.is_set():
        candidate: airsim.MultirotorClient | None = None
        try:
            # A timed-out msgpack client may retain a broken session. Create a
            # fresh client for every attempt instead of reusing that session.
            candidate = _new_client(args)
            if not candidate.ping():
                raise ConnectionError("AirSim ping returned false")
            with state_lock:
                ui["airsim_connected"] = True
                ui["airsim_error"] = ""
            set_flight_state(ui, state_lock, "CONNECTING")
            print(f"[FLIGHT] AirSim RPC 已连接：{endpoint}")
            return candidate
        except Exception as exc:
            _close_client(candidate)
            error_text = f"{type(exc).__name__}: {exc}"
            with state_lock:
                ui["airsim_connected"] = False
                ui["airsim_ready"] = False
                ui["airsim_error"] = error_text
            set_flight_state(ui, state_lock, "CONNECTING")
            if error_text != last_error:
                last_error = error_text
                print(f"[FLIGHT] AirSim RPC 连接失败（{endpoint}）：{error_text}")
            if stop_event.wait(delay):
                break
            delay = min(delay * 2, 10.0)
    return None


def _wait_for_ready(
    client: airsim.MultirotorClient,
    args: argparse.Namespace,
    stop_event: threading.Event,
    ui: dict[str, Any],
    state_lock: threading.Lock,
) -> airsim.MultirotorClient | None:
    """等待 AirSim 场景加载完成（RPC 通了但场景可能还在加载）。

    ``getMultirotorState`` 能返回有效位置即视为就绪；期间界面显示
    “等待场景就绪”，按 Q 可退出。就绪返回客户端，被停止返回 None。
    """
    endpoint = f"{args.airsim_ip}:{args.airsim_port}"
    print("[FLIGHT] AirSim 已连接，等待场景就绪…")
    ui["messages"].push("info", "AirSim 已连接，等待场景加载…")
    last_error = ""
    while not stop_event.is_set():
        try:
            state = client.getMultirotorState()
            position = state.kinematics_estimated.position
            if position is not None:
                with state_lock:
                    ui["airsim_ready"] = True
                    ui["airsim_error"] = ""
                set_flight_state(ui, state_lock, "READY")
                print(f"[FLIGHT] AirSim 场景已就绪：{endpoint}")
                return client
        except Exception as exc:
            error_text = f"{type(exc).__name__}: {exc}"
            if error_text != last_error:
                last_error = error_text
                print(f"[FLIGHT] 等待场景就绪失败（{endpoint}）：{error_text}")
            _close_client(client)
            client = None
            with state_lock:
                ui["airsim_ready"] = False
                ui["airsim_error"] = error_text
            set_flight_state(ui, state_lock, "CONNECTING")

            # Reconnect with a new RPC session instead of polling a client
            # whose request already timed out.
            client = _wait_for_connection(args, stop_event, ui, state_lock)
            if client is None:
                return None
        if stop_event.wait(1.0):
            break
    _close_client(client)
    return None


def _takeoff(
    client: airsim.MultirotorClient,
    args: argparse.Namespace,
    stop_event: threading.Event,
    ui: dict[str, Any],
    state_lock: threading.Lock,
) -> bool:
    """起飞并爬升至巡航高度（首次与“重新开始”复用）。"""
    if stop_event.is_set() or ui["stop_cruise"].is_set():
        return False
    set_flight_state(ui, state_lock, "TAKING_OFF")
    client.armDisarm(True)
    print("正在起飞无人机...")
    ui["messages"].push("info", "正在起飞…")
    try:
        client.takeoffAsync(timeout_sec=30).join()
    except Exception as exc:
        raise RuntimeError(f"起飞失败（30 秒超时或 RPC 错误）: {exc}") from exc
    if stop_event.is_set() or ui["stop_cruise"].is_set():
        safe_cancel(client)
        return False
    try:
        client.moveToZAsync(args.takeoff_z, 2.0).join()
    except Exception as exc:
        raise RuntimeError(f"爬升至巡航高度失败: {exc}") from exc
    with state_lock:
        ui["cruise_started"] = True
    set_flight_state(ui, state_lock, "CRUISING")
    return True


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
    client: airsim.MultirotorClient | None = None
    api_control = False
    landed = False
    try:
        client = _wait_for_connection(args, stop_event, ui, state_lock)
        if client is None:
            return  # 用户在连接等待期间按 Q 退出
        client = _wait_for_ready(client, args, stop_event, ui, state_lock)
        if client is None:
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
                set_flight_state(ui, state_lock, "READY")
                print("[FLIGHT] 任务待命：点击“多航点巡航”开始巡航")
                ui["messages"].push("info", "任务待命：点击“多航点巡航”开始巡航")
                while not ui["start_cruise"].wait(0.2) and not stop_event.is_set():
                    pass
                ui["start_cruise"].clear()
                if stop_event.is_set():
                    break
            first_run = False

            landed = False
            if not _takeoff(client, args, stop_event, ui, state_lock):
                break
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
                        set_flight_state(ui, state_lock, "STOPPING")
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
                        waypoint_index=waypoint_index,
                        monitor=monitor,
                        ui=ui,
                        state_lock=state_lock,
                    )
                if cruise_aborted:
                    break

            # 本次巡航结束：降落，回到待命
            landed = False
            set_flight_state(ui, state_lock, "LANDING")
            safe_cancel(client)
            try:
                client.landAsync().join()
                landed = True
            except Exception as exc:
                print(f"降落过程中出现问题：{exc}")
            try:
                client.armDisarm(False)
            except Exception:
                pass
            with state_lock:
                ui["cruise_started"] = False
            if result["error"] is None and not stop_event.is_set():
                set_flight_state(ui, state_lock, "READY")
            if not stop_event.is_set():
                print("[FLIGHT] 巡航结束，已降落；等待“多航点巡航”重新开始")
                ui["messages"].push("info", "巡航结束，已降落")
    except Exception as exc:
        result["error"] = exc
        set_flight_state(ui, state_lock, "ERROR")
        with state_lock:
            ui["cruise_started"] = False
        print(f"[FLIGHT] 巡航线程出错，准备降落: {type(exc).__name__}: {exc}")
        ui["messages"].push("error", f"巡航线程出错：{type(exc).__name__}")
    finally:
        if result["error"] is None:
            set_flight_state(ui, state_lock, "LANDING")
        safe_cancel(client)
        if api_control:
            if not landed:
                try:
                    client.landAsync().join()
                    landed = True
                except Exception as exc:
                    print(f"降落过程中出现问题：{exc}")
            try:
                client.armDisarm(False)
                client.enableApiControl(False)
            except Exception:
                pass
        with state_lock:
            ui["cruise_started"] = False
        if result["error"] is None:
            set_flight_state(ui, state_lock, "STOPPED")
        _close_client(client)
        ui["messages"].push("info", "任务结束，已降落")
        done_event.set()
