"""Benchmark raw AirSim Scene capture without loading YOLO or controlling flight."""

from __future__ import annotations

import argparse
import statistics
import time

import airsim

from airsim_connection import close_client
from capture import read_scene_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark raw AirSim Scene capture")
    parser.add_argument("--frames", type=int, default=120, help="Measured frames")
    parser.add_argument("--warmup", type=int, default=5, help="Warm-up frames excluded from statistics")
    parser.add_argument("--camera", default="0")
    parser.add_argument("--airsim-ip", default="127.0.0.1")
    parser.add_argument("--airsim-port", type=int, default=41451)
    parser.add_argument("--airsim-timeout", type=float, default=5.0)
    args = parser.parse_args()
    if args.frames <= 0 or args.warmup < 0:
        parser.error("--frames 必须大于 0，--warmup 不能为负数")
    return args


def main() -> None:
    args = parse_args()
    client = airsim.MultirotorClient(
        ip=args.airsim_ip,
        port=args.airsim_port,
        timeout_value=args.airsim_timeout,
    )
    try:
        client.confirmConnection()
        for _ in range(args.warmup):
            read_scene_frame(client, args.camera)

        rpc_ms: list[float] = []
        parse_ms: list[float] = []
        total_ms: list[float] = []
        successes = 0
        source_size = (0, 0)
        started = time.perf_counter()
        for _ in range(args.frames):
            frame_started = time.perf_counter()
            result = read_scene_frame(client, args.camera)
            total_ms.append((time.perf_counter() - frame_started) * 1000.0)
            rpc_ms.append(result.rpc_ms)
            parse_ms.append(result.parse_ms)
            if result.frame is not None:
                successes += 1
                source_size = (int(result.frame.shape[1]), int(result.frame.shape[0]))
        elapsed = max(0.001, time.perf_counter() - started)

        def summary(values: list[float]) -> str:
            return f"平均 {statistics.mean(values):.2f} ms / 最大 {max(values):.2f} ms"

        print("模式              : raw Scene")
        print(f"有效帧            : {successes}/{args.frames}")
        print(f"源分辨率          : {source_size[0]}×{source_size[1]}")
        print(f"实际采集 FPS      : {successes / elapsed:.2f}")
        print(f"simGetImages RPC  : {summary(rpc_ms)}")
        print(f"图像解析          : {summary(parse_ms)}")
        print(f"单帧总耗时        : {summary(total_ms)}")
    finally:
        close_client(client)


if __name__ == "__main__":
    main()
