"""Shared AirSim RPC client lifecycle helpers.

Each worker still owns its own ``MultirotorClient`` instance.  This module
only centralizes endpoint configuration and best-effort session cleanup so the
flight and camera workers cannot silently drift apart.
"""

from __future__ import annotations

import argparse
from typing import Any

import airsim


def new_client(args: argparse.Namespace) -> airsim.MultirotorClient:
    """Create a fresh client for one worker or one reconnect attempt."""
    return airsim.MultirotorClient(
        ip=args.airsim_ip,
        port=args.airsim_port,
        timeout_value=args.airsim_timeout,
    )


def close_client(client: Any | None) -> None:
    """Best-effort close of an AirSim msgpack session.

    AirSim's public client does not expose one stable ``close`` API across all
    versions, so try the public method first and then the underlying session.
    Cleanup must never mask the original connection or flight error.
    """
    if client is None:
        return
    try:
        client.close()
        return
    except Exception:
        pass
    try:
        client.client.close()
    except Exception:
        pass

