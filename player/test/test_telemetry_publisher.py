"""TelemetryPublisher enablement + sample projection (no real HTTP)."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from istream_player.modules.status.telemetry_publisher import TelemetryPublisher
from istream_player.modules.status.telemetry_schema import SAMPLE_VALUE_KEYS


class _FakeResp:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def text(self) -> str:
        return "ok"


class _FakeSession:
    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []

    def post(self, url: str, json: Optional[dict] = None):
        self.posts.append({"url": url, "json": json})
        return _FakeResp(200)


async def _async_return(value):
    return value


def test_publisher_posts_projected_sample():
    session = _FakeSession()
    config = SimpleNamespace(
        telemetry_run_id="r1",
        telemetry_arm_id="a",
        telemetry_session_id="s1",
        telemetry_interval_ms=1,
    )
    snap = {
        "state": "PLAYING",
        "position": 1.0,
        "stall_s_total": 0.5,
        "gpu_proc_mem_src": "torch",
        "enhanced_ready_inventory": None,
        "enhance_pending_inventory": None,
        "playing_segment": 7,
        "segment_index": None,
    }
    pub = TelemetryPublisher(
        get_snapshot=lambda: snap,
        get_config=lambda: config,  # type: ignore[arg-type]
        get_session=lambda: _async_return(session),
        server_base=lambda: "http://ctrl:8080",
    )
    assert pub.enabled()
    asyncio.run(pub.maybe_report())
    assert len(session.posts) == 1
    body = session.posts[0]["json"]
    assert body["run_id"] == "r1"
    assert body["arm_id"] == "a"
    sample = body["sample"]
    assert sample["state"] == "PLAYING"
    assert sample["stall_s_total"] == 0.5
    assert sample["gpu_proc_mem_src"] == "torch"
    assert sample["segment_index"] == 7
    assert sample["enhanced_ready_inventory"] == []
    assert sample["enhance_pending_inventory"] == []
    for key in SAMPLE_VALUE_KEYS:
        assert key in sample


def test_publisher_disabled_without_ids():
    pub = TelemetryPublisher(
        get_snapshot=lambda: {"state": "PLAYING"},
        get_config=lambda: SimpleNamespace(telemetry_run_id="", telemetry_arm_id=""),  # type: ignore[arg-type]
        get_session=lambda: _async_return(_FakeSession()),
        server_base=lambda: "http://ctrl:8080",
    )
    assert not pub.enabled()
    asyncio.run(pub.maybe_report())  # no-op
