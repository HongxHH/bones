"""StatusSnapshot build / stall / inventory — independent of HTTP shell."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from istream_player.models import State
from istream_player.modules.status.runtime_metrics import RuntimeMetrics
from istream_player.modules.status.snapshot import StatusSnapshot, jsonable


def test_jsonable_enum_and_nested():
    assert jsonable(State.BUFFERING) == "BUFFERING"
    assert jsonable({"a": State.READY, "b": [1, Path("x")]}) == {"a": "READY", "b": [1, "x"]}


def test_build_idle_defaults():
    runtime = RuntimeMetrics.__new__(RuntimeMetrics)
    runtime._interval_s = 1.0
    runtime._device_index = 0
    runtime._last_t = 0.0
    runtime._last = {}
    runtime._proc = None
    runtime._gpu = None
    runtime._pid = 1
    runtime.sample = lambda: {"cpu_pct": 2.5}  # type: ignore[method-assign]

    snap = StatusSnapshot(runtime=runtime)
    out = snap.build()
    assert out["state"] == "IDLE"
    assert out["download_buffer_s"] == 0.0
    assert out["stall_s_total"] == 0.0
    assert out["stall_active"] is False
    assert out["cpu_pct"] == 2.5
    assert out["enhanced_ready_inventory"] == []
    assert out["enhance_pending_inventory"] == []


def test_note_state_and_stall_in_build():
    runtime = RuntimeMetrics.__new__(RuntimeMetrics)
    runtime.sample = lambda: {}  # type: ignore[method-assign]
    snap = StatusSnapshot(runtime=runtime)
    snap.note_state(State.BUFFERING)
    assert snap.stalls.active
    snap.stalls.note_state(State.READY, now_mono=(snap.stalls._started_mono or 0.0) + 2.0)
    assert not snap.stalls.active
    assert abs(snap.stalls.total_s() - 2.0) < 1e-9
    assert snap.build()["stall_s_total"] == 2.0


def test_enhancing_segment_not_sticky_when_idle():
    runtime = RuntimeMetrics.__new__(RuntimeMetrics)
    runtime.sample = lambda: {}  # type: ignore[method-assign]
    runtime.set_enabled = lambda _e: None  # type: ignore[method-assign]
    snap = StatusSnapshot(runtime=runtime)
    snap.enhancer = SimpleNamespace(_current_enhancing_index=12)  # type: ignore[assignment]
    snap.enhancing_segments = {12: SimpleNamespace(duration=4.0, enhance_action=1, repr_id=0)}
    out = snap.build()
    assert out["enhance_busy"] is True
    assert out["enhancing_segment"] == 12
    snap.enhancer = SimpleNamespace(_current_enhancing_index=None)  # type: ignore[assignment]
    out = snap.build()
    assert out["enhance_busy"] is False
    assert out["enhancing_segment"] is None
    assert snap.last_enhanced_segment == 12


def test_ready_inventory_from_download_buffer():
    seg = SimpleNamespace(
        duration=4.0,
        enhance_action=2,
        repr_id=1,
        enhance_latency=0.5,
        decode_data=object(),
        is_enhance=True,
    )
    buf = SimpleNamespace(_segments={10: {0: seg}})
    buf.buffer_level = lambda continuous=True: 4.0

    runtime = RuntimeMetrics.__new__(RuntimeMetrics)
    runtime.sample = lambda: {}  # type: ignore[method-assign]

    snap = StatusSnapshot(runtime=runtime)
    snap.download_buffer = buf  # type: ignore[assignment]
    snap.config = SimpleNamespace(mod_enhancer="imdn_always")  # type: ignore[assignment]
    out = snap.build()
    assert out["enhanced_ready_s"] == 4.0
    assert len(out["enhanced_ready_inventory"]) == 1
    assert out["enhanced_ready_inventory"][0]["index"] == 10
    assert out["enhanced_ready_inventory"][0]["model"] == "IMDN"
    assert out["enhanced_ready_inventory"][0]["status"] == "ready"
