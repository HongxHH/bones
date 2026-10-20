from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from prism.control.app import create_app
from prism.control.config import ServerConfig
from prism.control.plugins.mpdutil import build_static_mpd
from prism.control.schemas import DEFAULT_LADDER
from prism.control.telemetry import summarize_arm


def _fake_dash(root: Path) -> Path:
    tree = root / "dash"
    tree.mkdir(parents=True)
    xml = build_static_mpd(
        title="fake",
        duration_s=12.0,
        segment_duration=4.0,
        ladder=DEFAULT_LADDER[:2],
    )
    (tree / "clip.mpd").write_text(xml, encoding="utf-8")
    for rid in (0, 1):
        stream = tree / f"stream{rid}"
        stream.mkdir()
        (stream / "segment_.mp4").write_bytes(b"init")
        (stream / "segment_1.m4s").write_bytes(b"seg")
    return tree


def test_summarize_rebuffer():
    """Legacy samples without stall_s_total: gap reconstruction (BUFFERING only)."""
    samples = [
        {"ts": 100.0, "state": "PLAYING", "bandwidth_bps": 1e6, "download_buffer_s": 5, "download_action": {"0": 1}, "position": 1},
        {"ts": 101.0, "state": "BUFFERING", "bandwidth_bps": 1e6, "download_buffer_s": 0.2, "download_action": {"0": 0}, "position": 1},
        {"ts": 103.0, "state": "PLAYING", "bandwidth_bps": 2e6, "download_buffer_s": 4, "download_action": {"0": 1}, "position": 3},
    ]
    s = summarize_arm(samples)
    assert s["samples"] == 3
    assert s["rebuffer_s"] == 2.0
    assert s["stall_s"] == 2.0
    assert s["avg_bandwidth_bps"] > 0


def test_summarize_prefers_client_stall_s_total():
    """Client StallTracker cumulative is canonical; gap estimate must not diverge."""
    samples = [
        {"ts": 100.0, "state": "PLAYING", "stall_s_total": 0.0},
        # Gap would say 2.0s, but client reports a precise 1.25s cumulative.
        {"ts": 101.0, "state": "BUFFERING", "stall_s_total": 0.5},
        {"ts": 103.0, "state": "PLAYING", "stall_s_total": 1.25},
    ]
    s = summarize_arm(samples)
    assert s["stall_s"] == 1.25
    assert s["rebuffer_s"] == 1.25
    assert s["rebuffer_ratio"] == round(1.25 / 3.0, 4)


def test_summarize_loading_not_counted_as_rebuffer_gap():
    samples = [
        {"ts": 0.0, "state": "LOADING"},
        {"ts": 5.0, "state": "PLAYING"},
    ]
    s = summarize_arm(samples)
    assert s["rebuffer_s"] == 0.0
    assert s["stall_s"] == 0.0


def test_telemetry_api_and_stop_summary(tmp_path: Path):
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    app = create_app(config)
    dash = _fake_dash(tmp_path)
    spec = {
        "apiVersion": "prism.lab/v1",
        "kind": "ExperimentSpec",
        "metadata": {"name": "tel"},
        "spec": {
            "content": {"source": f"path:{dash}"},
            "arms": [{"id": "a", "player": {"mod_renderer": "headless", "mod_status": "off"}}],
            "execution": {"arm_mode": "serial"},
        },
    }
    with TestClient(app) as client:
        run = client.post("/v1/runs", json={"spec": spec, "attach_players": False}).json()
        rid, arm = run["id"], "a"
        for i in range(5):
            res = client.post(
                "/v1/telemetry",
                json={
                    "run_id": rid,
                    "arm_id": arm,
                    "session_id": run["arms"][0]["session_id"],
                    "sample": {
                        "ts": 1000 + i,
                        "state": "BUFFERING" if i == 2 else "PLAYING",
                        "position": float(i),
                        "download_buffer_s": 3.0 - (0.5 if i == 2 else 0),
                        "enhanced_ready_s": 4.0,
                        "enhance_pending_s": 2.0,
                        "enhanced_ready_inventory": [
                            {
                                "index": 10 + i,
                                "duration_s": 4.0,
                                "enhance_level": 2,
                                "enhance_latency_s": 0.8,
                                "repr_id": 1,
                                "model": "IMDN",
                                "status": "ready",
                            }
                        ],
                        "enhance_pending_inventory": [
                            {
                                "index": 20 + i,
                                "duration_s": 4.0,
                                "enhance_level": 2,
                                "status": "queued",
                                "repr_id": 1,
                                "model": "IMDN",
                            }
                        ],
                        "render_fps": 29.5,
                        "stall_active": i == 2,
                        "stall_s_total": 0.5 if i >= 2 else 0.0,
                        "cpu_pct": 12.0,
                        "rss_mb": 512.0,
                        "mem_pct": 3.2,
                        "mem_total_mb": 16000.0,
                        "gpu_util_pct": 40.0,
                        "gpu_mem_used_mb": 2048.0,
                        "gpu_mem_total_mb": 16384.0,
                        "gpu_mem_pct": 12.5,
                        "gpu_proc_mem_mb": 1024.0,
                        "gpu_proc_mem_pct": 6.25,
                        "gpu_proc_mem_src": "torch",
                        "bandwidth_bps": 800000,
                        "download_action": {"0": 0},
                    },
                },
            )
            assert res.status_code == 200, res.text

        series = client.get(f"/v1/runs/{rid}/telemetry?arm_id=a&limit=10").json()
        assert len(series["series"]["a"]) == 5
        last = series["series"]["a"][-1]
        assert last["enhanced_ready_s"] == 4.0
        assert last["enhanced_ready_inventory"][0]["index"] == 14
        assert last["enhance_pending_inventory"][0]["status"] == "queued"
        assert last["metrics_v"] == 1
        assert last["render_fps"] == 29.5
        assert last["stall_s_total"] == 0.5
        assert last["cpu_pct"] == 12.0
        assert last["mem_pct"] == 3.2
        assert last["gpu_util_pct"] == 40.0
        assert last["gpu_mem_pct"] == 12.5
        assert last["gpu_proc_mem_mb"] == 1024.0
        assert last["gpu_proc_mem_src"] == "torch"
        metrics = client.get(f"/v1/runs/{rid}/metrics").json()
        assert metrics["arms"]["a"]["avg_render_fps"] == 29.5
        assert metrics["arms"]["a"]["stall_s"] == 0.5
        assert metrics["arms"]["a"]["samples"] == 5

        stopped = client.post(f"/v1/runs/{rid}/stop").json()
        assert stopped["status"] == "stopped"
        assert stopped["metrics"]["arms"]["a"]["samples"] == 5
        assert Path(stopped["metrics"]["path"]).exists()
