from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from prism.control.app import create_app
from prism.control.config import ServerConfig
from prism.control.plugins.mpdutil import build_static_mpd
from prism.control.schemas import DEFAULT_LADDER
from prism.control.telemetry_schema import (
    METRICS_VERSION,
    SAMPLE_VALUE_KEYS,
    project_row,
    project_sample,
    schema_path,
)


def test_schema_file_is_player_canonical():
    path = schema_path()
    assert path.name == "telemetry_schema.py"
    assert "istream_player" in str(path)
    assert path.exists()


def test_project_sample_includes_version_and_all_keys():
    snap = {
        "state": "PLAYING",
        "position": 1.5,
        "download_buffer_s": 4.0,
        "render_fps": 29.0,
        "cpu_pct": 8.0,
        "unknown_should_drop": 123,
    }
    sample = project_sample(snap, ts=1000.0, event="state:BUFFERING->PLAYING")
    assert sample["metrics_v"] == METRICS_VERSION
    assert sample["ts"] == 1000.0
    assert sample["event"] == "state:BUFFERING->PLAYING"
    assert sample["state"] == "PLAYING"
    assert sample["render_fps"] == 29.0
    assert sample["cpu_pct"] == 8.0
    assert "unknown_should_drop" not in sample
    for key in SAMPLE_VALUE_KEYS:
        assert key in sample


def test_project_row_matches_sample_keys():
    sample = project_sample({"state": "PLAYING", "stall_s_total": 1.25}, ts=42.0)
    row = project_row(sample, run_id="r1", arm_id="a", session_id="s1")
    assert row["run_id"] == "r1"
    assert row["arm_id"] == "a"
    assert row["session_id"] == "s1"
    assert row["metrics_v"] == METRICS_VERSION
    assert row["stall_s_total"] == 1.25
    for key in SAMPLE_VALUE_KEYS:
        assert key in row


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


def test_api_persists_schema_fields_including_formerly_dropped(tmp_path: Path):
    """playing_width used to be posted by the player but stripped by whitelist."""
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    app = create_app(config)
    dash = _fake_dash(tmp_path)
    spec = {
        "apiVersion": "prism.lab/v1",
        "kind": "ExperimentSpec",
        "metadata": {"name": "schema"},
        "spec": {
            "content": {"source": f"path:{dash}"},
            "arms": [{"id": "a", "player": {"mod_renderer": "headless", "mod_status": "off"}}],
            "execution": {"arm_mode": "serial"},
        },
    }
    with TestClient(app) as client:
        run = client.post("/v1/runs", json={"spec": spec, "attach_players": False}).json()
        rid = run["id"]
        res = client.post(
            "/v1/telemetry",
            json={
                "run_id": rid,
                "arm_id": "a",
                "session_id": run["arms"][0]["session_id"],
                "sample": project_sample(
                    {
                        "state": "PLAYING",
                        "position": 2.0,
                        "playing_width": 1280,
                        "playing_height": 720,
                        "render_fps": 30.0,
                        "gpu_proc_mem_mb": 512.0,
                        "gpu_proc_mem_src": "nvml",
                    },
                    ts=1234.5,
                ),
            },
        )
        assert res.status_code == 200, res.text
        series = client.get(f"/v1/runs/{rid}/telemetry?arm_id=a&limit=5").json()
        last = series["series"]["a"][-1]
        assert last["metrics_v"] == METRICS_VERSION
        assert last["playing_width"] == 1280
        assert last["playing_height"] == 720
        assert last["render_fps"] == 30.0
        assert last["gpu_proc_mem_mb"] == 512.0
        assert last["gpu_proc_mem_src"] == "nvml"
