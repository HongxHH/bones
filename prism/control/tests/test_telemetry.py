from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from prism.control.app import create_app
from prism.control.config import ServerConfig
from prism.control.orchestrator.state import PlatformState
from prism.control.plugins.mpdutil import build_static_mpd
from prism.control.registry import PluginRegistry
from prism.control.schemas import DEFAULT_LADDER, Experiment, ExperimentRun
from prism.control.telemetry import summarize_arm, telemetry_dir
from prism.control.metrics.catalog import resolve_content_id


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


def test_resolve_content_id_known_bbb_tree():
    assert resolve_content_id("path:player/data/video_mp4box", "") == "bbb"


def test_telemetry_dir_uses_store_when_recorded_archive_is_empty(tmp_path: Path):
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    state = PlatformState(config, PluginRegistry())
    exp = Experiment(
        id="exp_moved",
        name="bbb-play",
        artifact_dir=str(tmp_path / "server" / "var" / "experiments" / "exp_moved"),
    )
    state.experiments[exp.id] = exp
    run = ExperimentRun(id="run_moved", experiment_id=exp.id, name="bbb-play")
    real = tmp_path / "var" / "experiments" / "exp_moved" / "runs" / "run_moved" / "telemetry"
    real.mkdir(parents=True)
    (real / "play.jsonl").write_text('{"ts": 1}\n', encoding="utf-8")
    assert telemetry_dir(state, run) == real
    assert state.experiments[exp.id].artifact_dir == str(real.parent.parent.parent)


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


def test_end_telemetry_stops_running_run(tmp_path: Path):
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    app = create_app(config)
    dash = _fake_dash(tmp_path)
    spec = {
        "apiVersion": "prism.lab/v1",
        "kind": "ExperimentSpec",
        "metadata": {"name": "tel-end"},
        "spec": {
            "content": {"source": f"path:{dash}"},
            "arms": [{"id": "a", "player": {"mod_renderer": "headless", "mod_status": "off"}}],
            "execution": {"arm_mode": "serial"},
        },
    }
    with TestClient(app) as client:
        run = client.post("/v1/runs", json={"spec": spec, "attach_players": False}).json()
        assert run["status"] == "running"
        rid = run["id"]
        res = client.post(
            "/v1/telemetry",
            json={
                "run_id": rid,
                "arm_id": "a",
                "sample": {
                    "ts": 2000,
                    "state": "END",
                    "event": "state:READY->END",
                    "position": 12.0,
                    "download_buffer_s": 0.0,
                    "stall_s_total": 0.5,
                },
            },
        )
        assert res.status_code == 200, res.text
        assert res.json().get("run_status") == "finishing"
        got = None
        for _ in range(40):
            got = client.get(f"/v1/runs/{rid}").json()
            if got["status"] == "stopped":
                break
            import time
            time.sleep(0.05)
        assert got is not None
        assert got["status"] == "stopped"
        assert got["stopped_at"] is not None


def test_end_telemetry_does_not_kill_before_return(tmp_path: Path, monkeypatch):
    """END telemetry returns while the player is still alive. The file can already be read."""
    import time

    killed: list[int] = []
    monkeypatch.setattr("prism.control.runs.kill_pid", lambda pid: killed.append(pid))
    monkeypatch.setattr("prism.control.attach.pid_alive", lambda pid: True)
    monkeypatch.setattr("prism.control.runs.FINISH_WAIT_S", 0.4)
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    app = create_app(config)
    dash = _fake_dash(tmp_path)
    spec = {
        "apiVersion": "prism.lab/v1",
        "kind": "ExperimentSpec",
        "metadata": {"name": "tel-end-alive"},
        "spec": {
            "content": {"source": f"path:{dash}"},
            "arms": [{"id": "a", "player": {"mod_renderer": "headless", "mod_status": "off"}}],
        },
    }
    with TestClient(app) as client:
        run = client.post("/v1/runs", json={"spec": spec, "attach_players": False}).json()
        rid = run["id"]
        arm = client.app.state.platform.runs[rid].arms[0]
        arm.pid = 424242
        from prism.control.telemetry import telemetry_dir

        folder = telemetry_dir(client.app.state.platform, client.app.state.platform.runs[rid]).parent / "analyzer"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "play.json").write_text(
            '{"enhance_level_stats":{"1":{"count":2,"completed":1}},'
            '"abort_reason_count":{"deadline":1},'
            '"startup_metrics":{"time_to_first_play":0.4},'
            '"gpu_metrics":{"peak_memory_mb":512},'
            '"segments":[{}]}',
            encoding="utf-8",
        )
        res = client.post(
            "/v1/telemetry",
            json={"run_id": rid, "arm_id": "a", "sample": {"ts": 3, "state": "END", "event": "state:READY->END"}},
        )
        assert res.status_code == 200, res.text
        assert res.json().get("run_status") == "finishing"
        assert killed == []
        report = client.get(f"/v1/runs/{rid}/analyzer").json()
        assert report["available"] is True
        assert report["file"] == "play.json"
        assert report["enhance_levels"][0]["count"] == 2
        assert report["abort_reasons"]["deadline"] == 1
        assert report["startup"]["time_to_first_play"] == 0.4
        assert report["gpu"]["peak_memory_mb"] == 512
        time.sleep(0.7)
        assert 424242 in killed


def test_iter_samples_limit_reads_tail(tmp_path: Path):
    import json

    from prism.control.telemetry import iter_samples

    path = tmp_path / "arm.jsonl"
    lines = [json.dumps({"ts": i, "state": "PLAYING"}) for i in range(400)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rows = iter_samples(path, limit=3)
    assert [row["ts"] for row in rows] == [397, 398, 399]
    assert iter_samples(path, limit=3)[-1]["ts"] == 399


def test_live_view_skips_bundle(tmp_path: Path):
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    app = create_app(config)
    dash = _fake_dash(tmp_path)
    spec = {
        "apiVersion": "prism.lab/v1",
        "kind": "ExperimentSpec",
        "metadata": {"name": "live"},
        "spec": {
            "content": {"source": f"path:{dash}"},
            "arms": [{"id": "a", "player": {"mod_renderer": "headless", "mod_status": "off"}}],
        },
    }
    with TestClient(app) as client:
        run = client.post("/v1/runs", json={"spec": spec, "attach_players": False}).json()
        rid = run["id"]
        assert run["spec"]["provenance"]["renderer"] == "headless"
        posted = client.post(
            "/v1/telemetry",
            json={
                "run_id": rid,
                "arm_id": "a",
                "sample": {
                    "ts": 10,
                    "state": "PLAYING",
                    "stall_s_total": 0.25,
                    "frame_archive_dropped": 2,
                    "enhanced_ready_inventory": [{"index": 1}],
                },
            },
        )
        assert posted.status_code == 200, posted.text
        live = client.get(f"/v1/runs/{rid}/live?arm_id=a&limit=20").json()
        assert live["status"] == "running"
        assert live["arm_id"] == "a"
        assert live["summary"]["stall_s"] == 0.25
        assert live["summary"]["frame_archive_dropped"] == 2
        assert "bundle" not in live
        assert live["samples"][-1]["enhanced_ready_inventory"][0]["index"] == 1


def test_analyzer_report_projects_player_json(tmp_path: Path):
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    app = create_app(config)
    dash = _fake_dash(tmp_path)
    spec = {
        "apiVersion": "prism.lab/v1",
        "kind": "ExperimentSpec",
        "metadata": {"name": "analyzer"},
        "spec": {
            "content": {"source": f"path:{dash}"},
            "arms": [{"id": "a", "player": {"mod_renderer": "headless", "mod_status": "off"}}],
        },
    }
    with TestClient(app) as client:
        run = client.post("/v1/runs", json={"spec": spec, "attach_players": False}).json()
        rid = run["id"]
        missing = client.get(f"/v1/runs/{rid}/analyzer").json()
        assert missing["available"] is False
        from prism.control.telemetry import telemetry_dir

        state = client.app.state.platform
        stored = state.runs[rid]
        summary_path = telemetry_dir(state, stored).parent / "metrics_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(
            '{"arms":{"a":{"qoe":12.5,"stall_s":1.25,"avg_bandwidth_bps":800000,"num_quality_switches":3}}}',
            encoding="utf-8",
        )
        from_metrics = client.get(f"/v1/runs/{rid}/analyzer").json()
        assert from_metrics["available"] is True
        assert from_metrics["summary"]["qoe"] == 12.5
        assert from_metrics["summary"]["dur_stall"] == 1.25
        assert from_metrics["summary"]["avg_bitrate"] == 800000
        assert from_metrics["enhance_levels"] == []
        from prism.control.telemetry import telemetry_dir
        from prism.control.orchestrator.state import PlatformState

        state = client.app.state.platform
        assert isinstance(state, PlatformState)
        stored = state.runs[rid]
        folder = telemetry_dir(state, stored).parent / "analyzer"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "play.json").write_text(
            '{"QOE": 70.5, "num_stall": 2, "segments": [{"index": 3, "enhance_action": 2, '
            '"abort_reason": "Remaining time less than min threshold", "url": "http://secret", '
            '"fast_complete": false, "real_vmaf": 61.2}], '
            '"enhance_level_stats": {"1": {"count": 3, "completed": 2, "_latency_sum": 9}}, '
            '"startup_metrics": {"time_to_first_play": 1.2}, '
            '"bandwidth_estimate": [{"time": 0, "bandwidth": 1}]}',
            encoding="utf-8",
        )
        got = client.get(f"/v1/runs/{rid}/analyzer").json()
        assert got["available"] is True
        assert got["summary"]["qoe"] == 70.5
        assert got["summary"]["num_stall"] == 2
        assert got["summary"]["segments"] == 1
        assert got["enhance_levels"][0]["count"] == 3
        assert "_latency_sum" not in got["enhance_levels"][0]
        assert "bandwidth_estimate" not in got
        assert got["startup"]["time_to_first_play"] == 1.2
        assert got["segments"] == [{
            "index": 3,
            "enhance_action": 2,
            "abort_reason": "Remaining time less than min threshold",
            "fast_complete": False,
            "real_vmaf": 61.2,
        }]


def test_delete_runs_removes_record_and_analyzer(tmp_path: Path):
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    app = create_app(config)
    dash = _fake_dash(tmp_path)
    spec = {
        "apiVersion": "prism.lab/v1",
        "kind": "ExperimentSpec",
        "metadata": {"name": "del"},
        "spec": {
            "content": {"source": f"path:{dash}"},
            "arms": [{"id": "a", "player": {"mod_renderer": "headless", "mod_status": "off"}}],
        },
    }
    with TestClient(app) as client:
        run = client.post("/v1/runs", json={"spec": spec, "attach_players": False}).json()
        rid = run["id"]
        stopped = client.post(f"/v1/runs/{rid}/stop").json()
        assert stopped["status"] == "stopped"
        from prism.control.telemetry import telemetry_dir

        state = client.app.state.platform
        folder = telemetry_dir(state, state.runs[rid]).parent / "analyzer"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "play.json").write_text("{}", encoding="utf-8")
        result = client.post("/v1/runs/delete", json={"ids": [rid, rid]}).json()
        assert result["deleted"] == [rid]
        assert client.get(f"/v1/runs/{rid}").status_code == 404
        assert not folder.exists()
        assert rid not in client.app.state.platform.runs


def test_samples_show_playback_end():
    from prism.control.runs import samples_show_playback_end

    assert samples_show_playback_end([{"state": "END"}])
    assert samples_show_playback_end([{"state": "READY", "event": "state:READY->END"}])
    assert not samples_show_playback_end([{"state": "READY"}])
    assert not samples_show_playback_end([])
