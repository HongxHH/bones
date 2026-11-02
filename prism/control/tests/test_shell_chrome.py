from __future__ import annotations

import json

from prism.desk.picture import _fetch_hud, _pick_live_run
from prism.desk.scrub import (
    frame_caption,
    frames_page_url,
    pick_scrub_run,
    picture_put_body,
    picture_url,
    save_frames_url,
    split_left,
)


def test_picture_put_body_and_urls():
    body = json.loads(picture_put_body("run_1", "play", 12))
    assert body == {"run_id": "run_1", "arm_id": "play", "frame": 12}
    assert picture_url("127.0.0.1", 8080) == "http://127.0.0.1:8080/v1/system/picture"
    assert frames_page_url("127.0.0.1", 8080, "run_1", "play a") == (
        "http://127.0.0.1:8080/v1/runs/run_1/frames?arm_id=play%20a&limit=1"
    )
    assert save_frames_url("127.0.0.1", 8080, "run_1", "play") == (
        "http://127.0.0.1:8080/v1/runs/run_1/frames/save?arm_id=play"
    )


def test_frame_caption_matches_lab_line():
    text = frame_caption(
        {
            "frame": 4,
            "segment": 1,
            "representation": 2,
            "enhance_level": 3,
            "enhanced": True,
            "model": "imdn",
            "scale": "x2",
            "vmaf": 80.14,
        }
    )
    assert text == "#4 · 段 1 · 2 · 增强 3 · 已增强 · imdn · x2 · VMAF 80.1"
    assert frame_caption(None) == "—"
    plain = frame_caption({"frame": 0, "segment": 0, "representation": 0, "enhance_level": 0, "enhanced": False})
    assert "未增强" in plain
    assert "VMAF" not in plain


def test_pick_scrub_run_prefers_live_then_newest_stopped():
    runs = [
        {"id": "old", "status": "stopped", "updated_at": "2020-01-01", "active_arm_id": "a"},
        {"id": "new", "status": "stopped", "updated_at": "2024-01-01", "arms": [{"id": "b"}]},
        {"id": "live", "status": "running", "updated_at": "2024-06-01", "active_arm_id": "c"},
    ]
    assert pick_scrub_run(runs) == {"run_id": "live", "arm_id": "c"}
    assert pick_scrub_run(runs[:2]) == {"run_id": "new", "arm_id": "b"}
    assert pick_scrub_run([{"id": "prep", "status": "preparing", "active_arm_id": "d"}]) == {
        "run_id": "prep",
        "arm_id": "d",
    }
    assert pick_scrub_run([]) is None
    assert pick_scrub_run([{"id": "x", "status": "queued"}]) is None


def test_hud_keeps_bar_when_player_status_misses(monkeypatch):
    runs = [{
        "id": "run_1",
        "status": "running",
        "started_at": "2026-01-01T00:00:00",
        "active_arm_id": "play",
        "arms": [{"id": "play", "pid": 9, "status_port": 8765}],
    }]
    monkeypatch.setattr("prism.desk.picture._read_player_status", lambda port: None)
    monkeypatch.setattr(
        "prism.desk.picture._latest_live_sample",
        lambda host, port, run_id, arm_id: {
            "state": "PLAYING",
            "paused": False,
            "playing_segment": 4,
            "enhancing_segment": 6,
            "download_buffer_s": 3.5,
            "bandwidth_bps": 1_500_000,
            "compare_view": False,
        },
    )
    picked = _pick_live_run(runs, "127.0.0.1", 8080)
    assert picked["stale"] is False
    assert picked["source"] == "telemetry"
    assert picked["status"]["playing_segment"] == 4

    monkeypatch.setattr("prism.desk.picture._latest_live_sample", lambda *args: None)
    missed = _pick_live_run(runs, "127.0.0.1", 8080)
    assert missed["stale"] is True
    assert missed["status"] is None

    monkeypatch.setattr("prism.desk.picture._load_runs", lambda host, port: runs)
    monkeypatch.setattr("prism.desk.picture._scrub_snapshot", lambda host, port, rows: None)
    hud = _fetch_hud("127.0.0.1", 8080)
    assert hud["stale"] is True
    assert hud["facts"] is None


def test_split_left_review_uses_half_and_play_keeps_picture():
    assert split_left(1920, review=False) == 480
    assert split_left(1280, review=False) == 360
    assert split_left(1920, review=True) == 960
    assert split_left(100, review=False) == 100
    assert split_left(100, review=True) == 100
