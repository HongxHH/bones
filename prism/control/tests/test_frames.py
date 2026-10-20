import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "player"))

from istream_player.modules.renderer.frame_archive import FrameArchive
from prism.control.app import create_app
from prism.control.attach import player_cmd
from prism.control.config import ServerConfig
from prism.control.frames import claim_current_frames, current_frames_dir
from prism.control.plugins.mpdutil import build_static_mpd
from prism.control.schemas import DEFAULT_LADDER


def test_frame_archive_roundtrip(tmp_path: Path):
    archive = FrameArchive(str(tmp_path / "arm"))
    image = np.zeros((12, 16, 3), np.uint8)
    image[:, :] = (10, 40, 200)
    assert archive.submit(
        image,
        {
            "segment": 1,
            "frame_in_segment": 0,
            "pts": 0.0,
            "representation": 2,
            "enhance_level": 1,
            "enhanced": True,
            "download_buffer_s": 1.5,
            "enhance_buffer_s": 0.25,
            "enhance_latency_s": 0.02,
        },
    ) == 0
    archive.close()
    rows = (tmp_path / "arm" / "frames.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(rows) == 1
    assert '"enhanced": true' in rows[0]
    jpeg = (tmp_path / "arm" / "frames" / "000000.jpg").read_bytes()
    assert jpeg.startswith(b"\xff\xd8")
    assert (tmp_path / "arm" / "count.txt").read_text(encoding="ascii").strip() == "1"


def test_player_cmd_includes_frame_archive(tmp_path: Path):
    cmd = player_cmd(
        "http://127.0.0.1:8080/manifest.mpd",
        {"mod_nes": "always_greedy"},
        8765,
        frame_archive=str(tmp_path / "frames"),
    )
    assert cmd[cmd.index("--frame_archive") + 1] == str(tmp_path / "frames")


def test_frame_api(tmp_path: Path):
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    app = create_app(config)
    dash = tmp_path / "dash"
    dash.mkdir()
    (dash / "clip.mpd").write_text(
        build_static_mpd(title="fake", duration_s=8.0, segment_duration=4.0, ladder=DEFAULT_LADDER[:1]),
        encoding="utf-8",
    )
    (dash / "stream0").mkdir()
    (dash / "stream0" / "segment_.mp4").write_bytes(b"init")
    spec = {
        "apiVersion": "prism.lab/v1",
        "kind": "ExperimentSpec",
        "metadata": {"name": "frames"},
        "spec": {
            "content": {"source": f"path:{dash}"},
            "arms": [{"id": "main", "player": {"mod_renderer": "headless", "mod_status": "off"}}],
        },
    }
    with TestClient(app) as client:
        run = client.post("/v1/runs", json={"spec": spec, "attach_players": False}).json()
        rid = run["id"]
        root = current_frames_dir(client.app.state.platform)
        writer = FrameArchive(str(root))
        picture = np.full((8, 8, 3), 30, np.uint8)
        baseline = np.full((8, 8, 3), 180, np.uint8)
        writer.submit(picture, {"segment": 3, "pts": 8.0, "enhanced": False}, baseline)
        writer.close()
        listed = client.get(f"/v1/runs/{rid}/frames", params={"arm_id": "main"})
        assert listed.status_code == 200, listed.text
        body = listed.json()
        assert body["total"] == 1
        assert body["frames"][0]["segment"] == 3
        one = client.get(f"/v1/runs/{rid}/frames/0", params={"arm_id": "main"})
        assert one.status_code == 200
        assert one.json()["pts"] == 8.0
        image = client.get(f"/v1/runs/{rid}/frames/0/image", params={"arm_id": "main"})
        assert image.status_code == 200
        assert image.content.startswith(b"\xff\xd8")
        base = client.get(f"/v1/runs/{rid}/frames/0/base", params={"arm_id": "main"})
        assert base.status_code == 200
        assert base.content.startswith(b"\xff\xd8")
        assert one.json()["has_baseline"] is True
        missing = client.get(f"/v1/runs/{rid}/frames/4/image", params={"arm_id": "main"})
        assert missing.status_code == 404
        missing_base = client.get(f"/v1/runs/{rid}/frames/4/base", params={"arm_id": "main"})
        assert missing_base.status_code == 404
        idle = client.get("/v1/system/picture")
        assert idle.json()["active"] is False
        shown = client.put("/v1/system/picture", json={"run_id": rid, "arm_id": "main", "frame": 0})
        assert shown.status_code == 200, shown.text
        assert shown.json()["active"] is True
        assert shown.json()["frame"] == 0
        cleared = client.delete("/v1/system/picture")
        assert cleared.json()["active"] is False


def test_shared_frames_are_overwritten_unless_saved(tmp_path: Path):
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    app = create_app(config)
    dash = tmp_path / "dash"
    dash.mkdir()
    (dash / "clip.mpd").write_text(
        build_static_mpd(title="fake", duration_s=8.0, segment_duration=4.0, ladder=DEFAULT_LADDER[:1]),
        encoding="utf-8",
    )
    (dash / "stream0").mkdir()
    (dash / "stream0" / "segment_.mp4").write_bytes(b"init")
    spec = {
        "apiVersion": "prism.lab/v1",
        "kind": "ExperimentSpec",
        "metadata": {"name": "frames"},
        "spec": {
            "content": {"source": f"path:{dash}"},
            "arms": [{"id": "main", "player": {"mod_renderer": "headless", "mod_status": "off"}}],
        },
    }
    with TestClient(app) as client:
        state = client.app.state.platform
        first = client.post("/v1/runs", json={"spec": spec, "attach_players": False}).json()["id"]
        claim_current_frames(state, first, "main")
        writer = FrameArchive(str(current_frames_dir(state)))
        writer.submit(np.full((8, 8, 3), 30, np.uint8), {"segment": 1, "pts": 1.0, "enhanced": False})
        writer.close()
        kept = client.post(f"/v1/runs/{first}/frames/save", params={"arm_id": "main"})
        assert kept.status_code == 200, kept.text
        second = client.post("/v1/runs", json={"spec": spec, "attach_players": False}).json()["id"]
        claim_current_frames(state, second, "main")
        old = client.get(f"/v1/runs/{first}/frames", params={"arm_id": "main"})
        assert old.json()["storage"] == "saved"
        assert old.json()["total"] == 1
        fresh = client.get(f"/v1/runs/{second}/frames", params={"arm_id": "main"})
        assert fresh.json()["storage"] == "current"
        assert fresh.json()["total"] == 0
