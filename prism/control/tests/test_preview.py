from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from prism.control.app import create_app
from prism.control.config import ServerConfig
from prism.control.plugins.mpdutil import build_static_mpd
from prism.control.schemas import DEFAULT_LADDER

JPEG = b"\xff\xd8" + b"\x00" * 40 + b"\xff\xd9"


def _fake_dash(root: Path) -> Path:
    tree = root / "dash"
    tree.mkdir(parents=True)
    xml = build_static_mpd(
        title="fake",
        duration_s=8.0,
        segment_duration=4.0,
        ladder=DEFAULT_LADDER[:1],
    )
    (tree / "clip.mpd").write_text(xml, encoding="utf-8")
    stream = tree / "stream0"
    stream.mkdir()
    (stream / "segment_.mp4").write_bytes(b"init")
    return tree


def test_preview_roundtrip(tmp_path: Path):
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    app = create_app(config)
    dash = _fake_dash(tmp_path)
    spec = {
        "apiVersion": "prism.lab/v1",
        "kind": "ExperimentSpec",
        "metadata": {"name": "prev"},
        "spec": {
            "content": {"source": f"path:{dash}"},
            "arms": [{"id": "a", "player": {"mod_renderer": "headless", "mod_status": "off"}}],
        },
    }
    with TestClient(app) as client:
        run = client.post("/v1/runs", json={"spec": spec, "attach_players": False}).json()
        rid = run["id"]
        missing = client.get(f"/v1/runs/{rid}/preview.jpg?arm_id=a")
        assert missing.status_code == 404
        posted = client.post(
            "/v1/preview",
            content=JPEG,
            headers={"Content-Type": "image/jpeg", "X-Prism-Run": rid, "X-Prism-Arm": "a"},
        )
        assert posted.status_code == 200, posted.text
        got = client.get(f"/v1/runs/{rid}/preview.jpg?arm_id=a")
        assert got.status_code == 200
        assert got.content == JPEG
        assert got.headers["content-type"].startswith("image/jpeg")
