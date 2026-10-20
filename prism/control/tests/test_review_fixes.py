from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from prism.control.app import create_app
from prism.control.config import ServerConfig
from prism.control.plugins.mpdutil import build_static_mpd
from prism.control.schemas import DEFAULT_LADDER
from prism.control.security import is_loopback_host


def _fake_dash(root: Path) -> Path:
    tree = root / "dash"
    tree.mkdir(parents=True)
    xml = build_static_mpd(
        title="fake",
        duration_s=8.0,
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


def test_loopback_helpers():
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("testclient")
    assert not is_loopback_host("8.8.8.8")


def test_unsupported_publish_rejected(tmp_path: Path):
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    app = create_app(config)
    dash = _fake_dash(tmp_path)
    spec = {
        "apiVersion": "prism.lab/v1",
        "kind": "ExperimentSpec",
        "metadata": {"name": "bad-pub"},
        "spec": {
            "content": {"source": f"path:{dash}"},
            "arms": [{"id": "a", "publish": {"plugin": "no_such_recipe"}}],
        },
    }
    with TestClient(app) as client:
        res = client.post("/v1/runs", json={"spec": spec, "attach_players": False})
        assert res.status_code == 400
        assert "unknown publish.plugin" in res.text.lower() or "no_such_recipe" in res.text


def test_async_prepare_and_measure_on_stop(tmp_path: Path):
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    app = create_app(config)
    dash = _fake_dash(tmp_path)
    spec = {
        "apiVersion": "prism.lab/v1",
        "kind": "ExperimentSpec",
        "metadata": {"name": "prep"},
        "spec": {
            "content": {"source": f"path:{dash}"},
            "arms": [{"id": "a", "player": {"mod_renderer": "headless", "mod_status": "off"}}],
            "measure": [{"plugin": "noop", "params": {}}],
        },
    }
    with TestClient(app) as client:
        # wait=false returns quickly; poll until running
        res = client.post("/v1/runs", json={"spec": spec, "attach_players": False, "wait": False})
        assert res.status_code == 200
        run = res.json()
        assert run["status"] in ("preparing", "running", "failed")
        rid = run["id"]
        for _ in range(100):
            run = client.get(f"/v1/runs/{rid}").json()
            if run["status"] in ("running", "failed"):
                break
        assert run["status"] == "running"
        assert run["arms"]

        stopped = client.post(f"/v1/runs/{rid}/stop").json()
        assert stopped["status"] == "stopped"
        mids = (stopped.get("metrics") or {}).get("measure_job_ids") or []
        assert mids, "measure jobs should be enqueued on stop"
        job = client.get(f"/v1/jobs/{mids[0]}").json()
        assert job["plugin"] == "noop"


def test_specs_parse_yaml():
    config = ServerConfig(data_dir=Path("prism/var_test_parse"), public_base_url="http://testserver")
    # use tmp via create_app with tmp 鈥?simpler inline
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as td:
        app = create_app(ServerConfig(data_dir=Path(td), public_base_url="http://testserver"))
        yaml_text = """
apiVersion: prism.lab/v1
kind: ExperimentSpec
metadata:
  name: from-yaml
spec:
  content:
    source: publication:pub_x
  arms:
    - id: only
"""
        with TestClient(app) as client:
            res = client.post("/v1/specs/parse", json={"text": yaml_text, "format": "yaml"})
            assert res.status_code == 200, res.text
            assert res.json()["name"] == "from-yaml"
