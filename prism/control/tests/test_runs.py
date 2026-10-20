from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from prism.control.app import create_app
from prism.control.config import ServerConfig
from prism.control.experiment_spec import load_experiment_spec, parse_experiment_spec
from prism.control.plugins.mpdutil import build_static_mpd
from prism.control.schemas import DEFAULT_LADDER


def _fake_dash(root: Path) -> Path:
    tree = root / "dash"
    tree.mkdir(parents=True)
    xml = build_static_mpd(
        title="fake",
        duration_s=12.0,
        segment_duration=4.0,
        ladder=DEFAULT_LADDER[:3],
    )
    (tree / "clip.mpd").write_text(xml, encoding="utf-8")
    for rid in (0, 1, 2):
        stream = tree / f"stream{rid}"
        stream.mkdir()
        (stream / "segment_.mp4").write_bytes(b"init")
        (stream / "segment_1.m4s").write_bytes(b"seg")
    return tree


def test_parse_and_load_example_spec():
    repo = Path(__file__).resolve().parents[3]
    spec = load_experiment_spec(repo / "prism" / "experiments" / "bbb_play.yaml")
    assert spec.name() == "bbb-play"
    assert len(spec.spec.arms) == 1
    assert spec.spec.arms[0].id == "play"
    assert spec.spec.execution.arm_mode == "serial"


def test_create_run_serial_sessions(tmp_path: Path):
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    app = create_app(config)
    dash = _fake_dash(tmp_path)

    spec = {
        "apiVersion": "prism.lab/v1",
        "kind": "ExperimentSpec",
        "metadata": {"name": "unit-run"},
        "spec": {
            "content": {"source": f"path:{dash}"},
            "arms": [
                {
                    "id": "a",
                    "session": {"policy": {"allowed_representation_ids": [0], "mpd_type": "dynamic"}},
                    "player": {"mod_renderer": "headless", "mod_status": "off"},
                },
            ],
            "execution": {"arm_mode": "serial"},
        },
    }
    assert parse_experiment_spec(spec).name() == "unit-run"

    with TestClient(app) as client:
        bad = client.post("/v1/specs/validate", json={"kind": "Nope"})
        assert bad.status_code == 400

        ok = client.post("/v1/specs/validate", json=spec)
        assert ok.status_code == 200

        res = client.post("/v1/runs", json={"spec": spec, "attach_players": False})
        assert res.status_code == 200, res.text
        run = res.json()
        assert run["status"] == "running"
        assert len(run["arms"]) == 1
        assert run["active_arm_id"] == "a"
        assert run["arms"][0]["mpd_url"]
        assert run["arms"][0]["session_id"]

        mpd_a = client.get(f"/origin/sessions/{run['arms'][0]['session_id']}/manifest.mpd")
        assert mpd_a.status_code == 200
        assert 'type="dynamic"' in mpd_a.text or 'id="0"' in mpd_a.text
        assert 'id="1"' not in mpd_a.text

        listed = client.get("/v1/runs").json()
        assert any(r["id"] == run["id"] for r in listed)

        stopped = client.post(f"/v1/runs/{run['id']}/stop")
        assert stopped.status_code == 200
        assert stopped.json()["status"] == "stopped"
