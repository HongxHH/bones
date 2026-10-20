from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from prism.control.app import create_app
from prism.control.config import ServerConfig
from prism.control.plugins import build_registry
from prism.control.plugins.mpdutil import build_static_mpd
from prism.control.schemas import DEFAULT_LADDER, JobKind, RecipeOutput


def _wait_job(client: TestClient, job_id: str, timeout: float = 8.0) -> dict:
    deadline = time.time() + timeout
    last = {}
    while time.time() < deadline:
        res = client.get(f"/v1/jobs/{job_id}")
        assert res.status_code == 200
        last = res.json()
        if last["status"] in ("succeeded", "failed", "skipped"):
            return last
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish: {last}")


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


def test_import_session_dynamic_and_experiment(tmp_path: Path):
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    app = create_app(config)
    dash = _fake_dash(tmp_path)

    with TestClient(app) as client:
        health = client.get("/v1/health")
        assert health.status_code == 200
        catalog = health.json()["plugins"]
        assert "import_dash" in catalog["recipes"]
        assert "session_filter" in catalog["delivery"]
        assert "vmaf" in catalog["measurers"]

        boot = client.post(
            "/v1/bootstrap/import-dash",
            json={"name": "clip", "path": str(dash), "experiment_name": "exp-clip"},
        )
        assert boot.status_code == 200
        job = _wait_job(client, boot.json()["job"]["id"])
        assert job["status"] == "succeeded"
        pub_id = job["publication_id"]
        assert pub_id

        pubs = client.get("/v1/publications").json()
        assert len(pubs) == 1
        play = client.get(f"/v1/publications/{pub_id}/play").json()
        assert play["mpd_url"].endswith("/clip.mpd") or play["mpd_url"].endswith("clip.mpd")

        mpd = client.get(f"/origin/publications/{pub_id}/clip.mpd")
        assert mpd.status_code == 200
        assert "Representation" in mpd.text

        session = client.post("/v1/sessions", json={"publication_id": pub_id, "policy": {"allowed_representation_ids": [0, 1]}})
        assert session.status_code == 200
        sid = session.json()["id"]
        smpd = client.get(f"/origin/sessions/{sid}/manifest.mpd")
        assert smpd.status_code == 200
        assert 'id="2"' not in smpd.text
        assert 'id="0"' in smpd.text

        media = client.get(f"/origin/sessions/{sid}/stream0/segment_1.m4s")
        assert media.status_code == 200
        assert media.content == b"seg"

        patched = client.patch(
            f"/v1/sessions/{sid}/policy",
            json={"policy": {"allowed_representation_ids": [0], "mpd_type": "dynamic"}},
        )
        assert patched.status_code == 200
        _wait_job(client, client.get("/v1/jobs").json()[-1]["id"])
        dyn = client.get(f"/origin/sessions/{sid}/manifest.mpd")
        assert 'type="dynamic"' in dyn.text
        assert 'id="1"' not in dyn.text

        workers = client.post("/v1/workers/register", json={"labels": ["cpu", "gpu"], "capacity": 2})
        assert workers.status_code == 200
        wid = workers.json()["id"]
        hb = client.post(f"/v1/workers/{wid}/heartbeat")
        assert hb.status_code == 200

        measure = client.post(
            "/v1/jobs",
            json={"plugin": "noop", "kind": "content", "publication_id": pub_id, "steps": ["measure"]},
        )
        mj = _wait_job(client, measure.json()["id"])
        assert mj["status"] in ("succeeded", "skipped")

        logs = client.get(f"/v1/origin/logs?session_id={sid}")
        assert logs.status_code == 200
        assert len(logs.json()) >= 1

        exp_id = boot.json()["experiment_id"]
        exp = client.get(f"/v1/experiments/{exp_id}")
        assert exp.status_code == 200
        assert exp.json()["name"] == "exp-clip"


def test_pipeline_comes_from_plugin_not_orchestrator():
    registry = build_registry(ServerConfig())
    steps, labels = registry.pipeline_for("import_dash", JobKind.CONTENT)
    assert steps == ["import", "package"]
    assert labels == ["cpu"]
    assert registry.packager_name_for("import_dash", {}) == "identity"

    steps, labels = registry.pipeline_for("vmaf", JobKind.CONTENT)
    assert steps == ["measure"]
    assert "measure" in labels

    steps, labels = registry.pipeline_for("session_filter", JobKind.SESSION)
    assert steps == ["apply_policy"]

    class DummyVbr:
        name = "dummy_vbr"
        labels = ["cpu", "gpu"]
        steps = ["transcode", "package"]
        default_packager = "identity"

        async def run(self, *, asset_path: str, staging_dir: str, params: dict) -> RecipeOutput:
            return RecipeOutput(kind="dash_tree", dash_dir=staging_dir, mpd_name="x.mpd")

    registry.register_recipe(DummyVbr())
    steps, labels = registry.pipeline_for("dummy_vbr", JobKind.CONTENT)
    assert steps == ["transcode", "package"]
    assert labels == ["cpu", "gpu"]
    assert registry.packager_name_for("dummy_vbr", {}) == "identity"
    assert registry.labels_for_step("dummy_vbr", "package", {}) == ["cpu"]
