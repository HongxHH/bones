from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prism.control.app import create_app
from prism.control.config import ServerConfig
from prism.control.plugins.hooks import ArchiveExperimentHook, fingerprint_for_job
from prism.control.plugins.measurers import VmafMeasurer
from prism.control.plugins.mpdutil import build_static_mpd
from prism.control.schemas import DEFAULT_LADDER, Asset, WorkerTask
from prism.control.store.local import LocalAssetStore
from prism.control.workers.queue import InProcessQueue, RedisQueue, build_queue


def _wait_job(client: TestClient, job_id: str, timeout: float = 8.0) -> dict:
    import time

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


def _boot(client: TestClient, dash: Path) -> str:
    boot = client.post(
        "/v1/bootstrap/import-dash",
        json={"name": "clip", "path": str(dash)},
    )
    assert boot.status_code == 200
    job = _wait_job(client, boot.json()["job"]["id"])
    assert job["status"] == "succeeded"
    return job["publication_id"]


def test_dynamic_mpd_policy_version_and_cache(tmp_path: Path):
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    app = create_app(config)
    dash = _fake_dash(tmp_path)

    with TestClient(app) as client:
        pub_id = _boot(client, dash)
        session = client.post(
            "/v1/sessions",
            json={"publication_id": pub_id, "policy": {"allowed_representation_ids": [0, 1]}},
        )
        sid = session.json()["id"]
        first = client.get(f"/origin/sessions/{sid}/manifest.mpd")
        assert first.status_code == 200
        assert first.headers.get("Cache-Control", "").startswith("no-store")
        v0 = int(first.headers.get("X-Bones-Policy-Version", "0"))

        patched = client.patch(
            f"/v1/sessions/{sid}/policy",
            json={"policy": {"allowed_representation_ids": [0], "mpd_type": "dynamic"}},
        )
        assert patched.status_code == 200
        assert int(patched.json()["extra"].get("policy_version", 0)) == v0 + 1

        jobs = client.get("/v1/jobs").json()
        _wait_job(client, jobs[-1]["id"])
        second = client.get(f"/origin/sessions/{sid}/manifest.mpd")
        assert 'type="dynamic"' in second.text
        assert 'id="1"' not in second.text
        assert int(second.headers["X-Bones-Policy-Version"]) == v0 + 1


def test_dual_client_origin_concurrency(tmp_path: Path):
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    app = create_app(config)
    dash = _fake_dash(tmp_path)

    with TestClient(app) as client:
        pub_id = _boot(client, dash)
        s1 = client.post("/v1/sessions", json={"publication_id": pub_id}).json()["id"]
        s2 = client.post("/v1/sessions", json={"publication_id": pub_id}).json()["id"]

        def pull(sid: str) -> tuple[int, bytes]:
            mpd = client.get(f"/origin/sessions/{sid}/manifest.mpd")
            seg = client.get(f"/origin/sessions/{sid}/stream0/segment_1.m4s")
            return mpd.status_code, seg.content

        with ThreadPoolExecutor(max_workers=4) as pool:
            futs = [pool.submit(pull, s1) for _ in range(4)] + [pool.submit(pull, s2) for _ in range(4)]
            results = [f.result() for f in futs]

        assert all(code == 200 and body == b"seg" for code, body in results)
        logs = client.get("/v1/origin/logs").json()
        assert len(logs) >= 8


def test_persist_incremental_entity_files(tmp_path: Path):
    config = ServerConfig(data_dir=tmp_path / "var")
    store = LocalAssetStore(config)
    a1 = Asset(id="a1", name="one", source_path=str(tmp_path / "one.mp4"))
    a2 = Asset(id="a2", name="two", source_path=str(tmp_path / "two.mp4"))
    store.save_entity("assets", a1)
    store.save_entity("assets", a2)
    p1 = store.entity_path("assets", "a1")
    p2 = store.entity_path("assets", "a2")
    assert p1.exists() and p2.exists()
    assert p1 != store.meta_path("assets")

    loaded = store.load_collection("assets", Asset)
    assert {x.id for x in loaded} == {"a1", "a2"}

    a1.name = "one-updated"
    store.save_entity("assets", a1)
    again = store.load_collection("assets", Asset)
    by_id = {x.id: x for x in again}
    assert by_id["a1"].name == "one-updated"
    assert by_id["a2"].name == "two"


def test_inprocess_and_build_queue():
    async def _run():
        q = InProcessQueue()
        t = WorkerTask(id="t1", job_id="j1", step="package", plugin="identity", required_labels=["cpu"])
        await q.enqueue(t)
        assert q.pending_count() == 1
        claimed = await q.claim(["gpu"])
        assert claimed is None
        claimed = await q.claim(["cpu", "gpu"])
        assert claimed is not None and claimed.id == "t1"
        await q.ack("t1", {"ok": True})

        assert isinstance(build_queue(None), InProcessQueue)
        rq = build_queue("redis://localhost:6379/15")
        assert isinstance(rq, RedisQueue)

    asyncio.run(_run())


def test_redis_queue_with_fake():
    pytest.importorskip("redis")

    class FakeRedis:
        def __init__(self):
            self.lists: dict[str, list[str]] = {}
            self.hashes: dict[str, dict[str, str]] = {}

        async def rpush(self, key, value):
            self.lists.setdefault(key, []).append(value)

        async def lpush(self, key, *values):
            items = self.lists.setdefault(key, [])
            for v in reversed(values):
                items.insert(0, v)

        async def llen(self, key):
            return len(self.lists.get(key, []))

        async def lpop(self, key):
            items = self.lists.get(key) or []
            if not items:
                return None
            return items.pop(0)

        async def hset(self, key, field, value):
            self.hashes.setdefault(key, {})[field] = value

        async def hdel(self, key, field):
            self.hashes.get(key, {}).pop(field, None)

    async def _run():
        q = RedisQueue("redis://fake")
        fake = FakeRedis()
        q._redis = fake  # type: ignore[attr-defined]
        task = WorkerTask(id="rt1", job_id="j", step="measure", plugin="vmaf", required_labels=["measure"])
        await q.enqueue(task)
        assert await fake.llen("prism:tasks:pending") == 1
        assert await q.claim(["cpu"]) is None
        got = await q.claim(["measure", "cpu"])
        assert got is not None and got.id == "rt1"
        await q.ack("rt1", {})
        assert "rt1" not in fake.hashes.get("prism:tasks:inflight", {})

    asyncio.run(_run())


def test_vmaf_measurer_skips_without_inputs(tmp_path: Path):
    async def _run():
        m = VmafMeasurer()
        out = await m.measure(publication_dir=str(tmp_path), params={})
        assert out["status"] == "skipped"
        assert "reference" in out["reason"]

    asyncio.run(_run())


def test_experiment_fingerprint_and_archive(tmp_path: Path):
    job = {
        "id": "job_x",
        "plugin": "import_dash",
        "kind": "content",
        "steps": ["import", "package"],
        "params": {"name": "clip"},
        "asset_id": "a1",
        "publication_id": "p1",
        "status": "succeeded",
    }
    arts = {"publication_id": "p1", "status": "ok", "vmaf_mean": 90.5}
    fp = fingerprint_for_job(job, arts)
    assert len(fp["sha256"]) == 64
    assert fp["plugin"] == "import_dash"

    async def _run():
        hook = ArchiveExperimentHook()
        exp_dir = tmp_path / "exp"
        await hook.on_job_finished(job=job, artifacts=arts, experiment_dir=str(exp_dir))
        assert (exp_dir / "job_job_x.json").exists()
        manifest = json.loads((exp_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["jobs"][0]["fingerprint"]["sha256"] == fp["sha256"]
        metrics = json.loads((exp_dir / "metrics" / "measure_job_x.json").read_text(encoding="utf-8"))
        assert metrics["artifacts"]["vmaf_mean"] == 90.5

    asyncio.run(_run())
