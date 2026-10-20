"""Lab metrics.js unit tests + static serving."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from prism.control.app import create_app
from prism.control.config import ServerConfig

LAB_DIR = Path(__file__).resolve().parents[1] / "lab"
METRICS_JS = LAB_DIR / "metrics.js"
METRICS_TEST = LAB_DIR / "metrics.test.js"


def test_metrics_js_unit():
    node = shutil.which("node")
    if not node:
        pytest.skip("node not available")
    assert METRICS_JS.is_file()
    assert METRICS_TEST.is_file()
    proc = subprocess.run(
        [node, str(METRICS_TEST)],
        cwd=str(LAB_DIR),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    assert proc.returncode == 0, out
    assert "ok" in (proc.stdout or "")


def test_lab_serves_metrics_js(tmp_path: Path):
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    app = create_app(config)
    with TestClient(app) as client:
        lab = client.get("/lab/")
        assert lab.status_code == 200
        assert '/lab/static/metrics.js' in lab.text
        assert "PrismLabMetrics" in lab.text
        js = client.get("/lab/static/metrics.js")
        assert js.status_code == 200
        assert "formatCpuLine" in js.text
        assert "formatGpuLine" in js.text
        assert "gpu_proc_mem_src" in js.text
