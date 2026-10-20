from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from prism.control.app import create_app
from prism.control.config import ServerConfig


def test_lab_and_system_endpoints(tmp_path: Path):
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    app = create_app(config)

    with TestClient(app) as client:
        root = client.get("/", follow_redirects=False)
        assert root.status_code in (302, 307)
        assert "/lab" in root.headers.get("location", "")

        lab = client.get("/lab/")
        assert lab.status_code == 200
        assert "Prism" in lab.text
        assert "导入 BBB" in lab.text
        assert "附着播放器" not in lab.text
        assert "附着" not in lab.text
        assert "分类" not in lab.text
        assert "时间轴" in lab.text
        assert "保留" in lab.text
        assert "下载缓冲" in lab.text
        assert "已增强就绪" in lab.text
        assert "待增强" in lab.text
        assert 'id="btnReadyInv"' in lab.text
        assert 'id="invReady"' in lab.text
        assert 'id="btnPendingInv"' in lab.text
        assert 'id="invPending"' in lab.text
        assert 'id="valCpu"' in lab.text
        assert 'id="valGpu"' in lab.text
        assert "/lab/static/metrics.js" in lab.text
        assert "PrismLabMetrics" in lab.text
        assert 'id="valRss"' not in lab.text
        assert 'meter-name">CPU<' in lab.text
        assert "CPU / 内存" not in lab.text
        assert "系统</summary>" not in lab.text
        # Experiments pane IA: control → recipe → monitor(pipeline) → history
        assert 'id="recipePanel"' in lab.text
        assert 'id="recipeSummary"' in lab.text
        assert 'id="pipeline"' in lab.text
        assert 'id="machinePanel"' not in lab.text
        assert 'id="historyPanel"' in lab.text
        assert 'id="runStatus"' in lab.text
        assert 'class="pipeline"' in lab.text
        assert "body.in-shell .meters" not in lab.text
        assert "已经下载、还没播到的时长" not in lab.text
        assert "播放中卡顿" not in lab.text
        assert "runGlance" not in lab.text
        assert "左右对比" not in lab.text
        assert "对比" not in lab.text

        health = client.get("/v1/health")
        assert health.status_code == 200
        assert health.json()["product"] == "Prism"

        sysinfo = client.get("/v1/system")
        assert sysinfo.status_code == 200
        body = sysinfo.json()
        assert body["product"] == "Prism"
        assert "bbb_dash" in body["presets"]
        assert "data_dir" in body
