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
        assert "素材" not in lab.text
        assert "导入 BBB" not in lab.text
        assert 'data-panel="experiments"' in lab.text
        assert 'data-panel="jobs"' not in lab.text
        assert 'data-panel="system"' not in lab.text
        assert 'id="btnMore"' not in lab.text
        assert 'id="panel-jobs"' not in lab.text
        assert 'id="panel-system"' not in lab.text
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
        assert 'id="expList"' not in lab.text
        assert 'id="openedReport"' not in lab.text
        assert "播放器还没有写出分析报告" not in lab.text
        assert (
            "<th>时间</th><th>实验</th><th>状态</th>"
            "<th>重缓冲</th><th>带宽</th><th>QoE</th>"
        ) in lab.text
        assert 'id="runStatus"' in lab.text
        assert 'id="runChip"' not in lab.text
        assert "syncRunChip" in lab.text
        assert 'id="runNextHint"' not in lab.text
        assert "下方配方仅影响下次" not in lab.text
        assert 'id="telemWarn"' in lab.text
        assert 'id="recipeTitle"' in lab.text
        assert "播放中 · 停跑后计算 QoE / VMAF" in lab.text
        assert "setSysMetersVisible" in lab.text
        assert 'id="meterCpu"' in lab.text
        assert 'id="meterGpu"' in lab.text
        assert 'id="meterCpu" hidden' not in lab.text
        assert 'id="meterGpu" hidden' not in lab.text
        assert "system_metrics: true" in lab.text
        assert "recipeSummaryFromSpec" in lab.text
        assert 'id="monitorContext"' in lab.text
        assert "载入观测" in lab.text
        assert "载入下次配方" in lab.text
        assert "按此重跑" in lab.text
        assert "loadObservation" in lab.text
        assert "loadNextRecipe" in lab.text
        assert "rerunFromHistory" in lab.text
        assert "startCurrentRecipe" in lab.text
        assert 'id="btnKeepFrames"' in lab.text
        assert ">保留<" in lab.text
        assert 'id="timeline"' in lab.text
        assert 'id="chartDown"' in lab.text
        assert 'class="meter is-dup"' in lab.text
        assert "body.in-shell nav" in lab.text
        assert "body.in-shell #timeline" in lab.text
        assert 'body.in-shell[data-state="idle"] #pipeline .meter:not(.is-sys)' in lab.text
        assert 'body.in-shell[data-state="idle"] #monitor' not in lab.text
        assert 'body.in-shell[data-state="running"] #pipeline .meter.is-dup' not in lab.text
        assert "loadFullSparks" in lab.text
        assert "shellChrome" in lab.text
        assert "dataset.review" in lab.text
        assert "已保留到" in lab.text
        assert "详细分析" in lab.text
        assert "report-fold" in lab.text
        assert 'id="segmentDialog"' in lab.text
        assert "查看分段" in lab.text
        assert "只看 fast-complete" in lab.text
        assert 'score-group-label">播放质量<' not in lab.text
        assert 'class="pipeline"' in lab.text
        assert "body.in-shell .meters" not in lab.text
        assert "已经下载、还没播到的时长" not in lab.text
        assert "播放中卡顿" not in lab.text
        assert "runGlance" not in lab.text
        assert "左右对比" not in lab.text
        assert 'id="histCompare"' in lab.text
        assert 'id="comparePanel"' in lab.text
        assert "openCompare" in lab.text
        assert "renderCompareTable" in lab.text
        assert ">对比<" in lab.text

        health = client.get("/v1/health")
        assert health.status_code == 200
        assert health.json()["product"] == "Prism"

        sysinfo = client.get("/v1/system")
        assert sysinfo.status_code == 200
        body = sysinfo.json()
        assert body["product"] == "Prism"
        assert "bbb_dash" in body["presets"]
        assert "data_dir" in body
