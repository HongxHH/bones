from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from prism.control.app import create_app
from prism.control.attach import player_cmd
from prism.control.config import ServerConfig
from prism.control.spec_options import list_trace_datasets, recipe_options_payload


def test_recipe_options_payload_lists_modules(tmp_path: Path):
    repo = Path(__file__).resolve().parents[3]
    payload = recipe_options_payload(repo)
    assert any(x["id"] == "always_greedy" for x in payload["nes"])
    assert any(x["id"] == "imdn" and x["label"] == "IMDN" for x in payload["enhancer"])
    assert all(x["id"] != "imdn_always" for x in payload["enhancer"])
    assert any(x["id"] == "opencv" for x in payload["renderer"])
    datasets = list_trace_datasets(repo)
    assert any(d["id"] == "3Glogs" for d in datasets)


def test_specs_options_and_traces_api(tmp_path: Path):
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    app = create_app(config)
    with TestClient(app) as client:
        opts = client.get("/v1/specs/options")
        assert opts.status_code == 200, opts.text
        body = opts.json()
        assert body["nes"]
        assert body["trace_datasets"]

        ds = body["trace_datasets"][0]["id"]
        traces = client.get(f"/v1/specs/traces?dataset={ds}&limit=5")
        assert traces.status_code == 200, traces.text
        page = traces.json()
        assert page["total"] >= 1
        assert page["items"]
        assert page["items"][0]["path"].startswith(f"data/{ds}/")

        filtered = client.get(f"/v1/specs/traces?dataset={ds}&q=report&limit=10")
        assert filtered.status_code == 200
        assert filtered.json()["total"] >= 1

        rnd = client.get(f"/v1/specs/traces?dataset={ds}&random=1")
        assert rnd.status_code == 200, rnd.text
        one = rnd.json()
        assert one["random"] is True
        assert len(one["items"]) == 1
        assert one["items"][0]["path"].startswith(f"data/{ds}/")

        bad = client.get("/v1/specs/traces?dataset=nope")
        assert bad.status_code == 400


def test_player_cmd_forwards_fast_complete_and_metric_output():
    cmd = player_cmd(
        "http://x/m.mpd",
        {"use_fast_complete": False, "metric_output": "E:/runs/analyzer"},
        8765,
    )
    assert "--use_fast_complete" in cmd
    assert cmd[cmd.index("--use_fast_complete") + 1] == "false"
    assert cmd[cmd.index("--metric_output") + 1] == "E:/runs/analyzer"


def test_player_cmd_forwards_recovery_knobs():
    cmd = player_cmd(
        "http://x/m.mpd",
        {
            "use_fast_complete": True,
            "fixed_enhance_action": 3,
            "recovery_mode": "fallback",
            "fallback_enhance_action": 1,
        },
        8765,
    )
    assert cmd[cmd.index("--fixed_enhance_action") + 1] == "3"
    assert cmd[cmd.index("--recovery_mode") + 1] == "fallback"
    assert cmd[cmd.index("--fallback_enhance_action") + 1] == "1"


def test_player_cmd_passes_enhancer_and_trace():
    cmd = player_cmd(
        "http://x/m.mpd",
        {
            "mod_nes": "bones",
            "mod_enhancer": "pacnet",
            "mod_downloader": "tcp",
            "trace": "data/3Glogs/report.2010-09-13_1003CEST.json",
            "mod_renderer": "headless",
            "mod_status": "off",
        },
        8765,
    )
    assert "--mod_nes" in cmd and "bones" in cmd
    assert "--mod_enhancer" in cmd and "pacnet" in cmd
    assert "--mod_downloader" in cmd and "trace" in cmd
    assert "--trace" in cmd
    assert "data/3Glogs/report.2010-09-13_1003CEST.json" in cmd


def test_player_cmd_no_trace_keeps_tcp():
    cmd = player_cmd(
        "http://x/m.mpd",
        {"mod_downloader": "tcp", "trace": "", "mod_enhancer": "imdn_always"},
        8765,
    )
    assert cmd[cmd.index("--mod_downloader") + 1] == "tcp"
    assert "--trace" not in cmd
    assert cmd[cmd.index("--mod_enhancer") + 1] == "imdn"
