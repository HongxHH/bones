"""Quality catalog, QoE recipe, resolver cascade, and bundle export."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from prism.control.app import create_app
from prism.control.config import ServerConfig
from prism.control.experiment_spec import parse_experiment_spec
from prism.control.metrics.bundle import _score_observation, annotate_frame_quality, assemble_run_bundle, ingest_measure_job
from prism.control.metrics.catalog import QualityAsset, QualityCatalog, QualityKey
from prism.control.metrics.compute import apply_vmaf_scores
from prism.control.metrics.observation import enrich_frame_row, merge_analyzer_outcomes, segments_from_playback
from prism.control.metrics.recipes import bones_qoe_v1
from prism.control.metrics.bundle import fold_bundle
from prism.control.plugins.mpdutil import build_static_mpd
from prism.control.schemas import DEFAULT_LADDER
from prism.paths import repo_root


def _catalog(tmp_path: Path) -> QualityCatalog:
    return QualityCatalog(repo_root(), tmp_path / "quality_catalog")


def _arm(model: str = "imdn_always"):
    return SimpleNamespace(id="a", player={"mod_enhancer": model})


def _run(spec: dict):
    return SimpleNamespace(id="run1", experiment_id="exp", arms=[_arm()], spec=spec)


def _state(tmp_path: Path):
    return SimpleNamespace(config=SimpleNamespace(repo_root=repo_root(), data_dir=tmp_path))


def _spec(**quality) -> dict:
    body = {
        "apiVersion": "prism.lab/v1",
        "kind": "ExperimentSpec",
        "metadata": {"name": "qoe"},
        "spec": {
            "content": {"source": "path:player/data/video_mp4box"},
            "arms": [{"id": "a", "player": {"mod_enhancer": "imdn_always"}}],
            "metrics": {"recipes": ["bones_qoe_v1"], "quality": {"catalog_ref": "content:bbb", **quality}},
        },
    }
    return parse_experiment_spec(body).model_dump(mode="json")


def test_bones_qoe_matches_analyzer_formula():
    # avg 85, oscillation 10, two segments, 10s stall → freeze 5 → QoE 74.5
    result = bones_qoe_v1([80.0, 90.0], 10.0, 2)
    assert result["status"] == "ok"
    assert result["inputs"]["avg_vmaf"] == 85
    assert result["inputs"]["oscillation_vmaf"] == 10
    assert result["inputs"]["avg_freeze"] == 5
    assert result["value"] == 74.5


def test_qoe_unavailable_without_scores():
    result = bones_qoe_v1([None, None], 1.0, 2)
    assert result["status"] == "unavailable"
    assert result["value"] is None


def test_static_table_and_explicit_invalid_cell(tmp_path: Path):
    catalog = _catalog(tmp_path)
    hit = catalog.lookup(content_id="bbb", repr_index=0, model="imdn_always", level=0, segment_index=3)
    assert hit is not None
    assert hit.source == "static_table"
    assert hit.value == json.loads((repo_root() / "player" / "imdn_bbb_quality.json").read_text(encoding="utf-8"))[0][0]
    # Level 4 is -1 in the BBB table: do not fall through to the segment pack.
    assert catalog.lookup(content_id="bbb", repr_index=0, model="imdn", level=4, segment_index=0) is None


def test_segment_pack_cell(tmp_path: Path):
    catalog = _catalog(tmp_path)
    hit = catalog.lookup(
        content_id="bbb",
        repr_index=0,
        model="imdn",
        level=0,
        segment_index=0,
        sources=("segment_pack",),
    )
    assert hit is not None
    assert hit.value == 51.478979
    assert hit.provenance["purpose"] == "segment_measurement"


def test_frame_lookup_is_degraded_view_of_segment_table(tmp_path: Path):
    catalog = _catalog(tmp_path)
    hit = catalog.lookup(
        content_id="bbb",
        repr_index=0,
        model="imdn",
        level=1,
        segment_index=0,
        frame_in_segment=4,
        granularity="frame",
    )
    assert hit is not None and hit.degraded is True
    assert hit.value == json.loads((repo_root() / "player" / "imdn_bbb_quality.json").read_text(encoding="utf-8"))[0][1]


def test_frame_series_aggregates_to_segment(tmp_path: Path):
    catalog = _catalog(tmp_path)
    catalog.write_asset(
        QualityAsset(
            key=QualityKey(content_id="bbb", repr_index=1, model="imdn", level=2),
            by_frame={"0": [10.0, 30.0]},
            provenance={"source": "libvmaf"},
        )
    )
    segment = catalog.lookup(content_id="bbb", repr_index=1, model="imdn", level=2, segment_index=0)
    frame = catalog.lookup(
        content_id="bbb",
        repr_index=1,
        model="imdn",
        level=2,
        segment_index=0,
        frame_in_segment=1,
        granularity="frame",
    )
    assert segment is not None and segment.value == 20
    assert segment.source == "aggregated"
    assert frame is not None and frame.value == 30 and frame.degraded is False


def test_partial_enhance_mixes_base_and_enhanced(tmp_path: Path):
    catalog = _catalog(tmp_path)
    table = json.loads((repo_root() / "player" / "imdn_bbb_quality.json").read_text(encoding="utf-8"))
    obs = {
        "index": 0,
        "repr_index": 0,
        "model": "imdn",
        "level": 2,
        "partial": "fast_complete",
        "ratio": 0.25,
        "scale": "",
        "frames": [],
    }
    scored = _score_observation(catalog, "bbb", obs)
    assert scored["source"] == "mixed"
    assert scored["degraded"] is True
    assert scored["vmaf"] == table[0][0] * 0.75 + table[0][2] * 0.25


def test_observation_timeline_and_frame_metadata():
    samples = [
        {"segment_index": 0, "download_action": {"0": 1}, "enhance_action": {"0": 0}, "stall_s_total": 0},
        {"segment_index": 1, "download_action": {"1": 2}, "enhance_action": {"1": 3}, "stall_s_total": 1.5},
    ]
    frames = [
        {"segment": 1, "frame_in_segment": 0, "representation": 2, "enhance_level": 3, "enhanced": True, "width": 1280, "src_width": 640},
        {"segment": 1, "frame_in_segment": 1, "representation": 2, "enhance_level": 3, "enhanced": False, "width": 1280, "src_width": 640},
    ]
    obs = segments_from_playback(samples, frames, model_default="imdn_always")
    assert [row["index"] for row in obs] == [0, 1]
    assert obs[1]["partial"] == "fast_complete"
    assert obs[1]["ratio"] == 0.5
    assert obs[1]["scale"] == "x2"
    assert enrich_frame_row({"width": 1920, "src_width": 640, "model": ""}, model_default="pacnet")["scale"] == "x3"
    assert enrich_frame_row({"width": 1920, "src_width": 640, "model": ""}, model_default="pacnet")["model"] == "pacnet"


def test_abort_uses_playing_outcome_not_planned_enhance():
    samples = [
        {
            "segment_index": 48,
            "download_action": {"48": 1},
            "enhance_action": {"48": 2},
            "playing_enhanced": False,
            "playing_enhance_level": 0,
        }
    ]
    # Stale frame archive must not revive enhance credit after a live abort.
    frames = [
        {"segment": 48, "frame_in_segment": 0, "representation": 1, "enhance_level": 2, "enhanced": True},
        {"segment": 48, "frame_in_segment": 1, "representation": 1, "enhance_level": 2, "enhanced": True},
    ]
    obs = segments_from_playback(samples, frames, model_default="imdn")
    assert len(obs) == 1
    assert obs[0]["level"] == 0
    assert obs[0]["is_enhance"] is False
    assert obs[0]["planned_level"] == 2
    assert obs[0]["outcome_source"] == "telemetry_playing"


def test_action_value_does_not_borrow_other_segment():
    samples = [
        {
            "segment_index": 5,
            "download_action": {"4": 2},
            "enhance_action": {"4": 2},
            "playing_enhanced": True,
            "playing_enhance_level": 1,
        }
    ]
    obs = segments_from_playback(samples, model_default="imdn")
    assert obs[0]["repr_index"] == 0  # missing key → keep blank default
    assert obs[0]["level"] == 1  # from playing_enhance_level
    assert obs[0]["planned_level"] is None


def test_analyzer_abort_real_vmaf_wins_over_planned_enhance(tmp_path: Path):
    table = json.loads((repo_root() / "player" / "imdn_bbb_quality.json").read_text(encoding="utf-8"))
    samples = [
        {
            "segment_index": 2,
            "download_action": {"2": 0},
            "enhance_action": {"2": 2},
            "playing_enhanced": False,
            "playing_enhance_level": 0,
            "stall_s_total": 1.0,
        }
    ]
    analyzer_segments = [
        {
            "index": 2,
            "download_action": 0,
            "enhance_action": 2,
            "is_enhance": False,
            "abort_reason": "Already played in enhance",
            "real_vmaf": table[0][0],
            "base_vmaf": table[0][0],
            "fast_complete": False,
        }
    ]
    run = _run(_spec())
    analyzer_dir = tmp_path / "exp" / "runs" / "run1" / "analyzer"
    analyzer_dir.mkdir(parents=True)
    (analyzer_dir / "report.json").write_text(
        json.dumps({"segments": analyzer_segments, "QOE": 1.0}),
        encoding="utf-8",
    )
    state = _state(tmp_path)
    from prism.control import telemetry as tel_mod

    original = tel_mod.telemetry_dir

    def _tel_dir(_state, _run):
        return analyzer_dir.parent / "telemetry"

    tel_mod.telemetry_dir = _tel_dir  # type: ignore[assignment]
    try:
        bundle = assemble_run_bundle(
            state,
            run,
            {"a": {"stall_s": 1.0}},
            samples_for=lambda _arm: samples,
        )
    finally:
        tel_mod.telemetry_dir = original

    arm = bundle["arms"]["a"]
    seg = arm["quality"]["segments"][0]
    assert seg["level"] == 0
    assert seg["source"] == "analyzer"
    assert seg["vmaf"] == round(table[0][0], 6)
    assert arm["status"] == "partial"
    assert any("质量表推算" in reason for reason in arm["reasons"])
    assert arm["observations_summary"]["outcome_source"] == "analyzer"
    # Must not credit the planned enhance cell.
    assert seg["vmaf"] != round(table[0][2], 6)

    summary = {"arms": {"a": dict(arm.get("observations_summary") or {})}}
    fold_bundle(summary, bundle)
    assert summary["arms"]["a"]["quality_source"] == "analyzer"
    assert summary["arms"]["a"]["outcome_source"] == "analyzer"
    assert summary["arms"]["a"]["avg_vmaf"] == round(table[0][0], 6)


def test_execution_stats_from_analyzer_flatten_levels():
    from prism.control.metrics.bundle import _execution_stats_from_analyzer

    stats = _execution_stats_from_analyzer(
        {
            "enhance_level_stats": {
                "0": {"count": 8, "completed": 0, "completion_rate": 0.0},
                "2": {"count": 141, "completed": 129, "completion_rate": 0.9149, "avg_latency": 1.0333},
            },
            "enhancer_abort_total": 12,
            "abort_wasted_frames_total": 363,
            "abort_reason_count": {"Already played in enhance": 6},
            "num_stall": 9,
            "num_quality_switches": 40,
            "avg_bitrate": 681993.6,
        }
    )
    assert stats["enh_l0_count"] == 8
    assert stats["enh_l2_completed"] == 129
    assert stats["enh_l2_rate"] == 0.9149
    assert stats["abort_total"] == 12
    assert stats["abort_wasted_frames"] == 363
    assert stats["abort_reasons"]["Already played in enhance"] == 6
    assert stats["num_quality_switches"] == 40


def test_merge_analyzer_fast_complete_ratio():
    obs = segments_from_playback(
        [
            {
                "segment_index": 7,
                "download_action": {"7": 1},
                "enhance_action": {"7": 2},
                "playing_enhanced": True,
                "playing_enhance_level": 2,
            }
        ],
        model_default="imdn",
    )
    merged = merge_analyzer_outcomes(
        obs,
        [
            {
                "index": 7,
                "download_action": 1,
                "enhance_action": 2,
                "is_enhance": True,
                "fast_complete": True,
                "enhanced_cnt": 30,
                "interpolated_cnt": 90,
                "real_vmaf": 55.5,
            }
        ],
        model_default="imdn",
    )
    assert merged[0]["partial"] == "fast_complete"
    assert merged[0]["ratio"] == 0.25
    assert merged[0]["level"] == 2
    assert merged[0]["real_vmaf"] == 55.5
    assert merged[0]["outcome_source"] == "analyzer"


def test_bundle_local_table_then_miss(tmp_path: Path):
    spec = _spec()
    samples = [
        {"segment_index": 0, "download_action": {"0": 0}, "enhance_action": {"0": 0}, "stall_s_total": 0},
        {"segment_index": 1, "download_action": {"1": 1}, "enhance_action": {"1": 2}, "stall_s_total": 4},
    ]
    bundle = assemble_run_bundle(
        _state(tmp_path),
        _run(spec),
        {"a": {"stall_s": 4}},
        samples_for=lambda _arm: samples,
    )
    arm = bundle["arms"]["a"]
    assert arm["status"] == "partial"
    assert any("离线质量表" in reason for reason in arm["reasons"])
    recipe = arm["derived"]["bones_qoe_v1"]
    table = json.loads((repo_root() / "player" / "imdn_bbb_quality.json").read_text(encoding="utf-8"))
    expected = bones_qoe_v1([table[0][0], table[1][2]], 4, 2)
    assert recipe["value"] == expected["value"]
    assert arm["observations_summary"]["num_quality_switches"] == 1

    missing = _spec(catalog_ref="content:no-such-video")
    missed = assemble_run_bundle(
        _state(tmp_path),
        _run(missing),
        {"a": {"stall_s": 0}},
        samples_for=lambda _arm: samples[:1],
    )
    assert missed["arms"]["a"]["status"] == "unavailable"
    assert "质量表里没有这次播放的分数" in missed["arms"]["a"]["reasons"]
    assert missed["arms"]["a"]["derived"]["bones_qoe_v1"]["value"] is None


def test_compute_writeback_is_not_repeated(tmp_path: Path, monkeypatch):
    calls = []

    def fake(catalog, ctx):
        calls.append(ctx["content_id"])
        catalog.write_asset(
            QualityAsset(
                key=QualityKey(content_id=ctx["content_id"], repr_index=0, model="imdn", level=0),
                scalar=61.0,
                provenance={"source": "libvmaf"},
            )
        )
        return {"status": "ok", "reason": "", "written": 1, "pooled_mean": 61.0}

    monkeypatch.setattr("prism.control.metrics.bundle.run_compute", fake)
    spec = _spec(catalog_ref="content:fresh-clip", prefer=["local_catalog", "compute"], reference="ref.mp4", distorted="dist.mp4")
    samples = [{"segment_index": 0, "download_action": {"0": 0}, "enhance_action": {"0": 0}}]
    state = _state(tmp_path)
    first = assemble_run_bundle(state, _run(spec), {"a": {"stall_s": 0}}, samples_for=lambda _arm: samples)
    second = assemble_run_bundle(state, _run(spec), {"a": {"stall_s": 0}}, samples_for=lambda _arm: samples)
    assert calls == ["fresh-clip"]
    assert first["arms"]["a"]["derived"]["bones_qoe_v1"]["value"] == 61
    assert second["arms"]["a"]["quality"]["segments"][0]["source"] == "libvmaf"


def test_remote_catalog_then_local_hit(tmp_path: Path):
    document = {
        "entries": [
            {"metric": "vmaf", "granularity": "segment", "repr_index": 0, "model": "imdn", "level": 0, "values": [70.0, 80.0]}
        ]
    }
    remote = tmp_path / "remote.json"
    remote.write_text(json.dumps(document), encoding="utf-8")
    spec = _spec(catalog_ref="content:remote-clip", prefer=["local_catalog", "remote_catalog"], remote_url=remote.as_uri())
    samples = [
        {"segment_index": 0, "download_action": {"0": 0}, "enhance_action": {"0": 0}},
        {"segment_index": 1, "download_action": {"1": 0}, "enhance_action": {"1": 0}},
    ]
    state = _state(tmp_path)
    bundle = assemble_run_bundle(state, _run(spec), {"a": {"stall_s": 0}}, samples_for=lambda _arm: samples)
    assert [row["vmaf"] for row in bundle["arms"]["a"]["quality"]["segments"]] == [70.0, 80.0]
    assert bundle["arms"]["a"]["quality"]["segments"][0]["source"] == "remote"
    again = assemble_run_bundle(state, _run(spec), {"a": {"stall_s": 0}}, samples_for=lambda _arm: samples)
    assert again["arms"]["a"]["quality"]["segments"][1]["vmaf"] == 80.0


def test_apply_vmaf_log_splits_frames(tmp_path: Path):
    catalog = _catalog(tmp_path)
    observations = [
        {"index": 0, "repr_index": 0, "level": 1, "partial": "none", "scale": "x2", "model": "imdn", "frames": [{}, {}]},
    ]
    written = apply_vmaf_scores(
        catalog,
        content_id="clip",
        model="imdn",
        frames=[40.0, 60.0],
        pooled_mean=50.0,
        observations=observations,
        provenance={"source": "libvmaf"},
    )
    assert written == 1
    hit = catalog.lookup(
        content_id="clip",
        repr_index=0,
        model="imdn",
        level=1,
        segment_index=0,
        frame_in_segment=1,
        granularity="frame",
        scale="x2",
    )
    assert hit is not None and hit.value == 60 and hit.source == "libvmaf"


def test_parse_snapshots_workspace_cards():
    from prism.control.experiment_spec import parse_experiment_spec

    spec = parse_experiment_spec({
        "apiVersion": "prism.lab/v1",
        "kind": "ExperimentSpec",
        "metadata": {"name": "kit"},
        "spec": {
            "workspace": "baseline",
            "content": {"source": "path:player/data/video_mp4box"},
            "arms": [{"id": "a"}],
        },
    })
    assert [card["key"] for card in spec.spec.cards] == ["qoe", "avg_vmaf", "oscillation_vmaf", "avg_freeze"]
    assert spec.spec.arms[0].player.use_fast_complete is False
    assert spec.spec.arms[0].player.latency_adapt is False


def test_fast_complete_recipe_does_not_treat_missing_as_zero():
    from prism.control.metrics.recipes import fast_complete_v1
    from prism.control.workspaces import cards_for_run

    missing = fast_complete_v1([{"stall_s_total": 1}])
    assert missing["status"] == "unavailable"
    assert missing["inputs"]["segments"] is None
    present = fast_complete_v1([
        {"facts": {"fast_complete": {"enabled": True, "segments": 0, "saved_frames": 0, "quality_drop_avg": None}}},
        {"facts": {"fast_complete": {"enabled": True, "segments": 2, "saved_frames": 40, "quality_drop_avg": 1.25}}},
    ])
    assert present["status"] == "ok"
    assert present["inputs"]["segments"] == 2
    assert present["inputs"]["saved_frames"] == 40
    ws_id, cards = cards_for_run({"spec": {"workspace": "fast_complete"}})
    assert ws_id == "fast_complete"
    assert [card["key"] for card in cards if card["key"].startswith("fc_")] == [
        "fc_segments",
        "fc_saved_frames",
        "fc_quality_drop_avg",
    ]
    plain_id, plain_cards = cards_for_run({"spec": {}})
    assert plain_id == ""
    assert [card["key"] for card in plain_cards] == ["qoe", "avg_vmaf", "oscillation_vmaf", "avg_freeze"]
    frozen_id, frozen = cards_for_run({
        "spec": {"workspace": "fast_complete", "cards": [{"key": "qoe", "label": "当时的 QoE"}]},
    })
    assert frozen_id == "fast_complete"
    assert frozen == [{"key": "qoe", "label": "当时的 QoE"}]


def test_annotate_frame_for_timeline(tmp_path: Path):
    row = annotate_frame_quality(
        _state(tmp_path),
        _run(_spec()),
        {
            "frame": 3,
            "segment": 0,
            "frame_in_segment": 1,
            "representation": 0,
            "enhance_level": 1,
            "enhanced": True,
            "width": 1920,
            "height": 1080,
            "src_width": 640,
            "model": "imdn_always",
        },
    )
    table = json.loads((repo_root() / "player" / "imdn_bbb_quality.json").read_text(encoding="utf-8"))
    assert row["scale"] == "x3"
    assert row["vmaf"] == round(table[0][1], 4)
    assert row["vmaf_degraded"] is True


def test_export_csv_includes_qoe(tmp_path: Path):
    config = ServerConfig(data_dir=tmp_path / "var", public_base_url="http://testserver")
    app = create_app(config)
    dash = tmp_path / "dash"
    dash.mkdir()
    (dash / "clip.mpd").write_text(
        build_static_mpd(title="fake", duration_s=8, segment_duration=4, ladder=DEFAULT_LADDER[:1]),
        encoding="utf-8",
    )
    spec = {
        "apiVersion": "prism.lab/v1",
        "kind": "ExperimentSpec",
        "metadata": {"name": "csv"},
        "spec": {
            "content": {"source": f"path:{dash}"},
            "arms": [{"id": "a", "player": {"mod_renderer": "headless", "mod_status": "off", "mod_enhancer": "imdn_always"}}],
            "metrics": {"quality": {"catalog_ref": "content:bbb", "prefer": ["local_catalog"]}},
        },
    }
    with TestClient(app) as client:
        run = client.post("/v1/runs", json={"spec": spec, "attach_players": False}).json()
        rid = run["id"]
        posted = client.post(
            "/v1/telemetry",
            json={
                "run_id": rid,
                "arm_id": "a",
                "sample": {
                    "ts": 10,
                    "state": "PLAYING",
                    "segment_index": 0,
                    "stall_s_total": 0,
                    "download_action": {"0": 0},
                    "enhance_action": {"0": 0},
                },
            },
        )
        assert posted.status_code == 200, posted.text
        metrics = client.get(f"/v1/runs/{rid}/metrics").json()
        # Offline quality-table estimates are labeled partial, not measured-complete.
        assert metrics["arms"]["a"]["metrics_status"] == "partial"
        assert metrics["arms"]["a"]["qoe"] is not None
        csv = client.get(f"/v1/runs/{rid}/export").text
        assert "qoe" in csv.splitlines()[0]
        assert "avg_vmaf" in csv

        log = tmp_path / "vmaf.json"
        log.write_text(json.dumps({"pooled_metrics": {"vmaf": {"mean": 61.0}}, "frames": []}), encoding="utf-8")
        job = SimpleNamespace(plugin="vmaf", id="job-vmaf", params={"run_id": rid}, artifacts={"vmaf_mean": 61.0, "log": str(log)})
        ingest_measure_job(client.app.state.platform, job)
        refreshed = client.get(f"/v1/runs/{rid}/metrics").json()
        assert refreshed["arms"]["a"]["qoe"] == 61
        assert refreshed["bundle"]["arms"]["a"]["quality"]["segments"][0]["source"] == "libvmaf"
        assert refreshed["arms"]["a"]["metrics_status"] == "complete"
