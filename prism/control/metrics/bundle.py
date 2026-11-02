"""Assemble a MetricsBundle from observations, the quality resolver, and recipes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..frames import iter_frame_rows, locate_frames
from ..schemas import ExperimentRun
from .catalog import QualityCatalog, QualityHit, catalog_for, normalize_model, resolve_content_id
from .compute import apply_vmaf_scores, parse_vmaf_log, run_compute
from .observation import enrich_frame_row, observations_from_analyzer, segments_from_playback
from .recipes import RECIPES, bones_qoe_v1, fast_complete_v1
from .remote import pull_remote_catalog

_ANALYZER_TABLE_REASON = (
    "画质来自播放器质量表推算（与 analyzer 一致；abort/FC 按播出结局），不是实测 libvmaf"
)


def metrics_settings(run: ExperimentRun) -> dict[str, Any]:
    spec = (run.spec or {}).get("spec") or {}
    metrics = spec.get("metrics") or {}
    quality = metrics.get("quality") or {}
    compute = quality.get("compute") or {}
    prefer = list(quality.get("prefer") or ["local_catalog"])
    recipes = list(metrics.get("recipes") or ["bones_qoe_v1"])
    return {
        "recipes": recipes,
        "prefer": prefer,
        "catalog_ref": str(quality.get("catalog_ref") or ""),
        "remote_url": str(quality.get("remote_url") or ""),
        "model": str(quality.get("model") or ""),
        "reference": str(compute.get("reference") or ""),
        "distorted": str(compute.get("distorted") or ""),
        "writeback": bool(compute.get("writeback", True)),
        "content_source": str((spec.get("content") or {}).get("source") or ""),
    }


def assemble_run_bundle(
    state,
    run: ExperimentRun,
    arm_summaries: dict[str, dict[str, Any]],
    *,
    samples_for,
    allow_network: bool = True,
) -> dict[str, Any]:
    settings = metrics_settings(run)
    catalog = catalog_for(state.config.repo_root, state.config.data_dir)
    _install_catalog_ref(catalog, settings)
    content_id = resolve_content_id(settings["content_source"], settings["catalog_ref"])
    arms: dict[str, Any] = {}
    analyzer_report = _load_analyzer_report(state, run)
    analyzer_segments = list(analyzer_report.get("segments") or [])
    for arm in run.arms:
        summary = arm_summaries.get(arm.id) or {}
        samples = samples_for(arm.id)
        model_default = _arm_model(arm, settings)
        # Prefer the player analyzer as the sole outcome source when present.
        # That matches the paper/player QoE path and avoids stale shared frames.
        if analyzer_segments:
            observations = observations_from_analyzer(
                analyzer_segments,
                model_default=model_default,
            )
            frame_rows: list[dict[str, Any]] = []
        else:
            frame_rows = _load_frames(state, run, arm.id, model_default)
            observations = segments_from_playback(
                samples,
                frame_rows,
                model_default=model_default,
            )
        quality = _resolve_arm(
            catalog,
            settings,
            content_id,
            observations,
            frame_rows,
            allow_network=allow_network,
        )
        stall_s = _stall_seconds(summary, analyzer_report)
        derived = _derive(
            settings["recipes"],
            quality["segment_vmafs"],
            stall_s,
            len(observations),
            samples,
        )
        _align_fast_complete_recipe(derived, analyzer_report, settings["recipes"])
        status, reasons = _status(observations, quality, derived)
        dropped = int(summary.get("frame_archive_dropped") or 0)
        if dropped > 0 and not analyzer_segments:
            if status in ("complete", "partial"):
                status = "degraded"
            reasons.append(f"帧归档丢了 {dropped} 帧，画面统计不完整")
        obs_summary = {
            **summary,
            "stall_s": stall_s,
            "num_segments": len(observations),
            "num_quality_switches": _quality_switches(observations),
            "outcome_source": _outcome_source_label(observations),
            "analyzer_segments": len(analyzer_segments),
        }
        obs_summary.update(_execution_stats_from_analyzer(analyzer_report))
        arms[arm.id] = {
            "status": status,
            "reasons": reasons,
            "observations_summary": obs_summary,
            "quality": {
                "content_id": content_id,
                "segments": quality["segments"],
                "frame_count": quality["frame_count"],
                "frames_path": quality.get("frames_path") or "",
            },
            "derived": derived,
            "artifacts": {},
        }
    player = {}
    if run.arms:
        player = run.arms[0].player or {}
    spec_prov = ((run.spec or {}).get("provenance") or {}) if isinstance(run.spec, dict) else {}
    sources = [_quality_source_label(arm.get("quality") or {}) for arm in arms.values()]
    quality_source = next((item for item in sources if item != "unavailable"), "unavailable")
    outcomes = [
        str((arm.get("observations_summary") or {}).get("outcome_source") or "")
        for arm in arms.values()
    ]
    outcome_source = next((item for item in outcomes if item and item != "unavailable"), "unavailable")
    return {
        "run_id": run.id,
        "experiment_id": run.experiment_id,
        "arms": arms,
        "provenance": {
            "quality_source": quality_source,
            "outcome_source": outcome_source,
            "renderer": str(player.get("mod_renderer") or spec_prov.get("renderer") or ""),
            "latency_adapt": bool(player.get("latency_adapt", spec_prov.get("latency_adapt", True))),
            "use_fast_complete": bool(player.get("use_fast_complete", spec_prov.get("use_fast_complete", False))),
            "git_sha": str(spec_prov.get("git_sha") or ""),
        },
    }


def fold_bundle(summary: dict[str, Any], bundle: dict[str, Any]) -> None:
    """Copy recipe outputs onto arm summaries so Lab and CSV stay flat."""
    summary["bundle"] = bundle
    for arm_id, arm in (bundle.get("arms") or {}).items():
        slot = summary.setdefault("arms", {}).setdefault(arm_id, {})
        recipe = (arm.get("derived") or {}).get("bones_qoe_v1") or {}
        inputs = recipe.get("inputs") or {}
        obs_summary = arm.get("observations_summary") or {}
        slot["qoe"] = recipe.get("value")
        slot["avg_vmaf"] = inputs.get("avg_vmaf")
        slot["oscillation_vmaf"] = inputs.get("oscillation_vmaf")
        slot["avg_freeze"] = inputs.get("avg_freeze")
        slot["num_quality_switches"] = obs_summary.get("num_quality_switches")
        slot["metrics_status"] = arm.get("status")
        reasons = arm.get("reasons") or []
        slot["metrics_reason"] = "; ".join(reasons)
        slot["outcome_source"] = obs_summary.get("outcome_source")
        slot["quality_source"] = _quality_source_label(arm.get("quality") or {})
        for key in (
            "enh_l0_count",
            "enh_l0_completed",
            "enh_l0_rate",
            "enh_l0_latency",
            "enh_l1_count",
            "enh_l1_completed",
            "enh_l1_rate",
            "enh_l1_latency",
            "enh_l2_count",
            "enh_l2_completed",
            "enh_l2_rate",
            "enh_l2_latency",
            "abort_total",
            "abort_wasted_frames",
            "abort_reasons",
            "num_stall",
            "avg_bitrate",
        ):
            if key in obs_summary and obs_summary.get(key) is not None:
                slot[key] = obs_summary.get(key)
        if obs_summary.get("num_quality_switches") is not None:
            slot["num_quality_switches"] = obs_summary.get("num_quality_switches")
        fc = (arm.get("derived") or {}).get("fast_complete_v1") or {}
        fc_inputs = fc.get("inputs") or {}
        slot["fc_segments"] = fc_inputs.get("segments")
        slot["fc_saved_frames"] = fc_inputs.get("saved_frames")
        slot["fc_quality_drop_avg"] = fc_inputs.get("quality_drop_avg")
        if fc.get("status") == "unavailable":
            slot["fc_reason"] = fc.get("reason") or "unavailable"


def annotate_frame_quality(state, run: ExperimentRun, row: dict[str, Any]) -> dict[str, Any]:
    """Read-only local lookup so scrubbing does not trigger compute or network."""
    settings = metrics_settings(run)
    model_default = settings["model"] or "imdn"
    enriched = enrich_frame_row(row, model_default=model_default)
    catalog = catalog_for(state.config.repo_root, state.config.data_dir)
    content_id = resolve_content_id(settings["content_source"], settings["catalog_ref"])
    enhanced = bool(enriched.get("enhanced"))
    level = enriched.get("enhance_level")
    try:
        level_i = int(level) if enhanced and level is not None else 0
    except (TypeError, ValueError):
        level_i = 0
    repr_index = enriched.get("representation")
    try:
        repr_i = int(repr_index) if repr_index is not None else 0
    except (TypeError, ValueError):
        repr_i = 0
    segment_index = enriched.get("segment")
    frame_in_segment = enriched.get("frame_in_segment")
    try:
        seg_i = int(segment_index) if segment_index is not None else None
    except (TypeError, ValueError):
        seg_i = None
    try:
        frame_i = int(frame_in_segment) if frame_in_segment is not None else None
    except (TypeError, ValueError):
        frame_i = None
    hit = catalog.lookup(
        content_id=content_id,
        repr_index=repr_i,
        model=normalize_model(str(enriched.get("model") or model_default)),
        level=level_i,
        segment_index=seg_i,
        frame_in_segment=frame_i,
        granularity="frame",
        scale=str(enriched.get("scale") or ""),
    )
    if hit is not None and hit.value is not None:
        enriched["vmaf"] = round(hit.value, 4)
        enriched["vmaf_source"] = hit.source
        enriched["vmaf_degraded"] = hit.degraded
    return enriched


def ingest_measure_job(state, job) -> None:
    """Fold a finished VMAF job into the catalog and rebuild that run's bundle."""
    params = getattr(job, "params", None) or {}
    artifacts = getattr(job, "artifacts", None) or {}
    plugin = getattr(job, "plugin", "") or ""
    if plugin not in {"vmaf", "libvmaf"} and "vmaf_mean" not in artifacts and "log" not in artifacts:
        return
    run_id = str(params.get("run_id") or artifacts.get("run_id") or "")
    if not run_id or run_id not in state.runs:
        return
    run = state.runs[run_id]
    settings = metrics_settings(run)
    if not settings.get("writeback", True):
        return
    catalog = catalog_for(state.config.repo_root, state.config.data_dir)
    content_id = resolve_content_id(settings["content_source"], settings["catalog_ref"])
    log_path = artifacts.get("log")
    frames: list[float] = []
    mean = artifacts.get("vmaf_mean")
    if isinstance(mean, (int, float)):
        mean = float(mean)
    else:
        mean = None
    if log_path and Path(str(log_path)).is_file():
        frames, parsed_mean = parse_vmaf_log(Path(str(log_path)))
        if parsed_mean is not None:
            mean = parsed_mean
    from ..telemetry import iter_samples, telemetry_path

    observations: list[dict] = []
    model = settings["model"] or "imdn"
    if run.arms:
        arm = run.arms[0]
        model = _arm_model(arm, settings)
        analyzer_report = _load_analyzer_report(state, run)
        analyzer_segments = list(analyzer_report.get("segments") or [])
        if analyzer_segments:
            observations = observations_from_analyzer(
                analyzer_segments,
                model_default=model,
            )
            frame_rows = []
        else:
            frame_rows = _load_frames(state, run, arm.id, model)
            observations = segments_from_playback(
                iter_samples(telemetry_path(state, run, arm.id)),
                frame_rows,
                model_default=model,
            )
    written = apply_vmaf_scores(
        catalog,
        content_id=content_id,
        model=model,
        frames=frames,
        pooled_mean=mean,
        observations=observations,
        provenance={
            "source": "libvmaf",
            "tool": "ffmpeg",
            "job_id": getattr(job, "id", ""),
            "approx": False,
            "log": str(log_path or ""),
        },
    )
    if written:
        from ..telemetry import persist_run_metrics

        persist_run_metrics(state, run)


def _resolve_arm(
    catalog,
    settings,
    content_id,
    observations,
    frame_rows,
    *,
    allow_network: bool = True,
) -> dict[str, Any]:
    scored = [_score_observation(catalog, content_id, obs) for obs in observations]
    misses = [row for row in scored if row.get("vmaf") is None]
    remote_note = ""
    compute_note = ""
    pooled = None
    if not allow_network:
        if misses and ("remote_catalog" in settings["prefer"] or "compute" in settings["prefer"]):
            remote_note = "network quality resolve skipped on the live path"
        misses = []
    if misses and "remote_catalog" in settings["prefer"]:
        remote = pull_remote_catalog(catalog, settings["remote_url"], content_id)
        if remote.get("written"):
            scored = [_score_observation(catalog, content_id, obs) for obs in observations]
            misses = [row for row in scored if row.get("vmaf") is None]
        elif remote.get("reason"):
            remote_note = str(remote["reason"])
    if misses and "compute" in settings["prefer"]:
        result = run_compute(
            catalog,
            {
                "content_id": content_id,
                "model": settings["model"] or (observations[0]["model"] if observations else "imdn"),
                "reference": settings["reference"],
                "distorted": settings["distorted"],
                "observations": observations,
            },
        )
        pooled = result.get("pooled_mean")
        if result.get("written"):
            scored = [_score_observation(catalog, content_id, obs) for obs in observations]
        elif result.get("reason"):
            compute_note = str(result["reason"])
    frames_path = ""
    return {
        "segments": [
            {
                "index": row["index"],
                "repr_index": row["repr_index"],
                "model": row["model"],
                "level": row["level"],
                "partial": row["partial"],
                "vmaf": None if row["vmaf"] is None else round(float(row["vmaf"]), 6),
                "source": row["source"],
                "degraded": row["degraded"],
            }
            for row in scored
        ],
        "segment_vmafs": [row["vmaf"] for row in scored],
        "frame_count": len(frame_rows),
        "frames_path": frames_path,
        "remote_note": remote_note,
        "compute_note": compute_note,
        "pooled_mean": pooled,
        "degraded": any(row["degraded"] for row in scored if row["vmaf"] is not None),
    }


def _score_observation(catalog: QualityCatalog, content_id: str, obs: dict[str, Any]) -> dict[str, Any]:
    frames = obs.get("frames") or []
    base = {
        "index": obs["index"],
        "repr_index": obs["repr_index"],
        "model": obs["model"],
        "level": obs["level"],
        "partial": obs["partial"],
        "vmaf": None,
        "source": "",
        "degraded": False,
    }
    # Player analyzer already applied abort / fast-complete blending.
    real = obs.get("real_vmaf")
    if isinstance(real, (int, float)) and obs.get("outcome_source") == "analyzer":
        base["vmaf"] = float(real)
        base["source"] = "analyzer"
        base["degraded"] = False
        return base
    if frames:
        hits: list[QualityHit] = []
        for row in frames:
            enhanced = bool(row.get("enhanced"))
            level = row.get("enhance_level")
            try:
                level_i = int(level) if enhanced and level is not None else 0
            except (TypeError, ValueError):
                level_i = 0
            try:
                frame_i = int(row.get("frame_in_segment")) if row.get("frame_in_segment") is not None else None
            except (TypeError, ValueError):
                frame_i = None
            hit = catalog.lookup(
                content_id=content_id,
                repr_index=int(obs["repr_index"]),
                model=str(row.get("model") or obs["model"]),
                level=level_i,
                segment_index=int(obs["index"]),
                frame_in_segment=frame_i,
                granularity="frame",
                scale=str(row.get("scale") or obs.get("scale") or ""),
                partial=str(obs.get("partial") or "none"),
            )
            if hit is None or hit.value is None:
                hits = []
                break
            hits.append(hit)
        if hits:
            base["vmaf"] = sum(hit.value or 0 for hit in hits) / len(hits)
            base["source"] = hits[0].source
            base["degraded"] = any(hit.degraded for hit in hits)
            return base
    ratio = obs.get("ratio")
    if isinstance(ratio, float) and 0 < ratio < 1 and int(obs["level"]) > 0:
        low = _lookup_level(catalog, content_id, obs, 0)
        high = _lookup_level(catalog, content_id, obs, int(obs["level"]))
        if low and high and low.value is not None and high.value is not None:
            base["vmaf"] = low.value * (1 - ratio) + high.value * ratio
            base["source"] = "mixed"
            base["degraded"] = True
            return base
    hit = _lookup_level(catalog, content_id, obs, int(obs["level"]))
    if hit and hit.value is not None:
        base["vmaf"] = hit.value
        base["source"] = hit.source
        base["degraded"] = hit.degraded
    return base


def _lookup_level(catalog, content_id, obs, level: int) -> Optional[QualityHit]:
    return catalog.lookup(
        content_id=content_id,
        repr_index=int(obs["repr_index"]),
        model=str(obs["model"]),
        level=level,
        segment_index=int(obs["index"]),
        granularity="segment",
        scale=str(obs.get("scale") or ""),
        partial="none" if level == 0 else str(obs.get("partial") or "none"),
    )


def _derive(recipe_names, segment_vmafs, stall_s: float, segment_count: int, samples: list[dict[str, Any]]) -> dict[str, Any]:
    derived: dict[str, Any] = {}
    for name in recipe_names:
        if name == "bones_qoe_v1":
            derived[name] = bones_qoe_v1(segment_vmafs, stall_s, segment_count)
            continue
        if name == "fast_complete_v1":
            derived[name] = fast_complete_v1(samples)
            continue
        fn = RECIPES.get(name)
        if fn is None:
            derived[name] = {"recipe": name, "value": None, "status": "unavailable", "reason": "unknown recipe"}
            continue
        derived[name] = fn(segment_vmafs, stall_s, segment_count)
    return derived


def _quality_source_label(quality: dict[str, Any]) -> str:
    labels: list[str] = []
    for row in quality.get("segments") or []:
        if row.get("vmaf") is None:
            continue
        src = str(row.get("source") or "")
        if src == "analyzer":
            labels.append("analyzer")
        elif src == "libvmaf" or "compute" in src:
            labels.append("compute")
        elif "remote" in src:
            labels.append("remote_catalog")
        elif src in {"static_table", "mixed", "aggregated", "segment_pack"} or src:
            labels.append("local_catalog")
    return _majority_label(labels, ("analyzer", "compute", "remote_catalog", "local_catalog"))


def _outcome_source_label(observations: list[dict[str, Any]]) -> str:
    labels = [str(obs.get("outcome_source") or "") for obs in observations if obs.get("outcome_source")]
    return _majority_label(
        labels,
        ("analyzer", "frames", "telemetry_playing", "telemetry_planned"),
    )


def _majority_label(labels: list[str], priority: tuple[str, ...]) -> str:
    if not labels:
        return "unavailable"
    counts: dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    best = ""
    best_n = -1
    for name in list(priority) + [key for key in counts if key not in priority]:
        n = counts.get(name, 0)
        if n > best_n:
            best = name
            best_n = n
    return best or "unavailable"


def _load_analyzer_report(state, run: ExperimentRun) -> dict[str, Any]:
    """Latest player analyzer JSON for this run, if present."""
    try:
        from ..telemetry import telemetry_dir

        root = telemetry_dir(state, run).parent / "analyzer"
    except (ImportError, AttributeError, TypeError, ValueError):
        return {}
    if not root.is_dir():
        return {}
    files = sorted(root.glob("*.json"), key=lambda path: path.stat().st_mtime)
    if not files:
        return {}
    try:
        payload = json.loads(files[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _stall_seconds(summary: dict[str, Any], analyzer_report: dict[str, Any]) -> float:
    """Prefer analyzer ``dur_stall`` so Lab QoE matches the player report."""
    dur = analyzer_report.get("dur_stall")
    if isinstance(dur, (int, float)):
        return float(dur)
    for key in ("stall_s", "rebuffer_s"):
        value = summary.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return 0.0


def _execution_stats_from_analyzer(analyzer_report: dict[str, Any]) -> dict[str, Any]:
    """Flatten enhance-level / abort aggregates for Lab compare and CSV."""
    if not analyzer_report:
        return {}
    out: dict[str, Any] = {}
    stats = analyzer_report.get("enhance_level_stats") or {}
    counts = analyzer_report.get("enhance_level_count") or {}
    completed_map = analyzer_report.get("enhance_level_completed_count") or {}
    if not isinstance(stats, dict):
        stats = {}
    if not isinstance(counts, dict):
        counts = {}
    if not isinstance(completed_map, dict):
        completed_map = {}
    for level in (0, 1, 2):
        key = str(level)
        row = stats.get(key) if isinstance(stats.get(key), dict) else {}
        count = row.get("count")
        completed = row.get("completed")
        rate = row.get("completion_rate")
        latency = row.get("avg_latency")
        if count is None and key in counts:
            count = counts.get(key)
        if completed is None and key in completed_map:
            completed = completed_map.get(key)
        if rate is None and isinstance(count, (int, float)) and count > 0 and isinstance(completed, (int, float)):
            rate = round(float(completed) / float(count), 4)
        prefix = f"enh_l{level}"
        if count is not None:
            out[f"{prefix}_count"] = int(count) if isinstance(count, (int, float)) else count
        if completed is not None:
            out[f"{prefix}_completed"] = int(completed) if isinstance(completed, (int, float)) else completed
        if isinstance(rate, (int, float)):
            out[f"{prefix}_rate"] = float(rate)
        if isinstance(latency, (int, float)):
            out[f"{prefix}_latency"] = float(latency)
    if analyzer_report.get("enhancer_abort_total") is not None:
        out["abort_total"] = analyzer_report.get("enhancer_abort_total")
    if analyzer_report.get("abort_wasted_frames_total") is not None:
        out["abort_wasted_frames"] = analyzer_report.get("abort_wasted_frames_total")
    reasons = analyzer_report.get("abort_reason_count")
    if isinstance(reasons, dict) and reasons:
        out["abort_reasons"] = {str(k): v for k, v in reasons.items()}
    if analyzer_report.get("num_stall") is not None:
        out["num_stall"] = analyzer_report.get("num_stall")
    if analyzer_report.get("num_quality_switches") is not None:
        out["num_quality_switches"] = analyzer_report.get("num_quality_switches")
    if isinstance(analyzer_report.get("avg_bitrate"), (int, float)):
        out["avg_bitrate"] = float(analyzer_report["avg_bitrate"])
    return out


def _align_fast_complete_recipe(
    derived: dict[str, Any],
    analyzer_report: dict[str, Any],
    recipe_names: list[str],
) -> None:
    """When the analyzer file exists, FC cards should match its aggregates."""
    if "fast_complete_v1" not in recipe_names or not analyzer_report:
        return
    if analyzer_report.get("enhancer_use_fast_complete") is False:
        derived["fast_complete_v1"] = {
            "recipe": "fast_complete_v1",
            "recipe_ver": "1",
            "value": 0,
            "status": "ok",
            "inputs": {"segments": 0, "saved_frames": 0, "quality_drop_avg": None},
        }
        return
    segments = analyzer_report.get("fast_complete_segments_count")
    if segments is None and not analyzer_report.get("enhancer_use_fast_complete"):
        return
    derived["fast_complete_v1"] = {
        "recipe": "fast_complete_v1",
        "recipe_ver": "1",
        "value": segments,
        "status": "ok" if segments is not None else "unavailable",
        "inputs": {
            "segments": segments,
            "saved_frames": analyzer_report.get("fast_complete_saved_frames_total"),
            "quality_drop_avg": analyzer_report.get("fast_complete_quality_drop_avg"),
        },
    }


def _status(observations, quality, derived) -> tuple[str, list[str]]:
    """Label how trustworthy the scores are.

    - ``complete``: measured path (libvmaf / compute), all segments scored
    - ``partial``: usable estimate (offline quality table / analyzer table /
      mixed), or some segments missing scores
    - ``unavailable``: nothing to show
    """
    reasons: list[str] = []
    if not observations:
        reasons.append("还没有可评分的播放段")
        return "unavailable", reasons
    vmafs = quality["segment_vmafs"]
    if not any(isinstance(v, (int, float)) for v in vmafs):
        reasons.append("质量表里没有这次播放的分数")
        if quality.get("remote_note"):
            reasons.append(quality["remote_note"])
        if quality.get("compute_note"):
            reasons.append(quality["compute_note"])
        return "unavailable", reasons
    if any(v is None for v in vmafs):
        reasons.append("有的段没有画质分")
        recipe = derived.get("bones_qoe_v1") or {}
        if recipe.get("status") != "ok" and not reasons:
            reasons.append("配方算不出结果")
        return "partial", reasons
    scored_sources = [
        str(row.get("source") or "")
        for row in (quality.get("segments") or [])
        if row.get("vmaf") is not None
    ]
    measured = bool(scored_sources) and all(
        src == "libvmaf" or src == "compute" or "libvmaf" in src for src in scored_sources
    )
    if measured and not quality.get("degraded"):
        return "complete", reasons
    # Offline table / analyzer / mixed estimates are usable but not measured.
    analyzer_all = bool(scored_sources) and all(src == "analyzer" for src in scored_sources)
    if analyzer_all:
        reasons.append(_ANALYZER_TABLE_REASON)
    else:
        reasons.append("画质分用的是离线质量表推算，不是实测 libvmaf")
    return "partial", reasons


def _quality_switches(observations: list[dict]) -> int:
    switches = 0
    prev = None
    for obs in observations:
        current = (obs.get("repr_index"), obs.get("level"))
        if prev is not None and current != prev:
            switches += 1
        prev = current
    return switches


def _arm_model(arm, settings: dict) -> str:
    if settings.get("model"):
        return normalize_model(settings["model"])
    player = getattr(arm, "player", None) or {}
    return normalize_model(str(player.get("mod_enhancer") or "imdn"))


def _load_frames(state, run, arm_id: str, model_default: str) -> list[dict]:
    try:
        storage, root = locate_frames(state, run, arm_id)
    except Exception:
        return []
    if storage == "none":
        return []
    return [enrich_frame_row(row, model_default=model_default) for row in iter_frame_rows(root)]


def _install_catalog_ref(catalog: QualityCatalog, settings: dict) -> None:
    ref = settings["catalog_ref"]
    if not ref or ref.startswith("content:"):
        return
    path = Path(ref)
    if not path.is_file():
        return
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    content_id = resolve_content_id(settings["content_source"], ref)
    model = normalize_model(settings["model"] or "imdn")
    catalog.register_document(content_id, model, document)

