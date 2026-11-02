from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from .orchestrator.state import PlatformState
from .schemas import ExperimentRun
from .telemetry_schema import project_row


def _has_run_data(run_dir: Path) -> bool:
    telemetry = run_dir / "telemetry"
    if telemetry.is_dir():
        for item in telemetry.iterdir():
            if item.is_file() and item.suffix == ".jsonl" and item.stat().st_size > 0:
                return True
    frames = run_dir / "frames"
    return frames.is_dir() and any(frames.rglob("frames.jsonl"))


def run_archive_dir(state: PlatformState, run: ExperimentRun) -> Path:
    """Directory that actually holds this run.

    Older experiments still record ``server/var`` after the store moved to
    ``prism/var``. Prefer whichever directory already contains telemetry.
    """
    exp = state.experiments.get(run.experiment_id)
    current = state.store.experiment_dir(run.experiment_id) / "runs" / run.id
    recorded = Path(exp.artifact_dir) / "runs" / run.id if exp and exp.artifact_dir else None
    if _has_run_data(current):
        chosen = current
    elif recorded is not None and _has_run_data(recorded):
        chosen = recorded
    elif recorded is not None and recorded.exists():
        chosen = recorded
    else:
        chosen = current
    if exp is not None and _has_run_data(chosen):
        corrected = str(chosen.parent.parent)
        if exp.artifact_dir != corrected:
            exp.artifact_dir = corrected
            state.store.save_entity("experiments", exp)
    return chosen


def telemetry_dir(state: PlatformState, run: ExperimentRun) -> Path:
    path = run_archive_dir(state, run) / "telemetry"
    path.mkdir(parents=True, exist_ok=True)
    return path


def telemetry_path(state: PlatformState, run: ExperimentRun, arm_id: str) -> Path:
    return telemetry_dir(state, run) / f"{arm_id}.jsonl"


def append_samples(
    state: PlatformState,
    *,
    run_id: str,
    arm_id: str,
    samples: list[dict[str, Any]],
    session_id: Optional[str] = None,
) -> dict[str, Any]:
    run = state.runs.get(run_id)
    if run is None:
        raise ValueError("run not found")
    if not any(a.id == arm_id for a in run.arms):
        raise ValueError("arm not found")
    path = telemetry_path(state, run, arm_id)
    now = time.time()
    written = 0
    with path.open("a", encoding="utf-8") as fh:
        for sample in samples:
            row = project_row(
                sample,
                run_id=run_id,
                arm_id=arm_id,
                session_id=session_id,
                ts_default=now,
            )
            fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            written += 1
    return {"ok": True, "written": written, "path": str(path)}


_TAIL_CACHE: dict[str, dict[str, Any]] = {}


def _parse_jsonl_lines(lines: list[bytes] | list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for line in lines:
        if isinstance(line, bytes):
            text = line.decode("utf-8", errors="replace").strip()
        else:
            text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


def _tail_bytes(path: Path, limit: int) -> bytes:
    """Read only enough of the file tail to cover ``limit`` JSONL rows."""
    size = path.stat().st_size
    if size <= 0 or limit < 1:
        return b""
    # Worst case a row is large; grow the window until enough newlines exist.
    window = min(size, max(4096, limit * 512))
    with path.open("rb") as handle:
        while True:
            start = max(0, size - window)
            handle.seek(start)
            blob = handle.read(size - start)
            # A mid-line start needs one extra newline so the partial first row can be dropped.
            needed = limit if start == 0 else limit + 1
            if blob.count(b"\n") >= needed or start == 0 or window >= size:
                return blob
            window = min(size, window * 2)


def iter_samples(path: Path, *, limit: Optional[int] = None) -> list[dict[str, Any]]:
    """Load telemetry rows.

    ``limit`` reads from the file tail (O(limit)), not the whole jsonl.
    ``limit is None`` still reads the full file for end-of-run bundles.
    """
    if not path.exists():
        return []
    try:
        stat = path.stat()
    except OSError:
        return []
    size = int(stat.st_size)
    if size <= 0:
        return []
    if limit is None or limit <= 0:
        try:
            return _parse_jsonl_lines(path.read_bytes().splitlines())
        except OSError:
            return []
    mtime = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1e9)))
    key = f"{path.resolve()}|{limit}"
    cached = _TAIL_CACHE.get(key)
    if cached and cached.get("size") == size and cached.get("mtime") == mtime:
        return list(cached["rows"])
    try:
        blob = _tail_bytes(path, limit)
    except OSError:
        return []
    if not blob:
        return []
    start = max(0, size - len(blob))
    lines = blob.splitlines()
    if start > 0 and lines:
        lines = lines[1:]
    if len(lines) > limit:
        lines = lines[-limit:]
    rows = _parse_jsonl_lines(lines)
    _TAIL_CACHE[key] = {"size": size, "mtime": mtime, "rows": rows}
    if len(_TAIL_CACHE) > 64:
        _TAIL_CACHE.pop(next(iter(_TAIL_CACHE)))
    return list(rows)


_LIVE_DROP_KEYS = frozenset({
    "enhanced_ready_inventory",
    "enhance_pending_inventory",
    "facts",
    "config",
})


def slim_live_samples(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep inventories only on the newest sample so Lab charts stay small."""
    if not samples:
        return []
    out: list[dict[str, Any]] = []
    last = len(samples) - 1
    for i, sample in enumerate(samples):
        if i == last:
            out.append(sample)
            continue
        out.append({k: v for k, v in sample.items() if k not in _LIVE_DROP_KEYS})
    return out


def build_live_view(
    state: PlatformState,
    run: ExperimentRun,
    *,
    arm_id: Optional[str] = None,
    limit: int = 120,
) -> dict[str, Any]:
    """Lightweight Lab payload: tail samples + scalar summary. No bundle, no disk write."""
    chosen = arm_id or getattr(run, "active_arm_id", None) or (run.arms[0].id if run.arms else "")
    if chosen and not any(a.id == chosen for a in run.arms):
        raise ValueError("arm not found")
    samples = iter_samples(telemetry_path(state, run, chosen), limit=max(1, int(limit))) if chosen else []
    summary = summarize_arm(samples)
    from .frames import frame_count_hint
    from .workspaces import cards_for_run

    workspace_id, cards = cards_for_run(run.spec)
    status = run.status.value if hasattr(run.status, "value") else str(run.status)
    return {
        "run_id": run.id,
        "status": status,
        "active_arm_id": getattr(run, "active_arm_id", None),
        "arm_id": chosen,
        "arms": [{"id": arm.id} for arm in run.arms],
        "samples": slim_live_samples(samples),
        "summary": summary,
        "workspace": workspace_id,
        "cards": cards,
        "frame_count": frame_count_hint(state, run, chosen) if chosen else 0,
    }


def _primary_action(action: Any) -> Optional[float]:
    if isinstance(action, dict) and action:
        try:
            return float(next(iter(action.values())))
        except (TypeError, ValueError):
            return None
    if isinstance(action, (int, float)):
        return float(action)
    return None


def _avg(nums: list[float]) -> Optional[float]:
    return round(sum(nums) / len(nums), 3) if nums else None


def summarize_arm(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        return {
            "samples": 0,
            "duration_s": 0.0,
            "rebuffer_s": 0.0,
            "rebuffer_ratio": 0.0,
            "avg_bandwidth_bps": 0.0,
            "avg_download_buffer_s": 0.0,
            "avg_download_action": None,
            "max_position": 0.0,
            "avg_render_fps": None,
            "stall_s": 0.0,
            "avg_cpu_pct": None,
            "avg_rss_mb": None,
            "avg_gpu_util_pct": None,
            "avg_gpu_mem_used_mb": None,
            "frame_archive_dropped": 0,
        }
    t0 = float(samples[0].get("ts") or 0)
    t1 = float(samples[-1].get("ts") or t0)
    duration = max(0.0, t1 - t0)
    gap_rebuffer = 0.0
    bw_sum = 0.0
    bw_n = 0
    buf_sum = 0.0
    buf_n = 0
    act_sum = 0.0
    act_n = 0
    max_pos = 0.0
    fps_vals: list[float] = []
    cpu_vals: list[float] = []
    rss_vals: list[float] = []
    gpu_vals: list[float] = []
    gpu_mem_vals: list[float] = []
    stall_s = 0.0
    has_client_stall = False
    archive_dropped = 0
    for i, s in enumerate(samples):
        st = str(s.get("state") or "").upper()
        # Fallback only: sample-gap estimate (BUFFERING only; matches StallTracker).
        if i + 1 < len(samples) and st == "BUFFERING":
            dt = float(samples[i + 1].get("ts") or 0) - float(s.get("ts") or 0)
            if dt > 0:
                gap_rebuffer += dt
        bw = s.get("bandwidth_bps")
        if isinstance(bw, (int, float)) and bw > 0:
            bw_sum += float(bw)
            bw_n += 1
        buf = s.get("download_buffer_s")
        if isinstance(buf, (int, float)):
            buf_sum += float(buf)
            buf_n += 1
        act = _primary_action(s.get("download_action"))
        if act is not None:
            act_sum += act
            act_n += 1
        pos = s.get("position")
        if isinstance(pos, (int, float)):
            max_pos = max(max_pos, float(pos))
        fps = s.get("render_fps")
        if isinstance(fps, (int, float)) and fps >= 0:
            fps_vals.append(float(fps))
        cpu = s.get("cpu_pct")
        if isinstance(cpu, (int, float)):
            cpu_vals.append(float(cpu))
        rss = s.get("rss_mb")
        if isinstance(rss, (int, float)):
            rss_vals.append(float(rss))
        gpu = s.get("gpu_util_pct")
        if isinstance(gpu, (int, float)):
            gpu_vals.append(float(gpu))
        gmem = s.get("gpu_mem_used_mb")
        if isinstance(gmem, (int, float)):
            gpu_mem_vals.append(float(gmem))
        stall = s.get("stall_s_total")
        if isinstance(stall, (int, float)):
            has_client_stall = True
            stall_s = max(stall_s, float(stall))
        dropped = s.get("frame_archive_dropped")
        if isinstance(dropped, (int, float)):
            archive_dropped = max(archive_dropped, int(dropped))
    # Prefer client StallTracker cumulative; gap reconstruction is legacy fallback.
    if has_client_stall:
        rebuffer = stall_s
    else:
        rebuffer = gap_rebuffer
        stall_s = gap_rebuffer
    return {
        "samples": len(samples),
        "duration_s": round(duration, 3),
        "rebuffer_s": round(rebuffer, 3),
        "rebuffer_ratio": round(rebuffer / duration, 4) if duration > 0 else 0.0,
        "avg_bandwidth_bps": round(bw_sum / bw_n, 1) if bw_n else 0.0,
        "avg_download_buffer_s": round(buf_sum / buf_n, 3) if buf_n else 0.0,
        "avg_download_action": round(act_sum / act_n, 3) if act_n else None,
        "max_position": round(max_pos, 3),
        "avg_render_fps": _avg(fps_vals),
        "stall_s": round(stall_s, 3),
        "avg_cpu_pct": _avg(cpu_vals),
        "avg_rss_mb": _avg(rss_vals),
        "avg_gpu_util_pct": _avg(gpu_vals),
        "avg_gpu_mem_used_mb": _avg(gpu_mem_vals),
        "frame_archive_dropped": archive_dropped,
    }


def build_run_metrics(state: PlatformState, run: ExperimentRun) -> dict[str, Any]:
    from .metrics.bundle import assemble_run_bundle, fold_bundle

    arms: dict[str, Any] = {}
    for arm in run.arms:
        samples = iter_samples(telemetry_path(state, run, arm.id))
        arms[arm.id] = summarize_arm(samples)
    metrics = {
        "run_id": run.id,
        "experiment_id": run.experiment_id,
        "arms": arms,
        "updated_at": time.time(),
    }
    bundle = assemble_run_bundle(
        state,
        run,
        arms,
        samples_for=lambda arm_id: iter_samples(telemetry_path(state, run, arm_id)),
    )
    fold_bundle(metrics, bundle)
    from .workspaces import cards_for_run

    workspace_id, cards = cards_for_run(run.spec)
    metrics["workspace"] = workspace_id
    metrics["cards"] = cards
    return metrics


def persist_run_metrics(state: PlatformState, run: ExperimentRun) -> dict[str, Any]:
    metrics = build_run_metrics(state, run)
    out_dir = telemetry_dir(state, run).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics_summary.json"
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    bundle_path = out_dir / "metrics_bundle.json"
    bundle_path.write_text(json.dumps(metrics.get("bundle") or {}, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics["path"] = str(path)
    metrics["bundle_path"] = str(bundle_path)
    return metrics


_SEGMENT_FIELDS = (
    "index",
    "download_action",
    "enhance_action",
    "is_enhance",
    "abort_reason",
    "fast_complete",
    "enhanced_cnt",
    "interpolated_cnt",
    "enhance_latency",
    "real_vmaf",
    "quality_drop_due_to_fast",
    "fast_complete_threshold",
    "enhance_start_to_play_time",
    "enhance_end_to_play_time",
)


def project_segments(segments: Any) -> list[dict[str, Any]]:
    """Slim per-segment rows for the Lab dialog. Drop URLs and byte counters."""
    if not isinstance(segments, list):
        return []
    rows: list[dict[str, Any]] = []
    for item in segments:
        if not isinstance(item, dict):
            continue
        row: dict[str, Any] = {}
        for key in _SEGMENT_FIELDS:
            if key not in item:
                continue
            value = item[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                row[key] = value
        if row:
            rows.append(row)
    rows.sort(key=lambda row: (row.get("index") is None, row.get("index") if isinstance(row.get("index"), (int, float)) else 0))
    return rows


def _public_fields(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, Any] = {}
    for key, item in value.items():
        if str(key).startswith("_"):
            continue
        if isinstance(item, (str, int, float, bool)) or item is None:
            out[str(key)] = item
    return out


def _arm_metrics(state: PlatformState, run: ExperimentRun) -> dict[str, Any]:
    """Numbers already computed for this run. Prefer the on-disk summary."""
    root = telemetry_dir(state, run).parent
    stored: dict[str, Any] = {}
    path = root / "metrics_summary.json"
    if path.is_file():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                stored = loaded
        except (OSError, json.JSONDecodeError):
            stored = {}
    if not (stored.get("arms") or {}) and isinstance(run.metrics, dict):
        stored = run.metrics
    arms = stored.get("arms") if isinstance(stored, dict) else None
    if not isinstance(arms, dict) or not arms:
        return {}
    arm_id = run.arms[0].id if run.arms else next(iter(arms))
    row = arms.get(arm_id)
    return row if isinstance(row, dict) else {}


def _summary_from_metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "qoe": row.get("qoe"),
        "avg_vmaf": row.get("avg_vmaf"),
        "oscillation_vmaf": row.get("oscillation_vmaf"),
        "avg_freeze": row.get("avg_freeze"),
        "dur_stall": row.get("stall_s"),
        "avg_bitrate": row.get("avg_bandwidth_bps"),
        "num_quality_switches": row.get("num_quality_switches"),
        "segments": (row.get("num_segments") if row.get("num_segments") is not None else None),
        "avg_render_fps": row.get("avg_render_fps"),
        "rebuffer_s": row.get("rebuffer_s"),
        "frame_archive_dropped": row.get("frame_archive_dropped"),
        "fast_complete_segments_count": row.get("fc_segments"),
        "fast_complete_saved_frames_total": row.get("fc_saved_frames"),
        "fast_complete_quality_drop_avg": row.get("fc_quality_drop_avg"),
    }


def analyzer_report(state: PlatformState, run: ExperimentRun) -> dict[str, Any]:
    """Lab report for one run.

    Prism metrics are always included. The player analyzer JSON adds stall counts,
    enhance-level completion, abort reasons, startup, and GPU memory when that file exists.
    """
    metrics_row = _arm_metrics(state, run)
    if not metrics_row and run.arms:
        tel = telemetry_path(state, run, run.arms[0].id)
        if tel.is_file() and tel.stat().st_size > 0:
            built = build_run_metrics(state, run)
            metrics_row = (built.get("arms") or {}).get(run.arms[0].id) or {}
    summary = {key: value for key, value in _summary_from_metrics(metrics_row).items() if value is not None}
    note = ""
    file_name = ""
    segment_rows: list[dict[str, Any]] = []
    level_rows: list[dict[str, Any]] = []
    abort_reasons: dict[str, Any] = {}
    startup: dict[str, Any] = {}
    gpu: dict[str, Any] = {}
    root = telemetry_dir(state, run).parent / "analyzer"
    files = sorted(root.glob("*.json"), key=lambda path: path.stat().st_mtime) if root.is_dir() else []
    if files:
        path = files[-1]
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            note = f"分析报告读不出来：{exc}"
            payload = None
        if isinstance(payload, dict):
            file_name = path.name
            levels = payload.get("enhance_level_stats") or {}
            if isinstance(levels, dict):
                for key, row in levels.items():
                    if isinstance(row, dict):
                        level_rows.append({"level": key, **_public_fields(row)})
            abort_reasons = _public_fields(payload.get("abort_reason_count"))
            startup = _public_fields(payload.get("startup_metrics"))
            gpu = _public_fields(payload.get("gpu_metrics"))
            overlay = {
                "qoe": payload.get("QOE"),
                "avg_vmaf": payload.get("avg_vmaf"),
                "oscillation_vmaf": payload.get("oscillation_vmaf"),
                "avg_freeze": payload.get("avg_freeze"),
                "num_stall": payload.get("num_stall"),
                "dur_stall": payload.get("dur_stall"),
                "avg_bitrate": payload.get("avg_bitrate"),
                "num_quality_switches": payload.get("num_quality_switches"),
                "enhancer_abort_total": payload.get("enhancer_abort_total"),
                "enhancer_use_fast_complete": payload.get("enhancer_use_fast_complete"),
                "fast_complete_segments_count": payload.get("fast_complete_segments_count"),
                "fast_complete_saved_frames_total": payload.get("fast_complete_saved_frames_total"),
                "fast_complete_quality_drop_avg": payload.get("fast_complete_quality_drop_avg"),
                "segments": len(payload.get("segments") or []) or None,
            }
            for key, value in overlay.items():
                if value is not None:
                    summary[key] = value
            segment_rows = project_segments(payload.get("segments"))
    elif summary:
        note = "增强级别、中止原因、启动时间和 GPU 显存在播放器分析文件里。这次运行的该文件还没写出来，下面是控制面已经汇总的指标。"
    if not summary and not level_rows:
        return {"available": False, "reason": note or "这次运行还没有可显示的指标"}
    return {
        "available": True,
        "file": file_name,
        "note": note,
        "summary": summary,
        "enhance_levels": level_rows,
        "segments": segment_rows,
        "abort_reasons": abort_reasons,
        "startup": startup,
        "gpu": gpu,
    }


METRIC_KEYS = [
    "samples",
    "duration_s",
    "rebuffer_s",
    "rebuffer_ratio",
    "stall_s",
    "avg_render_fps",
    "avg_bandwidth_bps",
    "avg_download_buffer_s",
    "avg_download_action",
    "max_position",
    "avg_cpu_pct",
    "avg_rss_mb",
    "avg_gpu_util_pct",
    "avg_gpu_mem_used_mb",
    "qoe",
    "avg_vmaf",
    "oscillation_vmaf",
    "avg_freeze",
    "num_quality_switches",
    "metrics_status",
]


def flatten_run_rows(state: PlatformState, run: ExperimentRun) -> list[dict[str, Any]]:
    metrics = build_run_metrics(state, run)
    rows: list[dict[str, Any]] = []
    for arm in run.arms:
        m = (metrics.get("arms") or {}).get(arm.id) or {}
        rows.append(
            {
                "run_id": run.id,
                "experiment_id": run.experiment_id,
                "run_name": run.name,
                "run_status": run.status.value if hasattr(run.status, "value") else str(run.status),
                "arm_id": arm.id,
                "session_id": arm.session_id or "",
                "publication_id": arm.publication_id or "",
                **{k: m.get(k) for k in METRIC_KEYS},
            }
        )
    return rows


def metrics_to_csv(rows: list[dict[str, Any]]) -> str:
    import csv
    import io

    if not rows:
        return "run_id,arm_id\n"
    fieldnames: list[str] = []
    for row in rows:
        for k in row:
            if k not in fieldnames:
                fieldnames.append(k)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    return buf.getvalue()


def export_run_csv(state: PlatformState, run: ExperimentRun) -> str:
    return metrics_to_csv(flatten_run_rows(state, run))
