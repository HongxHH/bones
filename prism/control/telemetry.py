from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

from .orchestrator.state import PlatformState
from .schemas import ExperimentRun
from .telemetry_schema import project_row


def telemetry_dir(state: PlatformState, run: ExperimentRun) -> Path:
    exp = state.experiments.get(run.experiment_id)
    base = Path(exp.artifact_dir) if exp and exp.artifact_dir else state.store.experiment_dir(run.experiment_id)
    path = base / "runs" / run.id / "telemetry"
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


def iter_samples(path: Path, *, limit: Optional[int] = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    if limit is not None and limit > 0:
        lines = lines[-limit:]
    out: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


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
    }


def build_run_metrics(state: PlatformState, run: ExperimentRun) -> dict[str, Any]:
    arms: dict[str, Any] = {}
    for arm in run.arms:
        samples = iter_samples(telemetry_path(state, run, arm.id))
        arms[arm.id] = summarize_arm(samples)
    return {
        "run_id": run.id,
        "experiment_id": run.experiment_id,
        "arms": arms,
        "updated_at": time.time(),
    }


def persist_run_metrics(state: PlatformState, run: ExperimentRun) -> dict[str, Any]:
    metrics = build_run_metrics(state, run)
    out_dir = telemetry_dir(state, run).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "metrics_summary.json"
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    metrics["path"] = str(path)
    return metrics


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
