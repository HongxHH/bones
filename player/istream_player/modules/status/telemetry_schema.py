"""Single source of truth for Prism L0 telemetry sample fields.

Publisher (StatusExporter) and control plane (append_samples) both project
through this schema so new metrics cannot be dropped by a stale whitelist.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional

# Bump when field semantics change in a breaking way (e.g. cpu_pct normalization).
METRICS_VERSION = 1

# Keys copied from StatusSnapshot.build() into each telemetry sample body.
# Order is the persisted jsonl column order (stable for diffs / CSV-minded tools).
SAMPLE_VALUE_KEYS: tuple[str, ...] = (
    "state",
    "position",
    "paused",
    "download_buffer_s",
    "enhance_buffer_s",
    "enhance_pending_s",
    "enhance_queue_s",
    "enhanced_ready_s",
    "enhanced_ready_inventory",
    "enhance_pending_inventory",
    "bandwidth_bps",
    "download_action",
    "enhance_action",
    "segment_index",
    "playing_segment",
    "enhancing_segment",
    "enhance_busy",
    "downloading_segment",
    "playing_enhanced",
    "playing_enhance_level",
    "playing_width",
    "playing_height",
    "render_fps",
    "stall_active",
    "stall_s_total",
    "cpu_pct",
    "cpu_pct_raw",
    "cpu_system_pct",
    "rss_mb",
    "mem_pct",
    "mem_total_mb",
    "gpu_util_pct",
    "gpu_mem_used_mb",
    "gpu_mem_total_mb",
    "gpu_mem_pct",
    "gpu_proc_mem_mb",
    "gpu_proc_mem_pct",
    # "nvml" = driver per-PID; "torch" = CUDA allocator reserved (may under-count).
    "gpu_proc_mem_src",
    # Optional mechanism bag. Null when this player has nothing extra to report.
    "facts",
    # Frames the JPEG archive dropped because its writer fell behind.
    "frame_archive_dropped",
)


def project_sample(
    snap: Mapping[str, Any],
    *,
    ts: float,
    event: Optional[str] = None,
) -> dict[str, Any]:
    """Build a telemetry sample body from a status snapshot."""
    out: dict[str, Any] = {
        "ts": ts,
        "metrics_v": METRICS_VERSION,
        "event": event,
    }
    for key in SAMPLE_VALUE_KEYS:
        out[key] = snap.get(key)
    return out


def project_row(
    sample: Mapping[str, Any],
    *,
    run_id: str,
    arm_id: str,
    session_id: Optional[str] = None,
    ts_default: Optional[float] = None,
) -> dict[str, Any]:
    """Project an inbound sample onto the persisted jsonl row (control plane)."""
    row: dict[str, Any] = {
        "ts": sample.get("ts", ts_default),
        "run_id": run_id,
        "arm_id": arm_id,
        "session_id": session_id if session_id is not None else sample.get("session_id"),
        "metrics_v": sample.get("metrics_v", METRICS_VERSION),
        "event": sample.get("event"),
    }
    for key in SAMPLE_VALUE_KEYS:
        row[key] = sample.get(key)
    return row


def sample_key_set() -> set[str]:
    return set(SAMPLE_VALUE_KEYS) | {"ts", "metrics_v", "event"}


def assert_keys_subset(keys: Iterable[str]) -> None:
    """Dev helper: ensure an ad-hoc key list stays inside the schema."""
    unknown = set(keys) - sample_key_set()
    if unknown:
        raise AssertionError(f"unknown telemetry keys: {sorted(unknown)}")
