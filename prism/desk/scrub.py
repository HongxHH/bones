"""Picture-pane scrub requests and shell split widths. No Qt."""

from __future__ import annotations

import json
from typing import Any
from urllib.parse import quote


def frames_page_url(host: str, port: int, run_id: str, arm_id: str) -> str:
    arm = quote(str(arm_id), safe="")
    return f"http://{host}:{port}/v1/runs/{run_id}/frames?arm_id={arm}&limit=1"


def save_frames_url(host: str, port: int, run_id: str, arm_id: str) -> str:
    arm = quote(str(arm_id), safe="")
    return f"http://{host}:{port}/v1/runs/{run_id}/frames/save?arm_id={arm}"


def picture_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1/system/picture"


def picture_put_body(run_id: str, arm_id: str, frame: int) -> bytes:
    return json.dumps(
        {"run_id": run_id, "arm_id": arm_id, "frame": int(frame)},
        ensure_ascii=False,
    ).encode("utf-8")


def frame_caption(row: dict | None) -> str:
    """Same line the Lab timeline shows for one archived frame."""
    if not row:
        return "—"
    level = "—" if row.get("enhance_level") is None else row.get("enhance_level")
    enhanced = "已增强" if row.get("enhanced") else "未增强"
    model = f" · {row['model']}" if row.get("model") else ""
    scale = f" · {row['scale']}" if row.get("scale") else ""
    vmaf = row.get("vmaf")
    try:
        vmaf_bit = "" if vmaf is None or vmaf == "" else f" · VMAF {float(vmaf):.1f}"
    except (TypeError, ValueError):
        vmaf_bit = ""
    frame = row.get("frame")
    segment = row.get("segment")
    representation = row.get("representation")
    return f"#{frame} · 段 {segment} · {representation} · 增强 {level} · {enhanced}{model}{scale}{vmaf_bit}"


def pick_scrub_run(runs: list[dict[str, Any]] | None) -> dict[str, str] | None:
    """Live run wins. Otherwise the newest stopped or failed run."""
    if not runs:
        return None
    live = [item for item in runs if item.get("status") in ("running", "preparing")]
    pool = live or [item for item in runs if item.get("status") in ("stopped", "failed")]
    if not pool:
        return None
    pool.sort(
        key=lambda item: str(item.get("updated_at") or item.get("started_at") or item.get("created_at") or ""),
        reverse=True,
    )
    run = pool[0]
    run_id = str(run.get("id") or "")
    if not run_id:
        return None
    arms = run.get("arms") or []
    arm_id = str(run.get("active_arm_id") or "")
    if not arm_id and arms:
        arm_id = str((arms[0] or {}).get("id") or "")
    if not arm_id:
        return None
    return {"run_id": run_id, "arm_id": arm_id}


def split_left(total: int, *, review: bool) -> int:
    """Left column width. Compare uses about half; otherwise the picture stays ~72%."""
    if total < 200:
        return max(0, total)
    cap = max(360, total - 160)
    if review:
        left = int(total * 0.50)
    else:
        left = min(480, max(360, int(total * 0.28)))
    return min(left, cap)
