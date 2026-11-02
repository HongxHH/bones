"""Metric recipes. bones_qoe_v1 matches the player analyzer formula."""

from __future__ import annotations

from typing import Any, Optional

RECIPE_VERSION = "1"


def bones_qoe_v1(segment_vmafs: list[Optional[float]], stall_s: float, segment_count: int) -> dict[str, Any]:
    """QoE = avg_vmaf − oscillation_vmaf − avg_freeze/10.

    ``avg_vmaf`` averages segments that have a score. ``avg_freeze`` divides
    total stall seconds by the number of played segments, including unscored ones.
    Quality switches are not part of the formula.
    """
    present = [float(v) for v in segment_vmafs if isinstance(v, (int, float))]
    count = segment_count if segment_count > 0 else len(segment_vmafs)
    if not present or count <= 0:
        return {
            "recipe": "bones_qoe_v1",
            "recipe_ver": RECIPE_VERSION,
            "value": None,
            "status": "unavailable",
            "inputs": {
                "avg_vmaf": None,
                "oscillation_vmaf": None,
                "avg_freeze": None,
                "stall_s": stall_s,
                "segments_scored": 0,
                "segments": count,
            },
        }
    avg = sum(present) / len(present)
    if len(present) > 1:
        osc = sum(abs(present[i] - present[i - 1]) for i in range(1, len(present))) / (len(present) - 1)
    else:
        osc = 0.0
    avg_freeze = float(stall_s) / count
    qoe = avg - osc - avg_freeze / 10.0
    return {
        "recipe": "bones_qoe_v1",
        "recipe_ver": RECIPE_VERSION,
        "value": round(qoe, 6),
        "status": "ok",
        "inputs": {
            "avg_vmaf": round(avg, 6),
            "oscillation_vmaf": round(osc, 6),
            "avg_freeze": round(avg_freeze, 6),
            "stall_s": round(float(stall_s), 6),
            "segments_scored": len(present),
            "segments": count,
        },
    }


def fast_complete_v1(samples: list[dict[str, Any]] | None) -> dict[str, Any]:
    """Read the latest fast_complete fact bag. Missing facts stay unavailable, not zero."""
    latest = None
    for sample in samples or []:
        facts = sample.get("facts") if isinstance(sample, dict) else None
        block = facts.get("fast_complete") if isinstance(facts, dict) else None
        if isinstance(block, dict):
            latest = block
    empty = {"segments": None, "saved_frames": None, "quality_drop_avg": None}
    if not samples:
        reason = "还没有播放遥测"
    else:
        reason = "还没有一段增强结束，统计会在段完成后出现"
    if latest is None:
        return {
            "recipe": "fast_complete_v1",
            "recipe_ver": RECIPE_VERSION,
            "value": None,
            "status": "unavailable",
            "reason": reason,
            "inputs": empty,
        }
    return {
        "recipe": "fast_complete_v1",
        "recipe_ver": RECIPE_VERSION,
        "value": latest.get("segments"),
        "status": "ok",
        "inputs": {
            "segments": latest.get("segments"),
            "saved_frames": latest.get("saved_frames"),
            "quality_drop_avg": latest.get("quality_drop_avg"),
        },
    }


RECIPES = {
    "bones_qoe_v1": bones_qoe_v1,
    "fast_complete_v1": fast_complete_v1,
}
