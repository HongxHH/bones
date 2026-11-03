"""Playback segment outcomes used by metric recipes.

Observations carry *what was played* (repr, enhance level, abort, fast-complete
ratio). They may also carry ``real_vmaf`` copied from the player analyzer; that
value is still a quality-table estimate, not measured libvmaf.
"""

from __future__ import annotations

from typing import Any, Optional


def normalize_model(name: str) -> str:
    from .catalog import normalize_model as _norm

    return _norm(name)


def enrich_frame_row(row: dict[str, Any], *, model_default: str = "") -> dict[str, Any]:
    """Fill model / source size / scale when the archive row is older or partial."""
    out = dict(row)
    if not out.get("model") and model_default:
        out["model"] = model_default
    scale = out.get("scale")
    if scale in (None, "", 0):
        src_w = _as_int(out.get("src_width"))
        width = _as_int(out.get("width"))
        if src_w and width and src_w > 0:
            ratio = width / src_w
            rounded = round(ratio)
            if rounded >= 1 and abs(ratio - rounded) < 0.08:
                out["scale"] = f"x{rounded}"
            else:
                out["scale"] = ""
        else:
            raw = _as_int(out.get("enhance_scale"))
            out["scale"] = f"x{raw}" if raw and raw > 0 else ""
    elif isinstance(scale, (int, float)) and not isinstance(scale, bool):
        out["scale"] = f"x{int(scale)}" if scale > 0 else ""
    return out


def segments_from_playback(
    samples: list[dict[str, Any]],
    frames: Optional[list[dict[str, Any]]] = None,
    *,
    model_default: str = "imdn",
) -> list[dict[str, Any]]:
    """One observation per played segment, refined by the frame archive when present.

    Telemetry ``enhance_action`` is the *planned* enhance level. For the segment
    currently playing, prefer ``playing_enhanced`` / ``playing_enhance_level`` so
    aborted or skipped enhance is not scored as a full enhance.
    """
    indexed = [(_segment_index(sample), sample) for sample in samples]
    indexed = [(index, sample) for index, sample in indexed if index is not None]
    if not indexed and samples:
        indexed = [(0, samples[-1])]
    by_index: dict[int, dict[str, Any]] = {}
    for index, sample in indexed:
        slot = by_index.get(index)
        if slot is None:
            slot = _blank(index, model_default)
            by_index[index] = slot
        repr_index = _action_value(sample.get("download_action"), index)
        planned = _action_value(sample.get("enhance_action"), index)
        if repr_index is not None:
            slot["repr_index"] = repr_index
        if planned is not None:
            slot["planned_level"] = planned
        _apply_playing_outcome(slot, sample, planned)
        model = _model_from_sample(sample)
        if model:
            slot["model"] = normalize_model(model)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in frames or []:
        enriched = enrich_frame_row(row, model_default=model_default)
        index = _as_int(enriched.get("segment"))
        if index is None:
            index = 0
        grouped.setdefault(index, []).append(enriched)
        if index not in by_index:
            by_index[index] = _blank(index, model_default)
    for index, rows in grouped.items():
        _apply_frames(by_index[index], rows, model_default)
    ordered = [by_index[i] for i in sorted(by_index)]
    if not samples and not frames:
        return []
    if not ordered and (samples or frames):
        ordered = [_blank(0, model_default)]
    return ordered


def observations_from_analyzer(
    analyzer_segments: list[dict[str, Any]] | None,
    *,
    model_default: str = "imdn",
    segment_origin: int = 1,
) -> list[dict[str, Any]]:
    """Build observations solely from the player analyzer segment list."""
    return merge_analyzer_outcomes(
        [],
        analyzer_segments,
        model_default=model_default,
        segment_origin=segment_origin,
    )


def merge_analyzer_outcomes(
    observations: list[dict[str, Any]],
    analyzer_segments: list[dict[str, Any]] | None,
    *,
    model_default: str = "imdn",
    segment_origin: int = 1,
) -> list[dict[str, Any]]:
    """Overlay player-analyzer segment outcomes onto telemetry/frame observations.

    Analyzer abort / fast-complete / ``real_vmaf`` match the player QoE report.
    Prefer ``observations_from_analyzer`` when a full analyzer report exists; this
    merge is the fallback when only some segments need correction.
    """
    if not analyzer_segments:
        return observations
    by_index: dict[int, dict[str, Any]] = {}
    for obs in observations:
        index = _as_int(obs.get("index"))
        if index is None:
            continue
        by_index[index] = dict(obs)
    for seg in analyzer_segments:
        if not isinstance(seg, dict):
            continue
        index = _as_int(seg.get("index"))
        if index is None:
            continue
        slot = by_index.get(index)
        if slot is None:
            slot = _blank(index, model_default)
            by_index[index] = slot
        _apply_analyzer_segment(slot, seg, segment_origin=segment_origin)
    return [by_index[i] for i in sorted(by_index)]


def _apply_playing_outcome(slot: dict[str, Any], sample: dict[str, Any], planned: Optional[int]) -> None:
    """Prefer live playback outcome over the scheduler's planned enhance action."""
    if "playing_enhanced" in sample:
        enhanced = sample.get("playing_enhanced")
        playing_level = _as_int(sample.get("playing_enhance_level"))
        if enhanced is False:
            slot["level"] = 0
            slot["partial"] = "none"
            slot["ratio"] = 0.0
            slot["is_enhance"] = False
            slot["outcome_source"] = "telemetry_playing"
            return
        if enhanced is True:
            level = playing_level if playing_level is not None else planned
            if level is not None:
                slot["level"] = level
            slot["is_enhance"] = True
            if slot.get("partial") is None:
                slot["partial"] = "none"
            slot["outcome_source"] = "telemetry_playing"
            return
    # Legacy samples without playing_enhanced: keep planned level, mark uncertain.
    if planned is not None:
        slot["level"] = planned
        if not slot.get("outcome_source"):
            slot["outcome_source"] = "telemetry_planned"


def _apply_analyzer_segment(
    slot: dict[str, Any],
    seg: dict[str, Any],
    *,
    segment_origin: int = 1,
) -> None:
    download = _as_int(seg.get("download_action"))
    if download is None:
        download = _as_int(seg.get("quality"))
    if download is not None:
        slot["repr_index"] = download
    planned = _as_int(seg.get("enhance_action"))
    if planned is not None:
        slot["planned_level"] = planned
    abort = seg.get("abort_reason")
    if isinstance(abort, str) and abort.strip():
        slot["abort_reason"] = abort.strip()
    else:
        abort = None
    is_enhance = seg.get("is_enhance")
    fast = bool(seg.get("fast_complete"))
    if abort or is_enhance is False:
        slot["level"] = 0
        slot["partial"] = "none"
        slot["ratio"] = 0.0
        slot["is_enhance"] = False
    elif is_enhance is True and fast:
        enhanced = _as_int(seg.get("enhanced_cnt")) or 0
        interpolated = _as_int(seg.get("interpolated_cnt")) or 0
        total = enhanced + interpolated
        ratio = (enhanced / total) if total > 0 else 0.0
        if ratio <= 0 or planned in (None, 0):
            slot["level"] = 0
            slot["partial"] = "none"
            slot["ratio"] = 0.0
            slot["is_enhance"] = False
        elif ratio >= 1:
            slot["level"] = planned
            slot["partial"] = "none"
            slot["ratio"] = 1.0
            slot["is_enhance"] = True
        else:
            slot["level"] = planned
            slot["partial"] = "fast_complete"
            slot["ratio"] = ratio
            slot["is_enhance"] = True
            slot["enhanced_cnt"] = enhanced
            slot["played_frames"] = total
            interval = _as_int(seg.get("enhance_frame_interval"))
            slot["enhance_frame_interval"] = 0 if interval is None else interval
    elif is_enhance is True:
        slot["level"] = planned if planned is not None else int(slot.get("level") or 0)
        slot["partial"] = "none"
        slot["ratio"] = 1.0 if slot["level"] > 0 else 0.0
        slot["is_enhance"] = True
    real = seg.get("real_vmaf")
    if isinstance(real, (int, float)):
        slot["real_vmaf"] = float(real)
    slot["outcome_source"] = "analyzer"
    index = _as_int(slot.get("index"))
    if index is not None:
        slot["quality_index"] = index - int(segment_origin)


def _apply_frames(slot: dict[str, Any], rows: list[dict[str, Any]], model_default: str) -> None:
    # Analyzer / live-playing outcomes already encode what was played. A shared
    # or stale frame archive must not revive enhance credit on aborted segments.
    if slot.get("outcome_source") == "analyzer" or (
        slot.get("outcome_source") == "telemetry_playing" and slot.get("is_enhance") is False
    ):
        slot["frames"] = rows
        return
    slot["frames"] = rows
    reprs = [_as_int(r.get("representation")) for r in rows]
    reprs = [r for r in reprs if r is not None]
    if reprs:
        slot["repr_index"] = _majority(reprs)
    models = [str(r.get("model") or "") for r in rows if r.get("model")]
    if models:
        slot["model"] = normalize_model(_majority(models))
    elif model_default:
        slot["model"] = normalize_model(model_default)
    scales = [str(r.get("scale") or "") for r in rows if r.get("scale")]
    if scales:
        slot["scale"] = _majority(scales)
    enhanced_flags = [bool(r.get("enhanced")) for r in rows]
    if not enhanced_flags:
        return
    enhanced_levels = []
    for row, flag in zip(rows, enhanced_flags):
        if not flag:
            continue
        level = _as_int(row.get("enhance_level"))
        if level is not None and level > 0:
            enhanced_levels.append(level)
    ratio = sum(1 for flag in enhanced_flags if flag) / len(enhanced_flags)
    if enhanced_levels and ratio < 1:
        slot["level"] = _majority(enhanced_levels)
        slot["ratio"] = ratio
        slot["partial"] = "fast_complete"
        slot["is_enhance"] = True
        slot["outcome_source"] = "frames"
    elif enhanced_levels:
        slot["level"] = _majority(enhanced_levels)
        slot["ratio"] = 1.0
        slot["partial"] = "none"
        slot["is_enhance"] = True
        slot["outcome_source"] = "frames"
    else:
        slot["level"] = 0
        slot["ratio"] = 0.0
        slot["partial"] = "none"
        slot["is_enhance"] = False
        slot["outcome_source"] = "frames"


def _blank(index: int, model_default: str) -> dict[str, Any]:
    return {
        "index": index,
        "repr_index": 0,
        "level": 0,
        "planned_level": None,
        "model": normalize_model(model_default),
        "scale": "",
        "partial": "none",
        "ratio": None,
        "enhanced_cnt": None,
        "played_frames": None,
        "enhance_frame_interval": None,
        "frames": [],
        "real_vmaf": None,
        "is_enhance": None,
        "abort_reason": None,
        "outcome_source": None,
    }


def _segment_index(sample: dict[str, Any]) -> Optional[int]:
    for key in ("segment_index", "playing_segment"):
        value = _as_int(sample.get(key))
        if value is not None:
            return value
    return None


def _action_value(action: Any, segment_index: int) -> Optional[int]:
    """Read the action for ``segment_index`` only. Never borrow another segment's value."""
    if isinstance(action, dict) and action:
        for key in (str(segment_index), segment_index):
            if key in action:
                return _as_int(action[key])
        return None
    return _as_int(action)


def _model_from_sample(sample: dict[str, Any]) -> str:
    for bucket in ("enhanced_ready_inventory", "enhance_pending_inventory"):
        rows = sample.get(bucket)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and row.get("model"):
                    return str(row["model"])
    return ""


def _majority(values: list[Any]) -> Any:
    best = values[0]
    best_n = -1
    for value in values:
        count = sum(1 for item in values if item == value)
        if count > best_n:
            best = value
            best_n = count
    return best


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None or isinstance(value, dict):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
