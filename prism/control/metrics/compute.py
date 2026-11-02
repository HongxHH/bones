"""Compute backend: reference + distorted video → VMAF series → catalog write-back.

JPEG frame archives are not treated as a true distorted source. Scoring them
would bake a second compression into the number. Callers must pass a real
distorted video, or keep using a precomputed table.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Optional

from .catalog import QualityAsset, QualityCatalog, QualityKey, _now, _valid, normalize_model


def parse_vmaf_log(path: Path) -> tuple[list[float], Optional[float]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return [], None
    return parse_vmaf_document(payload)


def parse_vmaf_document(payload: dict) -> tuple[list[float], Optional[float]]:
    frames: list[float] = []
    for row in payload.get("frames") or []:
        if not isinstance(row, dict):
            continue
        metrics = row.get("metrics") or {}
        value = _valid(metrics.get("vmaf"))
        if value is not None:
            frames.append(value)
    mean = _valid(((payload.get("pooled_metrics") or {}).get("vmaf") or {}).get("mean"))
    return frames, mean


def run_compute(catalog: QualityCatalog, ctx: dict[str, Any]) -> dict[str, Any]:
    reference = str(ctx.get("reference") or "")
    distorted = str(ctx.get("distorted") or "")
    content_id = str(ctx.get("content_id") or "unknown")
    attempt = f"{content_id}|{reference}|{distorted}"
    if attempt in catalog._compute_attempted:
        return {"status": "cached", "reason": "", "written": 0, "pooled_mean": None}
    catalog._compute_attempted.add(attempt)
    if not reference or not Path(reference).is_file():
        return {"status": "skipped", "reason": "reference video missing", "written": 0, "pooled_mean": None}
    if not distorted or distorted == "playback_frames":
        return {
            "status": "skipped",
            "reason": "playback JPEG frames are not a VMAF distorted source; set quality.compute.distorted to a video file",
            "written": 0,
            "pooled_mean": None,
        }
    if not Path(distorted).is_file():
        return {"status": "skipped", "reason": "distorted video missing", "written": 0, "pooled_mean": None}
    log_path = catalog.store_dir / "_logs" / f"{QualityKey(content_id=content_id).slug()}-{abs(hash(attempt)) & 0xfffffff:x}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    outcome = _run_ffmpeg(reference, distorted, log_path)
    if outcome.get("status") != "ok":
        return {**outcome, "written": 0, "pooled_mean": None}
    frames, mean = parse_vmaf_log(log_path)
    written = apply_vmaf_scores(
        catalog,
        content_id=content_id,
        model=str(ctx.get("model") or "imdn"),
        frames=frames,
        pooled_mean=mean,
        observations=list(ctx.get("observations") or []),
        provenance={
            "source": "libvmaf",
            "tool": "ffmpeg",
            "created_at": _now(),
            "reference": reference,
            "distorted": distorted,
            "log": str(log_path),
            "approx": False,
        },
    )
    return {"status": "ok", "reason": "", "written": written, "pooled_mean": mean, "log": str(log_path)}


def apply_vmaf_scores(
    catalog: QualityCatalog,
    *,
    content_id: str,
    model: str,
    frames: list[float],
    pooled_mean: Optional[float],
    observations: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> int:
    model = normalize_model(model)
    written = 0
    if frames and observations:
        cursor = 0
        groups: dict[tuple, list[tuple[int, list[float]]]] = {}
        for obs in observations:
            count = len(obs.get("frames") or [])
            if count <= 0:
                count = 1 if not any(o.get("frames") for o in observations) and len(frames) == len(observations) else 0
            if count <= 0:
                continue
            chunk = frames[cursor : cursor + count]
            cursor += count
            if len(chunk) != count:
                break
            ident = (
                int(obs.get("repr_index") or 0),
                int(obs.get("level") or 0),
                str(obs.get("partial") or "none"),
                str(obs.get("scale") or ""),
            )
            groups.setdefault(ident, []).append((int(obs.get("index") or 0), chunk))
        if cursor == len(frames) and groups:
            for (repr_index, level, partial, scale), chunks in groups.items():
                by_frame = {str(index): values for index, values in chunks}
                catalog.write_asset(
                    QualityAsset(
                        key=QualityKey(
                            content_id=content_id,
                            granularity="segment",
                            repr_index=repr_index,
                            model=model,
                            level=level,
                            partial=partial,
                            scale=scale,
                        ),
                        by_frame=by_frame,
                        provenance=dict(provenance),
                    )
                )
                written += 1
            return written
    if pooled_mean is not None and len(observations) == 1:
        obs = observations[0]
        catalog.write_asset(
            QualityAsset(
                key=QualityKey(
                    content_id=content_id,
                    repr_index=int(obs.get("repr_index") or 0),
                    model=model,
                    level=int(obs.get("level") or 0),
                    partial=str(obs.get("partial") or "none"),
                    scale=str(obs.get("scale") or ""),
                ),
                scalar=float(pooled_mean),
                provenance=dict(provenance),
            )
        )
        return 1
    return written


def _run_ffmpeg(reference: str, distorted: str, log_path: Path) -> dict[str, Any]:
    ffmpeg = _resolve_ffmpeg()
    if not ffmpeg:
        return {"status": "skipped", "reason": "ffmpeg not found"}
    lavfi = (
        "[0:v]setpts=PTS-STARTPTS[dist];"
        "[1:v]setpts=PTS-STARTPTS[ref];"
        f"[dist][ref]libvmaf=log_fmt=json:log_path={log_path.as_posix()}:n_threads=2"
    )
    cmd = [ffmpeg, "-hide_banner", "-y", "-i", distorted, "-i", reference, "-lavfi", lavfi, "-f", "null", "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=60 * 30, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "skipped", "reason": f"ffmpeg/libvmaf unavailable: {exc}"}
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace")[-500:]
        return {"status": "skipped", "reason": f"ffmpeg/libvmaf unavailable: {err}"}
    return {"status": "ok"}


def _resolve_ffmpeg() -> str:
    for env_name in ("PRISM_FFMPEG", "BONES_FFMPEG"):
        env = os.environ.get(env_name)
        if env and Path(env).exists():
            return env
    return shutil.which("ffmpeg") or ""
