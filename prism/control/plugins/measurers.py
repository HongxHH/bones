from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path
from typing import Any


class NoopMeasurer:
    name = "noop"
    labels = ["cpu", "measure"]
    steps = ["measure"]

    async def measure(self, *, publication_dir: str, params: dict[str, Any]) -> dict[str, Any]:
        return {"status": "skipped", "reason": "noop measurer", "publication_dir": publication_dir}


class VmafMeasurer:
    """libvmaf via ffmpeg, aligned with workspace gold-standard habits:

    - continuous time-order inputs (no frame-dropping packing)
    - n_threads=2
    - BONES_FFMPEG preferred over PATH stubs
    """

    name = "vmaf"
    labels = ["cpu", "measure"]
    steps = ["measure"]

    def __init__(self, ffmpeg_bin: str = "ffmpeg"):
        self.ffmpeg_bin = ffmpeg_bin

    def _resolve_ffmpeg(self) -> str:
        env = os.environ.get("BONES_FFMPEG")
        if env and Path(env).exists():
            return env
        which = shutil.which(self.ffmpeg_bin)
        return which or self.ffmpeg_bin

    async def measure(self, *, publication_dir: str, params: dict[str, Any]) -> dict[str, Any]:
        ffmpeg = self._resolve_ffmpeg()
        ref = params.get("reference")
        dist = params.get("distorted")
        # Optional: pick a representation file from publication tree
        if not dist and params.get("distorted_glob"):
            matches = sorted(Path(publication_dir).glob(params["distorted_glob"]))
            if matches:
                dist = str(matches[0])
        if not ref or not dist:
            return {
                "status": "skipped",
                "reason": "vmaf requires params.reference and params.distorted (or distorted_glob)",
            }
        if not Path(ref).exists() or not Path(dist).exists():
            return {"status": "skipped", "reason": "reference or distorted file missing"}

        out_dir = Path(params.get("metrics_dir") or publication_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        log_path = out_dir / "vmaf.json"
        duration = params.get("duration_s")
        # Gold-standard-ish: reset PTS then libvmaf; keep both streams in display order.
        lavfi = (
            "[0:v]setpts=PTS-STARTPTS[dist];"
            "[1:v]setpts=PTS-STARTPTS[ref];"
            f"[dist][ref]libvmaf=log_fmt=json:log_path={log_path.as_posix()}:n_threads=2"
        )
        cmd = [ffmpeg, "-hide_banner", "-y"]
        if duration:
            cmd.extend(["-t", str(duration)])
        cmd.extend(["-i", dist])
        if duration:
            cmd.extend(["-t", str(duration)])
        cmd.extend(["-i", ref, "-lavfi", lavfi, "-f", "null", "-"])
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace")[-2000:]
            return {"status": "skipped", "reason": f"ffmpeg/libvmaf unavailable: {err}", "cmd": cmd}
        mean = None
        if log_path.exists():
            try:
                payload = json.loads(log_path.read_text(encoding="utf-8"))
                mean = payload.get("pooled_metrics", {}).get("vmaf", {}).get("mean")
            except json.JSONDecodeError:
                mean = None
        return {
            "status": "ok",
            "vmaf_mean": mean,
            "log": str(log_path),
            "reference": ref,
            "distorted": dist,
            "pipeline": "setpts+libvmaf_n_threads=2",
        }
