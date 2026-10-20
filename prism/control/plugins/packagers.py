from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .mpdutil import parse_ladder_from_mpd
from ..schemas import PackageResult, RecipeOutput


class IdentityPackager:
    """Publication is already a DASH tree (import path)."""

    name = "identity"
    labels = ["cpu"]

    async def publish(self, *, artifact: RecipeOutput, output_dir: str, params: dict[str, Any]) -> PackageResult:
        if not artifact.dash_dir:
            raise ValueError("identity packager requires RecipeOutput.dash_dir")
        dash_dir = Path(artifact.dash_dir)
        dest = Path(output_dir)
        if dest.resolve() != dash_dir.resolve():
            if dest.exists():
                shutil.rmtree(dest)

            def _copy(src: str, dst: str) -> None:
                try:
                    os.link(src, dst)
                except OSError:
                    shutil.copy2(src, dst)

            shutil.copytree(dash_dir, dest, copy_function=_copy)
        mpd_name = artifact.mpd_name or params.get("mpd_name") or "manifest.mpd"
        mpd_path = dest / mpd_name
        if not mpd_path.exists():
            found = list(dest.glob("*.mpd"))
            if not found:
                raise FileNotFoundError(f"No MPD in {dest}")
            mpd_path = found[0]
            mpd_name = mpd_path.name
        ladder, duration = parse_ladder_from_mpd(mpd_path.read_text(encoding="utf-8"))
        return PackageResult(
            mpd_name=mpd_name,
            ladder=ladder,
            duration_s=duration,
            output_dir=str(dest),
            segment_duration=float(params.get("seg_duration") or artifact.seg_duration),
        )
