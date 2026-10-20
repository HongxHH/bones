from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from ..schemas import RecipeOutput


def _link_or_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _copy_dash_bundle(mpd_path: Path, dst: Path) -> str:
    """Copy MPD + sibling stream* dirs only (skip source mp4 / scripts)."""
    dst.mkdir(parents=True, exist_ok=True)
    _link_or_copy_file(mpd_path, dst / mpd_path.name)
    parent = mpd_path.parent
    for child in sorted(parent.iterdir()):
        if not child.is_dir():
            continue
        if not (child.name.startswith("stream") or child.name.lower() in {"video", "audio"}):
            continue
        for file in child.rglob("*"):
            if file.is_file():
                _link_or_copy_file(file, dst / child.name / file.relative_to(child))
    return mpd_path.name


class ImportDashRecipe:
    """Copy an existing DASH tree (MPD + segments) into staging. No ffmpeg."""

    name = "import_dash"
    labels = ["cpu"]
    steps = ["import", "package"]
    default_packager = "identity"

    async def run(self, *, asset_path: str, staging_dir: str, params: dict[str, Any]) -> RecipeOutput:
        src = Path(asset_path)
        dst = Path(staging_dir) / "dash"
        if dst.exists():
            shutil.rmtree(dst)
        if src.is_file() and src.suffix.lower() == ".mpd":
            mpd_name = _copy_dash_bundle(src, dst)
        elif src.is_dir():
            mpds = list(src.glob("*.mpd"))
            if not mpds:
                raise FileNotFoundError(f"No MPD found under {src}")
            mpd_name = _copy_dash_bundle(mpds[0], dst)
        else:
            raise FileNotFoundError(f"DASH source not found: {src}")
        return RecipeOutput(kind="dash_tree", dash_dir=str(dst), mpd_name=mpd_name)
