"""Adapt ``data/quality`` into the ``results`` dict the frame-level simulators expect.

Keys match the legacy ``per_frame_quality.json`` layout:
``{resolution}_{base|sr_low|sr_medium|sr_high}_{segment}``.
Only full segments (``frames_per_seg`` long) are emitted. 720p has bicubic only,
so a 720p lookup that requires all four levels still misses and the simulator
keeps its segment-level fallback.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from prism.control.metrics.frame_quality import FrameQualityIndex

_RES = {0: 240, 1: 360, 2: 480, 3: 720}
_NAMES = {0: "base", 1: "sr_low", 2: "sr_medium", 3: "sr_high"}


def load_frame_results(repo_root: str | os.PathLike | None = None, content_id: str = "bbb") -> dict:
    root = Path(repo_root) if repo_root is not None else _REPO
    index = FrameQualityIndex(root)
    results: dict[str, list[float]] = {}
    segment = 0
    while segment < 10000:
        anchor = index.segment_frames(content_id, 0, 0, segment)
        if not anchor:
            break
        width = len(anchor)
        for repr_index, resolution in _RES.items():
            for level, name in _NAMES.items():
                frames = index.segment_frames(content_id, repr_index, level, segment)
                if frames and len(frames) == width:
                    results[f"{resolution}_{name}_{segment}"] = frames
        segment += 1
    return {"results": results}


def catalog_decision_table(
    segment_index: int,
    repo_root: str | os.PathLike | None = None,
    content_id: str = "bbb",
):
    """5×5 decision table. Missing segments use the clip mean; illegal cells are -inf."""
    root = Path(repo_root) if repo_root is not None else _REPO
    matrix = FrameQualityIndex(root).decision_matrix(content_id, int(segment_index))
    if matrix is None:
        return None
    table = np.full((len(matrix), len(matrix[0])), -np.inf, dtype=float)
    for row_index, row in enumerate(matrix):
        for level, value in enumerate(row):
            if value is not None:
                table[row_index, level] = float(value)
    return table


def load_per_frame_data(args, repo_root: str | os.PathLike | None = None) -> dict:
    """Old json when ``args.per_frame_path`` is set; otherwise ``data/quality``."""
    path = getattr(args, "per_frame_path", None)
    if path:
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)
    return load_frame_results(repo_root)
