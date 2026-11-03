"""Read ``data/quality`` frame-level VMAF and expose segment / clip aggregates.

These files are measured libvmaf of bicubic or IMDN versus the original 1080p.
Playback decisions do not use this module; Lab scoring does.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

# Representation index in the player ladder → resolution token in data/quality.
_REPR_RES = {0: "240", 1: "360", 2: "480", 3: "720"}
_LEVEL_METHOD = {0: "bicubic", 1: "sr_low", 2: "sr_medium", 3: "sr_high"}
# Native 1080p is the reference itself. Matches the legacy static table.
_REF_VMAF = 100.0
_ILLEGAL = -1.0


def quality_segment_index(mpd_index: int, origin: int = 1) -> int:
    """Map an MPD segment number onto ``data/quality``.

    The files slice the video from frame 0. BBB and TOS templates start at 1,
    so MPD segment 1 is quality segment 0.
    """
    return int(mpd_index) - int(origin)


class FrameQualityIndex:
    """Lazy, process-local cache of ``data/quality/{content}_{res}.json``."""

    def __init__(self, repo_root: Path):
        self.repo_root = Path(repo_root)
        self._docs: dict[tuple[str, str], Optional[dict]] = {}

    def has_content(self, content_id: str, model: str = "imdn") -> bool:
        if not _is_imdn(model):
            return False
        return any(self._doc(content_id, res) is not None for res in _REPR_RES.values())

    def static_matrix(self, content_id: str, model: str = "imdn") -> Optional[list]:
        """5×5 clip means. Illegal cells are ``-1``. ``None`` if this content has no files."""
        if not self.has_content(content_id, model):
            return None
        matrix: list[list[float]] = []
        for repr_index in range(5):
            row = []
            for level in range(5):
                value = self._clip_mean(content_id, repr_index, level)
                row.append(_ILLEGAL if value is None else value)
            matrix.append(row)
        return matrix

    def segment_mean(
        self,
        content_id: str,
        repr_index: int,
        level: int,
        segment_index: int,
        model: str = "imdn",
    ) -> Optional[float]:
        frames = self._frames(content_id, repr_index, level, model)
        if frames is None:
            return self._reference_cell(repr_index, level)
        if isinstance(frames, float):
            return frames
        doc = self._doc_for_repr(content_id, repr_index)
        if doc is None:
            return None
        width = int(doc.get("frames_per_seg") or 120)
        start = int(segment_index) * width
        if start < 0 or start >= len(frames):
            return None
        chunk = frames[start : start + width]
        if not chunk:
            return None
        return sum(chunk) / len(chunk)

    def segment_frames(
        self,
        content_id: str,
        repr_index: int,
        level: int,
        segment_index: int,
        model: str = "imdn",
    ) -> Optional[list[float]]:
        """Measured frames for one segment. Illegal cells and the 1080 reference are None."""
        raw = self._frames(content_id, repr_index, level, model)
        if not isinstance(raw, list):
            return None
        return self._segment_frames(content_id, repr_index, level, segment_index, model)

    def cell_for_decision(
        self,
        content_id: str,
        repr_index: int,
        level: int,
        segment_index: int,
        model: str = "imdn",
    ) -> Optional[float]:
        """Segment mean, or the clip mean when this segment has no frames.

        Illegal cells (no segment frames and no clip mean) stay None.
        """
        value = self.segment_mean(content_id, repr_index, level, segment_index, model)
        if value is not None:
            return value
        return self._clip_mean(content_id, repr_index, level)

    def decision_matrix(
        self,
        content_id: str,
        segment_index: int,
        model: str = "imdn",
    ) -> Optional[list]:
        """5×5 means for a BONES decision. Illegal cells are None."""
        if not self.has_content(content_id, model):
            return None
        matrix: list[list[Optional[float]]] = []
        for repr_index in range(5):
            row = [
                self.cell_for_decision(content_id, repr_index, level, segment_index, model)
                for level in range(5)
            ]
            matrix.append(row)
        return matrix

    def played_score(
        self,
        content_id: str,
        repr_index: int,
        level: int,
        segment_index: int,
        *,
        enhanced_frames: Optional[int] = None,
        interval: int = 0,
        ratio: Optional[float] = None,
        model: str = "imdn",
    ) -> Optional[dict]:
        """Playback VMAF using the same ruler as Lab.

        Full enhance uses the segment cell. A prefix fast-complete uses the
        spliced frames. Interval enhance blends the bicubic and enhance cells.
        """
        if not self.has_content(content_id, model):
            return None
        base = self.cell_for_decision(content_id, repr_index, 0, segment_index, model)
        if base is None:
            return None
        if int(level) <= 0:
            return {"real": base, "base": base, "full": base}
        full = self.cell_for_decision(content_id, repr_index, level, segment_index, model)
        if full is None:
            return {"real": base, "base": base, "full": base}
        if enhanced_frames is not None and int(interval or 0) <= 0:
            played = self.playback_mean(
                content_id, repr_index, level, segment_index, int(enhanced_frames), model
            )
            if played is not None:
                return {"real": played, "base": base, "full": full}
        if ratio is not None and 0 < float(ratio) < 1:
            mixed = base * (1 - float(ratio)) + full * float(ratio)
            return {"real": mixed, "base": base, "full": full}
        return {"real": full, "base": base, "full": full}

    def playback_mean(
        self,
        content_id: str,
        repr_index: int,
        level: int,
        segment_index: int,
        enhanced_frames: int,
        model: str = "imdn",
    ) -> Optional[float]:
        """Mean of a prefix-enhanced segment: SR frames first, then bicubic.

        ``None`` when either series is missing. ``enhanced_frames`` is clamped
        to the segment length.
        """
        enhanced = self._segment_frames(content_id, repr_index, level, segment_index, model)
        plain = self._segment_frames(content_id, repr_index, 0, segment_index, model)
        if enhanced is None or plain is None:
            return None
        width = min(len(enhanced), len(plain))
        if width <= 0:
            return None
        keep = max(0, min(int(enhanced_frames), width))
        mixed = enhanced[:keep] + plain[keep:width]
        if not mixed:
            return None
        return sum(mixed) / len(mixed)

    def frame_score(
        self,
        content_id: str,
        repr_index: int,
        level: int,
        segment_index: int,
        frame_in_segment: int,
        model: str = "imdn",
    ) -> Optional[float]:
        frames = self._frames(content_id, repr_index, level, model)
        if frames is None:
            return self._reference_cell(repr_index, level)
        if isinstance(frames, float):
            return frames
        doc = self._doc_for_repr(content_id, repr_index)
        if doc is None:
            return None
        width = int(doc.get("frames_per_seg") or 120)
        index = int(segment_index) * width + int(frame_in_segment)
        if index < 0 or index >= len(frames):
            return None
        return float(frames[index])

    def _clip_mean(self, content_id: str, repr_index: int, level: int) -> Optional[float]:
        frames = self._frames(content_id, repr_index, level, "imdn")
        if frames is None:
            return self._reference_cell(repr_index, level)
        if isinstance(frames, float):
            return frames
        if not frames:
            return None
        return sum(frames) / len(frames)

    def _segment_frames(
        self,
        content_id: str,
        repr_index: int,
        level: int,
        segment_index: int,
        model: str,
    ) -> Optional[list[float]]:
        frames = self._frames(content_id, repr_index, level, model)
        if isinstance(frames, float):
            return [frames]
        if not isinstance(frames, list):
            return None
        doc = self._doc_for_repr(content_id, repr_index)
        width = int((doc or {}).get("frames_per_seg") or 120)
        start = int(segment_index) * width
        if start < 0 or start >= len(frames):
            return None
        chunk = [float(value) for value in frames[start : start + width]]
        return chunk or None

    def _frames(self, content_id: str, repr_index: int, level: int, model: str):
        """Return a frame list, a constant reference score, or None when illegal / missing."""
        if not _is_imdn(model):
            return None
        if int(repr_index) == 4:
            if int(level) == 0:
                return _REF_VMAF
            return None
        method = _LEVEL_METHOD.get(int(level))
        res = _REPR_RES.get(int(repr_index))
        if method is None or res is None:
            return None
        doc = self._doc(content_id, res)
        if doc is None:
            return None
        methods = doc.get("methods") or {}
        slot = methods.get(method)
        if not isinstance(slot, dict):
            return None
        frames = slot.get("frames")
        if not isinstance(frames, list) or not frames:
            return None
        return frames

    def _reference_cell(self, repr_index: int, level: int) -> Optional[float]:
        if int(repr_index) == 4 and int(level) == 0:
            return _REF_VMAF
        return None

    def _doc_for_repr(self, content_id: str, repr_index: int) -> Optional[dict]:
        res = _REPR_RES.get(int(repr_index))
        if res is None:
            return None
        return self._doc(content_id, res)

    def _doc(self, content_id: str, res: str) -> Optional[dict]:
        key = (content_id, res)
        if key in self._docs:
            return self._docs[key]
        path = self.repo_root / "data" / "quality" / f"{content_id}_{res}.json"
        doc = None
        if path.is_file():
            try:
                doc = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                doc = None
        self._docs[key] = doc
        return doc


def _is_imdn(model: str) -> bool:
    text = (model or "").strip().lower()
    return not text or "imdn" in text
