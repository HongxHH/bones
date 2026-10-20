"""Latest JPEG preview per run arm. In-memory only; the instrument pushes frames."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass


@dataclass
class PreviewFrame:
    jpeg: bytes
    updated_at: float


_lock = threading.Lock()
_frames: dict[tuple[str, str], PreviewFrame] = {}
_MAX_BYTES = 1_500_000


def put_preview(run_id: str, arm_id: str, jpeg: bytes) -> None:
    if not run_id or not arm_id:
        raise ValueError("run_id and arm_id required")
    if not jpeg or len(jpeg) < 32:
        raise ValueError("empty preview")
    if len(jpeg) > _MAX_BYTES:
        raise ValueError("preview too large")
    if jpeg[:2] != b"\xff\xd8":
        raise ValueError("preview must be jpeg")
    with _lock:
        _frames[(run_id, arm_id)] = PreviewFrame(jpeg=jpeg, updated_at=time.time())


def get_preview(run_id: str, arm_id: str) -> bytes | None:
    with _lock:
        frame = _frames.get((run_id, arm_id))
        return frame.jpeg if frame else None


def clear_preview(run_id: str | None = None) -> None:
    with _lock:
        if run_id is None:
            _frames.clear()
            return
        for key in [k for k in _frames if k[0] == run_id]:
            _frames.pop(key, None)
