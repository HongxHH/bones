"""Background per-frame index and JPEG archive. The render thread only enqueues."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path

import cv2
import numpy as np


class FrameArchive:
    def __init__(self, directory: str, *, jpeg_quality: int = 98) -> None:
        self.root = Path(directory)
        self.frame_dir = self.root / "frames"
        self.frame_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "frames.jsonl"
        self.quality = jpeg_quality
        self.dropped = 0
        self._next = 0
        self._written = 0
        self._q: queue.Queue = queue.Queue(maxsize=64)
        self._params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        # 4:4:4 keeps enhancement detail that 4:2:0 sampling would blur.
        factor = getattr(cv2, "IMWRITE_JPEG_SAMPLING_FACTOR", None)
        if factor is not None:
            self._params.extend([int(factor), 0x111111])
        self._thread = threading.Thread(target=self._loop, name="frame-archive", daemon=True)
        self._thread.start()

    def submit(self, frame_bgr: np.ndarray, meta: dict, baseline_bgr: np.ndarray | None = None) -> int | None:
        # Playback calls this on the render coroutine. A full queue means the
        # JPEG writer is behind; copying the frame and touching disk here would
        # stall the picture without ever entering BUFFERING.
        if self._q.full():
            self.dropped += 1
            return None
        frame_no = self._next
        row = dict(meta)
        row["frame"] = frame_no
        row["width"] = int(frame_bgr.shape[1])
        row["height"] = int(frame_bgr.shape[0])
        row["has_baseline"] = baseline_bgr is not None
        try:
            self._q.put_nowait((
                frame_no,
                np.ascontiguousarray(frame_bgr).copy(),
                None if baseline_bgr is None else np.ascontiguousarray(baseline_bgr).copy(),
                row,
            ))
        except queue.Full:
            self.dropped += 1
            return None
        self._next += 1
        return frame_no

    def close(self) -> None:
        try:
            self._q.put(None, timeout=1)
        except queue.Full:
            return
        self._thread.join(timeout=3)

    def _write_status(self) -> None:
        (self.root / "count.txt").write_text(str(self._written), encoding="ascii")
        (self.root / "dropped.txt").write_text(str(self.dropped), encoding="ascii")

    def _loop(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._write_status()
                return
            frame_no, frame, baseline, row = item
            ok, encoded = cv2.imencode(".jpg", frame, self._params)
            if not ok:
                continue
            (self.frame_dir / f"{frame_no:06d}.jpg").write_bytes(encoded.tobytes())
            if baseline is not None:
                ok_base, encoded_base = cv2.imencode(".jpg", baseline, self._params)
                if ok_base:
                    (self.frame_dir / f"{frame_no:06d}.base.jpg").write_bytes(encoded_base.tobytes())
                else:
                    row["has_baseline"] = False
            with self.index_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            self._written += 1
            # Status files are for observers. Rewriting them on every frame
            # keeps the disk busier than the render clock.
            if self._written == 1 or self._written % 15 == 0:
                self._write_status()
