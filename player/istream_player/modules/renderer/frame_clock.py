"""Per-frame presentation timing. The render coroutine only enqueues a small row."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path


class FrameClock:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.dropped = 0
        self._q: queue.Queue = queue.Queue(maxsize=8192)
        self._thread = threading.Thread(target=self._loop, name="frame-clock", daemon=True)
        self._thread.start()

    def write(self, row: dict) -> None:
        try:
            self._q.put_nowait(row)
        except queue.Full:
            self.dropped += 1

    def close(self) -> None:
        try:
            self._q.put(None, timeout=1)
        except queue.Full:
            return
        self._thread.join(timeout=3)

    def _loop(self) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            pending = 0
            while True:
                item = self._q.get()
                if item is None:
                    if self.dropped:
                        handle.write(
                            json.dumps({"kind": "clock_dropped", "n": self.dropped}, separators=(",", ":"))
                            + "\n"
                        )
                    handle.flush()
                    return
                handle.write(json.dumps(item, separators=(",", ":")) + "\n")
                pending += 1
                if pending >= 30:
                    handle.flush()
                    pending = 0
