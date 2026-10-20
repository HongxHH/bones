"""One full-resolution RGB frame shared between the player and the Prism window."""

from __future__ import annotations

import struct
from multiprocessing import shared_memory

import numpy as np

MAGIC = b"PRFM"
HEADER = struct.Struct("<4sIII")
MAX_W = 3840
MAX_H = 2160
NAME = "PrismFrame"
# Second plane for the bicubic baseline. Absent or stale means "no compare pair".
BASE_NAME = NAME + "Base"
CAPACITY = HEADER.size + MAX_W * MAX_H * 3


class FrameSink:
    def __init__(self, name: str = NAME, *, create: bool = False) -> None:
        self.name = name
        self._shm: shared_memory.SharedMemory | None = None
        self._create = create

    def open(self) -> None:
        if self._shm is not None:
            return
        if self._create:
            try:
                existing = shared_memory.SharedMemory(name=self.name, create=False)
            except FileNotFoundError:
                existing = None
            if existing is not None and existing.size < CAPACITY:
                existing.close()
                existing.unlink()
                existing = None
            if existing is None:
                self._shm = shared_memory.SharedMemory(name=self.name, create=True, size=CAPACITY)
            else:
                self._shm = existing
            self._shm.buf[: HEADER.size] = HEADER.pack(MAGIC, 0, 0, 0)
            return
        self._shm = shared_memory.SharedMemory(name=self.name, create=False)

    def close(self) -> None:
        shm = self._shm
        self._shm = None
        if shm is None:
            return
        shm.close()
        if self._create:
            try:
                shm.unlink()
            except FileNotFoundError:
                pass

    def write(self, rgb: np.ndarray) -> bool:
        if self._shm is None:
            self.open()
        assert self._shm is not None
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            return False
        frame = rgb
        h, w = frame.shape[:2]
        if w > MAX_W or h > MAX_H:
            return False
        if frame.dtype != np.uint8:
            frame = frame.astype(np.uint8, copy=False)
        nbytes = w * h * 3
        packed = np.ascontiguousarray(frame).tobytes()
        buf = self._shm.buf
        _magic, _w, _h, seq = HEADER.unpack(buf[: HEADER.size])
        writing = (int(seq) + 1) | 1
        buf[: HEADER.size] = HEADER.pack(MAGIC, w, h, writing)
        buf[HEADER.size : HEADER.size + nbytes] = packed
        buf[: HEADER.size] = HEADER.pack(MAGIC, w, h, (writing + 1) & 0xFFFFFFFE)
        return True

    def read(self) -> tuple[int, np.ndarray] | None:
        if self._shm is None:
            self.open()
        assert self._shm is not None
        buf = self._shm.buf
        magic, w, h, seq = HEADER.unpack(buf[: HEADER.size])
        if magic != MAGIC or seq == 0 or (seq & 1) or w <= 0 or h <= 0 or w > MAX_W or h > MAX_H:
            return None
        nbytes = w * h * 3
        raw = bytes(buf[HEADER.size : HEADER.size + nbytes])
        magic2, w2, h2, seq2 = HEADER.unpack(buf[: HEADER.size])
        if magic2 != MAGIC or seq2 != seq or w2 != w or h2 != h:
            return None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3))
        return int(seq), frame
