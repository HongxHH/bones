"""Shared stall (rebuffer) accumulation for live status and end-of-run analysis.

Semantics (single source of truth):
- A stall starts when state becomes BUFFERING.
- A stall ends on the next transition to any non-BUFFERING state
  (READY, PLAYING, PAUSED, END, …). LOADING is not counted as stall.
- Live total includes an in-flight BUFFERING interval (monotonic clock).
- Offline timeline uses relative experiment seconds from the state log.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

from istream_player.models import State


@dataclass(frozen=True)
class StallInterval:
    time_start: float
    time_end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.time_end - self.time_start)


def is_buffering_state(state: Union[State, str, None]) -> bool:
    if state is None:
        return False
    if isinstance(state, State):
        return state == State.BUFFERING
    return str(state).upper() == "BUFFERING"


def stalls_from_state_timeline(
    states: Sequence[Tuple[float, Union[State, str], float]],
    *,
    close_open_at: Optional[float] = None,
) -> Tuple[List[StallInterval], float, int]:
    """Compute stalls from ``(t_rel, state, position)`` timeline.

    If the log ends while still BUFFERING and ``close_open_at`` is set,
    the open interval is closed at that time (and counted).
    """
    stalls: List[StallInterval] = []
    start: Optional[float] = None
    for t, state, _pos in states:
        if is_buffering_state(state) and start is None:
            start = float(t)
        elif not is_buffering_state(state) and start is not None:
            end = float(t)
            stalls.append(StallInterval(start, end))
            start = None
    if start is not None and close_open_at is not None and close_open_at >= start:
        stalls.append(StallInterval(start, float(close_open_at)))
        start = None
    total = sum(s.duration for s in stalls)
    return stalls, total, len(stalls)


class StallTracker:
    """Live stall accumulator (monotonic clock) with optional relative intervals."""

    def __init__(self) -> None:
        self._accum_s: float = 0.0
        self._started_mono: Optional[float] = None
        self._started_rel: Optional[float] = None
        self._completed: List[StallInterval] = []
        self._count: int = 0

    @property
    def active(self) -> bool:
        return self._started_mono is not None

    @property
    def count(self) -> int:
        """Completed stall count (in-flight not included)."""
        return self._count

    @property
    def completed(self) -> Sequence[StallInterval]:
        return self._completed

    def total_s(self, *, now_mono: Optional[float] = None) -> float:
        total = self._accum_s
        if self._started_mono is not None:
            now = time.monotonic() if now_mono is None else now_mono
            total += max(0.0, now - self._started_mono)
        return total

    def note_state(
        self,
        state: Union[State, str, None],
        *,
        t_rel: Optional[float] = None,
        now_mono: Optional[float] = None,
    ) -> None:
        now = time.monotonic() if now_mono is None else now_mono
        buffering = is_buffering_state(state)
        if buffering and self._started_mono is None:
            self._started_mono = now
            self._started_rel = t_rel
            return
        if not buffering and self._started_mono is not None:
            dur = max(0.0, now - self._started_mono)
            self._accum_s += dur
            if self._started_rel is not None and t_rel is not None:
                self._completed.append(StallInterval(self._started_rel, float(t_rel)))
            self._count += 1
            self._started_mono = None
            self._started_rel = None

    def close_open(
        self,
        *,
        t_rel: Optional[float] = None,
        now_mono: Optional[float] = None,
    ) -> None:
        """Force-close an in-flight stall (e.g. session teardown)."""
        if self._started_mono is None:
            return
        self.note_state(State.END, t_rel=t_rel, now_mono=now_mono)
