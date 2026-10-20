"""StallTracker / timeline semantics — shared by Exporter and Analyzer."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from istream_player.models import State
from istream_player.modules.status.stall_tracker import (
    StallTracker,
    stalls_from_state_timeline,
)


def test_timeline_buffering_to_ready():
    states = [
        (0.0, State.READY, 0.0),
        (1.0, State.BUFFERING, 1.0),
        (3.5, State.READY, 1.0),
        (4.0, State.PLAYING, 1.2),
    ]
    stalls, total, n = stalls_from_state_timeline(states)
    assert n == 1
    assert abs(total - 2.5) < 1e-9
    assert stalls[0].time_start == 1.0
    assert stalls[0].time_end == 3.5


def test_timeline_ends_on_any_non_buffering_not_only_ready():
    """Regression: Analyzer used to ignore BUFFERING→END (never closed)."""
    states = [
        (0.0, State.BUFFERING, 0.0),
        (2.0, State.END, 0.0),
    ]
    stalls, total, n = stalls_from_state_timeline(states)
    assert n == 1
    assert abs(total - 2.0) < 1e-9


def test_timeline_close_open_at():
    states = [(1.0, State.BUFFERING, 0.0)]
    stalls, total, n = stalls_from_state_timeline(states, close_open_at=4.0)
    assert n == 1
    assert abs(total - 3.0) < 1e-9


def test_loading_is_not_stall():
    states = [
        (0.0, State.LOADING, 0.0),
        (5.0, State.READY, 0.0),
        (6.0, State.BUFFERING, 1.0),
        (7.0, State.READY, 1.0),
    ]
    stalls, total, n = stalls_from_state_timeline(states)
    assert n == 1
    assert abs(total - 1.0) < 1e-9


def test_live_tracker_accumulates_with_monotonic():
    t = StallTracker()
    t.note_state(State.READY, now_mono=100.0)
    t.note_state(State.BUFFERING, now_mono=101.0)
    assert t.active
    assert abs(t.total_s(now_mono=101.5) - 0.5) < 1e-9
    t.note_state(State.READY, now_mono=103.0)
    assert not t.active
    assert t.count == 1
    assert abs(t.total_s(now_mono=200.0) - 2.0) < 1e-9
    t.note_state(State.BUFFERING, now_mono=110.0)
    t.note_state(State.END, now_mono=110.25)
    assert t.count == 2
    assert abs(t.total_s() - 2.25) < 1e-9


def test_live_tracker_records_rel_intervals():
    t = StallTracker()
    t.note_state(State.BUFFERING, t_rel=1.0, now_mono=10.0)
    t.note_state(State.READY, t_rel=3.0, now_mono=12.0)
    assert len(t.completed) == 1
    assert t.completed[0].time_start == 1.0
    assert t.completed[0].time_end == 3.0
