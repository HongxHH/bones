# -*- coding: utf-8 -*-
"""Unit tests for phase-3 remaining-work recovery trigger (no GPU)."""
from __future__ import annotations

import time
from types import SimpleNamespace

from istream_player.modules.enhancer.enhancer_imdn_always import IMDNEnhancerAlways


def _enhancer(**kw):
    e = IMDNEnhancerAlways.__new__(IMDNEnhancerAlways)
    e._num_frames = kw.get("num_frames", 120)
    e.cnt = kw.get("cnt", 30)
    e.task_start = time.perf_counter() - kw.get("elapsed", 1.0)
    e._current_enhancing_index = 5
    e.recovery_work_margin_s = kw.get("margin", 1.0 / 30.0)
    e.log = SimpleNamespace(info=lambda *a, **k: None)
    until = kw.get("until", 0.5)
    e._time_until_play = lambda idx, seg: until
    return e


def test_work_deadline_triggers_when_need_exceeds_slack():
    # 30 frames in 1.0s → 33.3ms/frame; 90 left → need ≈ 3.0s; until=0.5 → trigger
    e = _enhancer(cnt=30, elapsed=1.0, until=0.5, num_frames=120)
    assert e._should_trigger_work_deadline(SimpleNamespace()) is True


def test_work_deadline_holds_when_plenty_of_time():
    # same speed but until=10s → no trigger
    e = _enhancer(cnt=30, elapsed=1.0, until=10.0, num_frames=120)
    assert e._should_trigger_work_deadline(SimpleNamespace()) is False


def test_work_deadline_false_before_any_frame():
    e = _enhancer(cnt=0, elapsed=0.0, until=0.1)
    e.task_start = time.perf_counter()
    assert e._should_trigger_work_deadline(SimpleNamespace()) is False
