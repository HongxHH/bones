# -*- coding: utf-8 -*-
"""D1 unit checks for tool_greedy NES (no GPU play)."""
import numpy as np

from istream_player.modules.enhancer.tool_ids import (
    LEGAL_BY_DOWNLOAD,
    TOOL_NONE,
    TOOL_RESTORE_SHARP,
    TOOL_SR_HIGH,
    TOOL_SR_X2_MED,
)
from istream_player.modules.nes.tool_greedy import ToolGreedyController


class _FakeEnhancer:
    def __init__(self, q, t):
        self._q = q
        self._t = t

    def get_quality_table(self):
        return self._q

    def get_latency_table(self, segment_index=None):
        return self._t


def test_legal_map_hd_has_restore_and_x2():
    assert TOOL_SR_X2_MED in LEGAL_BY_DOWNLOAD[3]
    assert TOOL_RESTORE_SHARP in LEGAL_BY_DOWNLOAD[3]
    assert TOOL_SR_HIGH not in LEGAL_BY_DOWNLOAD[3]
    assert TOOL_RESTORE_SHARP in LEGAL_BY_DOWNLOAD[4]
    assert TOOL_SR_X2_MED not in LEGAL_BY_DOWNLOAD[4]


def test_pick_prefers_restore_on_720_when_feasible():
    ctl = ToolGreedyController()
    ctl.enhance_safety = 0.9
    ctl.seg_time = 4000.0  # ms
    ctl.buff_down = 20000.0
    ctl.buff_enh = 2000.0
    ctl.tool_ignore_slack = False
    ctl.tool_mask_hd = False

    # latency table already in "segment seconds" as Always returns
    # after *1000/safety → ms compared to seg_time
    # Use small T so all feasible; Q: restore > x2 > none
    q = np.full((5, 6), -np.inf)
    t = np.full((5, 6), 10.0)  # 10s segment work → 10*1000/0.9 ms >> 4000 → infeasible!
    # Make HD tools fast
    t[:, :] = 0.5  # 0.5s → ~555ms < 4000
    for r in range(5):
        q[r, TOOL_NONE] = 80.0
    q[3, TOOL_SR_X2_MED] = 82.0
    q[3, TOOL_RESTORE_SHARP] = 85.0
    ctl.enhancer = _FakeEnhancer(q, t)

    pick = ctl._pick_tool(3, index=40)
    assert pick == TOOL_RESTORE_SHARP


def test_pick_none_when_slack_tight():
    ctl = ToolGreedyController()
    ctl.enhance_safety = 0.9
    ctl.seg_time = 4000.0
    ctl.buff_down = 5000.0
    ctl.buff_enh = 4000.0  # slack = 5000-4000-4000 < 0
    ctl.tool_ignore_slack = False
    ctl.tool_mask_hd = False

    q = np.full((5, 6), -np.inf)
    t = np.full((5, 6), 2.0)
    q[3, TOOL_NONE] = 80.0
    q[3, TOOL_RESTORE_SHARP] = 99.0
    ctl.enhancer = _FakeEnhancer(q, t)

    pick = ctl._pick_tool(3, index=40)
    assert pick == TOOL_NONE


def test_ignore_slack_picks_restore_when_tight():
    ctl = ToolGreedyController()
    ctl.enhance_safety = 0.9
    ctl.seg_time = 4000.0
    ctl.buff_down = 5000.0
    ctl.buff_enh = 4000.0
    ctl.tool_ignore_slack = True
    ctl.tool_mask_hd = False

    q = np.full((5, 6), -np.inf)
    t = np.full((5, 6), 2.0)
    q[3, TOOL_NONE] = 80.0
    q[3, TOOL_RESTORE_SHARP] = 99.0
    ctl.enhancer = _FakeEnhancer(q, t)

    pick = ctl._pick_tool(3, index=40)
    assert pick == TOOL_RESTORE_SHARP


def test_mask_hd_hides_restore():
    ctl = ToolGreedyController()
    ctl.enhance_safety = 0.9
    ctl.seg_time = 4000.0
    ctl.buff_down = 20000.0
    ctl.buff_enh = 2000.0
    ctl.tool_ignore_slack = False
    ctl.tool_mask_hd = True
    ctl.tool_slack_kappa = 1.0

    q = np.full((5, 6), -np.inf)
    t = np.full((5, 6), 0.5)
    q[3, TOOL_NONE] = 80.0
    q[3, TOOL_SR_X2_MED] = 82.0
    q[3, TOOL_RESTORE_SHARP] = 99.0
    ctl.enhancer = _FakeEnhancer(q, t)

    pick = ctl._pick_tool(3, index=40)
    assert pick == TOOL_NONE


def test_slack_kappa_zero_allows_when_gap_beats_t():
    """kappa=0: slack = buff_down - buff_enh; no full-segment withholding."""
    ctl = ToolGreedyController()
    ctl.enhance_safety = 0.9
    ctl.seg_time = 4000.0
    ctl.buff_down = 5000.0
    ctl.buff_enh = 4000.0  # gap=1s; kappa=1 → slack<0; kappa=0 → slack=1s
    ctl.tool_ignore_slack = False
    ctl.tool_mask_hd = False
    ctl.tool_slack_kappa = 0.0

    q = np.full((5, 6), -np.inf)
    t = np.full((5, 6), 0.5)  # ~555ms < 1s gap
    q[3, TOOL_NONE] = 80.0
    q[3, TOOL_RESTORE_SHARP] = 99.0
    ctl.enhancer = _FakeEnhancer(q, t)

    pick = ctl._pick_tool(3, index=40)
    assert pick == TOOL_RESTORE_SHARP
    assert ctl.last_pick_debug is not None
    assert ctl.last_pick_debug["kappa"] == 0.0
