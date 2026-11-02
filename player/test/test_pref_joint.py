# -*- coding: utf-8 -*-
"""Unit tests for pref_joint scoring (no GPU play)."""
import numpy as np

from istream_player.modules.enhancer.tool_ids import (
    TOOL_NONE,
    TOOL_RESTORE_SHARP,
    TOOL_SR_HIGH,
    TOOL_SR_MED,
    TOOL_SR_X2_MED,
)
from istream_player.modules.nes.pref_joint import (
    _DEFAULT_PARAMS,
    load_joint_params,
    pick_joint,
    plan_enhance_s,
    score_pair,
    taste_bonus_for,
)


def _tables():
    # 5x6 like multitask; illegal = -inf / huge T
    q = np.full((5, 6), -np.inf)
    t = np.full((5, 6), 1e6)
    for d in range(3):
        q[d, TOOL_NONE] = 50 + 10 * d
        q[d, TOOL_SR_MED] = 60 + 10 * d
        q[d, TOOL_SR_HIGH] = 65 + 10 * d
        t[d, TOOL_NONE] = 0.0
        t[d, TOOL_SR_MED] = 0.01
        t[d, TOOL_SR_HIGH] = 0.05
    q[3, TOOL_NONE] = 90.0
    q[3, TOOL_SR_X2_MED] = 93.0
    q[3, TOOL_RESTORE_SHARP] = 95.0
    t[3, TOOL_NONE] = 0.0
    t[3, TOOL_SR_X2_MED] = 0.01
    t[3, TOOL_RESTORE_SHARP] = 0.001
    q[4, TOOL_NONE] = 100.0
    t[4, TOOL_NONE] = 0.0
    bitrates = np.array([200_000, 400_000, 800_000, 1_500_000, 3_000_000], dtype=float)
    return q, t, bitrates


def test_load_params_has_courage_and_taste():
    p = load_joint_params()
    assert "balanced" in p["courage"]
    assert "sharp" in p["taste_bonus"]
    assert plan_enhance_s(0.01, TOOL_SR_HIGH, p) >= 2.0  # floor


def test_taste_bonus_sharp_vs_clean():
    p = _DEFAULT_PARAMS
    assert taste_bonus_for(TOOL_SR_HIGH, "sharp", p) == 3.0
    assert taste_bonus_for(TOOL_RESTORE_SHARP, "sharp", p) == 0.0
    assert taste_bonus_for(TOOL_RESTORE_SHARP, "clean", p) == 4.0


def test_budget_rejects_when_buffers_tight():
    p = _DEFAULT_PARAMS
    ok, sc, dbg = score_pair(
        q=80.0,
        download_s=3.0,
        enhance_s=2.0,
        buff_down_s=4.0,
        buff_enh_s=3.0,
        t_seg_s=4.0,
        courage="balanced",  # danger=4, relax=1 → budget = 1-4+1 = -2
        taste="balanced",
        tool=TOOL_SR_HIGH,
        params=p,
    )
    assert ok is False
    assert dbg["reject"] == "budget"


def test_aggressive_allows_more_than_conservative():
    q, t, br = _tables()
    # Mid buffers: conservative should fall back lighter; aggressive heavier
    common = dict(
        quality=q,
        latency=t,
        bitrates_bps=br,
        bw_bps=1_000_000,
        buff_down_s=8.0,
        buff_enh_s=2.0,
        t_seg_s=4.0,
        taste="balanced",
        params=_DEFAULT_PARAMS,
    )
    d_c, t_c, _ = pick_joint(courage="conservative", **common)
    d_a, t_a, _ = pick_joint(courage="aggressive", **common)
    # Aggressive should not pick a strictly lighter tool on same or higher rung
    # At least: conservative never more aggressive in (d, tool weight)
    weight = {TOOL_NONE: 0, TOOL_SR_MED: 2, TOOL_SR_HIGH: 3, TOOL_SR_X2_MED: 3, TOOL_RESTORE_SHARP: 3}
    assert weight.get(t_a, 0) >= weight.get(t_c, 0) or d_a >= d_c


def test_clean_taste_prefers_restore_on_720():
    q, t, br = _tables()
    # Plenty of budget so both x2 and restore feasible
    common = dict(
        quality=q,
        latency=t,
        bitrates_bps=br,
        bw_bps=5_000_000,
        buff_down_s=20.0,
        buff_enh_s=0.0,
        t_seg_s=4.0,
        courage="balanced",
        params=_DEFAULT_PARAMS,
    )
    _, tool_bal, _ = pick_joint(taste="balanced", **common)
    _, tool_clean, _ = pick_joint(taste="clean", **common)
    # clean should pick restore when comparing 720 options; overall may pick high Q
    # Force by checking score at d=3: clean restore beats x2
    assert taste_bonus_for(TOOL_RESTORE_SHARP, "clean", _DEFAULT_PARAMS) > 0
    # With clean, if 720 wins it should be restore; with high buffer, Q[4,none]=100 may win.
    # Narrow: only use first 4 rungs so 1080 none doesn't dominate
    q2 = q.copy()
    q2[4, :] = -np.inf
    _, tool_clean2, dbg = pick_joint(
        quality=q2,
        latency=t,
        bitrates_bps=br,
        bw_bps=5_000_000,
        buff_down_s=20.0,
        buff_enh_s=0.0,
        t_seg_s=4.0,
        courage="balanced",
        taste="clean",
        params=_DEFAULT_PARAMS,
    )
    assert tool_clean2 == TOOL_RESTORE_SHARP, dbg


def test_fallback_when_nothing_feasible():
    q, t, br = _tables()
    d, tool, dbg = pick_joint(
        quality=q,
        latency=t,
        bitrates_bps=br,
        bw_bps=50_000,
        buff_down_s=1.0,
        buff_enh_s=0.9,
        t_seg_s=4.0,
        courage="conservative",
        taste="balanced",
        params=_DEFAULT_PARAMS,
    )
    assert d == 0 and tool == TOOL_NONE
    assert dbg.get("fallback") is True
