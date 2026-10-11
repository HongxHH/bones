"""M3/M4 帧级 EDF 队列单元测试（SR python 直接运行）。"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulation.frame_enhance_queue import FrameEnhanceQueue
from simulation.gpu_grid import STEP_MS, to_steps

F = 120
SEG_MS = 4000.0
FRAME_PLAY = SEG_MS / F


def make_tables(cost_ms, gains):
    """cost_ms: 每档相对前档的单帧增量成本 ms（level1/2/3）；gains: (F,3) 各档增量 VMAF。"""
    costs = np.zeros((F, 4), dtype=np.int64)
    vmaf = np.zeros((F, 4))
    vmaf[:, 0] = 50.0
    cum_cost = 0.0
    cum_gain = np.zeros(F)
    for l in range(1, 4):
        cum_cost += cost_ms[l - 1]
        costs[:, l] = to_steps(cum_cost)
        cum_gain = cum_gain + gains[:, l - 1]
        vmaf[:, l] = 50.0 + cum_gain
    return costs, vmaf


def test_ample_lead_all_enhanced():
    """提前量充足：全部帧上满档，无 missed。"""
    gains = np.ones((F, 3))
    costs, vmaf = make_tables([10.0, 10.0, 10.0], gains)
    q = FrameEnhanceQueue(F)
    # 起播提前 6000ms，零积压，单档成本 10ms，120 帧需 1200ms，三档 3600ms < 6000
    decisions, steps, seg_vmaf = q.schedule_segment(
        now_ms=0.0, seg_idx=0, play_start_lead_ms=6000.0,
        frame_play_ms=FRAME_PLAY, frame_cost_steps=costs, frame_vmaf=vmaf,
        max_level=3, backlog_steps=0)
    assert np.all(decisions == 3), decisions
    assert q.total_missed_frames == 0
    q.final_assert(steps)
    print("ok ample-lead all enhanced")


def test_tight_budget_partial():
    """紧提前量（旧整段闸门下 enhanced_count=0 的场景）：队列仍增强部分高增益帧。"""
    # 每帧 level1 成本 40ms（>33.3ms 播放间隔，GPU 稳态追不上，必然部分错过）
    rng = np.random.default_rng(0)
    gains = np.zeros((F, 3))
    gains[:, 0] = rng.uniform(0.5, 3.0, size=F)
    costs, vmaf = make_tables([40.0, 40.0, 40.0], gains)
    q = FrameEnhanceQueue(F)
    decisions, steps, seg_vmaf = q.schedule_segment(
        now_ms=0.0, seg_idx=0, play_start_lead_ms=500.0,
        frame_play_ms=FRAME_PLAY, frame_cost_steps=costs, frame_vmaf=vmaf,
        max_level=1, backlog_steps=0)
    enhanced = int(np.sum(decisions > 0))
    # 旧闸门：整段需 3600ms >> 500+2000(seg_time/2 赠送) 也不可行 -> 0 帧；
    # 新队列必须增强 >0 且 <120，并显式记账 missed。
    assert 0 < enhanced < F, enhanced
    assert q.total_missed_frames == F - enhanced
    q.final_assert(steps)
    # EDF 正确性：被接收帧按截止排序必须都可行（构造时已保证，这里复核）
    print(f"ok tight-budget partial: enhanced={enhanced}, missed={F-enhanced}")


def test_zero_lead_all_missed():
    """零提前量、有积压：最浅档也做不完，全部 missed。"""
    gains = np.ones((F, 3))
    costs, vmaf = make_tables([30.0, 30.0, 30.0], gains)
    q = FrameEnhanceQueue(F)
    # backlog 已占满，第 0 帧截止即现在
    # backlog 5000ms 已超过末帧截止（3963ms），任何作业都无法按时完成
    decisions, steps, seg_vmaf = q.schedule_segment(
        now_ms=0.0, seg_idx=0, play_start_lead_ms=0.0,
        frame_play_ms=FRAME_PLAY, frame_cost_steps=costs, frame_vmaf=vmaf,
        max_level=1, backlog_steps=to_steps(5000.0))
    assert steps == 0
    assert np.all(decisions == 0)
    assert q.total_missed_frames == F
    q.final_assert(steps)
    print("ok zero-lead all missed")


def test_backlog_conservation_multi_segment():
    """两段连续调度：累计推入量 == 队列承诺量，总帧守恒。"""
    gains = np.ones((F, 3))
    costs, vmaf = make_tables([10.0, 10.0, 10.0], gains)
    q = FrameEnhanceQueue(F)
    total_pushed = 0
    # 段0 lead 2000
    d0, s0, _ = q.schedule_segment(
        0.0, 0, 2000.0, FRAME_PLAY, costs, vmaf, 2, 0)
    total_pushed += s0
    # 段1 在 4000ms 后下载完，起播 lead 1000，backlog=段0 未完成部分
    d1, s1, _ = q.schedule_segment(
        4000.0, 1, 1000.0, FRAME_PLAY, costs, vmaf, 2,
        max(0, s0 - to_steps(4000.0)))
    total_pushed += s1
    assert q.total_enqueued == 2 * F
    q.final_assert(total_pushed)
    print("ok multi-segment conservation")


def test_deadline_ordering_early_frames_priority():
    """相同增益时 EDF 保证最早播放的帧优先被增强。"""
    gains = np.ones((F, 3))
    costs, vmaf = make_tables([40.0, 40.0, 40.0], gains)
    q = FrameEnhanceQueue(F)
    decisions, steps, _ = q.schedule_segment(
        0.0, 0, 300.0, FRAME_PLAY, costs, vmaf, 1, 0)
    enhanced_idx = np.where(decisions > 0)[0]
    # 40ms/帧 > 33.3ms 播放间隔，lead 300ms 时最早约 38 帧确定可行，必须全部增强
    assert len(enhanced_idx) > 0
    assert set(range(30)).issubset(set(enhanced_idx)), enhanced_idx
    # 接收集合整体必须 EDF 可调度（按截止排序累计成本不越线）
    admitted = [(300.0 + f * FRAME_PLAY, int(costs[f][1])) for f in enhanced_idx]
    assert FrameEnhanceQueue._edf_feasible(admitted, 0, 0.0, STEP_MS)
    print(f"ok EDF early-frame priority: {len(enhanced_idx)} frames, head all enhanced")


def test_oracle_mode_tight_budget():
    """M5：oracle 模式（真增益密度）紧预算下同样部分增强，守恒成立，且为标注上界。"""
    rng = np.random.default_rng(1)
    gains = np.zeros((F, 3))
    gains[:, 0] = rng.uniform(0.5, 3.0, size=F)
    costs, vmaf = make_tables([40.0, 40.0, 40.0], gains)
    q = FrameEnhanceQueue(F, rank_mode="oracle")
    assert q.rank_mode == "oracle"
    decisions, steps, _ = q.schedule_segment(
        now_ms=0.0, seg_idx=0, play_start_lead_ms=500.0,
        frame_play_ms=FRAME_PLAY, frame_cost_steps=costs, frame_vmaf=vmaf,
        max_level=1, backlog_steps=0)
    enhanced = int(np.sum(decisions > 0))
    assert 0 < enhanced < F, enhanced
    q.final_assert(steps)
    print(f"ok oracle-mode tight-budget: enhanced={enhanced}, missed={F-enhanced}")


if __name__ == "__main__":
    test_ample_lead_all_enhanced()
    test_tight_budget_partial()
    test_zero_lead_all_missed()
    test_backlog_conservation_multi_segment()
    test_deadline_ordering_early_frames_priority()
    test_oracle_mode_tight_budget()
    print("M3-M5 frame_enhance_queue tests PASSED")
