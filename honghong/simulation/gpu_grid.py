# -*- coding: utf-8 -*-
"""
GPU 时间网格与帧级 0/1 背包 DP（M1/M2 统一成本表）。

历史问题（见 docs/frame_scheduling_fix_plan.md B1/J1/J10）：
- 三处帧级 DP 各自用 10ms 网格，eval/bones_frame_level 向下取整（最便宜增强档
  变成 0 成本"免费动作"），bones_combined 向上取整（3ms 记成 10ms，浅档超额
  计费偏向选深）；
- 决策按网格成本、缓冲记账按连续毫秒，两套数互不一致。

统一约定：
- 步长 STEP_MS=0.1ms；单帧成本 int(round(ms/STEP_MS))，最小正成本动作
  （约 5ms）至少 50 步，从表示上消灭免费动作；
- DP 决策、缓冲推入、统计全部以同一张 frame_cost_steps 表为准（M2）。
"""
import numpy as np

STEP_MS = 0.1
# 最小正成本动作（IMDN_RTC l2 约 5.1ms）在 0.1ms 网格下至少应有的步数
MIN_POSITIVE_STEPS = 50


def to_steps(time_ms):
    """连续毫秒 -> 0.1ms 网格步数（四舍五入，禁止 floor/ceil 偏向）。"""
    return int(round(float(time_ms) / STEP_MS))


def budget_to_steps(gpu_budget_ms):
    """预算毫秒 -> 状态列数（状态 c 取值 0..列数-1）。"""
    return to_steps(gpu_budget_ms) + 1


def assert_cost_table(frame_cost_steps, allow_zero_level0=True):
    """
    成本表断言：
    - level 0（不增强）成本必须为 0；
    - 其余增强级别成本必须 >0 且不低于 MIN_POSITIVE_STEPS（消灭免费动作）。
    """
    costs = np.asarray(frame_cost_steps, dtype=np.int64)
    if allow_zero_level0:
        if not np.all(costs[:, 0] == 0):
            raise AssertionError("level 0（不增强）成本必须为 0")
        positive = costs[:, 1:]
    else:
        positive = costs
    if np.any(positive <= 0):
        raise AssertionError("存在零/负成本增强动作，网格步长过大或取整错误")
    if int(positive.min()) < MIN_POSITIVE_STEPS:
        raise AssertionError(
            f"最小正成本动作仅 {int(positive.min())} 步 < {MIN_POSITIVE_STEPS}，"
            "可能未走统一网格或延迟表异常")


def solve_frame_dp(frame_cost_steps, frame_values, budget_steps, max_level=None):
    """
    向量化帧级 0/1 背包 DP：段内 GPU 预算约束下最大化总价值（VMAF/目标函数）。

    :param frame_cost_steps: (F, L) 每帧每级别的 GPU 成本（0.1ms 步数，非负整数）
    :param frame_values: (F, L) 每帧每级别的价值；不允许的级别填 -inf
    :param budget_steps: 状态列数（budget_to_steps 导出），允许状态 c 满足
                         累计成本 < budget_steps
    :param max_level: 允许选择的最大级别（含）；None 表示全部级别可选
    :return: decisions (F,) np.int64、used_steps int、total_value float
    """
    costs = np.asarray(frame_cost_steps, dtype=np.int64)
    values = np.asarray(frame_values, dtype=np.float64)
    n_frames, n_levels = costs.shape
    last_level = n_levels if max_level is None else int(max_level) + 1

    dp = np.full((n_frames + 1, budget_steps), -np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    choice = np.zeros((n_frames + 1, budget_steps), dtype=np.int64)

    for f in range(n_frames):
        prev = dp[f]
        best = np.full(budget_steps, -np.inf, dtype=np.float64)
        ch = np.zeros(budget_steps, dtype=np.int64)
        for level in range(last_level):
            w = int(costs[f, level])
            val = float(values[f, level])
            cand = np.full(budget_steps, -np.inf, dtype=np.float64)
            if w == 0:
                cand = prev + val
            elif w < budget_steps:
                cand[w:] = prev[:budget_steps - w] + val
            better = cand > best
            best[better] = cand[better]
            ch[better] = level
        dp[f + 1] = best
        choice[f + 1] = ch

    best_c = int(np.argmax(dp[n_frames]))
    decisions = np.zeros(n_frames, dtype=np.int64)
    c = best_c
    for f in range(n_frames - 1, -1, -1):
        level = int(choice[f + 1, c])
        decisions[f] = level
        c -= int(costs[f, level])
    if c != 0:
        raise AssertionError(f"DP 回溯成本不守恒: 残余 {c} 步")
    return decisions, best_c, float(dp[n_frames, best_c])


def grid_residual_ms(continuous_ms, used_steps):
    """
    M2 记账残差：连续毫秒真实成本与决策网格成本之差（毫秒）。
    每帧误差 <= STEP_MS/2，120 帧段理论上界 6ms。
    """
    return float(continuous_ms) - used_steps * STEP_MS
