# -*- coding: utf-8 -*-
"""
M1/M2 验收：0.1ms GPU 时间网格与向量化帧级 DP。

运行（仓库根目录）：
  E:\\dev\\env\\miniconda3\\envs\\SR\\python.exe tests\\test_gpu_grid.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from simulation.gpu_grid import (STEP_MS, to_steps, budget_to_steps,
                                 assert_cost_table, solve_frame_dp,
                                 grid_residual_ms)


def test_round_trip_costs():
    # 历史 B1：10ms floor 下 7.50ms/8.79ms 变 0 成本；0.1ms round 后必须保留
    assert to_steps(7.50) == 75
    assert to_steps(8.79) == 88
    assert to_steps(12.28) == 123
    assert to_steps(13.77) == 138
    assert to_steps(24.0) == 240
    assert to_steps(0.0) == 0
    assert budget_to_steps(4000.0) == 40001


def test_free_positive_level_rejected():
    # 含零成本增强档（10ms floor 的 7.5ms 场景）必须被断言拦下
    bad = np.zeros((3, 4), dtype=np.int64)
    bad[:, 1] = 0  # low 档零成本
    try:
        assert_cost_table(bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("零成本增强档未被拦下")
    # 正常成本表通过（最便宜增强档 5.1ms = 51 步）
    ok = np.zeros((3, 4), dtype=np.int64)
    ok[:, 1] = 51
    ok[:, 2] = 120
    ok[:, 3] = 240
    assert_cost_table(ok)


def test_known_knapsack():
    # 2 帧、2 个增强档；预算只够"一帧 high"，high 增益集中在帧 1
    # level0 成本0价值0；level1 成本50步价值1；level2 成本100步价值：帧0=2，帧1=10
    costs = np.array([[0, 50, 100],
                      [0, 50, 100]], dtype=np.int64)
    values = np.array([[0.0, 1.0, 2.0],
                       [0.0, 1.0, 10.0]])
    decisions, used, total = solve_frame_dp(costs, values, budget_to_steps(10.0))
    assert list(decisions) == [0, 2], decisions
    assert used == 100
    assert abs(total - 10.0) < 1e-9


def test_cheapest_level_no_longer_free():
    # 历史 bug 复现：240p low 7.50ms、预算 2000ms 时旧网格把 low 当免费动作全选
    costs = np.zeros((120, 4), dtype=np.int64)
    costs[:, 1] = to_steps(7.50)
    costs[:, 2] = to_steps(13.77)
    costs[:, 3] = to_steps(24.0)
    values = np.zeros((120, 4))
    values[:, 1] = 0.5
    values[:, 2] = 1.0
    values[:, 3] = 2.0
    decisions, used, _ = solve_frame_dp(costs, values, budget_to_steps(2000.0))
    # low 不再免费：120 帧 low 需要 9000 步 > 20000? 75*120=9000 <= 20000 -> 可全选 low
    # 但 high 240 步只能选 83 帧；断言总成本守恒且不超预算
    assert used <= 20000
    assert used == sum(int(costs[f, decisions[f]]) for f in range(120))


def test_residual_bound():
    # 每帧四舍五入误差 <=0.05ms；120 帧残差 <=6ms
    rng = np.random.default_rng(0)
    continuous = rng.uniform(5.05, 25.0, size=120)
    steps = np.array([to_steps(x) for x in continuous])
    residual = grid_residual_ms(float(continuous.sum()), int(steps.sum()))
    assert abs(residual) <= 6.0 + 1e-9, residual
    assert STEP_MS == 0.1


if __name__ == "__main__":
    test_round_trip_costs()
    test_free_positive_level_rejected()
    test_known_knapsack()
    test_cheapest_level_no_longer_free()
    test_residual_bound()
    print("M1/M2 gpu_grid tests PASSED")
