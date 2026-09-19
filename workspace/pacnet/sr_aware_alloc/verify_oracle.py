# -*- coding: utf-8 -*-
"""
alloc_compare 结论的独立复核（2026-09-20）：
1) 独立 LP 求解器（verify_oracle_lp2.lp_raw_points）：不做凹包络，在原始 6 个
   R-D 点上用累计变量建模有序填充，允许非凹曲线，HiGHS 求全局最优，与贪心凹包络
   注水（alloc_compare.waterfill）对照。两实现独立，结论一致才算数。
2) 机制分析：原始失真边际斜率排序 vs SR 后边际斜率排序的 Spearman（解释
   "SR 感知"为何不改变最优分配）；平方根复杂度权重与真实边际斜率的排序关系
   （解释 sqrt 等码率下为何输 CBR）；段间异质性量化。
运行：python verify_oracle.py
"""
import numpy as np
from scipy.stats import spearmanr
from alloc_compare import (BITS, BIC, SR, W, NSEG, CRF_ORDER,
                           waterfill, bounds_allocation,
                           eval_metric_at_alloc)
from verify_oracle_lp2 import lp_raw_points


def part1():
    print("=" * 86)
    print("1) 独立 LP（原始点、允许非凹）vs 贪心凹包络注水")
    print("=" * 86)
    for bc in [42, 38, 34, 30, 26]:
        B = float(BITS[:, CRF_ORDER.index(bc)].sum())
        eq = bounds_allocation(np.ones(NSEG), B)
        sr_eq = eval_metric_at_alloc(SR, eq).mean()
        line = [f"CRF{bc}"]
        for name, M in [("raw", BIC), ("sr", SR)]:
            a_g = waterfill(M, B)
            a_lp = lp_raw_points(M, B)
            q_g = eval_metric_at_alloc(M, a_g).mean()
            q_lp = eval_metric_at_alloc(M, a_lp).mean()
            line.append(f"{name}: 贪心={q_g:.3f} LP={q_lp:.3f} "
                        f"差={q_lp - q_g:+.3f} 花费={a_lp.sum() / B:.4f}")
        a_raw = lp_raw_points(BIC, B)
        a_sr = lp_raw_points(SR, B)
        novelty = eval_metric_at_alloc(SR, a_sr).mean() - \
            eval_metric_at_alloc(SR, a_raw).mean()
        ceiling = eval_metric_at_alloc(SR, a_sr).mean() - sr_eq
        line.append(f"|| SR新颖性={novelty:+.3f} 对CBR上限={ceiling:+.3f} "
                     f"两解分配差={np.abs(a_sr - a_raw).mean() / B * NSEG * 100:.2f}%/段")
        print(" | ".join(line))


def part2():
    print("=" * 86)
    print("2) 机制：SR 是否重排段间边际价值排序")
    print("=" * 86)
    k0, k1 = CRF_ORDER.index(34), CRF_ORDER.index(30)
    raw_slope = (BIC[:, k1] - BIC[:, k0]) / (BITS[:, k1] - BITS[:, k0])
    sr_slope = (SR[:, k1] - SR[:, k0]) / (BITS[:, k1] - BITS[:, k0])
    rho, p = spearmanr(raw_slope, sr_slope)
    print(f"工作点边际 VMAF/bit 排序 Spearman(raw, sr) = {rho:.4f} (p={p:.1e})")
    gain = (SR - BIC).mean(axis=1)
    print(f"SR 段增益 g_i：mean={gain.mean():.2f} std={gain.std():.2f} "
          f"[{gain.min():.2f},{gain.max():.2f}]")
    print(f"g_i 与段复杂度 Spearman = {spearmanr(gain, W)[0]:.3f}; "
          f"g_i 与 raw 边际斜率 = {spearmanr(gain, raw_slope)[0]:.3f}")
    print(f"sqrt 复杂度权重 vs sr 边际斜率 Spearman = "
          f"{spearmanr(W, sr_slope)[0]:.3f}（负=按复杂度加码率方向恰好做反）")
    print("段间异质性：")
    for k, c in enumerate(CRF_ORDER):
        print(f"  CRF{c}: 比特CV={BITS[:, k].std() / BITS[:, k].mean():.3f} "
              f"bic={BIC[:, k].mean():.1f}±{BIC[:, k].std():.1f} "
              f"sr={SR[:, k].mean():.1f}±{SR[:, k].std():.1f}")


if __name__ == "__main__":
    part1()
    print()
    part2()
