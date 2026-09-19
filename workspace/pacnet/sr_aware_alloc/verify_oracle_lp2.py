# -*- coding: utf-8 -*-
"""
正确的独立 LP 复核：不做凹包络，直接在每段原始 6 个 R-D 点的逐段线性曲线上，
用累计变量 z_{i,k}（段 i 吃到第 k 档边的累计比特）建模有序填充：
  0 <= z1 <= db1,  z_{k-1} <= z_k <= z_{k-1}+db_k
目标 max Σ slope_k*(z_k-z_{k-1})，总预算约束 Σ z <= B-Σbmin。
任意（含非凹）逐段线性曲线的全局最优，HiGHS 求解。与贪心凹包络注水对照。
"""
import numpy as np
from scipy.optimize import linprog
from alloc_compare import (BITS, BIC, SR, NSEG, CRF_ORDER, waterfill,
                           bounds_allocation, eval_metric_at_alloc)

K = len(CRF_ORDER) - 1  # 5 条边/段


def lp_raw_points(metric_mat, budget):
    # 变量：z[i,k] 累计 kbit，共 NSEG*K；最后加一个松弛? 预算约束 <= 即可
    n = NSEG * K
    # 目标：系数 = slope_k - slope_{k+1}（因为 Σ slope_k*(z_k-z_{k-1})
    # 展开后 z_k 的系数为 slope_k - slope_{k+1}，z_K 系数 slope_K）
    c = np.zeros(n)
    bmin = BITS[:, 0].sum() / 1000.0
    for i in range(NSEG):
        for k in range(K):
            db = (BITS[i, k + 1] - BITS[i, k]) / 1000.0
            s_k = (metric_mat[i, k + 1] - metric_mat[i, k]) / db
            s_next = ((metric_mat[i, k + 2] - metric_mat[i, k + 1]) /
                      ((BITS[i, k + 2] - BITS[i, k + 1]) / 1000.0)) if k + 1 < K else 0.0
            c[i * K + k] = -(s_k - s_next)  # linprog 最小化负目标
    A = np.zeros(n)
    for i in range(NSEG):
        A[i * K + K - 1] = 1.0  # 预算约束作用于每段最终累计量 z_K
    Aub, bub = [A], [budget / 1000.0 - bmin]
    for i in range(NSEG):
        for k in range(K):
            db = (BITS[i, k + 1] - BITS[i, k]) / 1000.0
            idx = i * K + k
            if k == 0:
                r = np.zeros(n); r[idx] = -1; Aub.append(r); bub.append(0.0)  # z0>=0
                r = np.zeros(n); r[idx] = 1; Aub.append(r); bub.append(db)    # z0<=db0
            else:
                r = np.zeros(n); r[idx - 1] = 1; r[idx] = -1; Aub.append(r); bub.append(0.0)
                r = np.zeros(n); r[idx] = 1; r[idx - 1] = -1; Aub.append(r); bub.append(db)
    bounds = [(0, None)] * n
    res = linprog(c, A_ub=np.vstack(Aub), b_ub=np.array(bub),
                  bounds=bounds, method="highs")
    assert res.success, res.message
    z = res.x.reshape(NSEG, K)
    alloc = BITS[:, 0] + z[:, -1] * 1000.0
    return alloc


def main():
    print("独立 LP（原始 6 点、累计变量、允许非凹）vs 贪心凹包络注水")
    print("=" * 88)
    for bc in [42, 38, 34, 30, 26]:
        B = float(BITS[:, CRF_ORDER.index(bc)].sum())
        eq = bounds_allocation(np.ones(NSEG), B)
        sr_eq = eval_metric_at_alloc(SR, eq).mean()
        out = [f"CRF{bc}"]
        for name, M in [("raw", BIC), ("sr", SR)]:
            a_g = waterfill(M, B)
            a_lp = lp_raw_points(M, B)
            q_g = eval_metric_at_alloc(M, a_g).mean()
            q_lp = eval_metric_at_alloc(M, a_lp).mean()
            out.append(f"{name}: 贪心(包络)={q_g:.3f} LP(原点)={q_lp:.3f} "
                       f"LP-贪心={q_lp - q_g:+.3f} LP花费={a_lp.sum() / B:.4f}")
        a_raw = lp_raw_points(BIC, B)
        a_sr = lp_raw_points(SR, B)
        sr_at_raw = eval_metric_at_alloc(SR, a_raw).mean()
        sr_at_sr = eval_metric_at_alloc(SR, a_sr).mean()
        out.append(f"|| SR新颖性={sr_at_sr - sr_at_raw:+.3f} "
                   f"SR天花板(vs CBR {sr_eq:.2f})={sr_at_sr - sr_eq:+.3f} "
                   f"raw/sr分配差={np.abs(a_sr - a_raw).mean() / B * NSEG * 100:.2f}%")
        print(" | ".join(out))


if __name__ == "__main__":
    main()
