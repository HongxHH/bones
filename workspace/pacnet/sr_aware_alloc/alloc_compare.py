# -*- coding: utf-8 -*-
"""
等总码率下五种码率分配策略对比（TOS，426 宽，post-l2-SR VMAF 为观众所见质量）：
  const_crf : 恒定 CRF（预算定义点，每段同 CRF）
  eq_bits   : 等码率 CBR（每段等比特，按段 R-D 线性插值）
  sqrt      : 等预算平方根分配 t_i = B*sqrt(w_i)/Σsqrt(w)，w_i=CRF30 逐段比特（内容复杂度）
  raw_oracle: 以"无增强 bicubic VMAF"为目标的率失真最优（凹包络+边际斜率注水），仍用 post-SR 评估
  sr_oracle : 以"post-SR VMAF"为目标的率失真最优（本提案上限）
判据：sr_oracle 相对 sqrt / eq_bits 是否在多预算下稳定 >= +1 VMAF；
      sr_oracle 与 raw_oracle 的差 = "SR 感知"相对经典最优 VBR 的新颖性上限。
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RD = json.load(open(os.path.join(HERE, "seg_rd.json"), "r", encoding="utf-8"))
CRF_ORDER = [42, 38, 34, 30, 26, 22]          # 低码率 -> 高码率
BUDGET_CRFS = [42, 38, 34, 30, 26]
NSEG = RD["crf34"]["nseg"]


def seg_arrays(metric):
    """返回 [nseg][6] 的 bits 与 metric，列按 CRF_ORDER。"""
    B = np.zeros((NSEG, len(CRF_ORDER)))
    M = np.zeros((NSEG, len(CRF_ORDER)))
    for k, c in enumerate(CRF_ORDER):
        for s in RD[f"crf{c}"]["segments"]:
            i = s["seg"]
            B[i, k] = s["bits"]
            M[i, k] = s[metric]
    return B, M


BITS, SR = seg_arrays("sr_vmaf")
_, BIC = seg_arrays("bic_vmaf")
W = BITS[:, CRF_ORDER.index(30)]              # 复杂度：CRF30 逐段比特


def interp_at(bits_row, met_row, t):
    """单调化后按 bits 线性插值，越界钳到端点。"""
    order = np.argsort(bits_row)
    b, m = bits_row[order], met_row[order]
    keep = np.concatenate([[True], np.diff(b) > 0])
    b, m = b[keep], m[keep]
    if t <= b[0]:  return m[0], b[0]
    if t >= b[-1]: return m[-1], b[-1]
    return float(np.interp(t, b, m)), float(t)


def concave_hull(b, m):
    """上凹包络（边际收益递减），返回顶点 (bits, metric)。"""
    pts = sorted(zip(b, m))
    hull_b, hull_m = [pts[0][0]], [pts[0][1]]
    for pb, pm in pts[1:]:
        if pb <= hull_b[-1] + 1e-9:
            if pm > hull_m[-1]:
                hull_b[-1], hull_m[-1] = pb, pm
            continue
        while len(hull_b) >= 2:
            s1 = (hull_m[-1] - hull_m[-2]) / (hull_b[-1] - hull_b[-2])
            s2 = (pm - hull_m[-1]) / (pb - hull_b[-1])
            if s2 >= s1:
                hull_b.pop(); hull_m.pop()
            else:
                break
        hull_b.append(pb); hull_m.append(pm)
    return np.array(hull_b), np.array(hull_m)


def waterfill(metric_mat, budget):
    """全局边际斜率注水：每段凹包络边按斜率降序吃预算，返回逐段分配比特。"""
    edges = []
    hulls = []
    for i in range(NSEG):
        hb, hm = concave_hull(BITS[i], metric_mat[i])
        hulls.append((hb, hm))
        for k in range(len(hb) - 1):
            db = hb[k + 1] - hb[k]
            dm = hm[k + 1] - hm[k]
            edges.append((dm / db, i, k, db, dm))
    edges.sort(key=lambda e: -e[0])
    alloc_b = np.array([h[0][0] for h in hulls], dtype=float)
    alloc_m = np.array([h[1][0] for h in hulls], dtype=float)
    rem = budget - alloc_b.sum()
    for slope, i, k, db, dm in edges:
        if rem <= 0: break
        take = min(db, rem)
        alloc_b[i] += take
        alloc_m[i] += dm * (take / db)
        rem -= take
    return alloc_b


def eval_metric_at_alloc(metric_mat, alloc_bits):
    return np.array([interp_at(BITS[i], metric_mat[i], alloc_bits[i])[0]
                     for i in range(NSEG)])


def bounds_allocation(weights, B):
    """按权重分配预算，固定点钳到每段 [bmin,bmax] 并在未触界段间再分配，恰花完 B。"""
    bmin = BITS[:, 0]
    bmax = BITS[:, -1]
    t = B * weights / weights.sum()
    active = np.ones(NSEG, dtype=bool)
    for _ in range(200):
        t = np.clip(t, bmin, bmax)
        spend = t[active].sum()
        target = B - t[~active].sum()
        w = weights[active]
        t[active] = target * w / w.sum()
        new_active = (t > bmin + 1e-6) & (t < bmax - 1e-6) | \
                     (~((t <= bmin + 1e-6) | (t >= bmax - 1e-6)))
        lo = t <= bmin + 1e-6
        hi = t >= bmax - 1e-6
        new_active = ~(lo | hi)
        if np.array_equal(new_active, active):
            break
        active = new_active
    t = np.clip(t, bmin, bmax)
    return t


def run():
    rows = []
    detail = {}
    for bc in BUDGET_CRFS:
        B = float(BITS[:, CRF_ORDER.index(bc)].sum())
        kbps = B / (NSEG * 4) / 1000.0
        r = {"budget_crf": bc, "total_bits": B, "kbps": kbps}

        # const CRF
        k = CRF_ORDER.index(bc)
        const_sr = SR[:, k].mean(); const_bic = BIC[:, k].mean()

        # eq bits CBR（预算约束内等比特，触界段固定后再分配）
        tt_eq = bounds_allocation(np.ones(NSEG), B)
        eq_sr = np.array([interp_at(BITS[i], SR[i], tt_eq[i])[0] for i in range(NSEG)])
        eq_bic = np.array([interp_at(BITS[i], BIC[i], tt_eq[i])[0] for i in range(NSEG)])

        # sqrt equal-budget
        tt = bounds_allocation(np.sqrt(W), B)
        sq_sr = np.array([interp_at(BITS[i], SR[i], tt[i])[0] for i in range(NSEG)])
        sq_bic = np.array([interp_at(BITS[i], BIC[i], tt[i])[0] for i in range(NSEG)])
        clamped = int(sum(tt[i] <= BITS[i, 0] + 1e-6 or tt[i] >= BITS[i, -1] - 1e-6
                          for i in range(NSEG)))

        # raw oracle（目标 bic），用 sr 评估
        raw_alloc = waterfill(BIC, B)
        raw_sr = eval_metric_at_alloc(SR, raw_alloc)
        raw_bic = eval_metric_at_alloc(BIC, raw_alloc)

        # sr oracle（目标 sr）
        sr_alloc = waterfill(SR, B)
        sro_sr = eval_metric_at_alloc(SR, sr_alloc)
        sro_bic = eval_metric_at_alloc(BIC, sr_alloc)

        r.update({
            "const_crf_sr": const_sr, "const_crf_bic": const_bic,
            "eq_bits_sr": float(eq_sr.mean()), "eq_bits_bic": float(eq_bic.mean()),
            "sqrt_sr": float(sq_sr.mean()), "sqrt_bic": float(sq_bic.mean()),
            "raw_oracle_sr": float(raw_sr.mean()), "raw_oracle_bic": float(raw_bic.mean()),
            "sr_oracle_sr": float(sro_sr.mean()), "sr_oracle_bic": float(sro_bic.mean()),
            "sr_oracle_vs_sqrt": float(sro_sr.mean() - sq_sr.mean()),
            "sr_oracle_vs_raw": float(sro_sr.mean() - raw_sr.mean()),
            "sr_oracle_vs_eq": float(sro_sr.mean() - eq_sr.mean()),
            "sqrt_vs_eq": float(sq_sr.mean() - eq_sr.mean()),
            "raw_vs_sqrt": float(raw_sr.mean() - sq_sr.mean()),
            "seg_win_sr_oracle_over_sqrt": int((sro_sr > sq_sr).sum()),
            "sqrt_clamped_segs": clamped,
            "eq_spend_ratio": float(tt_eq.sum() / B),
            "sqrt_spend_ratio": float(tt.sum() / B),
            "raw_spend_ratio": float(raw_alloc.sum() / B),
            "sr_spend_ratio": float(sr_alloc.sum() / B),
        })
        rows.append(r)
        detail[f"crf{bc}"] = {
            "eq_alloc_bits": tt_eq.tolist(), "sqrt_alloc_bits": tt.tolist(),
            "sr_oracle_alloc_bits": sr_alloc.tolist(),
            "raw_alloc_bits": raw_alloc.tolist(),
            "seg_sr_const": SR[:, k].tolist(), "seg_sr_sqrt": sq_sr.tolist(),
            "seg_sr_sr_oracle": sro_sr.tolist(), "seg_sr_raw_oracle": raw_sr.tolist(),
        }
        print(f"[budget=CRF{bc} {kbps:.0f}kbps] const={const_sr:.2f} eq={eq_sr.mean():.2f} "
              f"sqrt={sq_sr.mean():.2f} rawO={raw_sr.mean():.2f} srO={sro_sr.mean():.2f} | "
              f"srO-sqrt={r['sr_oracle_vs_sqrt']:+.2f} srO-raw={r['sr_oracle_vs_raw']:+.2f} "
              f"sqrt-eq={r['sqrt_vs_eq']:+.2f} win={r['seg_win_sr_oracle_over_sqrt']}/{NSEG}")

    json.dump({"summary": rows, "detail": detail},
              open(os.path.join(HERE, "alloc_results.json"), "w", encoding="utf-8"), indent=1)

    # 图：预算-postSR VMAF 前沿 + 增益柱
    kb = [r["kbps"] for r in rows]
    plt.figure(figsize=(12, 4.6))
    plt.subplot(1, 2, 1)
    for key, lab, mk in [("const_crf_sr", "const CRF", "s"), ("eq_bits_sr", "CBR equal-bits", "^"),
                         ("sqrt_sr", "sqrt VBR (equal budget)", "o"),
                         ("raw_oracle_sr", "raw-RD oracle", "D"),
                         ("sr_oracle_sr", "SR-aware RD oracle", "*")]:
        plt.plot(kb, [r[key] for r in rows], marker=mk, label=lab)
    plt.xlabel("total bitrate (kbps, equal budget)"); plt.ylabel("mean post-SR VMAF")
    plt.title("Equal-bitrate allocation (TOS 426-wide, l2)")
    plt.legend(fontsize=8); plt.grid(alpha=.3)
    plt.subplot(1, 2, 2)
    w = 0.35; x = np.arange(len(rows))
    plt.bar(x - w/2, [r["sr_oracle_vs_sqrt"] for r in rows], w, label="SR-oracle − sqrt")
    plt.bar(x + w/2, [r["sr_oracle_vs_raw"] for r in rows], w, label="SR-oracle − raw-RD oracle (SR novelty)")
    plt.axhline(1.0, color="r", ls="--", lw=1, label="+1 VMAF threshold")
    plt.xticks(x, [f"CRF{r['budget_crf']}\n{r['kbps']:.0f}k" for r in rows])
    plt.ylabel("delta post-SR VMAF"); plt.title("Headroom vs heuristic and vs classical oracle")
    plt.legend(fontsize=8); plt.grid(alpha=.3, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "alloc_pareto.png"), dpi=140)
    print("saved alloc_pareto.png")


if __name__ == "__main__":
    run()
