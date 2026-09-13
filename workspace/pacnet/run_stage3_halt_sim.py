"""阶段3：halt 控制器离线调度仿真。

先对 159 段每段抽 1 帧生成 per-seg e1/e2/e3 PSNR（缓存），再模拟播放场景：
每帧有随机计算预算（模拟网络波动/缓冲区状态），控制器决定停在哪个出口。
对比固定档 / 贪心（只看预算）/ 内容感知（预算+深算增益），输出质量-掉帧帕累托。
"""
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, os.path.join(ROOT, "workspace/enhancer_survey"))
sys.path.insert(0, os.path.join(ROOT, "workspace/pacnet"))
sys.path.insert(0, os.path.join(ROOT, "player"))
import run_stage1 as rs  # noqa: E402
from pacnet_model import PACNet  # noqa: E402

DEV = rs.DEV
PACDIR = rs.PACDIR
FRAMES_PER_SEG = 120
N_SEG = 159
SAMPLE_J = 75  # 每段抽第75帧
# 阶段0实测 240p 整帧中位时延（ms）
LATENCY = {"e1": 14.0, "e2": 26.0, "e3": 39.0}
EXIT_IDX = {"e1": 0, "e2": 1, "e3": 2}


def gen_per_seg_psnr(weight_path, out_json):
    """对 159 段每段 j=75 帧，测 e1/e2/e3 整帧 PSNR（vs 1080p GT）。"""
    if os.path.exists(out_json):
        print(f"per-seg data exists: {out_json}", flush=True)
        return json.load(open(out_json, encoding="utf-8"))
    model = PACNet(nf=32, upscale=4, exits=(2, 4, 6), cond=False).to(DEV).eval()
    model.load_state_dict(torch.load(weight_path, map_location="cpu"))
    cap_lr = cv2.VideoCapture(os.path.join(ROOT, "data/video_mp4box/output_240p.mp4"))
    cap_hr = cv2.VideoCapture(os.path.join(ROOT, "data/video_mp4box/output_1080p.mp4"))
    want = {s * FRAMES_PER_SEG + SAMPLE_J for s in range(N_SEG)}
    data = {}
    f = 0
    with torch.no_grad():
        while True:
            ok1, lr = cap_lr.read(); ok2, hr = cap_hr.read()
            if not (ok1 and ok2):
                break
            if f in want:
                seg = f // FRAMES_PER_SEG
                x = rs.to_t(lr)
                outs = model(x)
                row = {}
                for ei, name in enumerate(("e1", "e2", "e3")):
                    row[name] = round(rs.psnr(outs[ei], hr), 4)
                row["gain_e3_e1"] = round(row["e3"] - row["e1"], 4)
                data[str(seg)] = row
                if seg % 30 == 0:
                    print(f"  seg {seg}: {row}", flush=True)
            f += 1
    cap_lr.release(); cap_hr.release()
    json.dump(data, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"saved {out_json}, {len(data)} segs", flush=True)
    return data


def make_budget_trace(n, mean, noise=0.3, seed=0):
    """生成 n 帧的预算轨迹（ms/帧），随机游走围绕 mean。"""
    rng = random.Random(seed)
    b = mean
    trace = []
    for _ in range(n):
        b += rng.uniform(-mean * noise, mean * noise)
        b = max(5.0, min(mean * 2, b))
        trace.append(b)
    return trace


def simulate(per_seg, budget_trace, controller, buffer_limit=50.0, lat_noise=0.15, seed=1):
    """逐帧仿真。返回 avg_psnr, drop_rate, exit_usage, effective_quality。
    controller(seg_gain, account, latencies) -> exit_name。
    账户 < -buffer_limit 则该帧掉帧（质量不计），账户重置为 -buffer_limit。
    """
    rng = random.Random(seed)
    account = 0.0
    psnrs = []
    drops = 0
    usage = {"e1": 0, "e2": 0, "e3": 0}
    n = len(budget_trace)
    for i in range(n):
        seg = i % N_SEG
        gain = per_seg[str(seg)]["gain_e3_e1"]
        account += budget_trace[i]
        # 时延加随机噪声（模拟 GPU 负载波动）
        lat = {k: v * (1 + rng.uniform(-lat_noise, lat_noise)) for k, v in LATENCY.items()}
        if account < -buffer_limit:
            # 掉帧：不增强，账户不再继续欠债
            drops += 1
            account = -buffer_limit
            continue
        ex = controller(gain, account, lat)
        account -= lat[ex]
        usage[ex] += 1
        psnrs.append(per_seg[str(seg)][ex])
    avg = float(np.mean(psnrs)) if psnrs else 0.0
    drop_rate = drops / n
    total = sum(usage.values())
    usage_pct = {k: round(v / total * 100, 1) if total else 0 for k, v in usage.items()}
    effective = avg * (1 - drop_rate)  # 掉帧视为质量 0
    return dict(avg_psnr=round(avg, 3), drop_rate=round(drop_rate, 4),
                effective_quality=round(effective, 3), exit_usage=usage_pct)


# ---------- 控制器 ----------
def ctrl_fixed(e):
    return lambda gain, acc, lat: e


def ctrl_greedy(gain, acc, lat):
    """只看预算：够 e3 就 e3，够 e2 就 e2，否则 e1。"""
    if acc >= lat["e3"]:
        return "e3"
    if acc >= lat["e2"]:
        return "e2"
    return "e1"


def ctrl_content_aware(thresh=1.5):
    """内容感知：深算增益 > thresh 的帧优先 e3，否则降级省时间。
    预算够 e3 且 gain 高 -> e3；预算够 e2 -> e2（gain 低时 e2 也够）；否则 e1。
    """
    def fn(gain, acc, lat):
        if gain >= thresh and acc >= lat["e3"]:
            return "e3"
        if acc >= lat["e2"]:
            return "e2"
        return "e1"
    return fn


def ctrl_content_aware_v2(thresh=1.5):
    """更激进的内容感知：gain 低的帧即使预算够 e3 也只用 e1（省时间给困难帧）；
    gain 高的帧预算够 e3 就 e3，够 e2 就 e2。"""
    def fn(gain, acc, lat):
        if gain >= thresh:
            if acc >= lat["e3"]:
                return "e3"
            if acc >= lat["e2"]:
                return "e2"
            return "e1"
        else:
            # 简单段：最多 e2，预算紧张直接 e1
            if acc >= lat["e2"] and acc >= lat["e3"] * 0.8:
                return "e2"
            return "e1"
    return fn


def ctrl_adaptive_ca(base_thresh=1.5, suff_ratio=1.3):
    """自适应内容感知：账户很充足（>= e3时延×suff_ratio）时自动全 e3（退化为 greedy），
    预算一般时才按 gain 阈值内容感知。目标：高预算不劣于 greedy，低预算优于 greedy。"""
    def fn(gain, acc, lat):
        if acc >= lat["e3"] * suff_ratio:
            return "e3"
        if gain >= base_thresh and acc >= lat["e3"]:
            return "e3"
        if acc >= lat["e2"]:
            return "e2"
        return "e1"
    return fn


def ctrl_adaptive_cav2(thresh=2.0, suff_ratio=1.3):
    """最终版：预算充足全 e3（退化为 greedy），预算紧张用 CAv2 激进内容感知
    （低 gain 帧直接 e1 省时间，高 gain 帧才深算）。"""
    def fn(gain, acc, lat):
        if acc >= lat["e3"] * suff_ratio:
            return "e3"
        if gain >= thresh:
            if acc >= lat["e3"]:
                return "e3"
            if acc >= lat["e2"]:
                return "e2"
            return "e1"
        else:
            if acc >= lat["e2"] and acc >= lat["e3"] * 0.8:
                return "e2"
            return "e1"
    return fn


def main():
    weight = os.path.join(PACDIR, "weights/pacnet_x4_best.pth")
    per_seg_json = os.path.join(PACDIR, "per_seg_psnr.json")
    print("=== 生成/加载 per-seg PSNR ===", flush=True)
    data = gen_per_seg_psnr(weight, per_seg_json)
    gains = [v["gain_e3_e1"] for v in data.values()]
    print(f"深算增益 gain(e3-e1): mean={np.mean(gains):.3f} std={np.std(gains):.3f} "
          f"min={np.min(gains):.3f} max={np.max(gains):.3f} "
          f"p25={np.percentile(gains,25):.3f} p75={np.percentile(gains,75):.3f}", flush=True)

    n_frames = 1590  # 10 个完整周期（159段×10），足够统计
    budgets = {"low(20ms)": 20.0, "mid(28ms)": 28.0, "high(36ms)": 36.0}
    controllers = {
        "fixed_e1": ctrl_fixed("e1"),
        "fixed_e2": ctrl_fixed("e2"),
        "fixed_e3": ctrl_fixed("e3"),
        "greedy": ctrl_greedy,
        "CA_t1.0": ctrl_content_aware(1.0),
        "CA_t1.5": ctrl_content_aware(1.5),
        "CA_t2.0": ctrl_content_aware(2.0),
        "CAv2_t1.5": ctrl_content_aware_v2(1.5),
        "CAv2_t2.0": ctrl_content_aware_v2(2.0),
        "ADA_t1.0_s1.3": ctrl_adaptive_ca(1.0, 1.3),
        "ADA_t1.5_s1.3": ctrl_adaptive_ca(1.5, 1.3),
        "ADA_t1.5_s1.5": ctrl_adaptive_ca(1.5, 1.5),
        "ADA_t2.0_s1.3": ctrl_adaptive_ca(2.0, 1.3),
        "ADAv2_t1.5_s1.3": ctrl_adaptive_cav2(1.5, 1.3),
        "ADAv2_t2.0_s1.3": ctrl_adaptive_cav2(2.0, 1.3),
        "ADAv2_t2.0_s1.5": ctrl_adaptive_cav2(2.0, 1.5),
    }

    results = {}
    print("\n=== 调度仿真（10周期×159段=1590帧，3预算水平×多控制器）===", flush=True)
    for bname, bmean in budgets.items():
        results[bname] = {}
        trace = make_budget_trace(n_frames, bmean, noise=0.3, seed=42)
        print(f"\n--- 预算 {bname} (mean={bmean}ms) ---", flush=True)
        header = f"{'controller':>14s} {'avgPSNR':>8s} {'drop%':>7s} {'effQ':>7s} {'e1%':>5s} {'e2%':>5s} {'e3%':>5s}"
        print(header, flush=True)
        for cname, ctrl in controllers.items():
            r = simulate(data, trace, ctrl, seed=7)
            results[bname][cname] = r
            print(f"{cname:>14s} {r['avg_psnr']:8.3f} {r['drop_rate']*100:6.2f}% "
                  f"{r['effective_quality']:7.3f} {r['exit_usage']['e1']:4.0f}% "
                  f"{r['exit_usage']['e2']:4.0f}% {r['exit_usage']['e3']:4.0f}%", flush=True)

    out = os.path.join(PACDIR, "stage3_halt_sim.json")
    json.dump(results, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\nsaved {out}", flush=True)

    # 关键对比：内容感知最优 vs fixed_e3 vs greedy，按有效质量
    print("\n=== 关键对比（有效质量 effQ = avgPSNR×(1-drop率)）===", flush=True)
    for bname in budgets:
        row = results[bname]
        best_ca = max([(c, row[c]["effective_quality"]) for c in row if c.startswith("CA") or c.startswith("ADA")], key=lambda x: x[1])
        print(f"{bname}: fixed_e3 effQ={row['fixed_e3']['effective_quality']:.3f} "
              f"(drop={row['fixed_e3']['drop_rate']*100:.1f}%) | "
              f"greedy effQ={row['greedy']['effective_quality']:.3f} | "
              f"best CA({best_ca[0]}) effQ={best_ca[1]:.3f} "
              f"(Δvs fixed_e3={best_ca[1]-row['fixed_e3']['effective_quality']:+.3f}, "
              f"Δvs greedy={best_ca[1]-row['greedy']['effective_quality']:+.3f})", flush=True)


if __name__ == "__main__":
    main()
