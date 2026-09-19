# -*- coding: utf-8 -*-
"""H2 前置探针：不训练的时序配准融合 oracle，估多帧时序信息上限。

动机：H1 把 QP/MV 当条件通道喂单帧 l2 已证伪（见 h1_condnet_falsification.md）。
MV 的结构性用法是时序配准——从相邻帧把单帧物理上丢失的细节"搬"过来。本脚本不训练、
不引入学习模型，只用稠密光流（cv2 DIS，缺失时退 Farneback）把邻帧 warp 到中心帧，
做时序均值/中值融合，再过 stock l2（IMDN_RTC + IMDN_medium_DIV2K_x4），与单帧 l2
在 hard_survey 完全同口径（7 档退化 × 32 帧，x4 后 bicubic 到 1920×800 对源）比
逐帧 VMAF(libvmaf) + LPIPS(squeeze)，并做逐帧配对符号检验。

判读：这是"非学习时序融合"的慷慨上界（稠密光流比解码器块级 MV 更准）。
- 连它相对单帧 l2 都没有 ≥1~2 VMAF / 明确 LPIPS 优势 → 便宜 VSR 无价值区间，H2 止损；
- 中值/均值融合稳定 +2~3 以上 → 值得再测解码器免费 MV 能拿到其中多少、并训轻量 VSR。

方法：single_l2（单帧基线）、mean3/median3（±1 帧）、mean7/median7（±3 帧）。
"""
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, os.path.join(ROOT, "player"))
sys.path.insert(0, os.path.join(ROOT, "workspace/pacnet/hard_survey"))
sys.path.insert(0, os.path.join(ROOT, "workspace/pacnet/h1_condnet"))
from istream_player.modules.enhancer.imdn_model import IMDN_RTC  # noqa: E402
from cond_model import WARM_L2  # noqa: E402
import measure_hard_survey as H  # noqa: E402

DEV = "cuda:0"
TARGET = (1920, 800)
N = 32
R = 3
OUT_JSON = os.path.join(HERE, "warp_oracle.json")

# DIS 光流器（contrib）；不可用退 Farneback
try:
    _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    USE_DIS = True
except Exception:
    _DIS = None
    USE_DIS = False


def flow_center_to_neighbor(gray_c, gray_n):
    """中心帧 -> 邻帧的稠密光流（输出像素在中心帧，向量指向邻帧），用于 remap 把邻帧搬回中心。"""
    if USE_DIS:
        return _DIS.calc(gray_c, gray_n, None)
    return cv2.calcOpticalFlowFarneback(gray_c, gray_n, None, 0.5, 3, 21, 5, 7, 1.2, 0)


def warp_to_center(frame_n, flow):
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (gx + flow[..., 0]).astype(np.float32)
    map_y = (gy + flow[..., 1]).astype(np.float32)
    return cv2.remap(frame_n, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def read_windows(path, centers, r):
    """顺序解码一遍，缓存每个中心帧的 [t-r, t+r] 窗口（BGR），返回 {center: [frames]}。"""
    need = {}
    for t in centers:
        for k in range(t - r, t + r + 1):
            need[k] = True
    cap = cv2.VideoCapture(path)
    got, i = {}, 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        if i in need:
            got[i] = f
        i += 1
    cap.release()
    wins = {}
    for t in centers:
        win = [got[k] for k in range(t - r, t + r + 1) if k in got]
        wins[t] = win
    return wins


@torch.no_grad()
def sr_l2(model, bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(DEV) / 255.0
    o = model(t)
    o = torch.nn.functional.interpolate(o, size=(TARGET[1], TARGET[0]),
                                        mode="bicubic", align_corners=False)
    o = o.squeeze(0).permute(1, 2, 0).cpu().numpy()
    o = (o * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(o, cv2.COLOR_RGB2BGR)


def fuse(aligned_bgr, mode):
    """aligned_bgr: 已对齐到中心的帧栈（含中心帧），BGR。"""
    st = np.stack(aligned_bgr, axis=0)
    if mode == "mean":
        return np.round(st.mean(axis=0)).astype(np.uint8)
    return np.median(st, axis=0).astype(np.uint8)  # median


def fuse_gated(center, neighbors_bgr, tau):
    """置信度门控融合（非学习时序融合的最佳情况）：逐像素只在邻帧 warp 后
    与中心帧 Y 残差 < tau（配准可信、静态/平坦区）时纳入平均，否则保持中心帧。
    中心帧权重恒 1；运动/纹理边缘/光流错误处邻帧权重 0，避免鬼影与抹细节。"""
    cy = cv2.cvtColor(center, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float32)
    acc = center.astype(np.float32)
    wsum = np.ones(cy.shape, np.float32)
    for f in neighbors_bgr:
        fy = cv2.cvtColor(f, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float32)
        m3 = np.repeat((np.abs(fy - cy) < tau)[:, :, None], 3, axis=2).astype(np.float32)
        acc += f.astype(np.float32) * m3
        wsum += m3[:, :, 0]
    out = acc / np.repeat(wsum[:, :, None], 3, axis=2)
    return np.round(out).clip(0, 255).astype(np.uint8)


def vmaf_perframe(pairs_bgr, tmp, tag):
    ep, rp, lname = (os.path.join(tmp, f"e_{tag}.mp4"), os.path.join(tmp, f"r_{tag}.mp4"),
                     f"l_{tag}.json")
    h, w = pairs_bgr[0][0].shape[:2]
    fc = cv2.VideoWriter_fourcc(*"mp4v")
    we, wr = cv2.VideoWriter(ep, fc, H.FPS, (w, h)), cv2.VideoWriter(rp, fc, H.FPS, (w, h))
    for e, r in pairs_bgr:
        we.write(e); wr.write(r)
    we.release(); wr.release()
    res = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", ep, "-i", rp,
                          "-lavfi", f"libvmaf=log_path={lname}:log_fmt=json", "-f", "null", "-"],
                         capture_output=True, text=True, timeout=300, cwd=tmp)
    lp = os.path.join(tmp, lname)
    if not os.path.exists(lp):
        raise RuntimeError(f"libvmaf log missing {tag}: {res.stderr[-800:]}")
    log = json.load(open(lp, encoding="utf-8"))
    return [float(fr["metrics"]["vmaf"]) for fr in log["frames"]]


@torch.no_grad()
def lpips_perframe(lp, e_bgr, r_bgr):
    def tt(x):
        rgb = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(DEV) / 255.0
    return float(lp(tt(e_bgr) * 2 - 1, tt(r_bgr) * 2 - 1))


def sign_test(diff):
    from scipy.stats import binomtest
    d = np.asarray(diff, float)
    d = d[np.abs(d) > 1e-9]
    if len(d) == 0:
        return None, 1.0, 0
    pos = int((d > 0).sum())
    return pos, float(binomtest(pos, len(d), 0.5).pvalue), len(d)


def main():
    model = IMDN_RTC(upscale=4)
    model.load_state_dict(torch.load(WARM_L2, map_location="cpu"))
    model.to(DEV).eval()
    import lpips as lpips_mod
    lp = lpips_mod.LPIPS(net="squeeze").to(DEV).eval()
    print(f"flow={'DIS' if USE_DIS else 'Farneback'} R={R}", flush=True)

    cap = cv2.VideoCapture(H.REF_VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    idxs = np.linspace(0, total - 10, N, dtype=int)
    ref = H.read_frames(H.REF_VIDEO, idxs)
    print("ref frames", len(ref), "total", total, flush=True)

    methods = ["single_l2", "mean3_l2", "median3_l2", "mean7_l2", "median7_l2",
               "gated3_t6", "gated3_t12", "gated7_t6", "gated7_t12"]
    results = {}
    for name, path in H.DEG.items():
        if not os.path.exists(path):
            print("SKIP", path); continue
        centers = [int(i) for i in idxs]
        wins = read_windows(path, centers, R)
        n = min(len(centers), len(ref))
        print(f"\n=== {name} ({n}) ===", flush=True)
        # 预计算每中心对齐帧栈（按到中心距离排序，中心帧在中间）
        per_center = []
        for ci in range(n):
            t = centers[ci]
            win = wins[t]
            c = win[len(win) // 2]
            gc = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
            aligned = {0: c}
            for off in range(1, R + 1):
                for sign, j in ((-1, len(win) // 2 - off), (1, len(win) // 2 + off)):
                    if 0 <= j < len(win):
                        gn = cv2.cvtColor(win[j], cv2.COLOR_BGR2GRAY)
                        fl = flow_center_to_neighbor(gc, gn)
                        aligned[sign * off] = warp_to_center(win[j], fl)
            per_center.append((c, aligned))
        out = {}
        with tempfile.TemporaryDirectory() as tmp:
            for meth in methods:
                epairs, lps = [], []
                for ci in range(n):
                    c, aligned = per_center[ci]
                    if meth == "single_l2":
                        fused = c
                    elif meth.startswith("gated"):
                        # gated{3|7}_t{6|12}：窗口 ±1/±3，Y 残差阈值 tau=6/12
                        rr = 1 if meth.startswith("gated3") else 3
                        tau = 6.0 if meth.endswith("_t6") else 12.0
                        nbrs = []
                        for off in range(1, rr + 1):
                            for s in (-1, 1):
                                if s * off in aligned:
                                    nbrs.append(aligned[s * off])
                        fused = fuse_gated(c, nbrs, tau)
                    else:
                        # mean3/median3 用 ±1 帧(rr=1)，mean7/median7 用 ±3 帧(rr=3)
                        mode = "mean" if meth.startswith("mean") else "median"
                        rr = 1 if meth.endswith("3_l2") else 3
                        stack = [aligned[0]]
                        for off in range(1, rr + 1):
                            for s in (-1, 1):
                                if s * off in aligned:
                                    stack.append(aligned[s * off])
                        fused = fuse(stack, mode)
                    e = sr_l2(model, fused)
                    epairs.append((e, ref[ci])); lps.append(lpips_perframe(lp, e, ref[ci]))
                vv = vmaf_perframe(epairs, tmp, name + "_" + meth)
                out[meth] = {"vmaf_mean": round(float(np.mean(vv)), 3),
                             "lpips_mean": round(float(np.mean(lps)), 5),
                             "vmaf_pf": [round(v, 3) for v in vv],
                             "lpips_pf": [round(v, 5) for v in lps]}
                print(f"  {meth:11s} VMAF={np.mean(vv):6.2f}  LPIPS={np.mean(lps):.4f}", flush=True)
        # 配对检验：各多帧法相对 single_l2（VMAF 正=多帧好；LPIPS 用 single-multi，正=多帧好）
        base_v = np.array(out["single_l2"]["vmaf_pf"])
        base_l = np.array(out["single_l2"]["lpips_pf"])
        for meth in methods[1:]:
            dv = np.array(out[meth]["vmaf_pf"]) - base_v
            dl = base_l - np.array(out[meth]["lpips_pf"])
            _, pv, nv = sign_test(dv); _, pl, nl = sign_test(dl)
            out[meth]["vs_single"] = {"dVMAF": round(float(dv.mean()), 3), "vmaf_p": round(pv, 5),
                                      "dLPIPS": round(float(dl.mean()), 5), "lpips_p": round(pl, 5),
                                      "n": nv}
        results[name] = out
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== multi-frame vs single_l2 summary ===")
    print(f"{'degradation':16s} {'method':11s} {'dVMAF':>7} {'vmaf_p':>8} {'dLPIPS':>8} {'lpips_p':>8}")
    for name, o in results.items():
        for meth in methods[1:]:
            s = o[meth]["vs_single"]
            print(f"{name:16s} {meth:11s} {s['dVMAF']:7.2f} {s['vmaf_p']:8.4f} "
                  f"{s['dLPIPS']:8.4f} {s['lpips_p']:8.4f}")
    print("WARP_ORACLE_DONE")


if __name__ == "__main__":
    main()
