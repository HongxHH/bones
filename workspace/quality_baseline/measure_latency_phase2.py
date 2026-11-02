# -*- coding: utf-8 -*-
"""
阶段2：测 IMDN 全路径单帧 wall 耗时，并按 burn K 扫描选出 (C, F)。

口径对齐播放器 enhance_one_frame：
  forward（可 burn）+ bicubic 对齐 1080p + uint8 + .cpu()
burn 只放大 forward（与 enhancer_imdn_always 一致），因此用
  T(K) = K * fwd_ms + post_ms
而不是 K * 全路径。

用法：
  python measure_latency_phase2.py
  python measure_latency_phase2.py --warmup 5 --measure 30 --primary-q 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch

PLAYER = r"E:\dev\Code\Python\bones\player"
sys.path.insert(0, PLAYER)
from istream_player.modules.enhancer.imdn_model import IMDN, IMDN_RTC
from istream_player.modules.enhancer import _gpuburn as gpuburn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r"E:\dev\Code\Python\bones"
OUT = os.path.join(HERE, "phase2_select_cf.json")
QUALITY = os.path.join(REPO, "data", "quality")
WEIGHT_DIR = os.path.join(PLAYER, r"istream_player\modules\enhancer\imdn_weight")
DEVICE = "cuda:0"
H, W = 1080, 1920
FPS = 30
SEG_FRAMES = 120
SEG_S = SEG_FRAMES / FPS

RES_GRID = {
    0: ("240p", 4, (426, 240)),
    1: ("360p", 3, (640, 360)),
    2: ("480p", 2, (854, 480)),
}
LEVELS = {
    1: ("sr_low", "low", lambda s: IMDN_RTC(upscale=s, num_modules=3, nf=6)),
    2: ("sr_medium", "medium", lambda s: IMDN_RTC(upscale=s)),
    3: ("sr_high", "high", lambda s: IMDN(upscale=s, nf=32)),
}
K_SWEEP = [1.0, 1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 4.0, 5.0]


def reset_burn(k: float = 0.0):
    os.environ.pop("PACNET_EXTRA_MS", None)
    if k and k > 1.0:
        os.environ["PACNET_BURN_K"] = str(k)
    else:
        os.environ.pop("PACNET_BURN_K", None)
    gpuburn._state = {
        "inited": False, "k": 1.0, "extra": 0.0,
        "single": 0.5, "a": None, "b": None,
    }


def weight_path(level: str, scale: int) -> str:
    return os.path.join(WEIGHT_DIR, f"IMDN_{level}_bbb_x{scale}.pth")


def load_model(level: int, scale: int) -> torch.nn.Module:
    _name, wtag, factory = LEVELS[level]
    model = factory(scale)
    model.load_state_dict(torch.load(weight_path(wtag, scale), map_location="cpu"))
    return model.to(DEVICE).eval()


def one_frame_split(model, tensor):
    """返回 (fwd_ms, post_ms, total_ms)，无 burn。"""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.cuda.synchronize()
        tf0 = time.perf_counter()
        out = model(tensor)
        torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - tf0) * 1000.0
        out = torch.nn.functional.interpolate(
            out, size=(H, W), mode="bicubic", align_corners=False
        )
        out = out.mul(255).clamp_(0, 255).to(dtype=torch.uint8)
    out = out.cpu()
    torch.cuda.synchronize()
    _ = float(out[0, 0, 0, 0])
    total = (time.perf_counter() - t0) * 1000.0
    post_ms = max(total - fwd_ms, 0.0)
    return fwd_ms, post_ms, total


def one_frame_burn(model, tensor):
    """整路径 wall ms；调用前须已 reset_burn(K) 且只 init 一次。"""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        torch.cuda.synchronize()
        tf0 = time.perf_counter()
        out = model(tensor)
        torch.cuda.synchronize()
        fwd_ms = (time.perf_counter() - tf0) * 1000.0
        gpuburn.burn(fwd_ms)
        out = torch.nn.functional.interpolate(
            out, size=(H, W), mode="bicubic", align_corners=False
        )
        out = out.mul(255).clamp_(0, 255).to(dtype=torch.uint8)
    out = out.cpu()
    torch.cuda.synchronize()
    _ = float(out[0, 0, 0, 0])
    return (time.perf_counter() - t0) * 1000.0


def measure_cell(level: int, scale: int, lh: int, lw: int, warmup: int, n: int):
    reset_burn(0)
    model = load_model(level, scale)
    data = [torch.rand((1, 3, lh, lw), device=DEVICE) for _ in range(warmup + n)]
    for i in range(warmup):
        one_frame_split(model, data[i])
    fwds, posts, totals = [], [], []
    for i in range(n):
        f, p, t = one_frame_split(model, data[warmup + i])
        fwds.append(f); posts.append(p); totals.append(t)
    fwd = float(np.mean(fwds))
    post = float(np.mean(posts))
    tot = float(np.mean(totals))
    cell = {
        "fwd_ms": fwd,
        "post_ms": post,
        "total_ms": tot,
        "std_total_ms": float(np.std(totals, ddof=1)) if n > 1 else 0.0,
        "p50_total_ms": float(np.median(totals)),
        "n": n,
        "seg_s_k1": tot * SEG_FRAMES / 1000.0,
        "realtime_ratio_k1": (tot * SEG_FRAMES / 1000.0) / SEG_S,
    }
    del model
    torch.cuda.empty_cache()
    return cell


def quality_gaps():
    gaps = {}
    for q_idx, (label, _scale, _) in RES_GRID.items():
        res = [240, 360, 480][q_idx]
        path = os.path.join(QUALITY, f"bbb_{res}.json")
        d = json.load(open(path, encoding="utf-8"))
        means = {k: float(np.mean(v["frames"])) for k, v in d["methods"].items()}
        gaps[q_idx] = {
            "label": label,
            "means": means,
            "low_minus_bic": means["sr_low"] - means["bicubic"],
            "med_minus_bic": means["sr_medium"] - means["bicubic"],
            "high_minus_bic": means["sr_high"] - means["bicubic"],
        }
    return gaps


def t_ms(cell, K: float) -> float:
    return K * cell["fwd_ms"] + cell["post_ms"]


def select_cf(cells_by_ql, gaps, primary_q, min_vmaf_gap=2.0):
    candidates = []
    for K in K_SWEEP:
        for C in (2, 3):
            for F in (1,):
                if F >= C:
                    continue
                c_cell = cells_by_ql[(primary_q, C)]
                f_cell = cells_by_ql[(primary_q, F)]
                c_seg = t_ms(c_cell, K) * SEG_FRAMES / 1000.0
                f_seg = t_ms(f_cell, K) * SEG_FRAMES / 1000.0
                gap = gaps[primary_q]["low_minus_bic"]
                ok_time = c_seg > SEG_S and f_seg < SEG_S
                ok_vmaf = gap >= min_vmaf_gap
                candidates.append({
                    "K": K,
                    "C": C,
                    "F": F,
                    "primary_q": primary_q,
                    "primary_label": RES_GRID[primary_q][0],
                    "C_ms": round(t_ms(c_cell, K), 3),
                    "F_ms": round(t_ms(f_cell, K), 3),
                    "C_seg_s": round(c_seg, 3),
                    "F_seg_s": round(f_seg, 3),
                    "F_minus_bic_vmaf": round(gap, 3),
                    "ok_time": ok_time,
                    "ok_vmaf": ok_vmaf,
                    "ok": bool(ok_time and ok_vmaf),
                })
    winners = [c for c in candidates if c["ok"]]
    # 最小 K，且优先 C=2（always_greedy 默认）
    winners.sort(key=lambda c: (c["K"], c["C"]))
    return (winners[0] if winners else None), candidates


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--measure", type=int, default=30)
    ap.add_argument("--primary-q", type=int, default=None)
    ap.add_argument("--burn-probe-k", type=float, default=1.5)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")

    primary_q = args.primary_q
    if primary_q is None:
        primary_q = int(os.environ.get("PHASE2_PRIMARY_Q", "1"))  # 默认 360p

    gaps = quality_gaps()
    print("quality gaps:", json.dumps(
        {str(k): {kk: (round(vv, 3) if isinstance(vv, float) else vv)
                  for kk, vv in v.items() if kk != "means"}
         for k, v in gaps.items()}, indent=2), flush=True)

    cells_by_ql = {}
    cells_out = {}
    for q, (label, scale, (lw, lh)) in RES_GRID.items():
        for level in (1, 2, 3):
            print(f"measure {label} L{level} x{scale} ...", flush=True)
            cell = measure_cell(level, scale, lh, lw, args.warmup, args.measure)
            cells_by_ql[(q, level)] = cell
            cells_out[f"{label}_L{level}"] = cell
            print(
                f"  fwd={cell['fwd_ms']:.3f} post={cell['post_ms']:.3f} "
                f"tot={cell['total_ms']:.3f} ms | seg_k1={cell['seg_s_k1']:.3f}s "
                f"({cell['realtime_ratio_k1']:.2f}x RT)",
                flush=True,
            )

    burn_check = None
    if args.burn_probe_k > 1.0:
        label, scale, (lw, lh) = RES_GRID[primary_q]
        model = load_model(2, scale)
        data = [torch.rand((1, 3, lh, lw), device=DEVICE) for _ in range(args.warmup + args.measure)]
        reset_burn(args.burn_probe_k)
        gpuburn._ensure()  # matmul 标定只做一次
        for i in range(args.warmup):
            one_frame_burn(model, data[i])
        samples = [one_frame_burn(model, data[args.warmup + i]) for i in range(args.measure)]
        burn_ms = float(np.mean(samples))
        base = cells_by_ql[(primary_q, 2)]
        pred = t_ms(base, args.burn_probe_k)
        burn_check = {
            "K": args.burn_probe_k,
            "measured_total_ms": burn_ms,
            "predicted_total_ms": pred,
            "base_fwd_ms": base["fwd_ms"],
            "base_post_ms": base["post_ms"],
            "base_total_ms": base["total_ms"],
            "meas_over_pred": None if pred <= 0 else burn_ms / pred,
        }
        print(f"burn probe {label} L2 K={args.burn_probe_k}: "
              f"meas={burn_ms:.3f} pred={pred:.3f} ms "
              f"(ratio={burn_check['meas_over_pred']:.3f})", flush=True)
        del model
        torch.cuda.empty_cache()
        reset_burn(0)

    choice, candidates = select_cf(cells_by_ql, gaps, primary_q)
    # 若主分辨率无解，尝试其他分辨率报告
    alt = {}
    for q in RES_GRID:
        if q == primary_q:
            continue
        c, _ = select_cf(cells_by_ql, gaps, q)
        alt[RES_GRID[q][0]] = c

    result = {
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "device": torch.cuda.get_device_name(0),
        "seg_frames": SEG_FRAMES,
        "seg_s": SEG_S,
        "warmup": args.warmup,
        "measure": args.measure,
        "path": "fwd + interpolate1080 + uint8 + cpu; burn on fwd only",
        "primary_q": primary_q,
        "primary_label": RES_GRID[primary_q][0],
        "quality_gaps": gaps,
        "cells_k1": cells_out,
        "burn_probe": burn_check,
        "selection": choice,
        "alt_selection_by_res": alt,
        "candidates_primary": candidates,
        "verdict": "GO" if choice else "EMPTY",
    }
    json.dump(result, open(OUT, "w", encoding="utf-8"), indent=2)
    print("Wrote", OUT, flush=True)
    print("VERDICT", result["verdict"], flush=True)
    print(json.dumps(choice, indent=2), flush=True)


if __name__ == "__main__":
    main()
