# -*- coding: utf-8 -*-
"""
从 Phase1 / H0 写 multitask 先验 Q，并实测写先验 T（禁手填增益）。

用法：
  cd workspace/multi_enhance
  python scripts/build_priors.py                 # BBB（默认）
  python scripts/build_priors.py --dataset tos   # 需 data/quality/tos_*.json + h0_tos_*.json
  python scripts/build_priors.py --dataset tos --skip-latency
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# .../workspace/multi_enhance/scripts/build_priors.py → repo root = parents[3]
REPO = Path(__file__).resolve().parents[3]
WS = Path(__file__).resolve().parents[1]  # multi_enhance/
PLAYER = REPO / "player"
sys.path.insert(0, str(PLAYER))

from istream_player.modules.enhancer.imdn_model import IMDN, IMDN_RTC  # noqa: E402
from istream_player.modules.enhancer.tool_ids import (  # noqa: E402
    N_TOOLS,
    TOOL_NONE,
    TOOL_RESTORE_SHARP,
    TOOL_SR_HIGH,
    TOOL_SR_LOW,
    TOOL_SR_MED,
    TOOL_SR_X2_MED,
)

QUALITY_DIR = REPO / "data" / "quality"
WEIGHT_DIR = PLAYER / "istream_player" / "modules" / "enhancer" / "imdn_weight"
DEVICE = "cuda:0"
INVALID_Q = -1.0
ILLEGAL_T = 1e6

DATASETS = {
    "bbb": {
        "prefix": "bbb",
        "h0_glob": "h0_720_segs_20_30_40_50_60_80_90_100_120_140.json",
        "h0_dir": REPO / "workspace" / "hd_enhance_probe",
        "out_q": PLAYER / "multitask_bbb_quality.json",
        "out_t": PLAYER / "multitask_latency.json",
        "meta": WS / "priors" / "meta_bbb.json",
        "display": (1080, 1920),  # H, W
        # row -> (label, (h, w), scale)  scale 对齐 get_resolution_set
        "res": {
            0: ("240p", (240, 426), 4),
            1: ("360p", (360, 640), 3),
            2: ("480p", (480, 854), 2),
            3: ("720p", (720, 1280), 1),
            4: ("1080p", (1080, 1920), 1),
        },
    },
    "tos": {
        "prefix": "tos",
        "h0_glob": None,  # resolved at runtime: prefer longest h0_tos_segs_*.json
        "h0_dir": REPO / "workspace" / "hd_enhance_probe",
        "out_q": PLAYER / "multitask_tos_quality.json",
        "out_t": PLAYER / "multitask_latency_tos.json",
        "meta": WS / "priors" / "meta_tos.json",
        "display": (800, 1920),
        "res": {
            0: ("240p", (176, 426), 4),
            1: ("360p", (268, 640), 2),
            2: ("480p", (356, 854), 2),
            3: ("720p", (532, 1280), 1),
            4: ("1080p", (800, 1920), 1),
        },
    },
}

SR_LEVELS = {
    TOOL_SR_LOW: ("low", lambda s: IMDN_RTC(upscale=s, num_modules=3, nf=6)),
    TOOL_SR_MED: ("medium", lambda s: IMDN_RTC(upscale=s)),
    TOOL_SR_HIGH: ("high", lambda s: IMDN(upscale=s, nf=32)),
}


def _mean_frames(method_block: dict) -> float:
    frames = method_block["frames"]
    return float(sum(frames) / len(frames))


def load_phase1_row(prefix: str, res: int) -> dict:
    data = json.loads((QUALITY_DIR / f"{prefix}_{res}.json").read_text(encoding="utf-8"))
    return {k: _mean_frames(v) for k, v in data["methods"].items()}


def find_h0_json(cfg: dict) -> Path:
    h0_dir: Path = cfg["h0_dir"]
    if cfg["h0_glob"]:
        p = h0_dir / cfg["h0_glob"]
        if not p.exists():
            raise FileNotFoundError(p)
        return p
    cands = sorted(h0_dir.glob("h0_tos_segs_*.json"), key=lambda p: len(p.name), reverse=True)
    if not cands:
        smoke = h0_dir / "smoke_h0_tos.json"
        if smoke.exists():
            return smoke
        raise FileNotFoundError(f"no h0_tos_segs_*.json under {h0_dir}")
    return cands[0]


def load_h0_means(h0_path: Path) -> dict:
    h0 = json.loads(h0_path.read_text(encoding="utf-8"))
    keys = ["bicubic", "restore_unsharp", "sr_x2_medium"]
    if "per_seg" in h0:
        acc = {k: [] for k in keys}
        for seg in h0["per_seg"]:
            for k in keys:
                acc[k].append(float(seg["means"][k]))
        return {k: float(sum(v) / len(v)) for k, v in acc.items()}
    # single-seg smoke file
    means = h0.get("per_seg", [h0])[0]["means"] if "per_seg" in h0 else h0["means"]
    return {k: float(means[k]) for k in keys}


def build_quality_table(dataset: str) -> tuple[list, dict]:
    cfg = DATASETS[dataset]
    prefix = cfg["prefix"]
    h0_path = find_h0_json(cfg)
    q = np.full((5, N_TOOLS), INVALID_Q, dtype=float)
    phase1 = {}
    for r, height in enumerate((240, 360, 480)):
        means = load_phase1_row(prefix, height)
        phase1[height] = means
        q[r, TOOL_NONE] = means["bicubic"]
        q[r, TOOL_SR_LOW] = means["sr_low"]
        q[r, TOOL_SR_MED] = means["sr_medium"]
        q[r, TOOL_SR_HIGH] = means["sr_high"]

    h0 = load_h0_means(h0_path)
    q[3, TOOL_NONE] = h0["bicubic"]
    q[3, TOOL_RESTORE_SHARP] = h0["restore_unsharp"]
    q[3, TOOL_SR_X2_MED] = h0["sr_x2_medium"]
    q[4, TOOL_NONE] = 100.0
    # native display restore：若 H0 未测 800p，保持 INVALID

    meta = {
        "dataset": dataset,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "script": "workspace/multi_enhance/scripts/build_priors.py",
        "shape": [5, N_TOOLS],
        "cols": ["none", "sr_low", "sr_med", "sr_high", "sr_x2_med", "restore_sharp"],
        "rows": ["240p", "360p", "480p", "720p", "1080p"],
        "display_hw": list(cfg["display"]),
        "sources": {
            "240_360_480": f"data/quality/{prefix}_{{240,360,480}}.json",
            "720": str(h0_path.relative_to(REPO)),
            "1080_none": "convention 100.0",
            "1080_restore": "unmeasured → -1",
        },
        "phase1_means": phase1,
        "h0_means": h0,
        "h0_mean_delta_vs_bicubic": {
            "restore_sharp": h0["restore_unsharp"] - h0["bicubic"],
            "sr_x2_med": h0["sr_x2_medium"] - h0["bicubic"],
        },
        "invalid_sentinel": INVALID_Q,
    }
    return q.tolist(), meta


def _apply_unsharp(tensor: torch.Tensor, amount: float = 0.8, radius: int = 3) -> torch.Tensor:
    channels = tensor.shape[1]
    k = radius * 2 + 1
    sigma = max(radius / 2.0, 0.5)
    coords = torch.arange(k, device=tensor.device, dtype=tensor.dtype) - (k // 2)
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    kernel_2d = (g[:, None] * g[None, :]).expand(channels, 1, k, k).contiguous()
    pad = k // 2
    blurred = torch.nn.functional.conv2d(tensor, kernel_2d, padding=pad, groups=channels)
    return (tensor + amount * (tensor - blurred)).clamp(0.0, 1.0)


def measure_latency_table(dataset: str, warmup: int, measure: int) -> tuple[list, dict]:
    cfg = DATASETS[dataset]
    display = cfg["display"]
    res = cfg["res"]
    t = np.full((5, N_TOOLS), ILLEGAL_T, dtype=float)
    t[:, TOOL_NONE] = 0.0
    cells = {}

    def finish(out: torch.Tensor) -> torch.Tensor:
        out = torch.nn.functional.interpolate(
            out, size=display, mode="bicubic", align_corners=False
        )
        out = out.mul(255).clamp_(0, 255).to(dtype=torch.uint8)
        return out.cpu()

    def time_path(fn) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(measure):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / measure

    for r in (0, 1, 2):
        label, (h, w), scale = res[r]
        for tool in (TOOL_SR_LOW, TOOL_SR_MED, TOOL_SR_HIGH):
            tag, factory = SR_LEVELS[tool]
            model = factory(scale)
            path = WEIGHT_DIR / f"IMDN_{tag}_bbb_x{scale}.pth"
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model = model.to(DEVICE).eval()
            x = torch.rand((1, 3, h, w), device=DEVICE)

            def run(m=model, inp=x):
                with torch.no_grad():
                    _ = finish(m(inp))

            sec = time_path(run)
            t[r, tool] = sec
            cells[f"{label}_tool{tool}"] = {"s_per_frame": sec, "ms": sec * 1000.0}
            del model
            torch.cuda.empty_cache()

    # HD row: x2
    label3, (h3, w3), _ = res[3]
    model_x2 = IMDN_RTC(upscale=2)
    model_x2.load_state_dict(torch.load(WEIGHT_DIR / "IMDN_medium_bbb_x2.pth", map_location="cpu"))
    model_x2 = model_x2.to(DEVICE).eval()
    x_hd = torch.rand((1, 3, h3, w3), device=DEVICE)

    def run_x2():
        with torch.no_grad():
            _ = finish(model_x2(x_hd))

    sec_x2 = time_path(run_x2)
    t[3, TOOL_SR_X2_MED] = sec_x2
    cells[f"{label3}_sr_x2_med"] = {"s_per_frame": sec_x2, "ms": sec_x2 * 1000.0}
    del model_x2
    torch.cuda.empty_cache()

    for r in (3, 4):
        label, (h, w), _ = res[r]
        x = torch.rand((1, 3, h, w), device=DEVICE)

        def run_restore(inp=x):
            with torch.no_grad():
                out = torch.nn.functional.interpolate(
                    inp, size=display, mode="bicubic", align_corners=False
                )
                out = _apply_unsharp(out)
                out = out.mul(255).clamp_(0, 255).to(dtype=torch.uint8)
                _ = out.cpu()

        sec = time_path(run_restore)
        t[r, TOOL_RESTORE_SHARP] = sec
        cells[f"{label}_restore_sharp"] = {"s_per_frame": sec, "ms": sec * 1000.0}

    meta = {
        "dataset": dataset,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "script": "workspace/multi_enhance/scripts/build_priors.py",
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "display_hw": list(display),
        "res_hw": {str(k): {"label": v[0], "hw": list(v[1]), "scale": v[2]} for k, v in res.items()},
        "path": "enhance_one_frame-aligned: model|unsharp + bicubic(display) + uint8 + cpu",
        "warmup": warmup,
        "measure": measure,
        "unit": "seconds_per_frame",
        "illegal_sentinel": ILLEGAL_T,
        "cells": cells,
    }
    return t.tolist(), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), default="bbb")
    ap.add_argument("--skip-latency", action="store_true")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--measure", type=int, default=30)
    args = ap.parse_args()
    cfg = DATASETS[args.dataset]

    q_meta = t_meta = None
    if not args.skip_quality:
        q, q_meta = build_quality_table(args.dataset)
        cfg["out_q"].write_text(json.dumps(q, indent=4) + "\n", encoding="utf-8")
        print(f"wrote {cfg['out_q']}")
        print("quality deltas@720:", q_meta["h0_mean_delta_vs_bicubic"])

    if not args.skip_latency:
        if not torch.cuda.is_available():
            raise SystemExit("CUDA required for latency measurement")
        t, t_meta = measure_latency_table(args.dataset, args.warmup, args.measure)
        cfg["out_t"].write_text(json.dumps(t, indent=4) + "\n", encoding="utf-8")
        print(f"wrote {cfg['out_t']}")
        for k, v in sorted(t_meta["cells"].items()):
            print(f"  {k}: {v['ms']:.3f} ms/frame")

    cfg["meta"].write_text(
        json.dumps({"quality": q_meta, "latency": t_meta}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {cfg['meta']}")


if __name__ == "__main__":
    main()
