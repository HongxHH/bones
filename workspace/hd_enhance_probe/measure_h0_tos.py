# -*- coding: utf-8 -*-
"""
H0-TOS：1280×532 → 1920×800 显示尺子下，bicubic / unsharp / IMDN×2 轻 SR。

口径对齐 quality_baseline：双输入 trim 锁帧 + libvmaf n_threads=1。
TOS 24fps、段长 4s → 96 帧/段；默认 --smoke 用 seg=40。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch

PLAYER = r"E:\dev\Code\Python\bones\player"
sys.path.insert(0, PLAYER)
from istream_player.modules.enhancer.imdn_model import IMDN_RTC

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r"E:\dev\Code\Python\bones"
TMP = os.path.join(HERE, "tmp_tos")
os.makedirs(TMP, exist_ok=True)

TOS_DIR = os.path.join(REPO, r"data\tears_of_steel")
REF_VIDEO = os.path.join(TOS_DIR, "output_1080p.mp4")
LR_VIDEO = os.path.join(TOS_DIR, "output_720p.mp4")
W, H = 1920, 800
LW, LH = 1280, 532
FPS = 24
FRAMES_PER_SEG = 96
DEFAULT_SMOKE_SEG = 40
DEVICE = "cuda:0"
FFMPEG = os.environ.get("BONES_FFMPEG", "ffmpeg")
WEIGHT_DIR = os.path.join(PLAYER, r"istream_player\modules\enhancer\imdn_weight")

UNSHARP = "unsharp=5:5:0.8:5:5:0.0"

SR_ARMS = ("sr_x2_low", "sr_x2_medium")
SR_SPEC = {
    "sr_x2_low": ("low", IMDN_RTC(upscale=2, num_modules=3, nf=6)),
    "sr_x2_medium": ("medium", IMDN_RTC(upscale=2)),
}


def md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weight_path(level: str) -> str:
    return os.path.join(WEIGHT_DIR, f"IMDN_{level}_bbb_x2.pth")


def err_file(name: str):
    return open(os.path.join(TMP, f"err_{name}.log"), "wb")


def trim_range(start: int, n_frames: int) -> str:
    return f"trim=start_frame={start}:end_frame={start + n_frames},setpts=PTS-STARTPTS"


def start_ffmpeg_arm(tag: str, lavfi_distort: str, start: int, n_frames: int):
    """纯 ffmpeg 臂：LR + REF → libvmaf。lavfi_distort 接在 trim 之后，产出 [d]。"""
    log_name = f"vmaf_{tag}.json"
    p = os.path.join(TMP, log_name)
    if os.path.exists(p):
        os.remove(p)
    tr = trim_range(start, n_frames)
    lavfi = (
        f"[0:v]{tr},{lavfi_distort}[d];"
        f"[1:v]{tr}[r];"
        f"[d][r]libvmaf=n_threads=1:log_path={log_name}:log_fmt=json"
    )
    cmd = [
        FFMPEG, "-v", "error", "-i", LR_VIDEO, "-i", REF_VIDEO,
        "-lavfi", lavfi, "-f", "null", "-",
    ]
    proc = subprocess.Popen(
        cmd, cwd=TMP, stdout=subprocess.DEVNULL, stderr=err_file(tag),
    )
    return proc, log_name


def start_lr_producer(start: int, n_frames: int):
    tr = trim_range(start, n_frames)
    cmd = [
        FFMPEG, "-v", "error", "-i", LR_VIDEO,
        "-vf", f"{tr},format=rgb24",
        "-frames:v", str(n_frames),
        "-f", "rawvideo", "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_file("lr"))


def start_sr_consumer(tag: str, start: int, n_frames: int):
    """stdin 已是截好的 n 帧；参考侧必须 trim 到同一段。"""
    log_name = f"vmaf_{tag}.json"
    p = os.path.join(TMP, log_name)
    if os.path.exists(p):
        os.remove(p)
    tr_ref = trim_range(start, n_frames)
    lavfi = (
        f"[0:v]format=yuv420p,trim=end_frame={n_frames},setpts=PTS-STARTPTS[d];"
        f"[1:v]{tr_ref}[r];"
        f"[d][r]libvmaf=n_threads=1:log_path={log_name}:log_fmt=json"
    )
    cmd = [
        FFMPEG, "-v", "error",
        "-f", "rawvideo", "-pixel_format", "rgb24",
        "-video_size", f"{W}x{H}", "-framerate", str(FPS), "-i", "-",
        "-i", REF_VIDEO, "-lavfi", lavfi, "-f", "null", "-",
    ]
    proc = subprocess.Popen(
        cmd, cwd=TMP, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=err_file(tag),
    )
    q: queue.Queue = queue.Queue(maxsize=4)

    def feeder():
        while True:
            item = q.get()
            if item is None:
                proc.stdin.close()
                return
            proc.stdin.write(item)

    th = threading.Thread(target=feeder, daemon=True)
    th.start()
    return proc, q, th, log_name


def build_sr_models():
    models = {}
    meta = {}
    for arm, (level, model) in SR_SPEC.items():
        wp = weight_path(level)
        model.load_state_dict(torch.load(wp, map_location="cpu"))
        models[arm] = model.to(DEVICE).eval()
        meta[arm] = {
            "kind": f"IMDN_RTC_{level}_x2_then_bicubic_800",
            "weights": os.path.relpath(wp, REPO),
            "weights_md5": md5_file(wp),
        }
    return models, meta


def infer_x2_to_display(model, rgb: np.ndarray) -> np.ndarray:
    t = (
        torch.from_numpy(np.array(rgb, copy=True))
        .permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
    )
    with torch.no_grad():
        o = model(t)
        o = torch.nn.functional.interpolate(
            o, size=(H, W), mode="bicubic", align_corners=False,
        )
    o = o.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (o * 255).clip(0, 255).astype(np.uint8)


def read_vmaf_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        log = json.load(f)
    return [float(fr["metrics"]["vmaf"]) for fr in log["frames"]]


def check_rc(proc, name: str):
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg rc={rc} for {name}; see tmp/err_{name}.log")


def run_probe(seg: int, n_frames: int, models=None, sr_meta=None) -> dict:
    start = seg * FRAMES_PER_SEG
    tag = f"h0_tos_seg{seg}"
    if models is None:
        models, sr_meta = build_sr_models()

    p_bic, log_bic = start_ffmpeg_arm(
        f"{tag}_bicubic",
        f"scale={W}:{H}:flags=bicubic,format=yuv420p",
        start, n_frames,
    )
    p_restore, log_restore = start_ffmpeg_arm(
        f"{tag}_restore_unsharp",
        f"scale={W}:{H}:flags=bicubic,{UNSHARP},format=yuv420p",
        start, n_frames,
    )

    p_lr = start_lr_producer(start, n_frames)
    sr_state = {}
    for arm in SR_ARMS:
        p, q, th, lg = start_sr_consumer(f"{tag}_{arm}", start, n_frames)
        sr_state[arm] = (p, q, th, lg)

    n = 0
    nbytes = LW * LH * 3
    while n < n_frames:
        buf = p_lr.stdout.read(nbytes)
        if len(buf) < nbytes:
            break
        lr = np.frombuffer(buf, dtype=np.uint8).reshape(LH, LW, 3)
        for arm in SR_ARMS:
            sr_state[arm][1].put(infer_x2_to_display(models[arm], lr).tobytes())
        n += 1
        if n % 48 == 0:
            print(f"  seg{seg} frames {n}/{n_frames}", flush=True)

    if n != n_frames:
        raise RuntimeError(f"LR frames {n} != {n_frames} (seg={seg} start={start})")

    check_rc(p_lr, "lr")
    for arm in SR_ARMS:
        p, q, th, _ = sr_state[arm]
        q.put(None)
        th.join()
        check_rc(p, f"{tag}_{arm}")
    check_rc(p_bic, f"{tag}_bicubic")
    check_rc(p_restore, f"{tag}_restore_unsharp")

    scores = {
        "bicubic": read_vmaf_json(os.path.join(TMP, log_bic)),
        "restore_unsharp": read_vmaf_json(os.path.join(TMP, log_restore)),
    }
    for arm in SR_ARMS:
        scores[arm] = read_vmaf_json(os.path.join(TMP, sr_state[arm][3]))

    for arm, s in scores.items():
        if len(s) != n_frames:
            raise RuntimeError(f"vmaf {arm} len {len(s)} != {n_frames}")

    means = {arm: float(np.mean(s)) for arm, s in scores.items()}
    base = means["bicubic"]
    deltas = {arm: means[arm] - base for arm in means}

    result = {
        "probe": "H0_TOS_532",
        "seg": seg,
        "start_frame": start,
        "n_frames": n_frames,
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "ref_video": os.path.relpath(REF_VIDEO, REPO),
        "input_video": os.path.relpath(LR_VIDEO, REPO),
        "input_size": [LW, LH],
        "target_size": [W, H],
        "ffmpeg": FFMPEG,
        "unsharp": UNSHARP,
        "means": {k: round(v, 4) for k, v in means.items()},
        "delta_vs_bicubic": {k: round(v, 4) for k, v in deltas.items()},
        "gate": {
            "go_threshold_delta": 1.0,
            "go_arms": [
                arm for arm, d in deltas.items()
                if arm != "bicubic" and d >= 1.0
            ],
        },
        "methods": {
            "bicubic": {"kind": "ffmpeg_scale_bicubic_yuv"},
            "restore_unsharp": {"kind": f"bicubic+{UNSHARP}"},
            **sr_meta,
        },
    }

    for lg in [log_bic, log_restore] + [sr_state[a][3] for a in SR_ARMS]:
        try:
            os.remove(os.path.join(TMP, lg))
        except OSError:
            pass
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--smoke", action="store_true",
        help=f"单段 {FRAMES_PER_SEG} 帧；默认 seg={DEFAULT_SMOKE_SEG}",
    )
    ap.add_argument("--seg", type=int, default=None, help="4s 段号（0-based）")
    ap.add_argument(
        "--segs", type=str, default=None,
        help="逗号分隔多段，如 40,70,100",
    )
    ap.add_argument("--frames", type=int, default=FRAMES_PER_SEG)
    ap.add_argument("--ffmpeg", default=None)
    args = ap.parse_args()
    global FFMPEG
    if args.ffmpeg:
        FFMPEG = args.ffmpeg
    elif os.environ.get("BONES_FFMPEG"):
        FFMPEG = os.environ["BONES_FFMPEG"]

    if args.segs:
        segs = [int(x) for x in args.segs.split(",") if x.strip() != ""]
    elif args.seg is not None:
        segs = [args.seg]
    elif args.smoke:
        segs = [DEFAULT_SMOKE_SEG]
    else:
        segs = [DEFAULT_SMOKE_SEG]

    n = args.frames
    print(f"H0-TOS 532p→800 segs={segs} n={n} ffmpeg={FFMPEG}", flush=True)
    models, sr_meta = build_sr_models()
    t0 = datetime.now()
    per_seg = []
    for seg in segs:
        print(f"=== seg {seg} (frames {seg * FRAMES_PER_SEG}..{seg * FRAMES_PER_SEG + n - 1}) ===",
              flush=True)
        r = run_probe(seg, n, models=models, sr_meta=sr_meta)
        per_seg.append(r)
        print("  means:", r["means"], flush=True)
        print("  delta:", r["delta_vs_bicubic"], flush=True)

    dt = (datetime.now() - t0).total_seconds()
    # 跨段平均 delta
    arms = list(per_seg[0]["means"].keys())
    mean_delta = {
        arm: float(np.mean([s["delta_vs_bicubic"][arm] for s in per_seg]))
        for arm in arms
    }
    go_arms = [
        arm for arm, d in mean_delta.items()
        if arm != "bicubic" and d >= 1.0
    ]
    out_obj = {
        "probe": "H0_TOS_532",
        "note": "TOS 1280x532→1920x800; 默认/冒烟用 seg>=40",
        "segs": segs,
        "n_frames_per_seg": n,
        "wall_s": round(dt, 1),
        "mean_delta_vs_bicubic": {k: round(v, 4) for k, v in mean_delta.items()},
        "gate": {"go_threshold_delta": 1.0, "go_arms_by_mean_delta": go_arms},
        "per_seg": per_seg,
    }
    out = os.path.join(
        HERE,
        "smoke_h0_tos.json" if len(segs) == 1 else f"h0_tos_segs_{'_'.join(map(str, segs))}.json",
    )
    with open(out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    print("mean_delta_vs_bicubic:", out_obj["mean_delta_vs_bicubic"], flush=True)
    print(
        f"GO arms (>=+1 mean): {go_arms if go_arms else '(none)'} → "
        f"{'GO expand' if go_arms else 'NO-GO for these arms'}",
        flush=True,
    )
    print(f"wall {dt:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
