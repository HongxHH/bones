# -*- coding: utf-8 -*-
"""
TOS 全片帧级质量基线（对齐播放器 scale / 显示 1920×800）。

与 measure_quality_baseline.py 同口径：
  bicubic：ffmpeg YUV scale；SR：IMDN forward + bicubic 对齐显示尺寸；
  双输入 libvmaf + trim 锁帧。

scale = min(int(1920/w), int(800/h))，与 enhancer.get_resolution_set 一致：
  426×176 → 4；640×268 → 2；854×356 → 2；1280×532 → 1（仅 bicubic）。

用法：
  set BONES_FFMPEG=E:\\dev\\tool\\ffmpeg-6.1.1-full_build-shared\\bin\\ffmpeg.exe
  python measure_quality_tos.py --smoke
  python measure_quality_tos.py
  python measure_quality_tos.py --only 360
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
from istream_player.modules.enhancer.imdn_model import IMDN, IMDN_RTC

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = r"E:\dev\Code\Python\bones"
TMP = os.path.join(HERE, "tmp_tos")
os.makedirs(TMP, exist_ok=True)
OUT_DIR = os.path.join(REPO, "data", "quality")
os.makedirs(OUT_DIR, exist_ok=True)

TOS_DIR = os.path.join(REPO, "data", "tears_of_steel")
REF_VIDEO = os.path.join(TOS_DIR, "output_1080p.mp4")
W, H = 1920, 800
FPS = 24
FRAMES_PER_SEG = 96  # 4s * 24
TOTAL_FRAMES = 17620
FULL_SEGMENTS = TOTAL_FRAMES // FRAMES_PER_SEG  # 183 + remainder
DEVICE = "cuda:0"
FFMPEG = os.environ.get("BONES_FFMPEG", "ffmpeg")
WEIGHT_DIR = os.path.join(PLAYER, r"istream_player\modules\enhancer\imdn_weight")


def player_scale(lw: int, lh: int) -> int:
    return min(int(W / lw), int(H / lh))


# 名义档名保留 240/360/480/720，尺寸为 TOS 真实 ladder
SR_GRID = {
    240: (os.path.join(TOS_DIR, "output_240p.mp4"), (426, 176)),
    360: (os.path.join(TOS_DIR, "output_360p.mp4"), (640, 268)),
    480: (os.path.join(TOS_DIR, "output_480p.mp4"), (854, 356)),
}
BIC_ONLY = {
    720: (os.path.join(TOS_DIR, "output_720p.mp4"), (1280, 532)),
}
SR_ARMS = ["sr_low", "sr_medium", "sr_high"]
METHOD_META = {
    "bicubic": ("ffmpeg_scale_bicubic_yuv", None),
    "sr_low": ("IMDN_RTC num_modules=3 nf=6", "low"),
    "sr_medium": ("IMDN_RTC default nf=12 num_modules=5", "medium"),
    "sr_high": ("IMDN nf=32", "high"),
}


def md5_file(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def weight_path(level, scale):
    return os.path.join(WEIGHT_DIR, f"IMDN_{level}_bbb_x{scale}.pth")


def build_models(scale):
    models = {}
    specs = {
        "sr_low": IMDN_RTC(upscale=scale, num_modules=3, nf=6),
        "sr_medium": IMDN_RTC(upscale=scale),
        "sr_high": IMDN(upscale=scale, nf=32),
    }
    for name, model in specs.items():
        level = METHOD_META[name][1]
        model.load_state_dict(torch.load(weight_path(level, scale), map_location="cpu"))
        models[name] = model.to(DEVICE).eval()
    return models


def infer_rgb(model, rgb):
    t = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
    with torch.no_grad():
        o = model(t)
        o = torch.nn.functional.interpolate(
            o, size=(H, W), mode="bicubic", align_corners=False
        )
    o = o.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (o * 255).clip(0, 255).astype(np.uint8)


def err_file(name):
    return open(os.path.join(TMP, f"err_{name}.log"), "wb")


def start_lr_producer(lr_path, n_frames):
    cmd = [
        FFMPEG, "-v", "error", "-i", lr_path,
        "-vf", "format=rgb24", "-frames:v", str(n_frames),
        "-f", "rawvideo", "-",
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err_file("lr"))


def start_bic_consumer(lr_path, tag, n_frames):
    log_name = f"vmaf_{tag}_bicubic.json"
    p = os.path.join(TMP, log_name)
    if os.path.exists(p):
        os.remove(p)
    lavfi = (
        f"[0:v]trim=end_frame={n_frames},setpts=PTS-STARTPTS,"
        f"scale={W}:{H}:flags=bicubic,format=yuv420p[d];"
        f"[1:v]trim=end_frame={n_frames},setpts=PTS-STARTPTS[r];"
        f"[d][r]libvmaf=n_threads=1:log_path={log_name}:log_fmt=json"
    )
    cmd = [
        FFMPEG, "-v", "error", "-i", lr_path, "-i", REF_VIDEO,
        "-lavfi", lavfi, "-f", "null", "-",
    ]
    return subprocess.Popen(
        cmd, cwd=TMP, stdout=subprocess.DEVNULL, stderr=err_file(f"{tag}_bicubic")
    ), log_name


def start_sr_consumer(tag, arm, n_frames):
    log_name = f"vmaf_{tag}_{arm}.json"
    p = os.path.join(TMP, log_name)
    if os.path.exists(p):
        os.remove(p)
    lavfi = (
        f"[0:v]format=yuv420p,trim=end_frame={n_frames},setpts=PTS-STARTPTS[d];"
        f"[1:v]trim=end_frame={n_frames},setpts=PTS-STARTPTS[r];"
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
        stdout=subprocess.DEVNULL, stderr=err_file(f"{tag}_{arm}"),
    )
    q = queue.Queue(maxsize=4)

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


def read_vmaf_json(path):
    with open(path, "r", encoding="utf-8") as f:
        log = json.load(f)
    return [float(fr["metrics"]["vmaf"]) for fr in log["frames"]]


def check_rc(proc, name):
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg rc={rc} for {name}; see {TMP}/err_{name}.log")


def measure_sr_input(res, smoke=False):
    lr_path, (lw, lh) = SR_GRID[res]
    scale = player_scale(lw, lh)
    if scale < 2:
        raise RuntimeError(f"tos{res} scale={scale} unexpected for SR row")
    tag = f"tos{res}" + ("_smoke" if smoke else "")
    n_target = FRAMES_PER_SEG if smoke else TOTAL_FRAMES
    print(f"START {tag} size={lw}x{lh} scale={scale} frames={n_target}", flush=True)

    models = build_models(scale)
    p_lr = start_lr_producer(lr_path, n_target)
    p_bic, bic_log = start_bic_consumer(lr_path, tag, n_target)
    sr_procs, sr_qs, sr_threads, sr_logs = {}, {}, {}, {}
    for arm in SR_ARMS:
        p, q, th, lg = start_sr_consumer(tag, arm, n_target)
        sr_procs[arm], sr_qs[arm], sr_threads[arm], sr_logs[arm] = p, q, th, lg

    n = 0
    lr_nbytes = lw * lh * 3
    while n < n_target:
        buf = p_lr.stdout.read(lr_nbytes)
        if len(buf) < lr_nbytes:
            break
        lr = np.frombuffer(buf, dtype=np.uint8).reshape(lh, lw, 3)
        for arm in SR_ARMS:
            sr_qs[arm].put(infer_rgb(models[arm], lr).tobytes())
        n += 1
        if n % 2000 == 0:
            print(f"  tos{res} frames {n}/{n_target}", flush=True)

    if n != n_target:
        raise RuntimeError(f"LR frame count {n} != {n_target} tos{res}")
    check_rc(p_lr, "lr")
    for arm in SR_ARMS:
        sr_qs[arm].put(None)
    for arm in SR_ARMS:
        sr_threads[arm].join()
        check_rc(sr_procs[arm], f"{tag}_{arm}")
    check_rc(p_bic, f"{tag}_bicubic")

    arm_logs = {"bicubic": bic_log}
    arm_logs.update({arm: sr_logs[arm] for arm in SR_ARMS})
    frame_scores = {}
    for arm, lg in arm_logs.items():
        s = read_vmaf_json(os.path.join(TMP, lg))
        if len(s) != n_target:
            raise RuntimeError(f"vmaf {arm} len {len(s)} != {n_target} tos{res}")
        frame_scores[arm] = s

    if smoke:
        print(f"SMOKE tos{res} {n} frames OK; arm means:", flush=True)
        for arm in ["bicubic"] + SR_ARMS:
            print(f"  {arm}: {np.mean(frame_scores[arm]):.3f}", flush=True)
        for lg in arm_logs.values():
            try:
                os.remove(os.path.join(TMP, lg))
            except OSError:
                pass
        return frame_scores

    methods = {}
    for arm in ["bicubic"] + SR_ARMS:
        kind, level = METHOD_META[arm]
        if level is None:
            wp, wmd5 = None, None
        else:
            wp_full = weight_path(level, scale)
            wp = os.path.relpath(wp_full, REPO)
            wmd5 = md5_file(wp_full)
        methods[arm] = {
            "kind": kind,
            "weights": wp,
            "weights_md5": wmd5,
            "frames": [round(float(x), 4) for x in frame_scores[arm]],
        }

    result = {
        "video": "tos",
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "ref_video": os.path.relpath(REF_VIDEO, REPO),
        "input_video": os.path.relpath(lr_path, REPO),
        "input_size": [lw, lh],
        "target_size": [W, H],
        "scale": scale,
        "scale_rule": "min(int(display_w/w), int(display_h/h))",
        "fps": FPS,
        "frames_total": TOTAL_FRAMES,
        "full_segments": FULL_SEGMENTS,
        "frames_per_seg": FRAMES_PER_SEG,
        "libvmaf_model": "default vmaf_v0.6.1",
        "libvmaf_n_threads": 1,
        "methods": methods,
    }
    out_path = os.path.join(OUT_DIR, f"tos_{res}.json")
    json.dump(result, open(out_path, "w", encoding="utf-8"), separators=(",", ":"))
    for lg in arm_logs.values():
        try:
            os.remove(os.path.join(TMP, lg))
        except OSError:
            pass
    print(f"DONE tos{res} -> {out_path}", flush=True)


def measure_bic_only(res):
    lr_path, (lw, lh) = BIC_ONLY[res]
    tag = f"tos{res}"
    print(f"START {tag} bicubic-only {lw}x{lh}", flush=True)
    proc, log_name = start_bic_consumer(lr_path, tag, TOTAL_FRAMES)
    check_rc(proc, f"{tag}_bicubic")
    frames = read_vmaf_json(os.path.join(TMP, log_name))
    if len(frames) != TOTAL_FRAMES:
        raise RuntimeError(f"vmaf len {len(frames)} != {TOTAL_FRAMES} tos{res}")
    result = {
        "video": "tos",
        "measured_at": datetime.now().isoformat(timespec="seconds"),
        "ref_video": os.path.relpath(REF_VIDEO, REPO),
        "input_video": os.path.relpath(lr_path, REPO),
        "input_size": [lw, lh],
        "target_size": [W, H],
        "scale": 1,
        "fps": FPS,
        "frames_total": TOTAL_FRAMES,
        "full_segments": FULL_SEGMENTS,
        "frames_per_seg": FRAMES_PER_SEG,
        "libvmaf_model": "default vmaf_v0.6.1",
        "libvmaf_n_threads": 1,
        "methods": {
            "bicubic": {
                "kind": "ffmpeg_scale_bicubic_yuv",
                "weights": None,
                "weights_md5": None,
                "frames": [round(float(x), 4) for x in frames],
            },
        },
    }
    out_path = os.path.join(OUT_DIR, f"tos_{res}.json")
    json.dump(result, open(out_path, "w", encoding="utf-8"), separators=(",", ":"))
    try:
        os.remove(os.path.join(TMP, log_name))
    except OSError:
        pass
    print(f"DONE tos{res} -> {out_path}", flush=True)


def output_done(res):
    p = os.path.join(OUT_DIR, f"tos_{res}.json")
    if not os.path.exists(p):
        return False
    d = json.load(open(p, "r", encoding="utf-8"))
    narms = 1 if res == 720 else 4
    return (
        d.get("video") == "tos"
        and d.get("frames_total") == TOTAL_FRAMES
        and d.get("target_size") == [W, H]
        and len(d["methods"]) == narms
        and all(len(m["frames"]) == TOTAL_FRAMES for m in d["methods"].values())
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--ffmpeg", default=None)
    args = ap.parse_args()
    global FFMPEG
    if args.ffmpeg:
        FFMPEG = args.ffmpeg
    elif os.environ.get("BONES_FFMPEG"):
        FFMPEG = os.environ["BONES_FFMPEG"]

    if args.smoke:
        t0 = datetime.now()
        measure_sr_input(args.only or 240, smoke=True)
        dt = (datetime.now() - t0).total_seconds()
        print(
            f"SMOKE wall {dt:.1f}s for {FRAMES_PER_SEG} frames; "
            f"extrapolated per SR input full film ~{dt * TOTAL_FRAMES / FRAMES_PER_SEG / 60:.1f} min",
            flush=True,
        )
        return

    for res in (240, 360, 480):
        if args.only and args.only != res:
            continue
        if output_done(res):
            print(f"SKIP tos{res} (done)")
            continue
        measure_sr_input(res)
    if not args.only or args.only == 720:
        if output_done(720):
            print("SKIP tos720 (done)")
        else:
            measure_bic_only(720)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
