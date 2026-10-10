# -*- coding: utf-8 -*-
"""一次性对齐核查：ref_c{i}.mkv 下采样帧 vs lr1080_crf34 同时间戳帧 PSNR。
另测 ±1 帧偏移作灵敏度对照。2026-10-10 审计用，跑完可删。"""
import subprocess
import numpy as np

HERE = r"E:\dev\Code\Python\bones\workspace\pacnet\modern_sr"
STARTS = [60, 220, 380, 540]
W, H = 1920, 1080


def grab(args):
    p = subprocess.run(args, capture_output=True)
    if p.returncode != 0 or len(p.stdout) < W * H * 3:
        raise RuntimeError(f"grab failed rc={p.returncode} len={len(p.stdout)}: {p.stderr[-300:].decode('utf-8','ignore')}")
    return np.frombuffer(p.stdout[:W*H*3], np.uint8).reshape(H, W, 3).astype(np.float32)


def ref_frame(ci, t_in):
    return grab(["ffmpeg", "-v", "error", "-ss", str(t_in), "-i",
                 f"{HERE}/tmp/ref_c{ci}.mkv", "-frames:v", "1",
                 "-vf", "scale=1920:1080:flags=bicubic",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])


def lr_frame(ci, t_abs):
    return grab(["ffmpeg", "-v", "error", "-ss", str(t_abs), "-i",
                 f"{HERE}/ladder/lr1080_crf34.mp4", "-frames:v", "1",
                 "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])


def psnr(a, b):
    mse = ((a - b) ** 2).mean()
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


for ci, st in enumerate(STARTS):
    for t_in in (0.0, 10.0):
        r = ref_frame(ci, t_in)
        l = lr_frame(ci, st + t_in)
        print(f"c{ci} t+{t_in:>4.1f}s aligned PSNR={psnr(r, l):6.2f} dB", flush=True)

# 灵敏度：c0 t=10，故意偏移 ±1 帧（±1/60s）
r = ref_frame(0, 10.0)
for off in (-1/60, 1/60):
    l = lr_frame(0, 60 + 10.0 + off)
    print(f"sensitivity off={off*1000:+.1f}ms PSNR={psnr(r, l):6.2f} dB")
