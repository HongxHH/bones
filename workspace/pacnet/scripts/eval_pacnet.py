"""阶段0证伪评测：16 个 uniform 留出段，整帧 PSNR（与闭环 val_psnr 同口径）+ 真实帧时延。

方法：base(bicubic) / PAC e1,e2,e3 / plain2(与e1同算力固定浅模型) /
      imdn_div(通用6-block) / imdn_lo(对症6-block，e3同算力上界)。
逐帧结果落 eval_pac16.json，判据由 analyze_pacnet.py 计算。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import cv2
import torch
import torch.nn.functional as F

ROOT = r"E:\dev\Code\Python\bones"
PACDIR = os.path.join(ROOT, "workspace/pacnet")
SURVEY = os.path.join(ROOT, "workspace/enhancer_survey")
WDIR = os.path.join(PACDIR, "weights")
sys.path.insert(0, os.path.join(ROOT, "player"))
sys.path.insert(0, PACDIR)
from pacnet_model import PACNet, PlainIMDN  # noqa: E402
from istream_player.modules.enhancer.imdn_model import IMDN  # noqa: E402

DEV = "cuda:0"
FPS_SEG = 120
TEST_SEGS = list(range(0, 159, 10))           # 16 段，含困难段 130/140/150
FRAME_JS = (15, 45, 75, 105)
HARD = (130, 140, 150)


def to_tensor(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(DEV)


def psnr_to_gt(out_t, hr_bgr):
    up = F.interpolate(out_t, size=(1080, 1920), mode="bicubic", align_corners=False).clamp(0, 1)
    g = torch.from_numpy(cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                         ).float().div(255).unsqueeze(0).to(DEV)
    mse = ((up - g) ** 2).mean().item()
    return 10 * np.log10(1.0 / max(mse, 1e-12))


@torch.no_grad()
def latency_ms(fn, lr_t, reps=30):
    for _ in range(5):
        fn(lr_t)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.time(); fn(lr_t); torch.cuda.synchronize(); ts.append((time.time() - t0) * 1000)
    return round(float(np.median(ts)), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pac_w", default="pacnet_x4_best.pth")
    ap.add_argument("--plain_w", default="plain2_x4_best.pth")
    ap.add_argument("--use_cond", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(PACDIR, "eval_pac16.json"))
    args = ap.parse_args()

    pac = PACNet(nf=32, upscale=4, exits=(2, 4, 6), cond=bool(args.use_cond)).to(DEV).eval()
    pac.load_state_dict(torch.load(os.path.join(WDIR, args.pac_w), map_location="cpu"))
    plain = PlainIMDN(nf=32, num_modules=2, upscale=4).to(DEV).eval()
    plain.load_state_dict(torch.load(os.path.join(WDIR, args.plain_w), map_location="cpu"))
    wdir = os.path.join(ROOT, "player/istream_player/modules/enhancer/imdn_weight")
    imdn_div = IMDN(upscale=4, nf=32).to(DEV).eval()
    imdn_div.load_state_dict(torch.load(os.path.join(wdir, "IMDN_high_DIV2K_x4.pth"), map_location="cpu"))
    imdn_lo = IMDN(upscale=4, nf=32).to(DEV).eval()
    imdn_lo.load_state_dict(torch.load(
        os.path.join(SURVEY, "weights/IMDN_high_bbbLO_x4_best.pth"), map_location="cpu"))

    cap_lr = cv2.VideoCapture(os.path.join(ROOT, "data/video_mp4box/output_240p.mp4"))
    cap_hr = cv2.VideoCapture(os.path.join(ROOT, "data/video_mp4box/output_1080p.mp4"))
    want = {(s, j) for s in TEST_SEGS for j in FRAME_JS}
    results = {}
    lat = {}
    f = 0
    t0 = time.time()
    while True:
        ok1, lr = cap_lr.read(); ok2, hr = cap_hr.read()
        if not (ok1 and ok2):
            break
        seg, j = divmod(f, FPS_SEG)
        if (seg, j) in want:
            x = to_tensor(lr)
            base = cv2.resize(lr, (1920, 1080), interpolation=cv2.INTER_CUBIC)
            row = {"base": round(psnr_to_gt(to_tensor(base), hr), 4)}
            outs = pac(x)
            for k, o in enumerate(outs):
                row[f"e{k+1}"] = round(psnr_to_gt(o, hr), 4)
            row["plain2"] = round(psnr_to_gt(plain(x), hr), 4)
            row["imdn_div"] = round(psnr_to_gt(imdn_div(x), hr), 4)
            row["imdn_lo"] = round(psnr_to_gt(imdn_lo(x), hr), 4)
            results.setdefault(str(seg), {})[str(j)] = row
            if not lat:  # 第一张真实帧测时延
                lat["e1"] = latency_ms(lambda t: pac(t, exit_idx=0), x)
                lat["e2"] = latency_ms(lambda t: pac(t, exit_idx=1), x)
                lat["e3"] = latency_ms(lambda t: pac(t, exit_idx=2), x)
                lat["plain2"] = latency_ms(plain, x)
                lat["imdn6"] = latency_ms(imdn_lo, x)
            print(f"seg{seg} f{j}: " + " ".join(f"{k}={v}" for k, v in row.items()), flush=True)
        f += 1
    cap_lr.release(); cap_hr.release()

    out = {"frame_psnr": results, "latency_ms_240p": lat,
           "test_segs": TEST_SEGS, "frame_js": list(FRAME_JS), "hard": list(HARD),
           "sec": round(time.time() - t0, 1)}
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("saved", args.out, "latency", lat, flush=True)


if __name__ == "__main__":
    main()
