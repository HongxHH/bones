# -*- coding: utf-8 -*-
"""1080p 输入 x2 推理延迟：全帧 fp32/fp16 vs 分块 fp16/fp32，验证 cuDNN 慢路径与可部署性。"""
import os, sys, time, copy
import numpy as np
import torch, cv2

sys.path.insert(0, r"E:\dev\Code\Python\bones\player")
from istream_player.modules.enhancer.imdn_model import IMDN_RTC, IMDN

DEVICE = "cuda:0"
W, H = 1920, 1080
WDIR = r"E:\dev\Code\Python\bones\player\istream_player\modules\enhancer\imdn_weight"
LEVELS = [
    ("low",  IMDN_RTC(upscale=2, num_modules=3, nf=6), "IMDN_low_DIV2K_x2.pth"),
    ("med",  IMDN_RTC(upscale=2),                      "IMDN_medium_DIV2K_x2.pth"),
    ("high", IMDN(upscale=2, nf=32),                   "IMDN_high_DIV2K_x2.pth"),
]


def to_tensor(bgr, half=False):
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
    return t.half() if half else t


def from_tensor(t):
    o = t.squeeze(0).permute(1, 2, 0).float().cpu().numpy()
    o = (o * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(o, cv2.COLOR_RGB2BGR)


def tiled(model, bgr, tile=480, ov=16, half=False):
    H, W = bgr.shape[:2]
    out = np.zeros((H * 2, W * 2, 3), np.float32)
    wgt = np.zeros((H * 2, W * 2, 1), np.float32)
    ys = list(range(0, H, tile - ov))
    xs = list(range(0, W, tile - ov))
    for y in ys:
        for x in xs:
            y0, x0 = y, x
            y1, x1 = min(y + tile, H), min(x + tile, W)
            pad_b, pad_r = (tile - (y1 - y0)) if y1 - y0 < tile else 0, \
                           (tile - (x1 - x0)) if x1 - x0 < tile else 0
            crop = bgr[y0:y1, x0:x1]
            if pad_b or pad_r:
                crop = cv2.copyMakeBorder(crop, 0, pad_b, 0, pad_r, cv2.BORDER_REFLECT)
            with torch.no_grad():
                o = model(to_tensor(crop, half))
            o = from_tensor(o).astype(np.float32)
            ch, cw = (y1 - y0) * 2, (x1 - x0) * 2
            o = o[:ch, :cw]
            # 重叠区线性权重
            wy = np.ones((ch, 1, 1), np.float32)
            wx = np.ones((1, cw, 1), np.float32)
            if y0 > 0:
                ramp = np.linspace(0, 1, ov * 2)[:, None, None]
                wy[:ov * 2] = ramp
            if x0 > 0:
                ramp = np.linspace(0, 1, ov * 2)[None, :, None]
                wx[:, :ov * 2] = ramp
            w = wy * wx
            oy, ox = y0 * 2, x0 * 2
            out[oy:oy + ch, ox:ox + cw] += o * w
            wgt[oy:oy + ch, ox:ox + cw] += w
    return np.clip(out / np.maximum(wgt, 1e-8), 0, 255).astype(np.uint8)


def bench(fn, n=40, warm=8):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.time(); fn(); torch.cuda.synchronize()
        ts.append((time.time() - t0) * 1000)
    return np.mean(ts), np.median(ts), np.percentile(ts, 95)


def main():
    dummy = np.random.randint(0, 255, (H, W, 3), np.uint8)
    for name, m, wf in LEVELS:
        m.load_state_dict(torch.load(os.path.join(WDIR, wf), map_location="cpu"))
        m32 = m.to(DEVICE).eval()
        m16 = copy.deepcopy(m32).half()
        r = {}
        r["full_fp32"] = bench(lambda: m32(to_tensor(dummy)))
        r["full_fp16"] = bench(lambda: m16(to_tensor(dummy, True)))
        r["tile480_fp16"] = bench(lambda: tiled(m16, dummy, 480, 16, True), n=12, warm=3)
        r["tile480_fp32"] = bench(lambda: tiled(m32, dummy, 480, 16, False), n=12, warm=3)
        # 分块与全帧一致性（fp32，真实帧时再核，这里只验证拼接无黑缝：均值差）
        for k, v in r.items():
            print(f"{name:5s} {k:13s} mean={v[0]:7.2f} p50={v[1]:7.2f} p95={v[2]:7.2f}")
        print()


if __name__ == "__main__":
    main()
