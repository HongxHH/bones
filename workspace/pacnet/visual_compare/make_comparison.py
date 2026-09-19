# -*- coding: utf-8 -*-
"""生成模型前/后/参考的真实帧放大对比图（TOS，hard_survey 同口径）。

每档退化输出一张 PNG：4 列 = Bicubic（模型前）/ IMDN l2（轻量主力）/ PAC e3（我们的最深出口）
/ Reference（1080p 源）；每列上为整帧缩略、下为同一高细节区域的放大裁切。
裁切在 LR 上先取区域再各自放大，保证看到真实输入块粒与各方法差异。
"""
import os
import sys

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, os.path.join(ROOT, "workspace/pacnet/hard_survey"))
sys.path.insert(0, os.path.join(ROOT, "workspace/pacnet"))
sys.path.insert(0, os.path.join(ROOT, "player"))
import measure_hard_survey as H  # noqa: E402
from pacnet_model import PACNet  # noqa: E402

DEV = "cuda:0"
OUT = HERE
CW, CH = 620, 360           # 目标坐标系下放大裁切尺寸
COLW = 470                  # 图中每列宽
N = 32

CASES = {
    "easy_426crf26": "426_crf26",
    "mid_426crf34": "426_crf34",
    "hard_320crf34": "320_crf34",
}


def build_models():
    l2 = H.IMDN_RTC(upscale=4)
    l2.load_state_dict(torch.load(os.path.join(H.WEIGHT_DIR, "IMDN_medium_DIV2K_x4.pth"),
                                  map_location="cpu"))
    pac = PACNet(nf=32, upscale=4, exits=(2, 4, 6), cond=False)
    pac.load_state_dict(torch.load(H.PAC_W, map_location="cpu"))
    return l2.to(DEV).eval(), pac.to(DEV).eval()


@torch.no_grad()
def sr_to(model, bgr, size, pac_exit=None):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(DEV) / 255.0
    o = model(t, exit_idx=pac_exit) if pac_exit is not None else model(t)
    o = torch.nn.functional.interpolate(o, size=(size[1], size[0]),
                                        mode="bicubic", align_corners=False)
    o = o.squeeze(0).permute(1, 2, 0).cpu().numpy()
    o = (o * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(o, cv2.COLOR_RGB2BGR)


def bic_to(bgr, size):
    return cv2.resize(bgr, size, interpolation=cv2.INTER_CUBIC)


def pick_crop(ref, cw, ch):
    """在参考帧上选高细节且尽量靠近画面中心的窗口。"""
    g = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    e = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) ** 2 + cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) ** 2
    Hh, Ww = ref.shape[:2]
    cx, cy = Ww / 2, Hh / 2
    best, bs = None, -1
    for y in range(0, Hh - ch, 24):
        for x in range(0, Ww - cw, 24):
            energy = float(e[y:y + ch, x:x + cw].mean())
            dist = np.hypot(x + cw / 2 - cx, y + ch / 2 - cy) / np.hypot(cx, cy)
            score = energy * (1.0 - 0.35 * dist)
            if score > bs:
                bs, best = score, (x, y)
    return best


def pick_frame(refs):
    """选中心区域细节较丰富的一帧。"""
    best, bi = -1, len(refs) // 2
    for i, r in enumerate(refs):
        h, w = r.shape[:2]
        c = r[int(h * 0.25):int(h * 0.8), int(w * 0.25):int(w * 0.8)]
        g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        s = float(cv2.Sobel(g, cv2.CV_32F, 1, 0).var())
        if s > best:
            best, bi = s, i
    return bi


def band(text, w, h=28):
    b = np.full((h, w, 3), 255, np.uint8)
    cv2.putText(b, text, (8, h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (20, 20, 20), 2, cv2.LINE_AA)
    return b


def panel(full_bgr, zoom_bgr, title):
    ov = cv2.resize(full_bgr, (COLW, int(COLW * H.TARGET[1] / H.TARGET[0])))
    zh = int(COLW * zoom_bgr.shape[0] / zoom_bgr.shape[1])
    z = cv2.resize(zoom_bgr, (COLW, zh))
    return np.vstack([band(title, COLW), ov, band("zoom", COLW, 22), z])


def build_fig(l2, pac, tag, deg, t, ref_full):
    path = H.DEG[deg]
    lr = H.read_frames(path, [t])[0]
    lh, lw = lr.shape[:2]
    sx, sy = H.TARGET[0] / lw, H.TARGET[1] / lh
    x, y = pick_crop(ref_full, CW, CH)
    lx0, ly0 = int(round(x / sx)), int(round(y / sy))
    lx1, ly1 = int(round((x + CW) / sx)), int(round((y + CH) / sy))
    lcrop = lr[max(0, ly0):ly1, max(0, lx0):lx1]
    rcrop = ref_full[y:y + CH, x:x + CW]
    cw_t, ch_t = rcrop.shape[1], rcrop.shape[0]

    full_bic = bic_to(lr, H.TARGET)
    full_l2 = sr_to(l2, lr, H.TARGET)
    full_e3 = sr_to(pac, lr, H.TARGET, pac_exit=2)
    z_bic = bic_to(lcrop, (cw_t, ch_t))
    z_l2 = sr_to(l2, lcrop, (cw_t, ch_t))
    z_e3 = sr_to(pac, lcrop, (cw_t, ch_t), pac_exit=2)

    cols = [
        panel(full_bic, z_bic, "Bicubic (no model)"),
        panel(full_l2, z_l2, "IMDN l2 (light SR)"),
        panel(full_e3, z_e3, "PAC e3 (deepest)"),
        panel(ref_full, rcrop, "Reference 1080p"),
    ]
    gap = np.full((cols[0].shape[0], 8, 3), 245, np.uint8)
    row = cols[0]
    for c in cols[1:]:
        row = np.hstack([row, gap, c])
    head = np.full((34, row.shape[1], 3), 255, np.uint8)
    cv2.putText(head, f"{deg}  (LR {lw}x{lh}, frame {t})", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    fig = np.vstack([head, row])
    fp = os.path.join(OUT, f"compare_{tag}.png")
    cv2.imwrite(fp, fig)
    print("saved", fp, fig.shape, flush=True)


def main():
    l2, pac = build_models()
    cap = cv2.VideoCapture(H.REF_VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    idxs = np.linspace(0, total - 10, N, dtype=int)
    refs = H.read_frames(H.REF_VIDEO, idxs)
    # 自然场景帧：只在前 60% 采样帧里选高细节（避开片尾字幕）
    fi_scene = pick_frame(refs[:int(N * 0.6)])
    ts = int(idxs[fi_scene])
    print("scene frame idx", ts, "#", fi_scene, flush=True)
    for tag, deg in CASES.items():
        build_fig(l2, pac, tag, deg, ts, refs[fi_scene])
    # 另出一张片尾字幕（细节最密）展示锐化/伪影权衡
    fi_txt = pick_frame(refs)
    tt = int(idxs[fi_txt])
    print("text frame idx", tt, "#", fi_txt, flush=True)
    build_fig(l2, pac, "text_426crf26", "426_crf26", tt, refs[fi_txt])
    print("COMPARE_DONE")


if __name__ == "__main__":
    main()
