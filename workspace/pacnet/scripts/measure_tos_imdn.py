"""
IMDN DIV2K 基线在 Tears of Steel 上的 VMAF 测量

加载播放器基线使用的三个通用 DIV2K 模型（low/medium/high），
测量 240/360/480p 下的 VMAF，生成 imdn_tos_quality.json，
与 PAC-Net 的 tos_quality.json 保持相同 no_enh 基准以公平对比。
"""
import os
import sys
import json
import subprocess
import tempfile
import numpy as np
import torch
import cv2

sys.path.insert(0, r"E:\dev\Code\Python\bones\player")

from istream_player.modules.enhancer.imdn_model import IMDN, IMDN_RTC

DEVICE = "cuda:0"
VIDEO_DIR = r"E:\dev\Code\Python\bones\data\tears_of_steel"
WEIGHT_DIR = r"E:\dev\Code\Python\bones\player\istream_player\modules\enhancer\imdn_weight"
REF_VIDEO = r"E:\dev\Data\tears_of_steel_1080p.mp4"
COMPRESSED = {
    "240p": os.path.join(VIDEO_DIR, "output_240p.mp4"),
    "360p": os.path.join(VIDEO_DIR, "output_360p.mp4"),
    "480p": os.path.join(VIDEO_DIR, "output_480p.mp4"),
    "720p": os.path.join(VIDEO_DIR, "output_720p.mp4"),
}
OUT_JSON = r"E:\dev\Code\Python\bones\workspace\pacnet\tos_imdn_vmaf.json"
QT_OUT = r"E:\dev\Code\Python\bones\data\tears_of_steel\imdn_tos_quality.json"

FPS = 24
TARGET = (1920, 800)
N = 16


def build_models():
    models = {}
    # level1: IMDN_RTC nf=6 num_modules=3
    m1 = IMDN_RTC(upscale=4, num_modules=3, nf=6)
    m1.load_state_dict(torch.load(os.path.join(WEIGHT_DIR, "IMDN_low_DIV2K_x4.pth"), map_location="cpu"))
    models["l1"] = m1.to(DEVICE).eval()
    # level2: IMDN_RTC 默认 nf=12 num_modules=5
    m2 = IMDN_RTC(upscale=4)
    m2.load_state_dict(torch.load(os.path.join(WEIGHT_DIR, "IMDN_medium_DIV2K_x4.pth"), map_location="cpu"))
    models["l2"] = m2.to(DEVICE).eval()
    # level3: IMDN nf=32
    m3 = IMDN(upscale=4, nf=32)
    m3.load_state_dict(torch.load(os.path.join(WEIGHT_DIR, "IMDN_high_DIV2K_x4.pth"), map_location="cpu"))
    models["l3"] = m3.to(DEVICE).eval()
    print("3 IMDN DIV2K models loaded")
    return models


def frame_at(path, idx):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, f = cap.read()
    cap.release()
    return f if ret else None


def enhance(model, bgr):
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
    with torch.no_grad():
        o = model(t)
        o = torch.nn.functional.interpolate(o, size=(TARGET[1], TARGET[0]),
                                            mode="bicubic", align_corners=False)
    o = o.squeeze(0).permute(1, 2, 0).cpu().numpy()
    o = (o * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(o, cv2.COLOR_RGB2BGR)


def bic(bgr):
    return cv2.resize(bgr, TARGET, interpolation=cv2.INTER_CUBIC)


def vmaf(pairs, tmp, tag):
    ep = os.path.join(tmp, f"e_{tag}.mp4")
    rp = os.path.join(tmp, f"r_{tag}.mp4")
    h, w = pairs[0][0].shape[:2]
    fc = cv2.VideoWriter_fourcc(*'mp4v')
    we, wr = cv2.VideoWriter(ep, fc, FPS, (w, h)), cv2.VideoWriter(rp, fc, FPS, (w, h))
    for e, r in pairs:
        we.write(e); wr.write(r)
    we.release(); wr.release()
    res = subprocess.run(["ffmpeg", "-i", ep, "-i", rp, "-lavfi", "libvmaf", "-f", "null", "-"],
                         capture_output=True, text=True, timeout=120)
    for line in res.stderr.split('\n'):
        if "VMAF score" in line:
            return float(line.split("VMAF score:")[1].strip())
    return None


def main():
    models = build_models()
    results = {}
    for res in ["240p", "360p", "480p", "720p"]:
        path = COMPRESSED[res]
        cap = cv2.VideoCapture(path)
        total = int(cap.get(7)); cap.release()
        idxs = np.linspace(0, total - 10, N, dtype=int)
        results[res] = {}
        methods = ["bicubic", "l1", "l2", "l3"] if res != "720p" else ["bicubic"]
        print(f"\n=== {res} ({total} frames) ===")
        with tempfile.TemporaryDirectory() as tmp:
            for meth in methods:
                pairs = []
                for fi in idxs:
                    lr = frame_at(path, int(fi))
                    rf = frame_at(REF_VIDEO, int(fi))
                    if lr is None or rf is None:
                        continue
                    if meth == "bicubic":
                        ef = bic(lr)
                    else:
                        ef = enhance(models[meth], lr)
                    pairs.append((ef, rf))
                v = vmaf(pairs, tmp, f"imdn_{res}_{meth}") if pairs else None
                results[res][meth] = v
                print(f"  {meth}: {v:.2f}" if v else f"  {meth}: FAILED")

    # 组装 5x5 表
    qt = [[-1.0]*5 for _ in range(5)]
    for row, res in enumerate(["240p", "360p", "480p"]):
        r = results[res]
        qt[row][0] = r["bicubic"]
        qt[row][1] = r["l1"]
        qt[row][2] = r["l2"]
        qt[row][3] = r["l3"]
    qt[3][0] = results["720p"]["bicubic"]
    qt[4][0] = 100.0
    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    with open(QT_OUT, "w") as f:
        json.dump(qt, f, indent=2)

    print("\n=== IMDN TOS Quality Table ===")
    for i, res in enumerate(["240p", "360p", "480p", "720p", "1080p"]):
        print("%-6s %8.2f %8.2f %8.2f %8.2f" % (res, qt[i][0], qt[i][1], qt[i][2], qt[i][3]))
    print(f"\nSaved {QT_OUT}")


if __name__ == "__main__":
    main()
