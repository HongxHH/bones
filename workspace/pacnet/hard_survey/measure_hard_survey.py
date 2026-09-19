# -*- coding: utf-8 -*-
"""
Anytime 价值区间证伪：在更难增强任务（更高倍率 + 更重压缩）上测轻量 vs 重量模型质量差。

退化：从 1080p TOS 源 bicubic 下采样到 426(4.5x)/320(6x) 宽，再 libx264 CRF26/34/42；
另以现有 output_240p.mp4(426宽/400kbps) 作标准锚点。
模型：IMDN l1(IMDN_RTC nf=6) / l2(IMDN_RTC nf=12) / l3(IMDN nf=32)，
      PAC e1/e2/e3（共享 nf=32 骨干，stage2_mix_nocond 权重），加 bicubic 基线。
统一 x4 超分后 bicubic 对齐 1920x800，对 1080p 源算 VMAF(libvmaf) 与 Y-PSNR。
判据：退化加重时若轻模型先饱和/塌陷、重模型继续涨（l2->e3 差拉大到 >=3~5 VMAF），
      anytime 在难任务上有价值区间；若差距不拉开，轻量 CNN 超分无 anytime 空间。
"""
import os, sys, json, subprocess, tempfile
import numpy as np
import torch, cv2

sys.path.insert(0, r"E:\dev\Code\Python\bones\player")
sys.path.insert(0, r"E:\dev\Code\Python\bones\workspace\pacnet")
from istream_player.modules.enhancer.imdn_model import IMDN, IMDN_RTC
from pacnet_model import PACNet

DEVICE = "cuda:0"
HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHT_DIR = r"E:\dev\Code\Python\bones\player\istream_player\modules\enhancer\imdn_weight"
PAC_W = r"E:\dev\Code\Python\bones\workspace\pacnet\weights\stage2_mix_nocond_x4_best.pth"
REF_VIDEO = r"E:\dev\Data\tears_of_steel_1080p.mp4"
ANCHOR_240 = r"E:\dev\Code\Python\bones\data\tears_of_steel\output_240p.mp4"
TARGET = (1920, 800)
FPS, N = 24, 32
OUT_JSON = os.path.join(HERE, "hard_survey.json")

DEG = {
    "anchor_426_400k": ANCHOR_240,
    "426_crf26": os.path.join(HERE, "lr_426_crf26.mp4"),
    "426_crf34": os.path.join(HERE, "lr_426_crf34.mp4"),
    "426_crf42": os.path.join(HERE, "lr_426_crf42.mp4"),
    "320_crf26": os.path.join(HERE, "lr_320_crf26.mp4"),
    "320_crf34": os.path.join(HERE, "lr_320_crf34.mp4"),
    "320_crf42": os.path.join(HERE, "lr_320_crf42.mp4"),
}


def build_models():
    m = {}
    m["l1"] = IMDN_RTC(upscale=4, num_modules=3, nf=6)
    m["l1"].load_state_dict(torch.load(os.path.join(WEIGHT_DIR, "IMDN_low_DIV2K_x4.pth"), map_location="cpu"))
    m["l2"] = IMDN_RTC(upscale=4)
    m["l2"].load_state_dict(torch.load(os.path.join(WEIGHT_DIR, "IMDN_medium_DIV2K_x4.pth"), map_location="cpu"))
    m["l3"] = IMDN(upscale=4, nf=32)
    m["l3"].load_state_dict(torch.load(os.path.join(WEIGHT_DIR, "IMDN_high_DIV2K_x4.pth"), map_location="cpu"))
    pac = PACNet(nf=32, upscale=4, exits=(2, 4, 6), cond=False)
    pac.load_state_dict(torch.load(PAC_W, map_location="cpu"))
    for k in m: m[k] = m[k].to(DEVICE).eval()
    pac = pac.to(DEVICE).eval()
    return m, pac


def read_frames(path, idxs):
    cap = cv2.VideoCapture(path)
    out = {}
    idx_set = set(int(i) for i in idxs)
    i = 0
    while True:
        ret, f = cap.read()
        if not ret: break
        if i in idx_set: out[i] = f
        i += 1
    cap.release()
    return [out[int(i)] for i in idxs if int(i) in out]


def infer(model, bgr, pac=None, exit_idx=None):
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0
    with torch.no_grad():
        o = model(t, exit_idx=exit_idx) if pac is not None else model(t)
        o = torch.nn.functional.interpolate(o, size=(TARGET[1], TARGET[0]),
                                            mode="bicubic", align_corners=False)
    o = o.squeeze(0).permute(1, 2, 0).cpu().numpy()
    o = (o * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(o, cv2.COLOR_RGB2BGR)


def bic(bgr):
    return cv2.resize(bgr, TARGET, interpolation=cv2.INTER_CUBIC)


def ypsnr(a_bgr, b_bgr):
    a = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float64)
    b = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2YUV)[:, :, 0].astype(np.float64)
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse < 1e-9 else float(10 * np.log10(255.0 ** 2 / mse))


def vmaf(pairs, tmp, tag):
    ep, rp = os.path.join(tmp, f"e_{tag}.mp4"), os.path.join(tmp, f"r_{tag}.mp4")
    h, w = pairs[0][0].shape[:2]
    fc = cv2.VideoWriter_fourcc(*"mp4v")
    we, wr = cv2.VideoWriter(ep, fc, FPS, (w, h)), cv2.VideoWriter(rp, fc, FPS, (w, h))
    for e, r in pairs: we.write(e); wr.write(r)
    we.release(); wr.release()
    res = subprocess.run(["ffmpeg", "-i", ep, "-i", rp, "-lavfi", "libvmaf", "-f", "null", "-"],
                         capture_output=True, text=True, timeout=180)
    for line in res.stderr.split("\n"):
        if "VMAF score" in line:
            return float(line.split("VMAF score:")[1].strip())
    return None


def main():
    mdl, pac = build_models()
    methods = ["bicubic", "l1", "l2", "l3", "e1", "e2", "e3"]
    results = {}
    # 参考帧索引（以源总帧数）
    cap = cv2.VideoCapture(REF_VIDEO); total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    idxs = np.linspace(0, total - 10, N, dtype=int)
    ref = read_frames(REF_VIDEO, idxs)
    print("ref frames", len(ref), "total", total)

    for name, path in DEG.items():
        if not os.path.exists(path):
            print("SKIP missing", path); continue
        lr = read_frames(path, idxs)
        n = min(len(lr), len(ref))
        lr, rf = lr[:n], ref[:n]
        results[name] = {}
        print(f"\n=== {name}  ({n} frames) ===")
        with tempfile.TemporaryDirectory() as tmp:
            for meth in methods:
                pairs, ps = [], []
                for x, r in zip(lr, rf):
                    if meth == "bicubic":
                        e = bic(x)
                    elif meth in ("l1", "l2", "l3"):
                        e = infer(mdl[meth], x)
                    else:
                        e = infer(pac, x, pac=True, exit_idx={"e1": 0, "e2": 1, "e3": 2}[meth])
                    pairs.append((e, r)); ps.append(ypsnr(e, r))
                v = vmaf(pairs, tmp, name + "_" + meth)
                results[name][meth] = {"vmaf": v, "psnr": round(float(np.mean(ps)), 3)}
                print(f"  {meth:8s} VMAF={v if v is None else round(v,2)}  PSNR={round(float(np.mean(ps)),2)}")
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # 汇总：l2->e3、l1->e3 差距
    print("\n=== gap summary (VMAF) ===")
    print(f"{'degradation':16s} {'bic':>6} {'l1':>6} {'l2':>6} {'l3':>6} {'e1':>6} {'e2':>6} {'e3':>6} {'l2-e3':>6} {'l1-e3':>6}")
    for name in DEG:
        if name not in results: continue
        r = results[name]
        def g(k): return r[k]["vmaf"]
        try:
            print(f"{name:16s} {g('bicubic'):6.2f} {g('l1'):6.2f} {g('l2'):6.2f} {g('l3'):6.2f} "
                  f"{g('e1'):6.2f} {g('e2'):6.2f} {g('e3'):6.2f} {g('l2')-g('e3'):6.2f} {g('l1')-g('e3'):6.2f}")
        except Exception as ex:
            print(name, "incomplete", ex)


if __name__ == "__main__":
    main()
