# -*- coding: utf-8 -*-
"""
复验"轻模型≈重模型"是不是 VMAF 失灵的假象。
在 hard_survey 同样的 7 档退化 x 7 方法 x 32 帧、统一 1920x800 对齐口径下，
除 VMAF 外补三类全参考感知指标：
  LPIPS(squeeze)  学习型感知距离，越低越好（SR 论文标配）
  CW-SSIM         复小波结构相似，越高越好（对模糊/纹理敏感，无需训练权重）
  VIF             视觉信息保真，越高越好（VMAF 底层特征同族）
输出 hard_survey_perceptual.json，并与 hard_survey.json 的 VMAF 并排，
重点看 l2(5.1ms 轻模型) -> e3(33.8ms 重出口) 在感知指标下差距是否拉开。
"""
import os, sys, json
import numpy as np
import torch, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_hard_survey as H  # 复用 build_models/read_frames/infer/bic/DEG/REF_VIDEO/TARGET/N

OUT_JSON = os.path.join(HERE, "hard_survey_perceptual.json")
DEVICE = "cuda:0"


def build_iqa():
    import lpips
    import pyiqa
    m = {}
    m["lpips"] = lpips.LPIPS(net="squeeze").to(DEVICE).eval()   # 越低越好
    m["cwssim"] = pyiqa.create_metric("cw_ssim", device=DEVICE)  # 越高越好
    m["vif"] = pyiqa.create_metric("vif", device=DEVICE)         # 越高越好
    # DISTS 需下载 VGG16(528MB)，默认不启用；权重就绪后设环境变量 ENABLE_DISTS=1 再补
    if os.environ.get("ENABLE_DISTS") == "1":
        m["dists"] = pyiqa.create_metric("dists", device=DEVICE)  # 越低越好
        print("[iqa] dists enabled")
    return m


def to_t(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0


@torch.no_grad()
def score(metrics, e_bgr, r_bgr):
    te, tr = to_t(e_bgr), to_t(r_bgr)
    out = {}
    out["lpips"] = float(metrics["lpips"](te * 2 - 1, tr * 2 - 1))
    out["cwssim"] = float(metrics["cwssim"](te, tr))
    out["vif"] = float(metrics["vif"](te, tr))
    if "dists" in metrics:
        out["dists"] = float(metrics["dists"](te, tr))
    return out


def main():
    mdl, pac = H.build_models()
    metrics = build_iqa()
    methods = ["bicubic", "l1", "l2", "l3", "e1", "e2", "e3"]
    cap = cv2.VideoCapture(H.REF_VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    idxs = np.linspace(0, total - 10, H.N, dtype=int)
    ref = H.read_frames(H.REF_VIDEO, idxs)
    print("ref frames", len(ref), "total", total)

    results = {}
    for name, path in H.DEG.items():
        if not os.path.exists(path):
            print("SKIP missing", path); continue
        lr = H.read_frames(path, idxs)
        n = min(len(lr), len(ref)); lr, rf = lr[:n], ref[:n]
        results[name] = {}
        print(f"\n=== {name} ({n} frames) ===")
        for meth in methods:
            acc = {}
            for x, r in zip(lr, rf):
                if meth == "bicubic":
                    e = H.bic(x)
                elif meth in ("l1", "l2", "l3"):
                    e = H.infer(mdl[meth], x)
                else:
                    e = H.infer(pac, x, pac=True, exit_idx={"e1": 0, "e2": 1, "e3": 2}[meth])
                s = score(metrics, e, r)
                for k, v in s.items(): acc.setdefault(k, []).append(v)
            results[name][meth] = {k: round(float(np.mean(v)), 5) for k, v in acc.items()}
            r0 = results[name][meth]
            print(f"  {meth:8s} lpips={r0['lpips']:.4f} cwssim={r0['cwssim']:.4f} vif={r0['vif']:.4f}"
                  + (f" dists={r0['dists']:.4f}" if "dists" in r0 else ""))
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # 汇总：l2 vs e3 差距（方向统一为"正=e3 更好"）
    vmaf = {}
    vj = os.path.join(HERE, "hard_survey.json")
    if os.path.exists(vj):
        vmaf = json.load(open(vj, encoding="utf-8"))
    print("\n=== l2 -> e3 gap (正=重模型 e3 更好) ===")
    hdr = f"{'degradation':16s} {'VMAFgap':>8s} {'LPIPSgap':>9s} {'CWgap':>8s} {'VIFgap':>8s}"
    print(hdr)
    for name in H.DEG:
        if name not in results: continue
        r = results[name]
        vg = (vmaf.get(name, {}).get("e3", {}).get("vmaf") or 0) - (vmaf.get(name, {}).get("l2", {}).get("vmaf") or 0)
        lg = r["l2"]["lpips"] - r["e3"]["lpips"]   # lpips 越低越好
        cg = r["e3"]["cwssim"] - r["l2"]["cwssim"]
        fg = r["e3"]["vif"] - r["l2"]["vif"]
        print(f"{name:16s} {vg:8.2f} {lg:9.4f} {cg:8.4f} {fg:8.4f}")


if __name__ == "__main__":
    main()
