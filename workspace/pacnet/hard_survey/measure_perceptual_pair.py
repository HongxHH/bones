# -*- coding: utf-8 -*-
"""
逐帧配对统计：在同一退化帧上比较轻量 l2 与重出口 e3（独立 l3 参照）的 LPIPS，
排除"均值差被少数离群帧拉动"。输出每档：
  mean lpips、配对差 d=lpips(l2)-lpips(e3)（正=e3 更好）、
  e3 更差帧占比（d<0）、符号检验 p 值（双侧）。
"""
import os, sys, json
import numpy as np
import torch, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import measure_hard_survey as H
import lpips

OUT = os.path.join(HERE, "perceptual_pair.json")
DEVICE = "cuda:0"


def to_t(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(DEVICE) / 255.0


def sign_p_2sided(w, n):
    # 双侧符号检验：H0 p=0.5，取较少侧 *2
    from math import comb
    k = min(w, n - w)
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


@torch.no_grad()
def frame_lpips(lp, e, r):
    return float(lp(to_t(e) * 2 - 1, to_t(r) * 2 - 1))


def main():
    mdl, pac = H.build_models()
    lp = lpips.LPIPS(net="squeeze").to(DEVICE).eval()
    cap = cv2.VideoCapture(H.REF_VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    idxs = np.linspace(0, total - 10, H.N, dtype=int)
    ref = H.read_frames(H.REF_VIDEO, idxs)
    res = {}
    print(f"{'degradation':16s} {'l2':>7} {'e3':>7} {'l3':>7} {'d(l2-e3)':>9} {'e3worse%':>9} {'p_sign':>8}")
    for name, path in H.DEG.items():
        if not os.path.exists(path): continue
        lr = H.read_frames(path, idxs)
        n = min(len(lr), len(ref)); lr, rf = lr[:n], ref[:n]
        per = {"l2": [], "e3": [], "l3": []}
        for x, r in zip(lr, rf):
            e_l2 = H.infer(mdl["l2"], x)
            e_e3 = H.infer(pac, x, pac=True, exit_idx=2)
            e_l3 = H.infer(mdl["l3"], x)
            per["l2"].append(frame_lpips(lp, e_l2, r))
            per["e3"].append(frame_lpips(lp, e_e3, r))
            per["l3"].append(frame_lpips(lp, e_l3, r))
        a, b = np.array(per["l2"]), np.array(per["e3"])
        d = a - b  # 正=e3 更接近 GT
        e3worse = int(np.sum(d < 0))  # d<0 => lpips(e3)>lpips(l2) => e3 更差
        p = sign_p_2sided(int(np.sum(d >= 0)), n)
        res[name] = {
            "lpips_l2": round(float(a.mean()), 5),
            "lpips_e3": round(float(b.mean()), 5),
            "lpips_l3": round(float(np.mean(per["l3"])), 5),
            "pair_d_mean": round(float(d.mean()), 5),
            "e3_worse_frames": e3worse, "n": n,
            "e3_worse_pct": round(100.0 * e3worse / n, 1),
            "sign_p": round(float(p), 5),
        }
        print(f"{name:16s} {a.mean():7.4f} {b.mean():7.4f} {np.mean(per['l3']):7.4f} "
              f"{d.mean():9.4f} {100.0*e3worse/n:8.1f}% {p:8.4f}")
        json.dump(res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
