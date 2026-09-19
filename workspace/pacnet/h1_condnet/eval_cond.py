# -*- coding: utf-8 -*-
"""H1 条件网 TOS 离线证伪评测（hard_survey 同口径，TOS 全程未参与训练）。

7 档退化（426/320 × CRF26/34/42 + 426/400k 锚点）× 32 帧，统一 x4 后 bicubic 到
1920x800 对 1080p 源。方法：bicubic / l2_stock(DIV2K 未微调) / base(微调无条件) /
cond(微调真条件) / shuf(乱序条件训练、评测喂真条件)。
每方法逐帧 VMAF(libvmaf json) + LPIPS(squeeze)，对 cond-base 做逐帧配对符号检验。
判据：cond 相对 base 在 VMAF 与 LPIPS 上同向稳定显著、且优于 shuf，H1 才成立。
"""
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "player"))
sys.path.insert(0, os.path.join(ROOT, "workspace/pacnet/hard_survey"))
import codec_lib  # noqa: E402
from istream_player.modules.enhancer.imdn_model import IMDN_RTC  # noqa: E402
from cond_model import CondIMDN_RTC, CondIMDNCat, WARM_L2  # noqa: E402
import measure_hard_survey as H  # noqa: E402

DEV = "cuda:0"
TARGET = (1920, 800)
N = 32


def load_models(wdir, model="add"):
    stock = IMDN_RTC(upscale=4); stock.load_state_dict(torch.load(WARM_L2, map_location="cpu"))
    CondNet = CondIMDNCat if model == "cat" else CondIMDN_RTC
    base = CondNet(upscale=4) if model == "cat" else IMDN_RTC(upscale=4)
    base.load_state_dict(torch.load(os.path.join(wdir, "base_finetuned.pth"), map_location="cpu"))
    cond = CondNet(upscale=4)
    cond.load_state_dict(torch.load(os.path.join(wdir, "cond_finetuned.pth"), map_location="cpu"))
    shuf = CondNet(upscale=4)
    shuf.load_state_dict(torch.load(os.path.join(wdir, "shuf_finetuned.pth"), map_location="cpu"))
    return {k: m.to(DEV).eval() for k, m in
            {"stock": stock, "base": base, "cond": cond, "shuf": shuf}.items()}


@torch.no_grad()
def infer_rgb(model, rgb, cond=None, is_cond=False):
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(DEV) / 255.0
    if is_cond:
        ct = None if cond is None else torch.from_numpy(cond).unsqueeze(0).to(DEV)
        o = model(t, ct)
    else:
        o = model(t)
    o = torch.nn.functional.interpolate(o, size=(TARGET[1], TARGET[0]),
                                        mode="bicubic", align_corners=False)
    o = o.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (o * 255).clip(0, 255).astype(np.uint8)  # RGB


def bic_rgb(rgb):
    return cv2.resize(rgb, TARGET, interpolation=cv2.INTER_CUBIC)


def vmaf_perframe(pairs_rgb, tmp, tag):
    """写 e/r mp4，libvmaf json 出逐帧 VMAF。cwd=tmp 用相对 log 名，避开 Windows 盘符冒号。"""
    ep, rp, lname = (os.path.join(tmp, f"e_{tag}.mp4"), os.path.join(tmp, f"r_{tag}.mp4"),
                     f"l_{tag}.json")
    h, w = pairs_rgb[0][0].shape[:2]
    fc = cv2.VideoWriter_fourcc(*"mp4v")
    we, wr = cv2.VideoWriter(ep, fc, H.FPS, (w, h)), cv2.VideoWriter(rp, fc, H.FPS, (w, h))
    for e, r in pairs_rgb:
        we.write(cv2.cvtColor(e, cv2.COLOR_RGB2BGR))
        wr.write(cv2.cvtColor(r, cv2.COLOR_RGB2BGR))
    we.release(); wr.release()
    res = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", ep, "-i", rp,
                          "-lavfi", f"libvmaf=log_path={lname}:log_fmt=json", "-f", "null", "-"],
                         capture_output=True, text=True, timeout=300, cwd=tmp)
    lp = os.path.join(tmp, lname)
    if not os.path.exists(lp):
        raise RuntimeError(f"libvmaf log missing for {tag}: {res.stderr[-800:]}")
    log = json.load(open(lp, encoding="utf-8"))
    return [float(fr["metrics"]["vmaf"]) for fr in log["frames"]]


@torch.no_grad()
def lpips_perframe(lp, e_rgb, r_rgb):
    def tt(x):
        return torch.from_numpy(x).permute(2, 0, 1).float().unsqueeze(0).to(DEV) / 255.0
    return float(lp(tt(e_rgb) * 2 - 1, tt(r_rgb) * 2 - 1))


def sign_test(diff):
    """正差个数 / 非零差总数，双侧二项符号检验 p 值。"""
    from scipy.stats import binomtest
    d = np.asarray(diff, float)
    d = d[np.abs(d) > 1e-9]
    if len(d) == 0:
        return None, 1.0, 0
    pos = int((d > 0).sum()); p = float(binomtest(pos, len(d), 0.5).pvalue)
    return pos, p, len(d)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["add", "cat"], default="add")
    ap.add_argument("--wdir", default=os.path.join(HERE, "weights_scratch"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out_json = args.out or os.path.join(
        HERE, os.path.basename(args.wdir.rstrip("/\\")) + "_eval.json")
    import lpips as lpips_mod
    lp = lpips_mod.LPIPS(net="squeeze").to(DEV).eval()
    models = load_models(args.wdir, args.model)

    cap = cv2.VideoCapture(H.REF_VIDEO)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    idxs = np.linspace(0, total - 10, N, dtype=int)
    ref_bgr = H.read_frames(H.REF_VIDEO, idxs)
    ref = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in ref_bgr]
    print("ref frames", len(ref), "total", total, flush=True)

    results = {}
    for name, path in H.DEG.items():
        if not os.path.exists(path):
            print("SKIP", path); continue
        fc, Wd, Hd = codec_lib.build_frame_cache(path, idxs)
        lrs = [fc[int(i)][0] for i in idxs if int(i) in fc]
        cds = {int(i): fc[int(i)][1] for i in idxs if int(i) in fc}
        n = min(len(lrs), len(ref))
        lrs, rf = lrs[:n], ref[:n]
        use_idx = [int(i) for i in idxs][:n]
        print(f"\n=== {name} ({n}) {Wd}x{Hd} ===", flush=True)
        out = {}
        with tempfile.TemporaryDirectory() as tmp:
            for meth in ["bicubic", "stock", "base", "cond", "shuf"]:
                epairs, lps = [], []
                for k, (x, r) in enumerate(zip(lrs, rf)):
                    if meth == "bicubic":
                        e = bic_rgb(x)
                    elif meth in ("stock", "base"):
                        e = infer_rgb(models[meth], x, is_cond=False)
                    else:
                        e = infer_rgb(models[meth], x, cds[use_idx[k]], is_cond=True)
                    epairs.append((e, r)); lps.append(lpips_perframe(lp, e, r))
                vv = vmaf_perframe(epairs, tmp, name + "_" + meth)
                out[meth] = {"vmaf_mean": round(float(np.mean(vv)), 3),
                             "lpips_mean": round(float(np.mean(lps)), 5),
                             "vmaf_pf": [round(v, 3) for v in vv],
                             "lpips_pf": [round(v, 5) for v in lps]}
                print(f"  {meth:7s} VMAF={np.mean(vv):6.2f}  LPIPS={np.mean(lps):.4f}", flush=True)
        # 配对检验：cond 相对 base（VMAF 正=cond好；LPIPS 负=cond好 → 用 base-cond）
        dv = np.array(out["cond"]["vmaf_pf"]) - np.array(out["base"]["vmaf_pf"])
        dl = np.array(out["base"]["lpips_pf"]) - np.array(out["cond"]["lpips_pf"])
        pv_pos, pv_p, nv = sign_test(dv); pl_pos, pl_p, nl = sign_test(dl)
        out["cond_vs_base"] = {
            "dVMAF_mean": round(float(dv.mean()), 3), "vmaf_win": pv_pos, "vmaf_n": nv,
            "vmaf_sign_p": round(pv_p, 5),
            "dLPIPS_mean": round(float((np.array(out["cond"]["lpips_pf"])
                                        - np.array(out["base"]["lpips_pf"])).mean()), 5),
            "lpips_win": pl_pos, "lpips_n": nl, "lpips_sign_p": round(pl_p, 5),
        }
        results[name] = out
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n=== cond vs base summary ===")
    print(f"{'degradation':16s} {'dVMAF':>7} {'vmaf p':>8} {'dLPIPS':>8} {'lpips p':>8}")
    for name, o in results.items():
        s = o["cond_vs_base"]
        print(f"{name:16s} {s['dVMAF_mean']:7.2f} {s['vmaf_sign_p']:8.4f} "
              f"{s['dLPIPS_mean']:8.4f} {s['lpips_sign_p']:8.4f}")
    print("EVAL_DONE")


if __name__ == "__main__":
    main()
