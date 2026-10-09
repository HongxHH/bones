"""阶段1：自蒸馏补浅出口弹性税。一次建缓存，顺序训练多组配置，再扫一遍 16 段统一评测对比。

配置通过 CONFIGS 指定；每组输出 weights/stage1_{name}_x4(_best).pth。
评测口径与 eval_pacnet.py 完全一致（x4 后 bicubic 到 1080 对 GT 算整帧 PSNR）。
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
from torch.utils.data import DataLoader

ROOT = r"E:\dev\Code\Python\bones"
SURVEY = os.path.join(ROOT, "workspace/enhancer_survey")
PACDIR = os.path.join(ROOT, "workspace/pacnet")
WDIR = os.path.join(PACDIR, "weights")
sys.path.insert(0, SURVEY); sys.path.insert(0, PACDIR); sys.path.insert(0, os.path.join(ROOT, "player"))
from train_leaveout import build_cache, PairDataset, UNIFORM_TEST  # noqa: E402
from train_pacnet import train_pac  # noqa: E402
from pacnet_model import PACNet, CascadePACNet, PlainIMDN  # noqa: E402
from istream_player.modules.enhancer.imdn_model import IMDN  # noqa: E402

DEV = "cuda:0"
FPS_SEG = 120
TEST_SEGS = list(range(0, 159, 10))
FRAME_JS = (15, 45, 75, 105)
HARD = (130, 140, 150)
IMDN_W = os.path.join(ROOT, "player/istream_player/modules/enhancer/imdn_weight/IMDN_high_DIV2K_x4.pth")
LO_W = os.path.join(SURVEY, "weights/IMDN_high_bbbLO_x4_best.pth")


def to_t(bgr):
    return torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                            ).float().div(255).unsqueeze(0).to(DEV)


def psnr(out_t, hr):
    up = F.interpolate(out_t, size=(1080, 1920), mode="bicubic", align_corners=False).clamp(0, 1)
    g = to_t(hr)
    return 10 * np.log10(1.0 / max(((up - g) ** 2).mean().item(), 1e-12))


@torch.no_grad()
def latency(fn, x, reps=20):
    for _ in range(3):
        fn(x)
    torch.cuda.synchronize(); ts = []
    for _ in range(reps):
        t0 = time.time(); fn(x); torch.cuda.synchronize(); ts.append((time.time() - t0) * 1000)
    return round(float(np.median(ts)), 2)


def build_models(pac_tags, kinds=None):
    """pac_tags: [(tag, 权重文件名)]；kinds={tag:'cas'} 标记级联模型；返回 {tag: 推理函数->dict出口}，外加固定对照。"""
    kinds = kinds or {}
    models = {}
    for tag, wf in pac_tags:
        kind = kinds.get(tag)
        cls = CascadePACNet if kind in ("cas", "caspe") else PACNet
        extra = dict(protect_e1=True) if kind == "caspe" else {}
        m = cls(nf=32, upscale=4, exits=(2, 4, 6), cond=False, **extra).to(DEV).eval()
        m.load_state_dict(torch.load(os.path.join(WDIR, wf), map_location="cpu"))
        models[tag] = (lambda mm: (lambda x: {f"e{i+1}": o for i, o in enumerate(mm(x))}))(m)
    plain = PlainIMDN(nf=32, num_modules=2, upscale=4).to(DEV).eval()
    plain.load_state_dict(torch.load(os.path.join(WDIR, "plain2_x4_best.pth"), map_location="cpu"))
    div = IMDN(upscale=4, nf=32).to(DEV).eval(); div.load_state_dict(torch.load(IMDN_W, map_location="cpu"))
    lo = IMDN(upscale=4, nf=32).to(DEV).eval(); lo.load_state_dict(torch.load(LO_W, map_location="cpu"))
    models["plain2"] = (lambda mm: (lambda x: {"v": mm(x)}))(plain)
    models["imdn_div"] = (lambda mm: (lambda x: {"v": mm(x)}))(div)
    models["imdn_lo"] = (lambda mm: (lambda x: {"v": mm(x)}))(lo)
    return models


def evaluate(pac_tags, out_json, kinds=None):
    models = build_models(pac_tags, kinds)
    tags = list(models.keys())
    pac_names = {t for t, _ in pac_tags}
    acc = {t: {"e1": [], "e2": [], "e3": [], "v": []} for t in tags}
    seg_of = []
    lat = {}
    cap_lr = cv2.VideoCapture(os.path.join(ROOT, "data/video_mp4box/output_240p.mp4"))
    cap_hr = cv2.VideoCapture(os.path.join(ROOT, "data/video_mp4box/output_1080p.mp4"))
    want = {(s, j) for s in TEST_SEGS for j in FRAME_JS}
    f = 0
    while True:
        ok1, lr = cap_lr.read(); ok2, hr = cap_hr.read()
        if not (ok1 and ok2):
            break
        seg, j = divmod(f, FPS_SEG)
        if (seg, j) in want:
            x = to_t(lr)
            base_up = cv2.resize(lr, (1920, 1080), interpolation=cv2.INTER_CUBIC)
            acc.setdefault("base", {"v": []})["v"].append(psnr(to_t(base_up), hr))
            for t in tags:
                od = models[t](x)
                for k, o in od.items():
                    acc[t][k].append(psnr(o, hr))
            seg_of.append(seg)
            if not lat:
                for t, fn in models.items():
                    if t in pac_names:
                        lat[t] = {f"e{i+1}": latency(lambda x, ei=i: models[t](x)[f"e{ei+1}"], x)
                                  for i in range(3)}
        f += 1
    cap_lr.release(); cap_hr.release()
    seg_of = np.array(seg_of); hard_m = np.isin(seg_of, HARD)

    def arr(t, k):
        return np.array(acc[t][k], dtype=float)

    plain = arr("plain2", "v"); lo = arr("imdn_lo", "v")
    summary = {"_base_mean": round(float(arr("base", "v").mean()), 3),
               "_plain2_mean": round(float(plain.mean()), 3),
               "_imdn_div_mean": round(float(arr("imdn_div", "v").mean()), 3),
               "_imdn_lo_mean": round(float(lo.mean()), 3),
               "latency_ms": lat, "pac": {}}
    for t in tags:
        if not acc[t]["e1"]:
            continue
        e1, e2, e3 = arr(t, "e1"), arr(t, "e2"), arr(t, "e3")
        tol = 0.05
        summary["pac"][t] = {
            "e1_mean": round(float(e1.mean()), 3), "e2_mean": round(float(e2.mean()), 3),
            "e3_mean": round(float(e3.mean()), 3),
            "e1_minus_plain2": round(float((e1 - plain).mean()), 3),
            "e3_minus_imdnLO": round(float((e3 - lo).mean()), 3),
            "mono_chain_ratio": round(float(np.mean((e2 >= e1 - tol) & (e3 >= e2 - tol))), 4),
            "hard_e3_minus_e1": round(float((e3 - e1)[hard_m].mean()), 3),
            "easy_e3_minus_e1": round(float((e3 - e1)[~hard_m].mean()), 3),
        }
    json.dump(summary, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\n================ 阶段1对比 ================")
    print(f"base {summary['_base_mean']}  plain2 {summary['_plain2_mean']}  "
          f"imdn_div {summary['_imdn_div_mean']}  imdn_lo {summary['_imdn_lo_mean']}")
    head = f"{'tag':8s} {'e1':>7s} {'e2':>7s} {'e3':>7s} {'e1-plain':>9s} {'e3-LO':>7s} {'mono':>6s} {'hardΔ':>7s}"
    print(head)
    for t, m in summary["pac"].items():
        print(f"{t:8s} {m['e1_mean']:7.3f} {m['e2_mean']:7.3f} {m['e3_mean']:7.3f} "
              f"{m['e1_minus_plain2']:9.3f} {m['e3_minus_imdnLO']:7.3f} "
              f"{m['mono_chain_ratio']:6.3f} {m['hard_e3_minus_e1']:7.3f}")
    print("saved", out_json)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--skip_train", action="store_true")
    args = ap.parse_args()

    # 配置：名称 -> train_pac 额外 kwargs
    configs = [
        ("d01", dict(dist_w=0.1)),
        ("d03", dict(dist_w=0.3)),
        ("d05", dict(dist_w=0.5)),
    ]
    if not args.skip_train:
        import random
        random.seed(1); torch.manual_seed(1)
        print("building cache once...", flush=True)
        train_pairs, val_pairs = build_cache(8, UNIFORM_TEST, [30, 90, 140])
        _lr0, _hr0 = train_pairs[0]
        sy, sx = _hr0.shape[0] / _lr0.shape[0], _hr0.shape[1] / _lr0.shape[1]
        dl = DataLoader(PairDataset(train_pairs, 192, 4, True, sy, sx), batch_size=16,
                        shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
        for name, kw in configs:
            out = os.path.join(WDIR, f"stage1_{name}_x4.pth")
            print(f"\n########## train {name} {kw} ##########", flush=True)
            train_pac(dl, val_pairs, args.epochs, 2e-4, 0.5, out,
                      weights=(0.3, 0.6, 1.0), use_cond=False, **kw)

    # 基线 d000=当前主版权重；d01/d03/d05=本次自蒸馏
    pac_tags = [("d000", "pacnet_x4_best.pth")] + \
               [(n, f"stage1_{n}_x4_best.pth") for n, _ in configs]
    evaluate(pac_tags, os.path.join(PACDIR, "stage1_sweep.json"))


if __name__ == "__main__":
    main()
