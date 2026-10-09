"""阶段1 Step4：多种子公平复测。对 seed=1/2/3 各训 alpha=0 基线、alpha=0.1 温和自蒸馏，
以及同种子 plain2，按同种子配对比较 e1-plain2，判断温和蒸馏的 0.1dB 级变化是真改进还是噪声。
一次扫完 16 段统一评测，输出每种子与跨种子汇总。"""
import argparse
import json
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import run_stage1 as rs
from train_leaveout import build_cache, PairDataset, UNIFORM_TEST
from train_pacnet import train_pac, train_plain
from pacnet_model import PACNet, PlainIMDN
from istream_player.modules.enhancer.imdn_model import IMDN

WDIR, DEV = rs.WDIR, rs.DEV
SEEDS = [1, 2, 3]


def eval_all(out_json):
    # 构建全部模型：tag -> fn(x)->{出口}
    models = {}
    for s in SEEDS:
        for tag, cls, wf, kind in [
            (f"s{s}_a0", PACNet, f"s{s}_a0_x4_best.pth", "pac"),
            (f"s{s}_a01", PACNet, f"s{s}_a01_x4_best.pth", "pac"),
            (f"s{s}_pl", PlainIMDN, f"s{s}_plain_x4_best.pth", "plain"),
        ]:
            kw = dict(num_modules=2) if kind == "plain" else dict(exits=(2, 4, 6), cond=False)
            m = cls(nf=32, upscale=4, **kw).to(DEV).eval()
            m.load_state_dict(torch.load(os.path.join(WDIR, wf), map_location="cpu"))
            if kind == "pac":
                models[tag] = (lambda mm: (lambda x: {f"e{i+1}": o for i, o in enumerate(mm(x))}))(m)
            else:
                models[tag] = (lambda mm: (lambda x: {"v": mm(x)}))(m)
    lo = IMDN(upscale=4, nf=32).to(DEV).eval(); lo.load_state_dict(torch.load(rs.LO_W, map_location="cpu"))
    models["imdn_lo"] = (lambda mm: (lambda x: {"v": mm(x)}))(lo)

    tags = list(models.keys())
    acc = {t: {"e1": [], "e2": [], "e3": [], "v": []} for t in tags}
    acc["base"] = {"v": []}
    seg_of = []
    cap_lr = cv2.VideoCapture(os.path.join(rs.ROOT, "data/video_mp4box/output_240p.mp4"))
    cap_hr = cv2.VideoCapture(os.path.join(rs.ROOT, "data/video_mp4box/output_1080p.mp4"))
    want = {(sg, j) for sg in rs.TEST_SEGS for j in rs.FRAME_JS}
    f = 0
    while True:
        ok1, lr = cap_lr.read(); ok2, hr = cap_hr.read()
        if not (ok1 and ok2):
            break
        seg, j = divmod(f, rs.FPS_SEG)
        if (seg, j) in want:
            x = rs.to_t(lr)
            acc["base"]["v"].append(rs.psnr(rs.to_t(cv2.resize(lr, (1920, 1080), interpolation=cv2.INTER_CUBIC)), hr))
            for t in tags:
                for k, o in models[t](x).items():
                    acc[t][k].append(rs.psnr(o, hr))
            seg_of.append(seg)
        f += 1
    cap_lr.release(); cap_hr.release()
    seg_of = np.array(seg_of); hard = np.isin(seg_of, rs.HARD)

    def arr(t, k):
        return np.array(acc[t][k], float)

    loa = arr("imdn_lo", "v")
    per_seed, rows = {}, []
    for s in SEEDS:
        e1_0, e3_0 = arr(f"s{s}_a0", "e1"), arr(f"s{s}_a0", "e3")
        e1_1, e2_1, e3_1 = arr(f"s{s}_a01", "e1"), arr(f"s{s}_a01", "e2"), arr(f"s{s}_a01", "e3")
        pl = arr(f"s{s}_pl", "v")
        tol = 0.05
        rec = {
            "seed": s,
            "a0_e1": round(e1_0.mean(), 3), "a0_e1_minus_plain": round((e1_0 - pl).mean(), 3),
            "a01_e1": round(e1_1.mean(), 3), "a01_e1_minus_plain": round((e1_1 - pl).mean(), 3),
            "a01_e2": round(e2_1.mean(), 3), "a01_e3": round(e3_1.mean(), 3),
            "a0_e3": round(e3_0.mean(), 3),
            "a01_e3_minus_LO": round((e3_1 - loa).mean(), 3),
            "a01_mono": round(float(np.mean((e2_1 >= e1_1 - tol) & (e3_1 >= e2_1 - tol))), 4),
            "a01_hard_e3_e1": round(float((e3_1 - e1_1)[hard].mean()), 3),
            "plain_mean": round(pl.mean(), 3),
        }
        per_seed[s] = rec; rows.append(rec)

    def ms(key):
        v = np.array([r[key] for r in rows]); return round(float(v.mean()), 3), round(float(v.std()), 3)

    summary = {"per_seed": per_seed,
               "mean": {"a0_e1_minus_plain": ms("a0_e1_minus_plain"),
                        "a01_e1_minus_plain": ms("a01_e1_minus_plain"),
                        "a0_e3": ms("a0_e3"), "a01_e3": ms("a01_e3"),
                        "a01_mono": ms("a01_mono"), "a01_hard": ms("a01_hard_e3_e1")},
               "base_mean": round(float(arr("base", "v").mean()), 3),
               "imdn_lo_mean": round(float(loa.mean()), 3)}
    json.dump(summary, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n==== 多种子复测（e1-plain 同种子配对，越接近0越好）====")
    for r in rows:
        print(f"seed{r['seed']}: a0 e1-plain {r['a0_e1_minus_plain']:+.3f} | "
              f"a01 e1-plain {r['a01_e1_minus_plain']:+.3f} | a01 e3 {r['a01_e3']} "
              f"mono {r['a01_mono']} hard {r['a01_hard_e3_e1']} | plain {r['plain_mean']}")
    m = summary["mean"]
    print(f"\n跨种子均值±标准差:  a0 e1-plain {m['a0_e1_minus_plain']}  |  "
          f"a01 e1-plain {m['a01_e1_minus_plain']}")
    print(f"e3: a0 {m['a0_e3']}  a01 {m['a01_e3']}   a01 mono {m['a01_mono']}  hard {m['a01_hard']}")
    print("saved", out_json)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--skip_train", action="store_true"); args = ap.parse_args()
    if not args.skip_train:
        print("building cache once...", flush=True)
        train_pairs, val_pairs = build_cache(8, UNIFORM_TEST, [30, 90, 140])
        lr0, hr0 = train_pairs[0]
        sy, sx = hr0.shape[0] / lr0.shape[0], hr0.shape[1] / lr0.shape[1]
        for s in SEEDS:
            random.seed(s); torch.manual_seed(s); np.random.seed(s)
            dl = DataLoader(PairDataset(train_pairs, 192, 4, True, sy, sx), batch_size=16,
                            shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
            print(f"\n########## seed {s}: alpha=0 ##########", flush=True)
            train_pac(dl, val_pairs, args.epochs, 2e-4, 0.5, os.path.join(WDIR, f"s{s}_a0_x4.pth"),
                      weights=(0.3, 0.6, 1.0), use_cond=False, dist_w=0.0)
            print(f"########## seed {s}: alpha=0.1 ##########", flush=True)
            train_pac(dl, val_pairs, args.epochs, 2e-4, 0.5, os.path.join(WDIR, f"s{s}_a01_x4.pth"),
                      weights=(0.3, 0.6, 1.0), use_cond=False, dist_w=0.1)
            print(f"########## seed {s}: plain2 ##########", flush=True)
            train_plain(dl, val_pairs, args.epochs, 2e-4, os.path.join(WDIR, f"s{s}_plain_x4.pth"))
    eval_all(os.path.join(rs.PACDIR, "stage4_multiseed.json"))


if __name__ == "__main__":
    main()
