"""阶段0证伪训练：PACNet 三出口 + 2-block 固定浅对照，数据管线与对症闭环完全一致。

- 复用 train_leaveout.build_cache / PairDataset（真实 4.5x 对齐、uniform 16 段留出）。
- PACNet：三出口 deep supervision L1（权重 0.3/0.6/1.0）+ 单调 margin（深出口误差不得更大）。
- PlainIMDN(2)：与出口1同算力的固定浅模型，单出口 L1，用于衡量"多出口共享的弹性税"。
- 两者均从 IMDN_high_DIV2K_x4 通用权重 warm-start，同数据、同优化配方，保证公平。
"""
import argparse
import json
import os
import random
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
sys.path.insert(0, SURVEY)
sys.path.insert(0, PACDIR)
from train_leaveout import build_cache, PairDataset, DEV  # noqa: E402
from pacnet_model import PACNet, PlainIMDN  # noqa: E402

WARM = os.path.join(ROOT, "player/istream_player/modules/enhancer/imdn_weight/IMDN_high_DIV2K_x4.pth")
WDIR = os.path.join(PACDIR, "weights")


@torch.no_grad()
def val_pac(model, val_pairs):
    model.eval()
    p1, p2, p3 = [], [], []
    for lr_bgr, hr_bgr in val_pairs:
        lr = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(lr.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(DEV)
        g = torch.from_numpy(cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                             ).float().div(255).unsqueeze(0).to(DEV)
        outs = model(t)
        for bucket, o in zip((p1, p2, p3), outs):
            up = F.interpolate(o, size=(1080, 1920), mode="bicubic", align_corners=False).clamp(0, 1)
            mse = ((up - g) ** 2).mean().item()
            bucket.append(10 * np.log10(1.0 / max(mse, 1e-12)))
    model.train()
    return float(np.mean(p1)), float(np.mean(p2)), float(np.mean(p3))


@torch.no_grad()
def val_plain(model, val_pairs):
    model.eval()
    ps = []
    for lr_bgr, hr_bgr in val_pairs:
        lr = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(lr.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(DEV)
        g = torch.from_numpy(cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
                             ).float().div(255).unsqueeze(0).to(DEV)
        up = F.interpolate(model(t), size=(1080, 1920), mode="bicubic", align_corners=False).clamp(0, 1)
        mse = ((up - g) ** 2).mean().item()
        ps.append(10 * np.log10(1.0 / max(mse, 1e-12)))
    model.train()
    return float(np.mean(ps))


def monotonic_margin(outs, g):
    """惩罚更深出口逐样本误差反而更大：relu(err_{k+1}-err_k) 均值。"""
    loss = 0.0
    errs = [(o - g).abs().mean(dim=(1, 2, 3)) for o in outs]
    for ea, eb in zip(errs[:-1], errs[1:]):
        loss = loss + F.relu(eb - ea).mean()
    return loss


def train_pac(dl, val_pairs, epochs, lr, mono_w, out, weights=(0.3, 0.6, 1.0), use_cond=True):
    model = PACNet(nf=32, upscale=4, exits=(2, 4, 6), cond=use_cond).to(DEV)
    n = model.load_imdn_warm(torch.load(WARM, map_location="cpu"), verbose=True)
    assert n > 20, "warm 迁移过少，检查键名"
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
    l1 = torch.nn.L1Loss()
    w = weights
    print("exit loss weights:", w, "mono_w", mono_w, flush=True)
    log, best = [], -1
    for ep in range(1, epochs + 1):
        t0 = time.time(); tot = 0.; nb = 0; mtot = 0.
        model.train()
        for lr_t, hr_t in dl:
            lr_t, hr_t = lr_t.to(DEV), hr_t.to(DEV)
            opt.zero_grad()
            outs = model(lr_t)
            main = sum(wi * l1(o, hr_t) for wi, o in zip(w, outs))
            mono = monotonic_margin(outs, hr_t)
            loss = main + mono_w * mono
            loss.backward(); opt.step()
            tot += main.item(); mtot += mono.item(); nb += 1
        sched.step()
        v1, v2, v3 = val_pac(model, val_pairs)
        row = dict(epoch=ep, main_l1=round(tot / nb, 5), mono=round(mtot / nb, 5),
                   val_e1=round(v1, 3), val_e2=round(v2, 3), val_e3=round(v3, 3),
                   sec=round(time.time() - t0, 1))
        log.append(row); print("PAC", row, flush=True)
        if v3 > best:
            best = v3; torch.save(model.state_dict(), out.replace(".pth", "_best.pth"))
    torch.save(model.state_dict(), out)
    json.dump(log, open(out.replace(".pth", "_trainlog.json"), "w", encoding="utf-8"), indent=2)
    print("PAC saved, best e3", round(best, 3), flush=True)


def train_plain(dl, val_pairs, epochs, lr, out):
    model = PlainIMDN(nf=32, num_modules=2, upscale=4).to(DEV)
    n = model.load_imdn_warm(torch.load(WARM, map_location="cpu"))
    print("plain warm-moved", n, flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
    l1 = torch.nn.L1Loss()
    log, best = [], -1
    for ep in range(1, epochs + 1):
        t0 = time.time(); tot = 0.; nb = 0
        model.train()
        for lr_t, hr_t in dl:
            lr_t, hr_t = lr_t.to(DEV), hr_t.to(DEV)
            opt.zero_grad(); loss = l1(model(lr_t), hr_t)
            loss.backward(); opt.step(); tot += loss.item(); nb += 1
        sched.step()
        v = val_plain(model, val_pairs)
        row = dict(epoch=ep, l1=round(tot / nb, 5), val=round(v, 3), sec=round(time.time() - t0, 1))
        log.append(row); print("PLAIN2", row, flush=True)
        if v > best:
            best = v; torch.save(model.state_dict(), out.replace(".pth", "_best.pth"))
    torch.save(model.state_dict(), out)
    json.dump(log, open(out.replace(".pth", "_trainlog.json"), "w", encoding="utf-8"), indent=2)
    print("PLAIN saved, best", round(best, 3), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--per_seg_step", type=int, default=8)
    ap.add_argument("--mono_w", type=float, default=0.5)
    ap.add_argument("--w1", type=float, default=0.3)
    ap.add_argument("--w2", type=float, default=0.6)
    ap.add_argument("--w3", type=float, default=1.0)
    ap.add_argument("--pac_out", default=os.path.join(WDIR, "pacnet_x4.pth"))
    ap.add_argument("--use_cond", type=int, default=1, help="是否启用退化条件 FiLM")
    ap.add_argument("--which", choices=["pac", "plain", "both"], default="both")
    args = ap.parse_args()
    random.seed(1); torch.manual_seed(1)
    os.makedirs(WDIR, exist_ok=True)

    # uniform 留出：训练排除 16 测试段，验证段与闭环一致
    from train_leaveout import UNIFORM_TEST
    hold = UNIFORM_TEST
    val_segs = [30, 90, 140]
    print("building aligned pairs (uniform test segs excluded)...", flush=True)
    train_pairs, val_pairs = build_cache(args.per_seg_step, hold, val_segs)
    _lr0, _hr0 = train_pairs[0]
    sy, sx = _hr0.shape[0] / _lr0.shape[0], _hr0.shape[1] / _lr0.shape[1]
    print(f"true scale sy={sy:.3f} sx={sx:.3f}", flush=True)
    dl = DataLoader(PairDataset(train_pairs, 192, 4, True, sy, sx), batch_size=args.batch,
                    shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

    if args.which in ("pac", "both"):
        train_pac(dl, val_pairs, args.epochs, args.lr, args.mono_w, args.pac_out,
                  weights=(args.w1, args.w2, args.w3), use_cond=bool(args.use_cond))
    if args.which in ("plain", "both"):
        train_plain(dl, val_pairs, args.epochs, args.lr,
                    os.path.join(WDIR, "plain2_x4.pth"))


if __name__ == "__main__":
    main()
