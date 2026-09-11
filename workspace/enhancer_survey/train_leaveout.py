"""阶段17 闭环：leave-segments-out 对症压缩微调 IMDN，判决"对症训练能否跨段泛化"。

训练对 = 真实链路退化：LR=output_240p.mp4（H.264 ~400kbps），GT=output_1080p.mp4，同帧号对齐。
严格排除普查测试段 TEST_SEGS（模型从未见过这些段），其余段稀疏采样训练。
从 IMDN_high_DIV2K_x4 通用权重 warm-start，L1 + Adam，与 step1 推理前处理完全一致。
"""
import argparse, os, sys, json, time, random
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, os.path.join(ROOT, "player"))
from istream_player.modules.enhancer.imdn_model import IMDN

FRAMES_PER_SEG = 120
UNIFORM_TEST = set(range(0, 159, 10))                 # 均匀抽段（0,10,...,150）
# 连续块留出：测试段及其相邻段全部不训练，排除"相邻帧近邻记忆"，检验真泛化
BLOCK_TEST = set(range(28, 33)) | set(range(88, 93)) | set(range(135, 159))
DEV = "cuda:0"


def build_cache(per_seg_step, hold_segs, val_segs):
    """顺序读两个对齐 mp4，缓存训练帧（排除 hold 段）与验证帧（仅 val 段）。"""
    cap_lr = cv2.VideoCapture(os.path.join(ROOT, "data/video_mp4box/output_240p.mp4"))
    cap_hr = cv2.VideoCapture(os.path.join(ROOT, "data/video_mp4box/output_1080p.mp4"))
    n = int(cap_lr.get(cv2.CAP_PROP_FRAME_COUNT))
    train, val = [], []
    f = 0
    t0 = time.time()
    while True:
        ok1, lr = cap_lr.read(); ok2, hr = cap_hr.read()
        if not (ok1 and ok2): break
        seg, j = divmod(f, FRAMES_PER_SEG)
        if seg in hold_segs:
            if seg in val_segs and j in (30, 60, 90):
                val.append((lr, hr))
        elif j % per_seg_step == 0:
            train.append((lr, hr))
        f += 1
        if f % 4000 == 0:
            print(f"  scanned {f}/{n}, train={len(train)} val={len(val)} {time.time()-t0:.0f}s", flush=True)
    cap_lr.release(); cap_hr.release()
    print(f"cache done: total={f} train_pairs={len(train)} val_pairs={len(val)}", flush=True)
    return train, val


class PairDataset(Dataset):
    def __init__(self, pairs, patch=192, scale=4, train=True, sy=4.5, sx=4.5):
        self.pairs, self.patch, self.scale, self.is_train = pairs, patch, scale, train
        # 真实 LR->HR 缩放比（240p->1080p 为 4.5/4.507，非整数 scale）；GT 必须按此裁剪才对齐
        self.sy, self.sx = sy, sx
    def __len__(self): return len(self.pairs)
    def __getitem__(self, i):
        lr_bgr, hr_bgr = self.pairs[i]
        lr = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB); hr = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB)
        if self.is_train:
            ip = self.patch // self.scale
            ih, iw = lr.shape[:2]
            iy = random.randrange(0, ih - ip + 1); ix = random.randrange(0, iw - ip + 1)
            lr = lr[iy:iy+ip, ix:ix+ip]
            # 按真实缩放比裁 GT，再 resize 到模型 scale 倍网格（patch×patch），保证内容严格对齐
            ty, tx = int(round(self.sy * iy)), int(round(self.sx * ix))
            th, tw = int(round(self.sy * ip)), int(round(self.sx * ip))
            hr = hr[ty:ty+th, tx:tx+tw]
            hr = cv2.resize(hr, (self.patch, self.patch), interpolation=cv2.INTER_CUBIC)
            if random.random() < 0.5: lr, hr = lr[:, ::-1], hr[:, ::-1]
            if random.random() < 0.5: lr, hr = lr[::-1], hr[::-1]
            if random.random() < 0.5: lr, hr = lr.transpose(1, 0, 2), hr.transpose(1, 0, 2)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).float().div(255)
        return t(lr), t(hr)


@torch.no_grad()
def val_psnr(model, val_pairs):
    """留出段整帧：x4 后 bicubic 到 1080，与 GT 算 RGB PSNR 均值。"""
    model.eval(); ps = []
    for lr_bgr, hr_bgr in val_pairs:
        lr = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(lr.transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(DEV)
        o = torch.nn.functional.interpolate(model(t), size=(1080, 1920), mode="bicubic", align_corners=False).clamp(0, 1)
        g = torch.from_numpy(cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)).float().div(255).unsqueeze(0).to(DEV)
        mse = ((o - g) ** 2).mean().item()
        ps.append(10 * np.log10(1.0 / max(mse, 1e-12)))
    model.train()
    return float(np.mean(ps))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--patch", type=int, default=192)
    ap.add_argument("--per_seg_step", type=int, default=8)
    ap.add_argument("--holdout", choices=["uniform", "block"], default="uniform")
    ap.add_argument("--out", default=os.path.join(ROOT, "workspace/enhancer_survey/weights/IMDN_high_bbbLO_x4.pth"))
    args = ap.parse_args()
    random.seed(1); torch.manual_seed(1)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    hold = UNIFORM_TEST if args.holdout == "uniform" else BLOCK_TEST
    val_segs = [30, 90, 140] if args.holdout == "uniform" else [30, 90, 140, 150]
    print(f"holdout={args.holdout}, hold {len(hold)} segs", flush=True)
    print("building aligned H.264 train pairs (test segs excluded)...", flush=True)
    train_pairs, val_pairs = build_cache(args.per_seg_step, hold, val_segs)
    _lr0, _hr0 = train_pairs[0]
    sy, sx = _hr0.shape[0] / _lr0.shape[0], _hr0.shape[1] / _lr0.shape[1]
    print(f"true LR->HR scale sy={sy:.3f} sx={sx:.3f}", flush=True)
    dl = DataLoader(PairDataset(train_pairs, args.patch, 4, True, sy, sx), batch_size=args.batch,
                    shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

    model = IMDN(upscale=4, nf=32)
    warm = os.path.join(ROOT, "player/istream_player/modules/enhancer/imdn_weight/IMDN_high_DIV2K_x4.pth")
    model.load_state_dict(torch.load(warm, map_location="cpu"))
    model.to(DEV).train()
    print(f"warm-start from {warm}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=15, gamma=0.5)
    l1 = nn.L1Loss()
    log = []
    best = -1
    for ep in range(1, args.epochs + 1):
        t0 = time.time(); tot = 0; nb = 0
        for lr_t, hr_t in dl:
            lr_t, hr_t = lr_t.to(DEV), hr_t.to(DEV)
            opt.zero_grad(); sr = model(lr_t); loss = l1(sr, hr_t); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        vp = val_psnr(model, val_pairs)
        row = dict(epoch=ep, train_l1=round(tot/nb, 5), val_psnr=round(vp, 3), sec=round(time.time()-t0, 1))
        log.append(row); print(row, flush=True)
        if vp > best:
            best = vp; torch.save(model.state_dict(), args.out.replace(".pth", "_best.pth"))
    torch.save(model.state_dict(), args.out)
    json.dump(log, open(args.out.replace(".pth", "_trainlog.json"), "w", encoding="utf-8"), indent=2)
    print("saved", args.out, "best val PSNR", round(best, 3), flush=True)


if __name__ == "__main__":
    main()
