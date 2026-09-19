# -*- coding: utf-8 -*-
"""H1 条件网离线小样本证伪训练（BBB 训练，TOS 全程留出）。

三臂，同数据 / 同退化 / 同优化预算 / 同骨干热启动，唯一差别是码流条件：
  base : IMDN_RTC(l2) 热启动后压缩域微调，不喂条件；
  cond : CondIMDN_RTC，热启动骨干 + 条件编码器，喂与帧对齐的真 QP/MV；
  shuf : 与 cond 完全同权重初始化，但喂“别的帧”的条件（固定乱序），
         作为容量对照——若真条件不优于乱序条件，增益只来自多的参数而非元数据信息。
HR 用 BBB output_1080p，LR/条件来自 6 个退化视频（426/320 × CRF26/34/42）。
"""
import argparse
import copy
import json
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "player"))
from istream_player.modules.enhancer.imdn_model import IMDN_RTC  # noqa: E402
from cond_model import CondIMDN_RTC, CondIMDNCat, WARM_L2, count_params  # noqa: E402

HR_VIDEO = os.path.join(ROOT, "data/video_mp4box/output_1080p.mp4")
CACHE = os.path.join(HERE, "cache")
WDIR = os.path.join(HERE, "weights")
DEV = "cuda:0"
PATCH = 192
VIDEOS = [(w, crf) for w in (426, 320) for crf in (26, 34, 42)]


def load_samples():
    """读 6 个 npz，组装样本；HR 按并集帧号顺序读 output_1080 一次，常驻内存。"""
    raw = []   # (lr, cond, idx, sy, sx)
    need = set()
    for w, crf in VIDEOS:
        d = np.load(os.path.join(CACHE, f"bbb_{w}_crf{crf}.npz"))
        idx, lr, cond = d["idx"], d["lr"], d["cond"]
        Hd, Wd = int(d["H"][0]), int(d["W"][0])
        sy, sx = 1080.0 / Hd, 1920.0 / Wd
        for j in range(len(idx)):
            raw.append((lr[j], cond[j], int(idx[j]), sy, sx))
            need.add(int(idx[j]))
    print(f"samples={len(raw)} reading HR for {len(need)} frames ...", flush=True)
    cap = cv2.VideoCapture(HR_VIDEO)
    hr_map, fi = {}, 0
    need_sorted = set(need)
    while need_sorted:
        ok, fr = cap.read()
        if not ok:
            break
        if fi in need_sorted:
            hr_map[fi] = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            need_sorted.discard(fi)
        fi += 1
    cap.release()
    samples = []
    for lr, cond, i, sy, sx in raw:
        samples.append((lr, cond.astype(np.float32), hr_map[i], sy, sx))
    return samples


class CondDataset(Dataset):
    def __init__(self, samples, train=True, cond_perm=None):
        self.s, self.is_train, self.perm = samples, train, cond_perm
        self.ip = PATCH // 4

    def __len__(self):
        return len(self.s)

    def _pack(self, lr, cond, hr, sy, sx):
        iy = random.randrange(0, lr.shape[0] - self.ip + 1)
        ix = random.randrange(0, lr.shape[1] - self.ip + 1)
        lr = lr[iy:iy + self.ip, ix:ix + self.ip]
        cond = cond[:, iy:iy + self.ip, ix:ix + self.ip]
        ty, tx = int(round(sy * iy)), int(round(sx * ix))
        th, tw = int(round(sy * self.ip)), int(round(sx * self.ip))
        hr = hr[ty:ty + th, tx:tx + tw]
        hr = cv2.resize(hr, (PATCH, PATCH), interpolation=cv2.INTER_CUBIC)
        # 翻转同步作用于 LR/HR/条件；MV 分量随翻转变号
        if random.random() < 0.5:  # 水平翻转
            lr = lr[:, ::-1]; hr = hr[:, ::-1]; cond = cond[:, :, ::-1]
            cond[1] = -cond[1]; cond[4] = -cond[4]
        if random.random() < 0.5:  # 垂直翻转
            lr = lr[::-1]; hr = hr[::-1]; cond = cond[:, ::-1, :]
            cond[2] = -cond[2]; cond[5] = -cond[5]
        tl = torch.from_numpy(np.ascontiguousarray(lr.transpose(2, 0, 1))).float().div(255)
        th_ = torch.from_numpy(np.ascontiguousarray(hr.transpose(2, 0, 1))).float().div(255)
        tc = torch.from_numpy(np.ascontiguousarray(cond)).float()
        return tl, tc, th_

    def __getitem__(self, i):
        lr, cond, hr, sy, sx = self.s[i]
        if self.perm is not None:
            _, cond, _, _, _ = self.s[self.perm[i]]   # 乱序条件：容量对照
        return self._pack(lr, cond, hr, sy, sx)


def make_loader(samples, perm, seed, batch, epochs_holder):
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(CondDataset(samples, True, perm), batch_size=batch, shuffle=True,
                      num_workers=0, drop_last=True, pin_memory=True, generator=g)


def train_arm(name, model, samples, perm, epochs, batch, lr, seed, mode, out):
    """mode: plain=model(x)；cond_zero=model(x,None)（cat 基线，恒零条件）；cond=model(x,条件)。"""
    random.seed(seed); torch.manual_seed(seed)
    loader = make_loader(samples, perm, seed, batch, None)
    model.to(DEV).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    l1 = nn.L1Loss()
    log = []
    for ep in range(1, epochs + 1):
        random.seed(seed + ep)  # 三臂每轮增广序列一致
        tot, nb = 0.0, 0
        for lr_t, cond_t, hr_t in loader:
            lr_t, cond_t, hr_t = lr_t.to(DEV), cond_t.to(DEV), hr_t.to(DEV)
            opt.zero_grad()
            if mode == "plain":
                pred = model(lr_t)
            elif mode == "cond_zero":
                pred = model(lr_t, None)
            else:
                pred = model(lr_t, cond_t)
            loss = l1(pred, hr_t)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        row = dict(epoch=ep, train_l1=round(tot / nb, 5))
        log.append(row); print(f"{name} {row}", flush=True)
    torch.save(model.state_dict(), out)
    json.dump(log, open(out.replace(".pth", "_trainlog.json"), "w", encoding="utf-8"), indent=2)
    print(f"{name} saved -> {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["add", "cat"], default="add",
                    help="add=条件编码器特征相加；cat=条件作首层额外输入通道（三臂同构）")
    ap.add_argument("--init", choices=["scratch", "warm"], default="scratch",
                    help="scratch=共享层同随机初值；warm=IMDN_medium 热启动微调")
    ap.add_argument("--wdir", default=None, help="权重输出目录，默认 weights_{tag}")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    tag = "cat_warm" if args.model == "cat" else args.init
    wdir = args.wdir or os.path.join(HERE, f"weights_{tag}")
    os.makedirs(wdir, exist_ok=True)
    samples = load_samples()

    random.seed(args.seed); torch.manual_seed(args.seed)
    if args.model == "cat":
        # 三臂同构 CondIMDNCat、同热启动初值；base 恒喂零，cond 喂真，shuf 喂乱序
        proto = CondIMDNCat(upscale=4)
        mu = proto.load_warm()
        assert all("model.0" in k for k in mu.missing_keys), mu.missing_keys
        base = copy.deepcopy(proto); cond = copy.deepcopy(proto)
        mode_base = "cond_zero"
    else:
        if args.init == "warm":
            base = IMDN_RTC(upscale=4)
            base.load_state_dict(torch.load(WARM_L2, map_location="cpu"))
            cond = CondIMDN_RTC(upscale=4)
            miss, _ = cond.load_warm(); assert not miss
        else:
            base = IMDN_RTC(upscale=4)
            cond = CondIMDN_RTC(upscale=4)
            cond.backbone.load_state_dict(base.state_dict())
        mode_base = "plain"
    shuf = copy.deepcopy(cond)
    print(f"model={args.model} init={args.init} params base/cond =",
          count_params(base), count_params(cond), flush=True)

    n = len(samples)
    # 乱序条件只在同分辨率组内配对（426/320 尺寸不同），保证裁剪尺寸一致
    groups = {}
    for i, (lr, _c, _h, _sy, _sx) in enumerate(samples):
        groups.setdefault((lr.shape[0], lr.shape[1]), []).append(i)
    perm = np.arange(n)
    rng = np.random.RandomState(7)
    for gidx in groups.values():
        gi = np.array(gidx)
        perm[gi] = gi[rng.permutation(len(gi))]
    common = (args.epochs, args.batch, args.lr, args.seed)
    train_arm("base", base, samples, None, *common, mode_base,
              os.path.join(wdir, "base_finetuned.pth"))
    train_arm("cond", cond, samples, None, *common, "cond",
              os.path.join(wdir, "cond_finetuned.pth"))
    train_arm("shuf", shuf, samples, perm, *common, "cond",
              os.path.join(wdir, "shuf_finetuned.pth"))
    print("TRAIN_ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
