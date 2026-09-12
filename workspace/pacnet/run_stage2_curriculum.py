"""阶段1 Step2：深度课程。在最优自蒸馏 alpha=0.1 上叠加先浅后深课程，
对比"纯课程"与"课程后冻结共享浅层"，看弹性税能否进一步收窄。"""
import os
import random
import sys

import torch
from torch.utils.data import DataLoader

import run_stage1 as rs
from train_leaveout import build_cache, PairDataset, UNIFORM_TEST
from train_pacnet import train_pac

WDIR = rs.WDIR


def main():
    random.seed(1); torch.manual_seed(1)
    print("building cache once...", flush=True)
    train_pairs, val_pairs = build_cache(8, UNIFORM_TEST, [30, 90, 140])
    _lr0, _hr0 = train_pairs[0]
    sy, sx = _hr0.shape[0] / _lr0.shape[0], _hr0.shape[1] / _lr0.shape[1]
    dl = DataLoader(PairDataset(train_pairs, 192, 4, True, sy, sx), batch_size=16,
                    shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

    configs = [
        ("cur10", dict(dist_w=0.1, curriculum_ep=10, freeze_shallow=False)),
        ("cur10fz", dict(dist_w=0.1, curriculum_ep=10, freeze_shallow=True)),
    ]
    for name, kw in configs:
        out = os.path.join(WDIR, f"stage2_{name}_x4.pth")
        print(f"\n########## train {name} {kw} ##########", flush=True)
        train_pac(dl, val_pairs, 30, 2e-4, 0.5, out,
                  weights=(0.3, 0.6, 1.0), use_cond=False, **kw)

    pac_tags = [("d000", "pacnet_x4_best.pth"),
                ("d01", "stage1_d01_x4_best.pth"),
                ("cur10", "stage2_cur10_x4_best.pth"),
                ("cur10fz", "stage2_cur10fz_x4_best.pth")]
    rs.evaluate(pac_tags, os.path.join(rs.PACDIR, "stage2_sweep.json"))


if __name__ == "__main__":
    main()
