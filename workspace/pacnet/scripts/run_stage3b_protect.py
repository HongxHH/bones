"""阶段1 Step3b：只保护保命档的折中级联。block1/2 只被 e1 训练（追平 plain2），
e2/e3 仍完整共享 block3..6 协同。对比三段全隔离级联（已失败）是否既补 e1 又保住深档。"""
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
    lr0, hr0 = train_pairs[0]
    sy, sx = hr0.shape[0] / lr0.shape[0], hr0.shape[1] / lr0.shape[1]
    dl = DataLoader(PairDataset(train_pairs, 192, 4, True, sy, sx), batch_size=16,
                    shuffle=True, num_workers=0, drop_last=True, pin_memory=True)

    out = os.path.join(WDIR, "stage3b_caspe_x4.pth")
    print("\n########## train CascadePACNet protect_e1 ##########", flush=True)
    train_pac(dl, val_pairs, 30, 2e-4, 0.5, out,
              weights=(0.3, 0.6, 1.0), use_cond=False, cascade=True, protect_e1=True)

    pac_tags = [("d000", "pacnet_x4_best.pth"),
                ("a01", "s1_a01_x4_best.pth"),
                ("caspe", "stage3b_caspe_x4_best.pth")]
    rs.evaluate(pac_tags, os.path.join(rs.PACDIR, "stage3b_sweep.json"),
                kinds={"caspe": "caspe"})


if __name__ == "__main__":
    main()
