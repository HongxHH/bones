"""阶段1 Step3：残差精修级联 CascadePACNet。训练侧手段（自蒸馏/课程/冻结）均无法消除
弹性税后，改用结构手段：s1 完整重建、s2=s1.detach+增量、s3=s2.detach+增量，浅层只被 e1 监督。"""
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

    out = os.path.join(WDIR, "stage3_cas_x4.pth")
    print("\n########## train CascadePACNet ##########", flush=True)
    train_pac(dl, val_pairs, 30, 2e-4, 0.5, out,
              weights=(0.3, 0.6, 1.0), use_cond=False, cascade=True)

    pac_tags = [("d000", "pacnet_x4_best.pth"),
                ("d01", "stage1_d01_x4_best.pth"),
                ("cas", "stage3_cas_x4_best.pth")]
    rs.evaluate(pac_tags, os.path.join(rs.PACDIR, "stage3_sweep.json"),
                kinds={"cas": "cas"})


if __name__ == "__main__":
    main()
