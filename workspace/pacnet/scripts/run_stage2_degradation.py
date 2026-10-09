"""阶段2：退化条件化验证。用播放器已有的多分辨率档（240/360/480/720p，天然携带不同压缩强度）
构造跨退化数据，训练混合 240/360/480p，留出 720p 作未见退化，对比：
  A 单退化专用（240p cond=False，复用阶段1主版权重）
  B 多退化无条件（混合 cond=False）
  C 多退化+条件化（混合 cond=True，CondEncoder 在线退化统计 FiLM）
判据：C 在未见 720p 上是否 >= B（条件化有用）；C 各档是否 >= A 对应档（一套权重可替代专用模型）。
"""
import json
import os
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, os.path.join(ROOT, "workspace/enhancer_survey"))
sys.path.insert(0, os.path.join(ROOT, "workspace/pacnet"))
sys.path.insert(0, os.path.join(ROOT, "player"))
import run_stage1 as rs  # noqa: E402
from train_leaveout import UNIFORM_TEST, FRAMES_PER_SEG  # noqa: E402
from train_pacnet import train_pac  # noqa: E402
from pacnet_model import PACNet  # noqa: E402

DEV = rs.DEV
WDIR = rs.WDIR
TRAIN_RES = [240, 360, 480]          # 训练混合三档
EVAL_RES = [240, 360, 480, 720]      # 评测含未见 720p
VAL_SEGS = [30, 90, 140]
TEST_SEGS = list(range(0, 159, 10))
FRAME_JS = (75,)  # 跨退化对比每段抽 1 帧，16 帧足够看模型间相对差异


@torch.no_grad()
def forward_tiled(model, x, tile=640, overlap=16):
    """大分辨率输入分块前向，避免 5120×2880 输出触发 cuDNN 慢速路径；重叠区取平均。"""
    B, C, H, W = x.shape
    if W <= tile and H <= tile:
        return model(x)
    step = tile - overlap
    outs = None
    for y in range(0, H, step):
        for x0 in range(0, W, step):
            y2, x2 = min(y + tile, H), min(x0 + tile, W)
            pouts = model(x[:, :, y:y2, x0:x2])
            if outs is None:
                outs = [torch.zeros(B, C, H * 4, W * 4, device=x.device) for _ in pouts]
                cnt = [torch.zeros(B, C, H * 4, W * 4, device=x.device) for _ in pouts]
            for ei, po in enumerate(pouts):
                outs[ei][:, :, y * 4:y2 * 4, x0 * 4:x2 * 4] += po
                cnt[ei][:, :, y * 4:y2 * 4, x0 * 4:x2 * 4] += 1
    return [o / c for o, c in zip(outs, cnt)]


def build_multi_cache(res_list, per_seg_step, hold, val_segs):
    """对每个分辨率档顺序扫 LR+GT，缓存 (lr,hr,res,sy,sx)。GT 重复扫，可接受。"""
    train, val = [], []
    for res in res_list:
        cap_lr = cv2.VideoCapture(os.path.join(ROOT, "data/video_mp4box", f"output_{res}p.mp4"))
        cap_hr = cv2.VideoCapture(os.path.join(ROOT, "data/video_mp4box", "output_1080p.mp4"))
        f = 0; t0 = time.time()
        while True:
            ok1, lr = cap_lr.read(); ok2, hr = cap_hr.read()
            if not (ok1 and ok2):
                break
            seg, j = divmod(f, FRAMES_PER_SEG)
            sy, sx = hr.shape[0] / lr.shape[0], hr.shape[1] / lr.shape[1]
            if seg in hold:
                if seg in val_segs and j in (30, 60, 90):
                    val.append((lr, hr, res, sy, sx))
            elif j % per_seg_step == 0:
                train.append((lr, hr, res, sy, sx))
            f += 1
        cap_lr.release(); cap_hr.release()
        print(f"  res {res}p scanned {f} frames, cum train={len(train)} val={len(val)} "
              f"{time.time()-t0:.0f}s", flush=True)
    print(f"multi cache done: train={len(train)} val={len(val)}", flush=True)
    return train, val


class MultiPairDataset(Dataset):
    """每对自带 sy/sx，按各自真实缩放比裁 GT，再 resize 到统一 patch×patch。"""

    def __init__(self, pairs, patch=192, scale=4, train=True):
        self.pairs, self.patch, self.scale, self.is_train = pairs, patch, scale, train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        lr_bgr, hr_bgr, _res, sy, sx = self.pairs[i]
        lr = cv2.cvtColor(lr_bgr, cv2.COLOR_BGR2RGB)
        hr = cv2.cvtColor(hr_bgr, cv2.COLOR_BGR2RGB)
        if self.is_train:
            ip = self.patch // self.scale
            ih, iw = lr.shape[:2]
            iy = random.randrange(0, ih - ip + 1); ix = random.randrange(0, iw - ip + 1)
            lr = lr[iy:iy + ip, ix:ix + ip]
            ty, tx = int(round(sy * iy)), int(round(sx * ix))
            th, tw = int(round(sy * ip)), int(round(sx * ip))
            hr = hr[ty:ty + th, tx:tx + tw]
            hr = cv2.resize(hr, (self.patch, self.patch), interpolation=cv2.INTER_CUBIC)
            if random.random() < 0.5: lr, hr = lr[:, ::-1], hr[:, ::-1]
            if random.random() < 0.5: lr, hr = lr[::-1], hr[::-1]
            if random.random() < 0.5: lr, hr = lr.transpose(1, 0, 2), hr.transpose(1, 0, 2)
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1))).float().div(255)
        return t(lr), t(hr)


def evaluate_cross_res(model_tags, out_json):
    """逐个模型评测（GPU 只留一个模型，避免 720p 大张量多模型共存触发 CUDA 慢速路径）。
    model_tags: [(tag, cond_bool, 权重文件名)]。"""
    tags = [t for t, _, _ in model_tags]
    acc = {res: {t: {"e1": [], "e2": [], "e3": []} for t in tags} for res in EVAL_RES}
    base_acc = {res: [] for res in EVAL_RES}
    base_done = set()
    for tag, cond, wf in model_tags:
        m = PACNet(nf=32, upscale=4, exits=(2, 4, 6), cond=cond).to(DEV).eval()
        m.load_state_dict(torch.load(os.path.join(WDIR, wf), map_location="cpu"))
        for res in EVAL_RES:
            cap_lr = cv2.VideoCapture(os.path.join(ROOT, "data/video_mp4box", f"output_{res}p.mp4"))
            cap_hr = cv2.VideoCapture(os.path.join(ROOT, "data/video_mp4box", "output_1080p.mp4"))
            want = {(s, j) for s in TEST_SEGS for j in FRAME_JS}
            f = 0
            while True:
                ok1, lr = cap_lr.read(); ok2, hr = cap_hr.read()
                if not (ok1 and ok2):
                    break
                seg, j = divmod(f, FRAMES_PER_SEG)
                if (seg, j) in want:
                    x = rs.to_t(lr)
                    if res not in base_done:
                        base_up = cv2.resize(lr, (1920, 1080), interpolation=cv2.INTER_CUBIC)
                        base_acc[res].append(rs.psnr(rs.to_t(base_up), hr))
                    outs = forward_tiled(m, x)
                    for ei, o in enumerate(outs):
                        acc[res][tag][f"e{ei+1}"].append(rs.psnr(o, hr))
                f += 1
            cap_lr.release(); cap_hr.release()
            print(f"  [{tag}] res {res}p done", flush=True)
            base_done.add(res)
        del m
        torch.cuda.empty_cache()

    summary = {"base": {res: round(float(np.mean(base_acc[res])), 3) for res in EVAL_RES},
               "per_res": {}}
    print("\n==== 跨退化评测（e3 深档 PSNR，行=退化档 列=模型）====")
    header = f"{'res':>6s} " + " ".join(f"{t:>10s}" for t in tags) + f" {'base':>8s}"
    print(header)
    for res in EVAL_RES:
        row = {t: {k: round(float(np.mean(acc[res][t][k])), 3) for k in ("e1", "e2", "e3")} for t in tags}
        summary["per_res"][res] = row
        line = f"{res:>4d}p " + " ".join(f"{row[t]['e3']:10.3f}" for t in tags) + f" {summary['base'][res]:8.3f}"
        print(line)
    print("\n==== e1 保命档（看浅档泛化）====")
    print(f"{'res':>6s} " + " ".join(f"{t:>10s}" for t in tags))
    for res in EVAL_RES:
        print(f"{res:>4d}p " + " ".join(f"{summary['per_res'][res][t]['e1']:10.3f}" for t in tags))
    json.dump(summary, open(out_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("saved", out_json)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--skip_train", action="store_true")
    args = ap.parse_args()

    if not args.skip_train:
        random.seed(1); torch.manual_seed(1)
        print("building multi-resolution cache (240/360/480)...", flush=True)
        train_pairs, val_pairs = build_multi_cache(TRAIN_RES, 8, set(UNIFORM_TEST), VAL_SEGS)
        dl = DataLoader(MultiPairDataset(train_pairs, 192, 4, True), batch_size=16,
                        shuffle=True, num_workers=0, drop_last=True, pin_memory=True)
        # val_pac 只做整帧 PSNR，只需 (lr,hr) 二元组
        val_2 = [(lr, hr) for lr, hr, _, _, _ in val_pairs]

        print("\n########## B: 多退化 无条件 ##########", flush=True)
        train_pac(dl, val_2, args.epochs, 2e-4, 0.5,
                  os.path.join(WDIR, "stage2_mix_nocond_x4.pth"),
                  weights=(0.3, 0.6, 1.0), use_cond=False)
        print("\n########## C: 多退化 + 条件化 ##########", flush=True)
        train_pac(dl, val_2, args.epochs, 2e-4, 0.5,
                  os.path.join(WDIR, "stage2_mix_cond_x4.pth"),
                  weights=(0.3, 0.6, 1.0), use_cond=True)

    # A 复用阶段1主版（240p 单退化 cond=False）；B/C 本次训练
    model_tags = [("A_single240", False, "pacnet_x4_best.pth"),
                  ("B_mix_nocond", False, "stage2_mix_nocond_x4_best.pth"),
                  ("C_mix_cond", True, "stage2_mix_cond_x4_best.pth")]
    evaluate_cross_res(model_tags, os.path.join(rs.PACDIR, "stage2_cross_res.json"))


if __name__ == "__main__":
    main()
