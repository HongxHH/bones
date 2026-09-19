# -*- coding: utf-8 -*-
"""H1 条件网训练缓存：对 6 个 BBB 退化视频均匀抽帧，提取 LR 与码流条件(QP+前后向MV)。

产物 cache/bbb_{w}_crf{c}.npz：
  idx   (K,)        显示序帧号（与 output_1080p.mp4 同帧号对齐）
  lr    (K,H,W,3)   uint8 RGB（取自同一 PyAV 帧，与 MV 天然对齐）
  cond  (K,7,H,W)   float16 条件场（见 codec_lib.NCOND）
HR（output_1080p）在训练脚本里按并集帧号一次性读入内存，不在此重复存。
"""
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import codec_lib  # noqa: E402

ROOT = r"E:\dev\Code\Python\bones"
DEG = os.path.join(HERE, "deg")
CACHE = os.path.join(HERE, "cache")
K = 180
VIDEOS = [(w, crf) for w in (426, 320) for crf in (26, 34, 42)]


def main():
    os.makedirs(CACHE, exist_ok=True)
    for w, crf in VIDEOS:
        name = f"bbb_{w}_crf{crf}"
        vp = os.path.join(DEG, name + ".mp4")
        outp = os.path.join(CACHE, name + ".npz")
        if os.path.exists(outp):
            print("skip existing", name, flush=True)
            continue
        cap = cv2.VideoCapture(vp)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
        # 每个退化视频用独立、跨视频不重叠的随机帧集，最大化源帧多样性（最多 6K 个不同 HR 帧）
        rng = np.random.RandomState(1000 + w * 7 + crf)
        wanted = np.sort(rng.choice(np.arange(60, n - 60), K, replace=False))
        t0 = time.time()
        print(f"[{name}] n={n} wanted={K} extracting MV+QP ...", flush=True)
        frame_cache, Wd, Hd = codec_lib.build_frame_cache(vp, wanted)
        lrs = np.zeros((K, Hd, Wd, 3), np.uint8)
        conds = np.zeros((K, codec_lib.NCOND, Hd, Wd), np.float16)
        miss = 0
        for j, i in enumerate(wanted):
            if int(i) not in frame_cache:
                miss += 1
                continue
            rgb, c = frame_cache[int(i)]
            lrs[j] = rgb
            conds[j] = c.astype(np.float16)
        np.savez_compressed(outp, idx=wanted.astype(np.int64), lr=lrs, cond=conds,
                            W=np.array([Wd]), H=np.array([Hd]))
        print(f"[{name}] done {Wd}x{Hd} miss={miss} {time.time()-t0:.0f}s", flush=True)
    print("BUILD_CACHE_DONE", flush=True)


if __name__ == "__main__":
    main()
