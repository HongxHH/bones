# -*- coding: utf-8 -*-
"""复测三模型 1080p->4K 单帧延迟稳定性（fp32/fp16，两轮独立计时）。"""
import os, sys, time
import numpy as np
import torch, cv2, copy
sys.path.insert(0, r"E:\dev\Code\Python\bones\player")
from istream_player.modules.enhancer.imdn_model import IMDN_RTC, IMDN
DEVICE = "cuda:0"
W, H = 1920, 1080
WDIR = r"E:\dev\Code\Python\bones\player\istream_player\modules\enhancer\imdn_weight"
LEVELS = [("low", IMDN_RTC(upscale=2, num_modules=3, nf=6), "IMDN_low_DIV2K_x2.pth"),
          ("med", IMDN_RTC(upscale=2), "IMDN_medium_DIV2K_x2.pth"),
          ("high", IMDN(upscale=2, nf=32), "IMDN_high_DIV2K_x2.pth")]

def tfunc(m, bgr, half=False):
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2,0,1).float().unsqueeze(0).to(DEVICE)/255.
    if half: t = t.half()
    with torch.no_grad(): o = m(t)
    torch.cuda.synchronize()

def bench(m, half, n=30):
    d = np.random.randint(0,255,(H,W,3),np.uint8)
    for _ in range(5): tfunc(m,d,half)
    ts=[]
    for _ in range(n):
        t0=time.time(); tfunc(m,d,half); ts.append((time.time()-t0)*1000)
    return np.mean(ts), np.median(ts), np.percentile(ts,95)

for name, m, wf in LEVELS:
    m.load_state_dict(torch.load(os.path.join(WDIR,wf),map_location="cpu"))
    m32 = m.to(DEVICE).eval()
    a = bench(m32, False)
    b = bench(m32, False)
    m16 = copy.deepcopy(m32).half()
    c = bench(m16, True)
    # fp16 数值检查
    d = np.random.randint(0,255,(H,W,3),np.uint8)
    img = cv2.cvtColor(d, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2,0,1).half().unsqueeze(0).to(DEVICE)/255.
    with torch.no_grad(): o16 = m16(t)
    finite = bool(torch.isfinite(o16).all().item())
    print(f"{name:5s} fp32 run1 mean={a[0]:8.2f} p50={a[1]:8.2f} | run2 mean={b[0]:8.2f} | fp16 mean={c[0]:8.2f} finite={finite}")
    del m32, m16
    torch.cuda.empty_cache()
