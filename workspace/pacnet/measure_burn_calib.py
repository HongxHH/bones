# -*- coding: utf-8 -*-
"""校准 burn：设 PACNET_EXTRA_MS 后，测三出口端到端(forward+interp+burn)延迟。"""
import os, sys, time, torch, torch.nn.functional as F
os.environ["PACNET_EXTRA_MS"] = "17"  # 目标：e3 33.8+17≈50.8ms (k≈1.5)
sys.path.insert(0, r"E:\dev\Code\Python\bones\player")
os.chdir(r"E:\dev\Code\Python\bones\player")
from istream_player.modules.enhancer.pacnet_model import PACNet
from istream_player.modules.enhancer import _gpuburn

w,h,DW,DH = 426,176,1920,800
model = PACNet(nf=32, upscale=4, exits=(2,4,6), cond=False)
model.load_state_dict(torch.load(r"E:\dev\Code\Python\bones\workspace\pacnet\weights\stage2_mix_nocond_x4_best.pth", map_location="cuda"))
model.cuda().eval()

for ei,name in [(0,"e1"),(1,"e2"),(2,"e3")]:
    with torch.no_grad():
        for _ in range(10):
            x=torch.rand(1,3,h,w).cuda(); y=model(x,exit_idx=ei)
            y=F.interpolate(y,size=(DH,DW),mode="bicubic",align_corners=False); _gpuburn.burn()
    torch.cuda.synchronize()
    t0=time.perf_counter()
    with torch.no_grad():
        for _ in range(30):
            x=torch.rand(1,3,h,w).cuda(); y=model(x,exit_idx=ei)
            y=F.interpolate(y,size=(DH,DW),mode="bicubic",align_corners=False); _gpuburn.burn()
    torch.cuda.synchronize()
    ms=(time.perf_counter()-t0)/30*1000
    print(f"{name}: 端到端={ms:.2f}ms  (帧间隔41.67, k={ms/[11.64,24.19,33.79][ei]:.2f})")
