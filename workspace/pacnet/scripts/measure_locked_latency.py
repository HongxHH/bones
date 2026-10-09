# -*- coding: utf-8 -*-
"""锁频后实测 PACNet 240p 三出口延迟，校准算力紧张系数 k。
不写缓存文件。"""
import os, sys, time, torch, torch.nn.functional as F
sys.path.insert(0, r"E:\dev\Code\Python\bones\player")
os.chdir(r"E:\dev\Code\Python\bones\player")
from istream_player.modules.enhancer.pacnet_model import PACNet

w, h = 426, 176  # TOS 240p
DW, DH = 1920, 800  # display
model = PACNet(nf=32, upscale=4, exits=(2,4,6), cond=False)
sd = torch.load(r"E:\dev\Code\Python\bones\workspace\pacnet\weights\stage2_mix_nocond_x4_best.pth", map_location="cuda")
model.load_state_dict(sd); model.cuda().eval()

for exit_idx, name in [(0,"e1"),(1,"e2"),(2,"e3")]:
    # warmup
    with torch.no_grad():
        for _ in range(10):
            x = torch.rand(1,3,h,w).cuda()
            y = model(x, exit_idx=exit_idx)
            y = F.interpolate(y, size=(DH,DW), mode="bicubic", align_corners=False)
    torch.cuda.synchronize()
    # pure forward
    t0=time.perf_counter()
    with torch.no_grad():
        for _ in range(30):
            x = torch.rand(1,3,h,w).cuda()
            y = model(x, exit_idx=exit_idx)
    torch.cuda.synchronize()
    fwd=(time.perf_counter()-t0)/30*1000
    # forward + interpolate
    t0=time.perf_counter()
    with torch.no_grad():
        for _ in range(30):
            x = torch.rand(1,3,h,w).cuda()
            y = model(x, exit_idx=exit_idx)
            y = F.interpolate(y, size=(DH,DW), mode="bicubic", align_corners=False)
    torch.cuda.synchronize()
    endtoend=(time.perf_counter()-t0)/30*1000
    print(f"{name}: pure_fwd={fwd:.2f}ms  fwd+interp={endtoend:.2f}ms")

print()
print("满血基准(缓存): e1=11.64 e2=24.19 e3=33.79 ms")
