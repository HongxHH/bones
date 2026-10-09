"""备用定位：同一 DIV2K warm 权重下，PACNet 最深出口 e3 与标准 IMDN 前向是否逐元素等价。"""
import os
import sys
import torch

ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, os.path.join(ROOT, "player"))
sys.path.insert(0, os.path.join(ROOT, "workspace/pacnet"))
from pacnet_model import PACNet  # noqa: E402
from istream_player.modules.enhancer.imdn_model import IMDN  # noqa: E402

WARM = os.path.join(ROOT, "player/istream_player/modules/enhancer/imdn_weight/IMDN_high_DIV2K_x4.pth")
sd = torch.load(WARM, map_location="cpu")

torch.manual_seed(0)
x = torch.randn(1, 3, 64, 64).cuda()

imdn = IMDN(upscale=4, nf=32).cuda().eval()
imdn.load_state_dict(sd)
pac = PACNet(nf=32, upscale=4, exits=(2, 4, 6), cond=False).cuda().eval()
n = pac.load_imdn_warm(sd)
print("warm moved", n)

with torch.no_grad():
    y_imdn = imdn(x)
    y_e3 = pac(x, exit_idx=2)
print("shape", y_imdn.shape, y_e3.shape)
print("max abs diff:", (y_imdn - y_e3).abs().max().item())
print("mean abs diff:", (y_imdn - y_e3).abs().mean().item())

# 逐层定位：比较共享 block 输出
hooks_imdn, hooks_pac, cap = {}, {}, {}
def grab(store, name):
    def h(m, inp, out):
        store[name] = out.detach()
    return h
for i in range(6):
    imdn.__getattr__(f"IMDB{i+1}").register_forward_hook(grab(cap, f"imdn_b{i}"))
    pac.blocks[i].register_forward_hook(grab(cap, f"pac_b{i}"))
with torch.no_grad():
    imdn(x); pac(x, exit_idx=2)
for i in range(6):
    d = (cap[f"imdn_b{i}"] - cap[f"pac_b{i}"]).abs().max().item()
    print(f"block{i+1} max diff:", d)
