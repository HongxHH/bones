"""阶段0自检：模型构造、warm 迁移、三出口 shape、exit 截断、参数量。不训练。"""
import os
import sys

import torch

ROOT = r"E:\dev\Code\Python\bones"
PACDIR = os.path.join(ROOT, "workspace/pacnet")
sys.path.insert(0, PACDIR)
from pacnet_model import PACNet, PlainIMDN  # noqa: E402

WARM = os.path.join(ROOT, "player/istream_player/modules/enhancer/imdn_weight/IMDN_high_DIV2K_x4.pth")
sd = torch.load(WARM, map_location="cpu")
print("IMDN warm keys:", len(sd))

pac = PACNet(nf=32, upscale=4, exits=(2, 4, 6), cond=True)
n = pac.load_imdn_warm(sd, verbose=True)
print("PAC warm-moved:", n)
x = torch.randn(2, 3, 48, 48)
pac.eval()
with torch.no_grad():
    outs = pac(x)
    print("train-mode exits:", [tuple(o.shape) for o in outs])
    for ei in range(3):
        o = pac(x, exit_idx=ei)
        print(f"  exit_idx={ei} ->", tuple(o.shape))
assert all(tuple(o.shape) == (2, 3, 192, 192) for o in outs)

plain = PlainIMDN(nf=32, num_modules=2, upscale=4)
n2 = plain.load_imdn_warm(sd)
with torch.no_grad():
    po = plain(x)
print("PLAIN2 warm-moved:", n2, "out:", tuple(po.shape))
assert tuple(po.shape) == (2, 3, 192, 192)


def count(m):
    return sum(p.numel() for p in m.parameters())


print("---- param counts ----")
print("PAC total:", count(pac))
print("  fea+blocks(shared):", count(pac.fea_conv) + sum(count(b) for b in pac.blocks))
for i, h in enumerate(pac.heads):
    print(f"  head{(2,4,6)[i]}:", count(h))
print("  cond:", count(pac.cond))
print("PLAIN2 total:", count(plain))

# 出口推理时延（240p 整帧尺寸 426x240），cuda
pac = pac.cuda().eval()
xx = torch.randn(1, 3, 240, 426, device="cuda")
for ei in range(3):
    for _ in range(3):
        pac(xx, exit_idx=ei)
    torch.cuda.synchronize()
    import time
    t0 = time.time()
    for _ in range(20):
        pac(xx, exit_idx=ei)
    torch.cuda.synchronize()
    print(f"exit {(2,4,6)[ei]} avg ms/frame:", round((time.time() - t0) / 20 * 1000, 2))
print("SELFTEST OK")
