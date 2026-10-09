"""CascadePACNet 结构自检：warm 迁移数、零增量起步、detach 梯度隔离、出口 shape/时延。"""
import os
import sys

import torch

ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, os.path.join(ROOT, "workspace/pacnet"))
from pacnet_model import CascadePACNet  # noqa: E402

WARM = os.path.join(ROOT, "player/istream_player/modules/enhancer/imdn_weight/IMDN_high_DIV2K_x4.pth")
DEV = "cuda:0"
m = CascadePACNet(nf=32, upscale=4, exits=(2, 4, 6), cond=False).to(DEV)
n = m.load_imdn_warm(torch.load(WARM, map_location="cpu"), verbose=True)
assert n >= 90, f"warm 迁移过少 {n}"

x = torch.randn(1, 3, 60, 106, device=DEV)
outs = m(x)
print("train outs shapes:", [tuple(o.shape) for o in outs])
for i in range(3):
    print("exit_idx", i, tuple(m(x, exit_idx=i).shape))

# 零增量起步：head2/3 最终卷积零化 => s2==s1, s3==s2
d21 = (outs[1] - outs[0]).abs().max().item()
d32 = (outs[2] - outs[1]).abs().max().item()
print("init |s2-s1| max", d21, " |s3-s2| max", d32)
assert d21 < 0.2 and d32 < 0.2, "增量头初始增量应≈0"

# 梯度隔离1：只回 e1，深层 block3..6 不应有梯度
m.zero_grad(); outs = m(x); outs[0].sum().backward()
g_shallow = m.blocks[0].c1.weight.grad.abs().sum().item()
g_deep = m.blocks[2].c1.weight.grad
print("e1 backward: block1 grad", round(g_shallow, 4), " block3 grad", None if g_deep is None else "NONZERO(错误)")
assert g_deep is None, "e1 梯度泄漏到深层"

# 梯度隔离2：只回 e3，段1/段2 不应有梯度，仅段3(block5/6)可训
m.zero_grad(); outs = m(x); outs[2].sum().backward()
def gnorm(mod):
    g = mod.c1.weight.grad
    return None if g is None else round(g.abs().sum().item(), 4)
g_b1, g_b3, g_b5 = gnorm(m.blocks[0]), gnorm(m.blocks[2]), gnorm(m.blocks[4])
g_head1 = m.heads[0].upsampler[0].weight.grad
def isolated(v):
    return v is None or v == 0.0
print("e3 backward: block1(段1)", g_b1, " block3(段2)", g_b3, " block5(段3)", g_b5,
      " head1", None if g_head1 is None else "NONZERO(错误)")
assert isolated(g_b1) and isolated(g_b3) and (g_b5 is not None and g_b5 > 0) and g_head1 is None, "e3 段隔离失败"

# 梯度隔离3：只回 e2，段1/段3 不应有梯度，仅段2(block3/4)可训
m.zero_grad(); outs = m(x); outs[1].sum().backward()
g_b1, g_b3, g_b5 = gnorm(m.blocks[0]), gnorm(m.blocks[2]), gnorm(m.blocks[4])
print("e2 backward: block1(段1)", g_b1, " block3(段2)", g_b3, " block5(段3)", g_b5)
assert isolated(g_b1) and (g_b3 is not None and g_b3 > 0) and isolated(g_b5), "e2 段隔离失败"
print("\nCASCADE SELFTEST PASS")
