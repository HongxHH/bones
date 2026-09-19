# -*- coding: utf-8 -*-
"""H1 条件网：在 l2 规模 IMDN_RTC（nf=12, num_modules=5）骨干上，加一条码流条件编码支路。

设计原则（保证对照干净）：
- 骨干与无条件基线完全相同，fea_conv / 残差块 / 上采样结构和权重逐字一致，可直接热加载
  IMDN_medium_DIV2K_x4.pth；
- 条件（QP+前后向MV，7 通道，LR 分辨率）经一个两层 3x3 小编码器映射到 nf 通道，
  在 fea_conv 输出处**逐元素相加**注入，再走原残差骨干；
- 新增参数仅条件编码器（约 2k），相对骨干可忽略，参数量差异在报告中给出；
- 唯一系统变量是“是否喂码流条件”，便于离线小样本证伪。
"""
import os
import sys

import torch
import torch.nn as nn

ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, os.path.join(ROOT, "player"))
from istream_player.modules.enhancer.imdn_model import IMDN_RTC  # noqa: E402

WARM_L2 = os.path.join(
    ROOT, "player/istream_player/modules/enhancer/imdn_weight/IMDN_medium_DIV2K_x4.pth")


class CondIMDN_RTC(nn.Module):
    def __init__(self, upscale=4, nf=12, num_modules=5, ncond=7, cond_hidden=12):
        super().__init__()
        self.backbone = IMDN_RTC(upscale=upscale, nf=nf, num_modules=num_modules)
        seq = list(self.backbone.model)
        self.fea_conv = seq[0]          # conv 3->nf
        self.body = seq[1]              # ShortcutBlock(IMDModule_speed x5 + LR_conv)
        self.upsampler = nn.Sequential(*seq[2:])
        self.cond_enc = nn.Sequential(
            nn.Conv2d(ncond, cond_hidden, 3, 1, 1),
            nn.LeakyReLU(0.05, inplace=True),
            nn.Conv2d(cond_hidden, nf, 3, 1, 1),
        )

    def load_warm(self, warm_path=WARM_L2):
        # IMDN_RTC 的参数都挂在 backbone.model.* 下，直接整体热加载骨干
        state = torch.load(warm_path, map_location="cpu")
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        return missing, unexpected

    def forward(self, x, cond=None):
        h = self.fea_conv(x)
        if cond is not None:
            h = h + self.cond_enc(cond)
        h = self.body(h)
        return self.upsampler(h)


class CondIMDNCat(nn.Module):
    """条件作为额外输入通道拼到首层（最强、最直接注入），排除“注入太弱”的假阴性。

    首层 fea_conv 由 3->nf 扩为 (3+ncond)->nf；前 3 通道拷贝热启动权重，
    多出的 ncond 通道零初始化，cond=全 0 时输出与热启动骨干严格一致。
    base 恒喂 0（等价纯 3 通道）、cond 喂真条件、shuf 喂乱序条件，三臂同结构同参数量。
    """

    def __init__(self, upscale=4, nf=12, ncond=7, warm_path=WARM_L2):
        super().__init__()
        self.ncond = ncond
        # 先热加载 3 通道骨干，再把首层扩成 3+ncond（多出通道零初始化）
        self.backbone = IMDN_RTC(upscale=upscale, nf=nf)
        self.backbone.load_state_dict(torch.load(warm_path, map_location="cpu"))
        old = self.backbone.model[0]
        conv = nn.Conv2d(3 + ncond, nf, 3, padding=1)
        with torch.no_grad():
            conv.weight.zero_()
            conv.weight[:, :3].copy_(old.weight)
            conv.bias.copy_(old.bias)
        self.backbone.model[0] = conv

    def load_warm(self, warm_path=WARM_L2):
        class _R:  # 已在 __init__ 内热加载，保持与 add 版相同的调用接口
            missing_keys, unexpected_keys = [], []
        return _R()

    def forward(self, x, cond=None):
        if cond is None:
            cond = torch.zeros(x.size(0), self.ncond, x.size(2), x.size(3), device=x.device)
        return self.backbone(torch.cat([x, cond], dim=1))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    base = IMDN_RTC(upscale=4)
    cond = CondIMDN_RTC(upscale=4)
    miss, unexp = cond.load_warm()
    print("warm missing:", miss)
    print("warm unexpected:", unexp)
    print("baseline params:", count_params(base))
    print("condnet  params:", count_params(cond))
    print("cond encoder params:", count_params(cond.cond_enc))
    x = torch.randn(2, 3, 48, 48)
    c = torch.randn(2, 7, 48, 48)
    cond.eval()
    with torch.no_grad():
        y_no = cond(x, cond=None)
        y_c = cond(x, cond=c)
    print("out no-cond", tuple(y_no.shape), "out cond", tuple(y_c.shape))
    # 未训练条件编码器初始接近 0（最后一层默认初始化非0），验证 cond=None 与骨干等价
    base.load_state_dict(torch.load(WARM_L2, map_location="cpu"))
    base.eval(); cond.eval()
    with torch.no_grad():
        a = base(x); b = cond(x, cond=None)
    print("cond=None equals backbone:", float((a - b).abs().max()))
