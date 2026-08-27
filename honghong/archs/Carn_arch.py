import torch
from torch import nn

from honghong.archs.ops import CARN_common


class CARN(nn.Module):
    def __init__(self, upscale = 4):
        super(CARN, self).__init__()


        self.scale = upscale
        # print(scale)
        # multi_scale = kwargs.get("multi_scale")
        # group = kwargs.get("group", 1)

        self.sub_mean = CARN_common.MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = CARN_common.MeanShift((0.4488, 0.4371, 0.4040), sub=False)

        self.entry = nn.Conv2d(3, 64, 3, 1, 1)

        self.b1 = Block(64, 64)
        self.b2 = Block(64, 64)
        self.b3 = Block(64, 64)
        self.c1 = CARN_common.BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = CARN_common.BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = CARN_common.BasicBlock(64 * 4, 64, 1, 1, 0)

        self.upsample = CARN_common.UpsampleBlock(64, scale=self.scale)
        self.exit = nn.Conv2d(64, 3, 3, 1, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.entry(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        out = self.upsample(o3, scale=self.scale)

        out = self.exit(out)
        out = self.add_mean(out)

        return out


class Block(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 group=1):
        super(Block, self).__init__()

        self.b1 = CARN_common.ResidualBlock(64, 64)
        self.b2 = CARN_common.ResidualBlock(64, 64)
        self.b3 = CARN_common.ResidualBlock(64, 64)
        self.c1 = CARN_common.BasicBlock(64 * 2, 64, 1, 1, 0)
        self.c2 = CARN_common.BasicBlock(64 * 3, 64, 1, 1, 0)
        self.c3 = CARN_common.BasicBlock(64 * 4, 64, 1, 1, 0)

    def forward(self, x):
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)

        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)

        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)

        return o3
