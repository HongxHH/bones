"""
免编译的分组调制可变形卷积（DCNv2），用 grid_sample 向量化实现。
正确性判据：offset=0、mask=1 时，数值等价于同权重的 F.conv2d（padding 相同）。
STDF 中 groups(conv)=1、deform_groups=in_nc、stride=1、dilation=1、kernel=3。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def modulated_deform_conv(x, offset, mask, weight, bias, stride=1, padding=1,
                          dilation=1, groups=1, deform_groups=1):
    N, C, H, W = x.shape
    Cout, _, kH, kW = weight.shape
    Cdg = C // deform_groups  # 每个可变形分组的输入通道
    Hout = (H + 2 * padding - dilation * (kH - 1) - 1) // stride + 1
    Wout = (W + 2 * padding - dilation * (kW - 1) - 1) // stride + 1
    dev, dt = x.device, x.dtype
    kN = kH * kW

    # 输出像素坐标
    oy = torch.arange(Hout, device=dev, dtype=dt).view(1, 1, Hout, 1)
    ox = torch.arange(Wout, device=dev, dtype=dt).view(1, 1, 1, Wout)
    dy = (torch.arange(kH, device=dev, dtype=dt) - kH // 2) * dilation
    dx = (torch.arange(kW, device=dev, dtype=dt) - kW // 2) * dilation
    # 基础采样位置 (1, kN, Hout, Wout)；越界由 grid_sample zeros 等价 padding，不再减 padding
    by = (oy * stride + dy.view(kH, 1).repeat(1, kW).reshape(kN, 1, 1))
    bx = (ox * stride + dx.view(1, kW).repeat(kH, 1).reshape(kN, 1, 1))

    # offset/mask 顺序：[g][m]，通道 index = g*kN+m
    off = offset.reshape(N, deform_groups, 2, kN, Hout, Wout)
    msk = mask.reshape(N, deform_groups, kN, Hout, Wout)
    sy = by + off[:, :, 0]  # (N,G,kN,Hout,Wout)
    sx = bx + off[:, :, 1]
    # 归一化到 grid_sample（align_corners=False：像素 i 中心为 (i+0.5)/size*2-1）
    gx = (sx + 0.5) / W * 2 - 1
    gy = (sy + 0.5) / H * 2 - 1
    grid = torch.stack([gx, gy], dim=-1).reshape(N * deform_groups * kN, Hout, Wout, 2)

    # 输入按可变形分组展开到 batch：(N,G,Cdg,H,W)->重复 kN ->(N*G*kN,Cdg,H,W)
    xg = x.reshape(N, deform_groups, Cdg, H, W).unsqueeze(2).expand(N, deform_groups, kN, Cdg, H, W)
    xg = xg.reshape(N * deform_groups * kN, Cdg, H, W)
    samp = F.grid_sample(xg, grid, mode="bilinear", padding_mode="zeros",
                         align_corners=False)  # (N*G*kN,Cdg,Hout,Wout)
    samp = samp.reshape(N, deform_groups, kN, Cdg, Hout, Wout)
    samp = samp * msk.unsqueeze(3)  # 广播 Cdg
    # -> columns (N, C, kN, Hout, Wout)，通道序按 [g][Cdg] 排回
    cols = samp.permute(0, 1, 3, 2, 4, 5).reshape(N, C, kN, Hout, Wout)
    cols = cols.reshape(N, C * kN, Hout, Wout)
    Ccg = C // groups
    w2 = weight.reshape(Cout, Ccg * kN, 1, 1) if groups == 1 else \
        weight.reshape(groups, Cout // groups, Ccg * kN, 1, 1)
    if groups == 1:
        y = F.conv2d(cols, w2, bias, stride=1, padding=0)
    else:
        ys = []
        for g in range(groups):
            cg = cols[:, g * Ccg * kN:(g + 1) * Ccg * kN]
            ys.append(F.conv2d(cg, w2[g], None))
        y = torch.cat(ys, dim=1)
        if bias is not None:
            y = y + bias.view(1, Cout, 1, 1)
    return y


class ModulatedDeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, deformable_groups=1, bias=True):
        super().__init__()
        self.stride, self.padding, self.dilation = stride, padding, dilation
        self.groups, self.deformable_groups = groups, deformable_groups
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.empty(out_channels)) if bias else None
        nn.init.kaiming_uniform_(self.weight, nonlinearity="relu")
        if bias:
            nn.init.constant_(self.bias, 0.0)

    def forward(self, x, offset, mask):
        return modulated_deform_conv(
            x, offset, mask, self.weight, self.bias, self.stride, self.padding,
            self.dilation, self.groups, self.deformable_groups)


if __name__ == "__main__":
    torch.manual_seed(0)
    N, C, Cout, G, k = 2, 7, 64, 7, 3
    H = W = 40
    m = ModulatedDeformConv(C, Cout, k, padding=1, deformable_groups=G)
    x = torch.randn(N, C, H, W)
    off = torch.zeros(N, 2 * G * k * k, H, W)
    msk = torch.ones(N, G * k * k, H, W)
    yd = m(x, off, msk)
    yc = F.conv2d(x, m.weight, m.bias, padding=1)
    print("shape", tuple(yd.shape), "max abs diff vs conv2d =", float((yd - yc).abs().max()))
