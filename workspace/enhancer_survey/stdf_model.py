"""
STDF (AAAI'20) 适配：用 torchvision 原生 deform_conv2d 替换需编译的 DCNv2，
权重键名(weight/bias)与官方一致，可直接加载官方 ckp。网络定义同 net_stdf.py。
只处理 Y 通道、时序半径 3（7 帧）、同分辨率压缩复原（不做放大）。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dcn_nocompile import ModulatedDeformConv  # 免编译 DCNv2（offset=0 时等价 conv2d）


class STDF(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb, base_ks=3, deform_ks=3):
        super().__init__()
        self.nb, self.in_nc, self.deform_ks = nb, in_nc, deform_ks
        self.size_dk = deform_ks ** 2
        self.in_conv = nn.Sequential(nn.Conv2d(in_nc, nf, base_ks, padding=base_ks // 2), nn.ReLU(inplace=True))
        for i in range(1, nb):
            setattr(self, f"dn_conv{i}", nn.Sequential(
                nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2), nn.ReLU(inplace=True),
                nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2), nn.ReLU(inplace=True)))
            setattr(self, f"up_conv{i}", nn.Sequential(
                nn.Conv2d(2 * nf, nf, base_ks, padding=base_ks // 2), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1), nn.ReLU(inplace=True)))
        self.tr_conv = nn.Sequential(
            nn.Conv2d(nf, nf, base_ks, stride=2, padding=base_ks // 2), nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(nf, nf, 4, stride=2, padding=1), nn.ReLU(inplace=True))
        self.out_conv = nn.Sequential(nn.Conv2d(nf, nf, base_ks, padding=base_ks // 2), nn.ReLU(inplace=True))
        self.offset_mask = nn.Conv2d(nf, in_nc * 3 * self.size_dk, base_ks, padding=base_ks // 2)
        self.deform_conv = ModulatedDeformConv(
            in_nc, out_nc, deform_ks, padding=deform_ks // 2, deformable_groups=in_nc)

    def forward(self, inputs):
        nb, in_nc, n_off = self.nb, self.in_nc, self.size_dk
        out_lst = [self.in_conv(inputs)]
        for i in range(1, nb):
            out_lst.append(getattr(self, f"dn_conv{i}")(out_lst[i - 1]))
        out = self.tr_conv(out_lst[-1])
        for i in range(nb - 1, 0, -1):
            out = getattr(self, f"up_conv{i}")(torch.cat([out, out_lst[i]], 1))
        off_msk = self.offset_mask(self.out_conv(out))
        off = off_msk[:, :in_nc * 2 * n_off]
        msk = torch.sigmoid(off_msk[:, in_nc * 2 * n_off:])
        return F.relu(self.deform_conv(inputs, off, msk), inplace=True)


class PlainCNN(nn.Module):
    def __init__(self, in_nc=64, nf=48, nb=8, out_nc=3, base_ks=3):
        super().__init__()
        self.in_conv = nn.Sequential(nn.Conv2d(in_nc, nf, base_ks, padding=1), nn.ReLU(inplace=True))
        hid = []
        for _ in range(nb - 2):
            hid += [nn.Conv2d(nf, nf, base_ks, padding=1), nn.ReLU(inplace=True)]
        self.hid_conv = nn.Sequential(*hid)
        self.out_conv = nn.Conv2d(nf, out_nc, base_ks, padding=1)

    def forward(self, x):
        return self.out_conv(self.hid_conv(self.in_conv(x)))


class MFVQE(nn.Module):
    def __init__(self, radius=3):
        super().__init__()
        self.radius = radius
        self.input_len = 2 * radius + 1
        in_nc = 1
        self.ffnet = STDF(in_nc * self.input_len, 64, 32, 3, base_ks=3, deform_ks=3)
        self.qenet = PlainCNN(in_nc=64, out_nc=1, nf=48, nb=8, base_ks=3)
        self.in_nc = in_nc

    def forward(self, x):
        out = self.qenet(self.ffnet(x))
        frm_lst = [self.radius + c * self.input_len for c in range(self.in_nc)]
        out = out + x[:, frm_lst]
        return out


OPTS = {"radius": 3}
