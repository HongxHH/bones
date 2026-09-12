"""PAC-Net 阶段0最小证伪模型。

结构：共享 IMDN 骨干（fea_conv + 6 个串联 IMDModule），在第 2/4/6 个 block 后各接一个
重建出口 ExitHead。出口越深算力越大；推理时可只跑到指定出口（anytime/可中断），到点必出
当前最优结果。另含压缩退化条件 FiLM（零初始化，开局恒等，不破坏 warm-start）。

对照模型 PlainIMDN：任意 block 数的标准 IMDN，用于构造与出口1同算力的"固定浅模型"，
检验多出口共享是否让保命档付出明显代价（弹性税）。
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, os.path.join(ROOT, "player"))
from istream_player.modules.enhancer import imdn_block as B  # noqa: E402


class CondEncoder(nn.Module):
    """从 LR 帧提取固定退化统计（无需 QP 标签、在线可得），生成 FiLM 的 gamma/beta 调制浅层特征。

    统计量：RGB 三通道均值(3) + 高频能量(1) + 下采样-上采样残差能量(1) = 5 维。
    最后一层零初始化 => 开局 gamma=0、beta=0，即恒等映射，不破坏 warm-start 权重。
    """

    def __init__(self, nf, cond_dim=5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, nf), nn.ReLU(inplace=True),
            nn.Linear(nf, nf * 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self._blur_k = None

    def stats(self, x):
        gray = x.mean(1, keepdim=True)
        mean = x.mean(dim=(2, 3))                                   # (N,3)
        if self._blur_k is None or self._blur_k.device != x.device:
            self._blur_k = torch.ones(1, 1, 5, 5, device=x.device) / 25.0
        blur = F.conv2d(F.pad(gray, (2, 2, 2, 2), mode="reflect"), self._blur_k)
        hf = (gray - blur).abs().mean(dim=(2, 3))                   # (N,1)
        h, w = gray.shape[-2:]
        small = F.interpolate(gray, size=(max(h // 4, 1), max(w // 4, 1)),
                              mode="bilinear", align_corners=False)
        back = F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)
        blur_energy = (gray - back).abs().mean(dim=(2, 3))         # (N,1)
        # 固定尺度归到相近量纲（不用 batch 维统计，保证单帧推理也稳定有效）
        z = torch.cat([mean, hf * 8.0, blur_energy * 8.0], dim=1)  # (N,5)
        return z

    def forward(self, x, fea):
        gb = self.mlp(self.stats(x))
        gamma, beta = gb.chunk(2, dim=1)
        gamma = (1.0 + gamma).unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return fea * gamma + beta


class ExitHead(nn.Module):
    """单个递进出口：1x1 融合 k*nf->nf，3x3 LR_conv + 浅层残差，pixelshuffle 重建。"""

    def __init__(self, k, nf=32, upscale=4):
        super().__init__()
        # 与标准 IMDN 的 c 对齐：1x1 融合后必须接 LeakyReLU（warm 权重即在该结构下训练）
        self.fuse = B.conv_block(nf * k, nf, kernel_size=1, act_type="lrelu")
        self.lr_conv = B.conv_layer(nf, nf, kernel_size=3)
        self.upsampler = B.pixelshuffle_block(nf, 3, upscale_factor=upscale)

    def forward(self, feats, fea0):
        x = self.fuse(torch.cat(feats, dim=1))
        x = self.lr_conv(x) + fea0
        return self.upsampler(x)


class PACNet(nn.Module):
    """共享 6-block 骨干 + exits 处递进出口 + 可选退化条件调制。"""

    def __init__(self, nf=32, upscale=4, exits=(2, 4, 6), cond=True):
        super().__init__()
        self.exits = tuple(exits)
        self.fea_conv = B.conv_layer(3, nf, kernel_size=3)
        self.blocks = nn.ModuleList([B.IMDModule(in_channels=nf) for _ in range(6)])
        self.heads = nn.ModuleList([ExitHead(k, nf, upscale) for k in self.exits])
        self.cond = CondEncoder(nf) if cond else None

    def forward(self, x, exit_idx=None):
        """exit_idx=None：训练，返回全部出口 list；exit_idx=0/1/2：只跑到该出口（真正省算力）。"""
        fea = self.fea_conv(x)
        if self.cond is not None:
            fea = self.cond(x, fea)
        bis, outs = [], []
        cur = fea
        for i, blk in enumerate(self.blocks):
            cur = blk(cur)
            bis.append(cur)
            if (i + 1) in self.exits:
                ei = self.exits.index(i + 1)
                if exit_idx is not None:
                    if ei == exit_idx:
                        return self.heads[ei](bis, fea)
                    # 非目标中间出口：继续向更深 block 计算
                else:
                    outs.append(self.heads[ei](bis, fea))
        return outs

    @torch.no_grad()
    def load_imdn_warm(self, imdn_sd, verbose=False):
        """把标准 IMDN(nf) 的 DIV2K 权重迁移到共享骨干与最深出口（e3），其余新初始化。"""
        own = self.state_dict()
        moved = []

        def cp(dst, src):
            if src in imdn_sd and tuple(imdn_sd[src].shape) == tuple(own[dst].shape):
                own[dst] = imdn_sd[src].clone(); moved.append(dst)

        cp("fea_conv.weight", "fea_conv.weight")
        cp("fea_conv.bias", "fea_conv.bias")
        for i in range(6):
            for key in imdn_sd:
                if key.startswith(f"IMDB{i+1}."):
                    dst = f"blocks.{i}." + key[len(f"IMDB{i+1}."):]
                    cp(dst, key)
        # 最深出口 e3（heads[-1]，k=6）对齐 IMDN 的 c / LR_conv / upsampler
        # conv_layer 是裸 Conv2d（键无 .conv）；conv_block 返回 Sequential（融合层键为 .0）
        h3 = "heads.%d." % (len(self.exits) - 1)
        cp(h3 + "fuse.0.weight", "c.0.weight")
        cp(h3 + "fuse.0.bias", "c.0.bias")
        cp(h3 + "lr_conv.weight", "LR_conv.weight")
        cp(h3 + "lr_conv.bias", "LR_conv.bias")
        cp(h3 + "upsampler.0.weight", "upsampler.0.weight")
        cp(h3 + "upsampler.0.bias", "upsampler.0.bias")
        self.load_state_dict(own)
        if verbose:
            print(f"warm-moved {len(moved)} tensors")
        return len(moved)


class PlainIMDN(nn.Module):
    """任意 block 数的标准 IMDN（播放器实现把 6 个 block 写死，这里参数化以便造 2-block 浅对照）。"""

    def __init__(self, nf=32, num_modules=2, upscale=4):
        super().__init__()
        self.m = num_modules
        self.fea_conv = B.conv_layer(3, nf, kernel_size=3)
        self.blocks = nn.ModuleList([B.IMDModule(in_channels=nf) for _ in range(num_modules)])
        self.c = B.conv_block(nf * num_modules, nf, kernel_size=1, act_type="lrelu")
        self.lr_conv = B.conv_layer(nf, nf, kernel_size=3)
        self.upsampler = B.pixelshuffle_block(nf, 3, upscale_factor=upscale)

    def forward(self, x):
        fea = self.fea_conv(x)
        bis = []
        cur = fea
        for blk in self.blocks:
            cur = blk(cur); bis.append(cur)
        out = self.c(torch.cat(bis, dim=1))
        out = self.lr_conv(out) + fea
        return self.upsampler(out)

    @torch.no_grad()
    def load_imdn_warm(self, imdn_sd):
        own = self.state_dict(); moved = []

        def cp(dst, src):
            if src in imdn_sd and tuple(imdn_sd[src].shape) == tuple(own[dst].shape):
                own[dst] = imdn_sd[src].clone(); moved.append(dst)

        cp("fea_conv.weight", "fea_conv.weight")
        cp("fea_conv.bias", "fea_conv.bias")
        for i in range(self.m):
            for key in imdn_sd:
                if key.startswith(f"IMDB{i+1}."):
                    dst = f"blocks.{i}." + key[len(f"IMDB{i+1}."):]
                    cp(dst, key)
        cp("c.0.weight", "c.0.weight")
        cp("c.0.bias", "c.0.bias")
        cp("lr_conv.weight", "LR_conv.weight")
        cp("lr_conv.bias", "LR_conv.bias")
        cp("upsampler.0.weight", "upsampler.0.weight")
        cp("upsampler.0.bias", "upsampler.0.bias")
        self.load_state_dict(own)
        return len(moved)
