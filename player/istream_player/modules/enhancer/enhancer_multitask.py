# -*- coding: utf-8 -*-
"""
MultiTaskEnhancer（探索线 / multi_enhance D1）

多类型工具：SR 档 + sr_x2_med + restore_sharp。
继承 IMDNEnhancerAlways，复用 recovery / latency_adapt / segment_stats；
**不**把业务逻辑写进 Prism 默认的 imdn 路径。

mod_enhancer=multitask
兼容别名：imdn_tools（见 enhancer_tools.py）

旧版 USM-only 五动作实现见 enhancer_multitask_legacy.py（multitask_legacy）。
"""
from __future__ import annotations

import asyncio
import logging
import os

import numpy as np
import torch
import PyNvCodec as nvc

from istream_player.config.config import PlayerConfig
from istream_player.core.module import ModuleOption
from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.core.scheduler import Scheduler
from istream_player.core.renderer import Renderer
from istream_player.core.player import Player
from istream_player.core.downloader import DownloadManager
from istream_player.modules.enhancer.compare_view import keep_input
from istream_player.modules.decoder import TensorConverter

from .enhancer_imdn_always import IMDNEnhancerAlways
from .imdn_model import IMDN_RTC
from .tool_ids import (
    LEGAL_BY_DOWNLOAD,
    N_TOOLS,
    TOOL_NONE,
    TOOL_RESTORE_SHARP,
    TOOL_SR_HIGH,
    TOOL_SR_LOW,
    TOOL_SR_MED,
    TOOL_SR_X2_MED,
    is_native_res_tool,
    tool_name,
)


_REQUIRES = [
    "model_downloader",
    DownloadBufferImpl,
    EnhanceBufferImpl,
    Scheduler,
    Renderer,
    Player,
]


@ModuleOption("multitask", default=False, requires=_REQUIRES)
class MultiTaskEnhancer(IMDNEnhancerAlways):
    log = logging.getLogger("MultiTaskEnhancer")

    async def setup(
        self,
        config: PlayerConfig,
        model_downloader: DownloadManager,
        download_buffer: DownloadBufferImpl,
        enhance_buffer: EnhanceBufferImpl,
        scheduler: Scheduler,
        renderer: Renderer,
        player: Player,
        **kwargs,
    ):
        await super().setup(
            config,
            model_downloader,
            download_buffer,
            enhance_buffer,
            scheduler,
            renderer,
            player,
            **kwargs,
        )
        self.restore_usm_amount = float(getattr(config, "restore_usm_amount", 0.8))
        self.restore_usm_radius = int(getattr(config, "restore_usm_radius", 3))
        self._current_tool_mode = "sr"

    async def load_model(self):
        model_pool = await super().load_model()
        weight = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "imdn_weight",
            "IMDN_medium_bbb_x2.pth",
        )
        model = IMDN_RTC(upscale=2)
        model.load_state_dict(torch.load(weight, map_location="cpu"))
        model_pool[("x2", 2)] = model.to(self.device).eval()
        self.log.info(f"Loaded sr_x2_med from {weight}")
        return model_pool

    def _allows_enhance_for_repr(self, repr_idx, level) -> bool:
        if repr_idx in self.resolution_set:
            return True
        return is_native_res_tool(int(level))

    def _enhance_scale_for_repr(self, repr_idx, level) -> int:
        if repr_idx in self.resolution_set:
            return int(self.resolution_set[repr_idx][2])
        return 1

    def _select_enhance_model(self, scale, level):
        level = int(level)
        self._current_tool_mode = "sr"
        if level == TOOL_RESTORE_SHARP:
            self._current_tool_mode = "restore"
            return None
        if level == TOOL_SR_X2_MED:
            self._current_tool_mode = "sr_x2"
            model = self.model_pool.get(("x2", 2))
            if model is None:
                raise KeyError("sr_x2_med model ('x2', 2) not loaded")
            return model
        if level in (TOOL_SR_LOW, TOOL_SR_MED, TOOL_SR_HIGH):
            return self.model_pool[(scale, level)]
        raise KeyError(f"unsupported multitask tool level={level} ({tool_name(level)})")

    def _apply_unsharp(self, tensor: torch.Tensor) -> torch.Tensor:
        amount = float(self.restore_usm_amount)
        radius = int(self.restore_usm_radius)
        channels = tensor.shape[1]
        k = radius * 2 + 1
        sigma = max(radius / 2.0, 0.5)
        coords = torch.arange(k, device=tensor.device, dtype=tensor.dtype) - (k // 2)
        g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
        g = g / g.sum()
        kernel_2d = (g[:, None] * g[None, :]).expand(channels, 1, k, k).contiguous()
        pad = k // 2
        blurred = torch.nn.functional.conv2d(tensor, kernel_2d, padding=pad, groups=channels)
        return (tensor + amount * (tensor - blurred)).clamp(0.0, 1.0)

    async def enhance_one_frame(
        self,
        surf: nvc.Surface,
        model: torch.nn.Module,
        tensor_converter: TensorConverter,
        compare_segment=None,
    ) -> torch.Tensor:
        tensor = tensor_converter.surface_to_tensor(surf)
        tensor = tensor.to(self.device)
        keep_input(compare_segment, tensor)
        await asyncio.sleep(0)
        mode = getattr(self, "_current_tool_mode", "sr")
        with torch.no_grad():
            if mode == "restore":
                tensor = torch.nn.functional.interpolate(
                    tensor,
                    size=(self.display_H, self.display_W),
                    mode="bicubic",
                    align_corners=False,
                )
                await asyncio.sleep(0)
                tensor = self._apply_unsharp(tensor)
            else:
                if model is None:
                    raise RuntimeError(f"enhance_one_frame: model is None for mode={mode}")
                from ._gpuburn import burn as _burn
                from ._gpuburn import enabled as _burn_enabled

                if _burn_enabled():
                    import time as _time

                    torch.cuda.synchronize()
                    _t0 = _time.perf_counter()
                    tensor = model(tensor)
                    torch.cuda.synchronize()
                    _burn((_time.perf_counter() - _t0) * 1000.0)
                else:
                    tensor = model(tensor)
                await asyncio.sleep(0)
                tensor = torch.nn.functional.interpolate(
                    tensor,
                    size=(self.display_H, self.display_W),
                    mode="bicubic",
                    align_corners=False,
                )
            await asyncio.sleep(0)
            tensor = tensor.mul(255).clamp_(0, 255).to(dtype=torch.uint8)
        return tensor.cpu()

    def _multitask_table_path(self, basename: str) -> str:
        """Resolve table next to CWD (player/) or beside this package."""
        if os.path.exists(basename):
            return basename
        here = os.path.dirname(os.path.abspath(__file__))
        # player/istream_player/modules/enhancer → player/
        player_root = os.path.abspath(os.path.join(here, "..", "..", "..", ".."))
        cand = os.path.join(player_root, basename)
        if os.path.exists(cand):
            return cand
        return basename

    async def measure_latency(
        self, resolution_set, model_pool, file_name="multitask_latency.json"
    ):
        """Load D2 measured 5×6 table; never invent restore/x2 latency."""
        if getattr(self, "video_dataset", "bbb") == "tos":
            file_name = "multitask_latency_tos.json"
        path = self._multitask_table_path(file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"multitask latency prior missing: {path}. "
                "Run: python workspace/multi_enhance/build_priors.py"
            )
        import json

        wide = np.asarray(json.load(open(path, encoding="utf-8")), dtype=float)
        if wide.ndim != 2 or wide.shape[1] < N_TOOLS:
            raise ValueError(f"bad multitask latency shape {wide.shape}, need (*,{N_TOOLS})")
        # Mask illegal (download, tool) even if file has leftover numbers.
        for r in range(wide.shape[0]):
            legal = LEGAL_BY_DOWNLOAD.get(r, {TOOL_NONE})
            for t in range(N_TOOLS):
                if t not in legal:
                    wide[r, t] = 1e6
            wide[r, TOOL_NONE] = 0.0
        # 1080 restore: T may be measured but Q invalid → keep T for completeness;
        # policy still blocked by −inf Q.
        self.latency_table = wide
        self.log.info(f"Loaded multitask latency prior from {path} shape={wide.shape}")
        return wide

    def get_quality_table(self):
        """Load D2 measured 5×6 VMAF prior (Phase1 + H0); no hand-filled deltas."""
        if self.quality_table is not None and np.asarray(self.quality_table).shape[-1] >= N_TOOLS:
            return self.quality_table

        import json

        file_name = "multitask_bbb_quality.json"
        if getattr(self, "video_dataset", "bbb") == "tos":
            # TOS priors not built yet — refuse silent fallback to estimates.
            file_name = "multitask_tos_quality.json"
        path = self._multitask_table_path(file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"multitask quality prior missing: {path}. "
                "Run: python workspace/multi_enhance/build_priors.py"
            )

        wide = np.asarray(json.load(open(path, encoding="utf-8")), dtype=float)
        if wide.ndim != 2 or wide.shape[1] < N_TOOLS:
            raise ValueError(f"bad multitask quality shape {wide.shape}, need (*,{N_TOOLS})")
        wide = wide.astype(float)
        wide[wide < 0] = -np.inf
        for r in range(wide.shape[0]):
            legal = LEGAL_BY_DOWNLOAD.get(r, {TOOL_NONE})
            for t in range(N_TOOLS):
                if t not in legal:
                    wide[r, t] = -np.inf
        if getattr(self, "enhance", True) is False:
            self.old_quality = wide.copy()
            wide = wide.copy()
            wide[:, 1:] = -np.inf
        self.quality_table = wide
        self.log.info(f"Loaded multitask quality prior from {path} shape={wide.shape}")
        return wide
