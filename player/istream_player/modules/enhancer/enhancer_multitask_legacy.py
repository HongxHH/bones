"""
MultiTaskEnhancer: 多任务视频增强器（v2：多级别SR + USM-only）

将 enhance_action (0-4) 解释为：
  0: none (无增强)
  1: sr_light (轻量SR，IMDN_low，级别1)
  2: sr_medium (中等SR，IMDN_medium，级别2)
  3: sr_heavy (重度SR，IMDN_high，级别3)
  4: usm_only (仅锐化，Unsharp Mask，无SR，适用于scale=1和低纹理段)

核心改动：
  1. 恢复BONES的多级别SR选择（轻量/中等/重度），这是BONES的核心优势
  2. 保留USM-only作为第5个选项，用于scale=1表示和低纹理段
  3. 移除SR+USM组合（增益不确定， rarely selected）和FI（未实现）
"""
import asyncio
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import PyNvCodec as nvc

from istream_player.config.config import PlayerConfig
from istream_player.modules.enhancer.compare_view import drop_segment, keep_input, keep_same, start_segment
from istream_player.core.module import Module, ModuleOption
from istream_player.core.enhancer import Enhancer, EnhancerEventListener
from istream_player.core.scheduler import Scheduler
from istream_player.models import State
from istream_player.utils.async_utils import critical_task

from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.modules.decoder import DecoderNvCodec, TensorConverter
from istream_player.core.player import PlayerEventListener, Player
from istream_player.models.mpd_objects import Segment
from istream_player.core.downloader import DownloadManager

from .enhancer_imdn import IMDNEnhancer
from .imdn_model import IMDN, IMDN_RTC


@ModuleOption("multitask_legacy", default=False,
              requires=["model_downloader", DownloadBufferImpl, EnhanceBufferImpl, Scheduler, Player])
class MultiTaskEnhancerLegacy(IMDNEnhancer):
    """旧版多任务增强器（SR 档 + usm_only）。探索线请用 multitask（Always 子类）。"""

    log = logging.getLogger("MultiTaskEnhancerLegacy")

    # 任务组合定义（v2：多级别SR + USM-only）
    TASK_NONE = 0
    TASK_SR_LIGHT = 1    # IMDN_low, 级别1
    TASK_SR_MEDIUM = 2   # IMDN_medium, 级别2
    TASK_SR_HEAVY = 3    # IMDN_high, 级别3
    TASK_USM_ONLY = 4    # 仅USM锐化，无SR

    def __init__(self):
        super().__init__()
        # USM 锐化参数
        self.usm_amount = 1.5
        self.usm_radius = 3
        # 是否启用锐化（可通过配置关闭）
        self.enable_sharpen = True
        # FI 模型（暂未实现）
        self.fi_model = None
        self.enable_fi = False
        # 任务组合名称映射
        self.task_names = {0: 'none', 1: 'sr_light', 2: 'sr_medium', 3: 'sr_heavy', 4: 'usm_only'}
        # 段级增强统计（供 Analyzer 使用）
        from typing import Dict, Any
        self.segment_stats: Dict[str, Dict[str, Any]] = {}
        # 安全因子（延迟表缩放）
        self.safe_factor = 1.0
        # 中止统计（供 Analyzer 使用）
        self.abort_total = 0
        # fast_complete 标志（供 Analyzer 使用）
        self.use_fast_complete = False
        # fast_complete 动态阈值（秒）
        self._fast_complete_threshold = 2.0
        # fast_complete 检查间隔（帧数）
        self._fast_complete_check_interval = 15
        # 当前正在增强的段索引（用于fast_complete判断）
        self._current_enhancing_index: Optional[int] = None
        # 按任务组合的最小阈值（秒）
        self._min_threshold_by_task = {0: 0.1, 1: 0.15, 2: 0.25, 3: 0.35, 4: 0.12}
        # 仅锐化的表示集合（scale=1，如1080p，不支持SR但支持锐化）
        self.sharpen_only_set: Dict[int, tuple] = {}

    async def setup(self, config: PlayerConfig, model_downloader: DownloadManager,
                    download_buffer: DownloadBufferImpl, enhance_buffer: EnhanceBufferImpl,
                    scheduler: Scheduler, player: Player, **kwargs):
        """初始化多任务增强器"""
        # 调用父类 setup
        await super().setup(config, model_downloader, download_buffer,
                            enhance_buffer, scheduler, player, **kwargs)
        # 读取多任务配置
        self.enable_sharpen = getattr(config, 'mt_enable_sharpen', True)
        self.usm_amount = getattr(config, 'mt_usm_amount', 1.5)
        self.usm_radius = getattr(config, 'mt_usm_radius', 3)
        self.enable_fi = getattr(config, 'mt_enable_fi', False)
        # 读取 fast_complete 配置
        self.use_fast_complete = getattr(config, 'use_fast_complete', False)
        self.log.info(f"MultiTaskEnhancer setup: sharpen={self.enable_sharpen}, "
                      f"usm_amount={self.usm_amount}, usm_radius={self.usm_radius}, fi={self.enable_fi}, "
                      f"fast_complete={self.use_fast_complete}")

    async def load_model(self):
        """加载模型池：确保所有3个SR级别都加载（不受content_aware标志限制）+ USM配置"""
        # 先加载父类的 SR 模型池
        model_pool = await super().load_model()
        self.log.info(f"SR model pool loaded: {len(model_pool)} models")

        # 确保level 3（IMDN_high）已加载（父类在content_aware=True时不加载level 3）
        missing_level3 = [(s, l) for (s, l) in [(2,3),(3,3),(4,3)] if (s, l) not in model_pool]
        if missing_level3:
            self.log.info(f"Loading missing SR level 3 models: {missing_level3}")
            from .imdn_model import IMDN
            file_name = "imdn_path.json"
            import json as _json
            model_path_cfg = _json.load(open(file_name))
            # 使用agnostic路径（DIV2K预训练），与父类非内容感知模式一致
            path_key = "agnostic" if not self.content_aware else "aware"
            model_paths = model_path_cfg[path_key]
            ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
            for scale, level in missing_level3:
                url = os.path.join(ABSOLUTE_PATH, model_paths[f"scale{scale}_level{level}"])
                model = IMDN(upscale=scale, nf=32)  # high
                model.load_state_dict(torch.load(url))
                model_pool[(scale, level)] = model.to(self.device)
                self.log.info(f"  Loaded IMDN_high scale={scale} from {url}")

        # USM 锐化无需加载模型，仅记录参数
        if self.enable_sharpen:
            self.log.info(f"USM sharpen enabled: amount={self.usm_amount}, radius={self.usm_radius} "
                          f"(PyTorch GPU implementation, no pretrained weights needed)")

        return model_pool

    def _needs_sharpen(self, task_combo: int) -> bool:
        """判断任务组合是否需要锐化（仅USM-only）"""
        return task_combo == self.TASK_USM_ONLY

    def _needs_sr(self, task_combo: int) -> bool:
        """判断任务组合是否需要超分（SR_light/medium/heavy）"""
        return task_combo in (self.TASK_SR_LIGHT, self.TASK_SR_MEDIUM, self.TASK_SR_HEAVY)

    def _sr_level(self, task_combo: int) -> int:
        """获取SR任务对应的模型级别（1/2/3），非SR任务返回0"""
        if task_combo == self.TASK_SR_LIGHT:
            return 1
        elif task_combo == self.TASK_SR_MEDIUM:
            return 2
        elif task_combo == self.TASK_SR_HEAVY:
            return 3
        return 0

    def get_resolution_set(self, adaptation_sets):
        """
        覆盖父类方法：除了scale>1的表示外，还将scale=1的表示加入sharpen_only_set
        scale=1的表示（如1080p）不支持SR，但支持锐化增强
        """
        resolution_set = {}
        self.sharpen_only_set = {}
        for as_idx in adaptation_sets:
            as_obj = adaptation_sets[as_idx]
            if as_obj.content_type != "video":
                continue
            for repr_idx in as_obj.representations:
                repr_obj = as_obj.representations[repr_idx]
                scale = min(int(self.display_W / repr_obj.width), int(self.display_H / repr_obj.height))
                if scale < 1:
                    scale = 1
                if scale != 1:
                    resolution_set[repr_idx] = (repr_obj.width, repr_obj.height, scale)
                elif self.enable_sharpen:
                    # scale=1的表示加入仅锐化集合
                    self.sharpen_only_set[repr_idx] = (repr_obj.width, repr_obj.height, scale)
            break
        self.log.info(f"Resolution set (SR-capable): {len(resolution_set)} reprs")
        self.log.info(f"Sharpen-only set (scale=1): {len(self.sharpen_only_set)} reprs")
        return resolution_set

    def _gaussian_kernel_2d(self, kernel_size: int, sigma: float, channels: int = 3) -> torch.Tensor:
        """构建2D高斯核，返回 [channels, 1, k, k] 用于depthwise conv2d"""
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)  # [k, k]
        kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, k, k]
        kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)  # [channels, 1, k, k]
        return kernel_2d.to(self.device)

    def _unsharp_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Unsharp Mask 锐化（手动Gaussian blur实现，兼容旧版PyTorch）

        Args:
            tensor: [B, C, H, W] GPU张量，值域[0,1]
        Returns:
            锐化后的张量，值域[0,1]
        """
        kernel_size = self.usm_radius * 2 + 1
        sigma = self.usm_radius / 2.0
        channels = tensor.shape[1]
        padding = kernel_size // 2

        # 构建高斯核（缓存）
        if not hasattr(self, '_usm_kernel') or self._usm_kernel.shape[0] != channels:
            self._usm_kernel = self._gaussian_kernel_2d(kernel_size, sigma, channels)

        # depthwise convolution 实现高斯模糊
        blurred = torch.nn.functional.conv2d(
            tensor, self._usm_kernel, padding=padding, groups=channels
        )
        sharpened = tensor + self.usm_amount * (tensor - blurred)
        return torch.clamp(sharpened, 0.0, 1.0)

    async def enhance_one_frame(self, surf: nvc.Surface, model: Optional[torch.nn.Module],
                                 tensor_converter: TensorConverter, task_combo: int = TASK_SR_LIGHT,
                                 compare_segment=None) -> torch.Tensor:
        """
        增强单帧，根据任务组合执行不同管线

        Args:
            surf: NVIDIA Surface (解码后的帧)
            model: SR 模型 (None 表示不需要SR)
            tensor_converter: 张量转换器
            task_combo: 任务组合ID
        """
        # 将 Surface 转换为 PyTorch 张量 (GPU)
        tensor = tensor_converter.surface_to_tensor(surf)
        tensor = tensor.to(self.device)
        keep_input(compare_segment, tensor)
        await asyncio.sleep(0)

        with torch.no_grad():
            # 步骤1: 锐化（在输入分辨率下，USM）
            if self._needs_sharpen(task_combo) and self.enable_sharpen:
                tensor = self._unsharp_mask(tensor)
                await asyncio.sleep(0)

            # 步骤2: 超分
            if self._needs_sr(task_combo) and model is not None:
                tensor = model(tensor)
                await asyncio.sleep(0)
                # 双三次插值到显示分辨率
                tensor = torch.nn.functional.interpolate(
                    tensor, size=(self.display_H, self.display_W),
                    mode='bicubic', align_corners=False
                )
            elif not self._needs_sr(task_combo):
                # 仅锐化：双三次插值到显示分辨率
                tensor = torch.nn.functional.interpolate(
                    tensor, size=(self.display_H, self.display_W),
                    mode='bicubic', align_corners=False
                )

            await asyncio.sleep(0)

        # 移回 CPU 节省显存
        tensor = tensor.cpu()
        await asyncio.sleep(0)
        return tensor

    def remain_task(self) -> float:
        """计算当前增强任务的剩余预期时间（秒）"""
        if self.task_start is None or self.task_total is None:
            return 0.0
        return max(self.task_total - (time.perf_counter() - self.task_start), 0.0)

    def check_fast_complete(self, index: int, task_combo: int) -> bool:
        """
        检查是否应该触发 fast_complete（基于距离播放的时间，而非仅判断是否是下一段）

        改进点：
        1. 计算当前段距离播放的实际时间（队列位置 × 段时长 + 当前段剩余播放时间）
        2. 使用任务组合相关的最小阈值
        3. 结合当前增强任务的剩余预期时间

        Returns:
            bool: 是否应该触发 fast_complete
        """
        if not self.use_fast_complete:
            return False

        try:
            next_index = self.download_buffer.get_next_render_segment_index()
        except Exception:
            return False

        # 当前段已经在播放或已播放过，不触发（会被abort逻辑处理）
        if index <= next_index:
            return False

        # 计算距离播放的时间：队列中前面的段数 × 段时长 + 当前渲染剩余时间
        queue_ahead = index - next_index - 1  # 当前段前面还有多少段
        try:
            render_remain = self.renderer.remain_task()
        except Exception:
            render_remain = 0.0

        time_to_play = render_remain + queue_ahead * self.seg_time

        # 动态阈值：max(任务最小阈值, 当前增强剩余预期时间)
        # 如果当前增强还需要很长时间，而距离播放很近，就应该触发
        min_threshold = self._min_threshold_by_task.get(task_combo, 0.2)
        remain_enhance_time = self.remain_task()
        threshold = max(min_threshold, remain_enhance_time * 0.5)

        if time_to_play <= threshold:
            self.log.info(
                f"Fast complete triggered: seg {index}, queue_ahead={queue_ahead}, "
                f"render_remain={render_remain:.2f}s, time_to_play={time_to_play:.2f}s, "
                f"threshold={threshold:.2f}s, task={self.task_names.get(task_combo, task_combo)}"
            )
            return True
        return False

    @critical_task()
    async def run(self):
        """主增强循环（重写以支持多任务）"""
        async with self._accessible:
            await self._accessible.wait()

        while self._is_ready:
            index, segments = await self.enhance_buffer.dequeue()
            if index is None or segments is None:
                self._is_ready = False
                self.log.info("MultiTaskEnhancer closed")
                return

            if self.enhance_buffer.is_empty():
                if self.scheduler.is_end:
                    self._is_ready = False
                    self.log.info("MultiTaskEnhancer closed")
                    return

            for listener in self.listeners:
                await listener.on_enhancement_start(segments)

            for as_idx in segments:
                segment = segments[as_idx]
                task_combo = segment.enhance_action  # 现在是任务组合ID
                abort = False

                self.log.info(f"Enhancing seg {index}: download={segment.download_action}, "
                              f"task={self.task_names.get(task_combo, task_combo)}")

                start_time = time.perf_counter()
                self.task_start = start_time
                self.task_total = self.get_latency_table()[segment.download_action, task_combo]

                # 无增强
                if task_combo == self.TASK_NONE:
                    abort = True
                repr_idx = segment.repr_id
                # 判断表示是否支持增强
                if repr_idx in self.resolution_set:
                    # 支持SR的表示：所有任务组合都允许
                    pass
                elif repr_idx in self.sharpen_only_set:
                    # scale=1的表示（如1080p）：仅允许锐化（combo 3）
                    if self._needs_sr(task_combo):
                        self.log.debug(f"Seg {index}: repr {repr_idx} is sharpen-only, abort SR task {task_combo}")
                        abort = True
                else:
                    # 不支持任何增强
                    abort = True
                if segment.url in self.played_urls:
                    abort = True
                if abort:
                    self.log.info(f"Abort seg {index}: task={task_combo}")
                    segment.is_enhance = False
                    segment.abort_reason = f"Aborted before processing (task={task_combo})"
                    segment.enhanced_cnt = 0
                    segment.wasted_enhanced_cnt = 0
                    segment.fast_complete = False
                    segment.fast_complete_threshold = self._fast_complete_threshold
                    segment.enhance_end_to_play_time = 0.0
                    segment.quality_drop_due_to_fast = 0.0
                    segment.quality_switches_within_segment = 0
                    self.segment_stats[segment.url] = {
                        "is_enhance": False,
                        "abort_reason": f"Aborted before processing (task={task_combo})",
                        "enhanced_cnt": 0,
                        "wasted_enhanced_cnt": 0,
                        "fast_complete_triggered": False,
                        "fast_complete_threshold": self._fast_complete_threshold,
                        "enhance_end_to_play_time": 0.0,
                        "enhance_frame_interval": 0,
                        "interpolated_cnt": 0,
                    }
                    break

                # 获取scale（从resolution_set或sharpen_only_set）
                if repr_idx in self.resolution_set:
                    _, _, scale = self.resolution_set[repr_idx]
                else:
                    _, _, scale = self.sharpen_only_set[repr_idx]  # scale=1

                # 创建解码器
                decoder = DecoderNvCodec(self.config, segment, resize=False)
                decode_W, decode_H = decoder.resolution()
                display_W, display_H = self.config.display_width, self.config.display_height
                tensor_converter = TensorConverter(decode_W, decode_H, display_W, display_H, gpu_id=0)

                # 选择 SR 模型（仅在需要SR时，根据任务组合选择级别）
                sr_model = None
                sr_level = self._sr_level(task_combo)
                if sr_level > 0:
                    sr_model = self.model_pool[(scale, sr_level)]

                result: List[torch.Tensor] = []
                start_segment(segment, self.config)
                cnt = 0
                fast_complete_triggered = False
                effective_task = task_combo  # fast_complete后可能降级为SR_light

                # 记录当前增强段索引（用于fast_complete判断）
                self._current_enhancing_index = index

                while True:
                    surf = decoder.decode_one_frame()
                    await asyncio.sleep(0)
                    if surf is None:
                        break

                    # fast_complete 检查（每N帧检查一次）
                    if (self.use_fast_complete and not fast_complete_triggered
                            and cnt > 0 and cnt % self._fast_complete_check_interval == 0):
                        if self.check_fast_complete(index, task_combo):
                            fast_complete_triggered = True
                            # 优雅降级：剩余帧保持基础质量（不增强）
                            # 注：曾尝试降级为SR_light，但增加了段间VMAF振荡（新质量层级），
                            # TASK_NONE更简单且振荡更低，故保留此方案
                            effective_task = self.TASK_NONE
                            self.log.info(f"Fast complete: switching task -> none for remaining frames")

                    # 根据 effective_task 决定是否需要SR模型
                    current_model = sr_model if self._needs_sr(effective_task) else None

                    if effective_task == self.TASK_NONE:
                        # 基础质量：直接clone surface（不做增强）
                        keep_same(segment)
                        surf_enh = surf.Clone(decoder.gpu_id)
                    else:
                        surf_enh = await self.enhance_one_frame(
                            surf, current_model, tensor_converter, effective_task, segment
                        )

                    result.append(surf_enh)
                    cnt += 1
                    await asyncio.sleep(0)

                    if self.check_abort and segment.url in self.played_urls:
                        abort = True
                        self.check_abort = False
                        break
                    else:
                        self.check_abort = False

                if abort:
                    drop_segment(segment)
                    self.log.info(f"Abort seg {index} during processing")
                    segment.is_enhance = False
                    segment.abort_reason = "Aborted during processing"
                    segment.enhanced_cnt = 0
                    segment.wasted_enhanced_cnt = cnt
                    segment.fast_complete = fast_complete_triggered
                    segment.fast_complete_threshold = self._fast_complete_threshold
                    segment.enhance_end_to_play_time = 0.0
                    segment.quality_drop_due_to_fast = 0.0
                    segment.quality_switches_within_segment = 0
                    self._current_enhancing_index = None
                    self.segment_stats[segment.url] = {
                        "is_enhance": False,
                        "abort_reason": "Aborted during processing",
                        "enhanced_cnt": 0,
                        "wasted_enhanced_cnt": cnt,
                        "fast_complete_triggered": fast_complete_triggered,
                        "fast_complete_threshold": self._fast_complete_threshold,
                        "enhance_end_to_play_time": 0.0,
                        "enhance_frame_interval": 0,
                        "interpolated_cnt": 0,
                    }
                    break

                segment.decode_data = result
                # 标记段已增强（关键：Analyzer依赖此属性计算增强VMAF）
                segment.is_enhance = True
                segment.enhanced_cnt = cnt
                segment.enhance_scale = scale
                segment.abort_reason = ""
                segment.wasted_enhanced_cnt = 0
                # fast_complete 相关属性（Analyzer依赖）
                segment.fast_complete = fast_complete_triggered
                segment.fast_complete_threshold = self._fast_complete_threshold
                segment.quality_drop_due_to_fast = 0.0
                segment.quality_switches_within_segment = 0
                # 计算增强结束到播放的时间
                try:
                    next_idx = self.download_buffer.get_next_render_segment_index()
                    render_remain = self.renderer.remain_task()
                    if index > next_idx:
                        segment.enhance_end_to_play_time = render_remain + (index - next_idx - 1) * self.seg_time
                    else:
                        segment.enhance_end_to_play_time = 0.0
                except Exception:
                    segment.enhance_end_to_play_time = 0.0

                await self.download_buffer.replace(index, {as_idx: segment})

                end_time = time.perf_counter()
                enhance_latency = end_time - start_time
                segment.enhance_latency = enhance_latency
                segment.enhance_fps = cnt / enhance_latency if enhance_latency > 0 else 0

                # 存储段级统计，供Analyzer按URL查询（必须包含Analyzer期望的所有键）
                self.segment_stats[segment.url] = {
                    "is_enhance": True,
                    "abort_reason": "",
                    "enhanced_cnt": cnt,
                    "wasted_enhanced_cnt": 0,
                    "enhance_latency": enhance_latency,
                    "enhance_scale": scale,
                    "enhance_fps": segment.enhance_fps,
                    "task_combo": task_combo,
                    "task_name": self.task_names.get(task_combo, str(task_combo)),
                    "fast_complete_triggered": fast_complete_triggered,
                    "fast_complete_threshold": self._fast_complete_threshold,
                    "enhance_start_to_play_time": 0.0,
                    "enhance_end_to_play_time": segment.enhance_end_to_play_time,
                    "enhance_frame_interval": 0,
                    "interpolated_cnt": 0,
                    "effective_task": self.task_names.get(effective_task, str(effective_task)),
                }

                # 重置当前增强段索引
                self._current_enhancing_index = None

                # 更新延迟表
                self.latency_table[segment.download_action, task_combo] = \
                    enhance_latency / (self.seg_time * self.frame_rate)

                fc_info = f", fast_complete={fast_complete_triggered}" if fast_complete_triggered else ""
                self.log.info(f"Complete seg {index}: task={self.task_names.get(task_combo)}, "
                              f"latency={enhance_latency:.3f}s, frames={cnt}, "
                              f"FPS={segment.enhance_fps:.1f}, scale={scale}{fc_info}")

    def get_quality_table(self):
        """
        构建多级别SR + USM-only质量表（v2）

        任务组合（列）：
          0: none (base VMAF)
          1: sr_light (IMDN_low, 级别1)
          2: sr_medium (IMDN_medium, 级别2)
          3: sr_heavy (IMDN_high, 级别3)
          4: usm_only (仅Unsharp Mask，无SR)

        码率级别（行）：
          0: 240p, 1: 360p, 2: 480p, 3: 720p(scale=1仅USM), 4: 1080p(scale=1)

        SR级别VMAF数据来自imdn_bbb_3060ti.json（159段平均，实测数据）
        """
        if self.quality_table is not None:
            return self.quality_table

        # 初始化5x5质量表
        quality_table = np.full((5, 5), -np.inf)

        # === 实测SR级别VMAF（来自imdn_bbb_3060ti.json，159段平均）===
        # Row 0: 240p
        quality_table[0, 0] = 39.30   # base
        quality_table[0, 1] = 48.53   # sr_light (L1, gain=+9.23)
        quality_table[0, 2] = 53.99   # sr_medium (L2, gain=+14.69)
        quality_table[0, 3] = 60.77   # sr_heavy (L3, gain=+21.47)
        quality_table[0, 4] = 44.00   # usm_only (base + 4.70, 估计值)

        # Row 1: 360p
        quality_table[1, 0] = 64.02   # base
        quality_table[1, 1] = 72.43   # sr_light (L1, gain=+8.41)
        quality_table[1, 2] = 75.87   # sr_medium (L2, gain=+11.85)
        quality_table[1, 3] = 79.89   # sr_heavy (L3, gain=+15.87)
        quality_table[1, 4] = 69.22   # usm_only (base + 5.20, 实测20段平均)

        # Row 2: 480p
        quality_table[2, 0] = 76.80   # base
        quality_table[2, 1] = 82.42   # sr_light (L1, gain=+5.62)
        quality_table[2, 2] = 85.24   # sr_medium (L2, gain=+8.44)
        quality_table[2, 3] = 87.16   # sr_heavy (L3, gain=+10.36)
        quality_table[2, 4] = 78.48   # usm_only (base + 1.68, 实测20段平均)

        # Row 3: 720p (scale=1, 仅支持USM)
        quality_table[3, 0] = 90.10   # base
        quality_table[3, 1] = -np.inf  # 无SR
        quality_table[3, 2] = -np.inf  # 无SR
        quality_table[3, 3] = -np.inf  # 无SR
        quality_table[3, 4] = 90.80    # usm_only (base + 0.70, 保守估计)

        # Row 4: 1080p (scale=1, 已达最高质量)
        quality_table[4, 0] = 100.0    # base
        quality_table[4, 1] = -np.inf
        quality_table[4, 2] = -np.inf
        quality_table[4, 3] = -np.inf
        quality_table[4, 4] = -np.inf   # USM无增益（已100分）

        self.quality_table = quality_table
        self.log.info(f"Multi-level SR quality table built: shape={quality_table.shape}")
        self.log.info(f"  SR gains (L1/L2/L3): "
                      f"240p=[+9.23/+14.69/+21.47], "
                      f"360p=[+8.41/+11.85/+15.87], "
                      f"480p=[+5.62/+8.44/+10.36]")
        self.log.info(f"  USM-only gains (实测20段): 240p=+4.70(估), 360p=+5.20, 480p=+1.68, 720p=+0.70(估)")
        return quality_table

    def get_latency_table(self, segment_index: int = None):
        """
        获取延迟表（兼容 CA-BONES 的 segment_index 参数）

        Args:
            segment_index: 段索引（暂未使用内容感知延迟预测，保留兼容）
        """
        try:
            return self.latency_table * self.time_factor * self.seg_time * self.frame_rate * self.safe_factor
        except Exception:
            return self.latency_table

    async def measure_latency(self, resolution_set, model_pool):
        """测量多级别SR + USM-only延迟（v2）

        父类measure_latency()已测量SR级别1-3的延迟，存储在列1-3。
        此处仅补充USM-only（列4）的延迟，不覆盖SR级别数据。
        """
        # 先测量父类的 SR 级别1-3延迟（列1-3）
        latency_table = await super().measure_latency(resolution_set, model_pool)

        # 测量USM-only延迟（列4）
        if self.enable_sharpen:
            self.log.info("Measuring USM-only latency (column 4)...")
            for repr_idx in resolution_set:
                (w, h, scale) = resolution_set[repr_idx]
                # 测量 USM 在该分辨率下的延迟（GPU）
                dummy = torch.randn(1, 3, h, w, device=self.device)
                with torch.no_grad():
                    for _ in range(self.warmup_epoch):
                        _ = self._unsharp_mask(dummy)
                    torch.cuda.synchronize()
                    lats = []
                    for _ in range(self.measure_epoch):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        _ = self._unsharp_mask(dummy)
                        torch.cuda.synchronize()
                        lats.append((time.perf_counter() - t0) * 1000)
                usm_lat = np.mean(lats)
                # USM-only = USM + bicubic upscale（bicubic开销很小，暂忽略）
                latency_table[repr_idx, self.TASK_USM_ONLY] = usm_lat
                self.log.info(f"  repr {repr_idx} ({w}x{h}): usm_only={usm_lat:.2f}ms")

            # 测量 sharpen_only_set（scale=1，如1080p）的USM延迟
            for repr_idx in self.sharpen_only_set:
                (w, h, scale) = self.sharpen_only_set[repr_idx]
                dummy = torch.randn(1, 3, h, w, device=self.device)
                with torch.no_grad():
                    for _ in range(self.warmup_epoch):
                        _ = self._unsharp_mask(dummy)
                    torch.cuda.synchronize()
                    lats = []
                    for _ in range(self.measure_epoch):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        _ = self._unsharp_mask(dummy)
                        torch.cuda.synchronize()
                        lats.append((time.perf_counter() - t0) * 1000)
                usm_lat = np.mean(lats)
                # scale=1只支持USM-only（列4），SR任务设为极大值防止选择
                INVALID_LAT = 1e6
                latency_table[repr_idx, self.TASK_SR_LIGHT] = INVALID_LAT
                latency_table[repr_idx, self.TASK_SR_MEDIUM] = INVALID_LAT
                latency_table[repr_idx, self.TASK_SR_HEAVY] = INVALID_LAT
                latency_table[repr_idx, self.TASK_USM_ONLY] = usm_lat
                self.log.info(f"  repr {repr_idx} ({w}x{h}, sharpen-only): usm_only={usm_lat:.2f}ms")

        return latency_table
