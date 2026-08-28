"""
MultiTaskEnhancer: 多任务视频增强器

扩展 IMDNEnhancer，将 enhance_action (0-4) 重新解释为任务组合ID：
  0: none (无增强)
  1: sr (仅超分，NanoVSR/IMDN)
  2: sr_deblur (去模糊 + 超分，NAFNet → SR)
  3: deblur (仅去模糊，NAFNet → bicubic upscale)
  4: sr_deblur_fi (去模糊 + 超分 + 插帧，FI暂未实现，降级为sr_deblur)

核心改动：
  1. setup/start: 加载 NAFNet-Small 去模糊模型 (FP16)
  2. enhance_one_frame: 根据任务组合执行不同管线
  3. measure_latency: 测量多任务组合延迟
  4. run: 模型选择逻辑适配任务组合
"""
import asyncio
import logging
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import torch
import PyNvCodec as nvc

from istream_player.config.config import PlayerConfig
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


# NAFNet 路径
NAFNET_PATH = r'E:\dev\Code\Python\bones\enhance\NAFNet'
if NAFNET_PATH not in sys.path:
    sys.path.insert(0, NAFNET_PATH)


@ModuleOption("multitask", default=False,
              requires=["model_downloader", DownloadBufferImpl, EnhanceBufferImpl, Scheduler, Player])
class MultiTaskEnhancer(IMDNEnhancer):
    """多任务视频增强器：支持 SR + Deblur 组合"""

    log = logging.getLogger("MultiTaskEnhancer")

    # 任务组合定义
    TASK_NONE = 0
    TASK_SR = 1
    TASK_SR_DEBLUR = 2
    TASK_DEBLUR = 3
    TASK_SR_DEBLUR_FI = 4

    def __init__(self):
        super().__init__()
        # NAFNet 去模糊模型
        self.deblur_model = None
        self.deblur_device = None
        # 是否启用去模糊（可通过配置关闭）
        self.enable_deblur = True
        # FI 模型（暂未实现）
        self.fi_model = None
        self.enable_fi = False
        # 任务组合名称映射
        self.task_names = {0: 'none', 1: 'sr', 2: 'sr_deblur', 3: 'deblur', 4: 'sr_deblur_fi'}
        # 段级增强统计（供 Analyzer 使用）
        from typing import Dict, Any
        self.segment_stats: Dict[str, Dict[str, Any]] = {}
        # 安全因子（延迟表缩放）
        self.safe_factor = 1.0
        # 中止统计（供 Analyzer 使用）
        self.abort_total = 0
        # fast_complete 标志（供 Analyzer 使用）
        self.use_fast_complete = False
        # 仅去模糊的表示集合（scale=1，如1080p，不支持SR但支持去模糊）
        self.deblur_only_set: Dict[int, tuple] = {}

    async def setup(self, config: PlayerConfig, model_downloader: DownloadManager,
                    download_buffer: DownloadBufferImpl, enhance_buffer: EnhanceBufferImpl,
                    scheduler: Scheduler, player: Player, **kwargs):
        """初始化多任务增强器"""
        # 调用父类 setup
        await super().setup(config, model_downloader, download_buffer,
                            enhance_buffer, scheduler, player, **kwargs)
        # 读取多任务配置
        self.enable_deblur = getattr(config, 'mt_enable_deblur', True)
        self.enable_fi = getattr(config, 'mt_enable_fi', False)
        self.log.info(f"MultiTaskEnhancer setup: deblur={self.enable_deblur}, fi={self.enable_fi}")

    async def load_model(self):
        """加载模型池 + NAFNet 去模糊模型"""
        # 先加载父类的 SR 模型池
        model_pool = await super().load_model()
        self.log.info(f"SR model pool loaded: {len(model_pool)} models")

        # 加载 NAFNet-Small 去模糊模型
        if self.enable_deblur:
            try:
                from basicsr.models.archs.NAFNet_arch import NAFNet
                self.log.info("Loading NAFNet-Small deblur model (FP16)...")
                self.deblur_model = NAFNet(
                    img_channel=3, width=16, middle_blk_num=1,
                    enc_blk_nums=[1, 1, 1, 14], dec_blk_nums=[1, 1, 1, 1]
                )
                self.deblur_model = self.deblur_model.to(self.device).half().eval()
                deblur_params = sum(p.numel() for p in self.deblur_model.parameters())
                self.log.info(f"NAFNet-Small loaded: {deblur_params:,} params, "
                              f"GPU mem: {torch.cuda.memory_allocated()/1024/1024:.1f} MB")
            except Exception as e:
                self.log.error(f"Failed to load NAFNet: {e}, deblur disabled")
                self.enable_deblur = False
                self.deblur_model = None

        return model_pool

    def _needs_deblur(self, task_combo: int) -> bool:
        """判断任务组合是否需要去模糊"""
        return task_combo in (self.TASK_SR_DEBLUR, self.TASK_DEBLUR, self.TASK_SR_DEBLUR_FI)

    def _needs_sr(self, task_combo: int) -> bool:
        """判断任务组合是否需要超分"""
        return task_combo in (self.TASK_SR, self.TASK_SR_DEBLUR, self.TASK_SR_DEBLUR_FI)

    def get_resolution_set(self, adaptation_sets):
        """
        覆盖父类方法：除了scale>1的表示外，还将scale=1的表示加入deblur_only_set
        scale=1的表示（如1080p）不支持SR，但支持去模糊增强
        """
        resolution_set = {}
        self.deblur_only_set = {}
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
                elif self.enable_deblur:
                    # scale=1的表示加入仅去模糊集合
                    self.deblur_only_set[repr_idx] = (repr_obj.width, repr_obj.height, scale)
            break
        self.log.info(f"Resolution set (SR-capable): {len(resolution_set)} reprs")
        self.log.info(f"Deblur-only set (scale=1): {len(self.deblur_only_set)} reprs")
        return resolution_set

    async def enhance_one_frame(self, surf: nvc.Surface, model: Optional[torch.nn.Module],
                                 tensor_converter: TensorConverter, task_combo: int = TASK_SR) -> torch.Tensor:
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
        await asyncio.sleep(0)

        with torch.no_grad():
            # 步骤1: 去模糊（在输入分辨率下）
            if self._needs_deblur(task_combo) and self.deblur_model is not None:
                tensor = self.deblur_model(tensor.half()).float()
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
                # 仅去模糊：双三次插值到显示分辨率
                tensor = torch.nn.functional.interpolate(
                    tensor, size=(self.display_H, self.display_W),
                    mode='bicubic', align_corners=False
                )

            await asyncio.sleep(0)

        # 移回 CPU 节省显存
        tensor = tensor.cpu()
        await asyncio.sleep(0)
        return tensor

    @critical_task()
    async def run(self):
        """主增强循环（重写以支持多任务）"""
        async with self._accessible:
            await self._accessible.wait()

        while self._is_ready:
            index, segments = await self.enhance_buffer.dequeue()

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

                start_time = time.time()
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
                elif repr_idx in self.deblur_only_set:
                    # scale=1的表示（如1080p）：仅允许去模糊（combo 3）
                    if self._needs_sr(task_combo):
                        self.log.debug(f"Seg {index}: repr {repr_idx} is deblur-only, abort SR task {task_combo}")
                        abort = True
                else:
                    # 不支持任何增强
                    abort = True
                if segment.url in self.played_urls:
                    abort = True
                if abort:
                    self.log.info(f"Abort seg {index}: task={task_combo}")
                    break

                # 获取scale（从resolution_set或deblur_only_set）
                if repr_idx in self.resolution_set:
                    _, _, scale = self.resolution_set[repr_idx]
                else:
                    _, _, scale = self.deblur_only_set[repr_idx]  # scale=1

                # 创建解码器
                decoder = DecoderNvCodec(self.config, segment, resize=False)
                decode_W, decode_H = decoder.resolution()
                display_W, display_H = self.config.display_width, self.config.display_height
                tensor_converter = TensorConverter(decode_W, decode_H, display_W, display_H, gpu_id=0)

                # 选择 SR 模型（仅在需要SR时）
                sr_model = None
                if self._needs_sr(task_combo):
                    sr_model = self.model_pool[(scale, 1)]  # 使用级别1的SR模型

                result: List[torch.Tensor] = []
                cnt = 0

                while True:
                    surf = decoder.decode_one_frame()
                    await asyncio.sleep(0)
                    if surf is None:
                        break

                    surf_enh = await self.enhance_one_frame(surf, sr_model, tensor_converter, task_combo)
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
                    self.log.info(f"Abort seg {index} during processing")
                    break

                segment.decode_data = result
                await self.download_buffer.replace(index, {as_idx: segment})

                end_time = time.time()
                # 更新延迟表
                self.latency_table[segment.download_action, task_combo] = \
                    (end_time - start_time) / (self.seg_time * self.frame_rate)

                self.log.info(f"Complete seg {index}: task={self.task_names.get(task_combo)}, "
                              f"latency={end_time-start_time:.3f}s, frames={cnt}, "
                              f"FPS={cnt/(end_time-start_time):.1f}")

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
        """测量多任务组合延迟"""
        # 先测量父类的 SR 延迟
        latency_table = await super().measure_latency(resolution_set, model_pool)

        # 测量去模糊相关组合的延迟
        if self.enable_deblur and self.deblur_model is not None:
            self.log.info("Measuring multi-task latency (deblur combinations)...")
            for repr_idx in resolution_set:
                (w, h, scale) = resolution_set[repr_idx]
                # 测量 NAFNet 在该分辨率下的延迟
                dummy = torch.randn(1, 3, h, w, device=self.device).half()
                with torch.no_grad():
                    for _ in range(self.warmup_epoch):
                        _ = self.deblur_model(dummy)
                    torch.cuda.synchronize()
                    lats = []
                    for _ in range(self.measure_epoch):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        _ = self.deblur_model(dummy)
                        torch.cuda.synchronize()
                        lats.append((time.perf_counter() - t0) * 1000)
                deblur_lat = np.mean(lats)

                # SR 延迟（从父类延迟表获取级别1）
                sr_lat = latency_table[repr_idx, 1] if repr_idx < latency_table.shape[0] else 10.0

                # 更新各任务组合延迟
                # combo 2 (sr_deblur): deblur + sr
                latency_table[repr_idx, self.TASK_SR_DEBLUR] = deblur_lat + sr_lat
                # combo 3 (deblur): deblur only
                latency_table[repr_idx, self.TASK_DEBLUR] = deblur_lat
                # combo 4 (sr_deblur_fi): deblur + sr + fi (FI暂用0)
                latency_table[repr_idx, self.TASK_SR_DEBLUR_FI] = deblur_lat + sr_lat

                self.log.info(f"  repr {repr_idx} ({w}x{h}): deblur={deblur_lat:.1f}ms, "
                              f"sr={sr_lat:.1f}ms, sr_deblur={deblur_lat+sr_lat:.1f}ms")

            # 测量 deblur_only_set（scale=1，如1080p）的去模糊延迟
            for repr_idx in self.deblur_only_set:
                (w, h, scale) = self.deblur_only_set[repr_idx]
                dummy = torch.randn(1, 3, h, w, device=self.device).half()
                with torch.no_grad():
                    for _ in range(self.warmup_epoch):
                        _ = self.deblur_model(dummy)
                    torch.cuda.synchronize()
                    lats = []
                    for _ in range(self.measure_epoch):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        _ = self.deblur_model(dummy)
                        torch.cuda.synchronize()
                        lats.append((time.perf_counter() - t0) * 1000)
                deblur_lat = np.mean(lats)
                # scale=1只支持deblur-only（combo 3），其他组合设为极大值防止控制器选择
                INVALID_LAT = 1e6
                latency_table[repr_idx, self.TASK_SR] = INVALID_LAT
                latency_table[repr_idx, self.TASK_SR_DEBLUR] = INVALID_LAT
                latency_table[repr_idx, self.TASK_SR_DEBLUR_FI] = INVALID_LAT
                latency_table[repr_idx, self.TASK_DEBLUR] = deblur_lat
                self.log.info(f"  repr {repr_idx} ({w}x{h}, deblur-only): deblur={deblur_lat:.1f}ms")

        return latency_table
