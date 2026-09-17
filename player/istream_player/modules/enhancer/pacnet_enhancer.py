"""PAC-Net 可渐进增强器：单套权重三出口(e1/e2/e3)，帧级 halt 控制器根据
渲染缓冲区水位 + 段内容难度(per-seg gain)决定停在哪个出口。

注册模块名 "pacnet"，配置 mod_enhancer="pacnet" 即可启用。
忽略 NES 的 enhance_action，增强决策完全由 halt 控制器帧级接管；
NES 仍正常做码率选择(download_action)。
"""
import json
import logging
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from istream_player.config.config import PlayerConfig
from istream_player.core.module import Module, ModuleOption
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.modules.enhancer.enhancer_imdn_always import IMDNEnhancerAlways
from istream_player.modules.enhancer.pacnet_model import PACNet
from istream_player.modules.decoder import TensorConverter

from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.core.scheduler import Scheduler
from istream_player.core.renderer import Renderer
from istream_player.core.player import Player

logger = logging.getLogger("PACNetEnhancer")


@ModuleOption("pacnet", default=True, requires=["model_downloader", DownloadBufferImpl, EnhanceBufferImpl, Scheduler, Renderer, Player])
class PACNetEnhancer(IMDNEnhancerAlways):
    log = logger

    def __init__(self):
        super().__init__()
        self.pacnet = None
        self.pacnet_latency = [0.014, 0.026, 0.039]  # e1/e2/e3 每帧秒数，阶段0实测，measure后覆盖
        self.per_seg_gain = None  # 段索引 -> e3-e1 增益
        self._current_exit_idx = 2  # 默认最深
        self._halt_safe_factor = 1.5  # 预算充足判定安全因子（更保守，避免e2/e3积压）
        self._halt_gain_thresh = 1.5  # 内容难度阈值（gain>此值视为困难段）

    async def load_model(self):
        """只加载一个 PACNet 模型（多退化混合训练权重，cond=False，nf=32）。"""
        weight_candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
                           "workspace", "pacnet", "weights", "stage2_mix_nocond_x4_best.pth"),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
                           "workspace", "pacnet", "weights", "pacnet_x4_best.pth"),
        ]
        weight_path = None
        for p in weight_candidates:
            if os.path.exists(p):
                weight_path = p
                break
        if weight_path is None:
            raise FileNotFoundError(f"PACNet weight not found in {weight_candidates}")
        self.log.info(f"Loading PACNet from {weight_path}")
        model = PACNet(nf=32, upscale=4, exits=(2, 4, 6), cond=False)
        state = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state)
        model = model.to(self.device).eval()
        self.log.info(f"PACNet loaded, params={sum(p.numel() for p in model.parameters())/1e6:.2f}M")

        # 加载 per-seg gain
        gain_file = "per_seg_psnr_tos.json" if getattr(self, "video_dataset", "bbb") == "tos" else "per_seg_psnr.json"
        gain_candidates = [
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..",
                           "workspace", "pacnet", gain_file),
        ]
        for p in gain_candidates:
            if os.path.exists(p):
                data = json.load(open(p, encoding="utf-8"))
                self.per_seg_gain = {int(k): v.get("gain_e3_e1", 1.0) for k, v in data.items()}
                gains = list(self.per_seg_gain.values())
                self.log.info(f"per-seg gain loaded: {len(gains)} segs, "
                              f"mean={np.mean(gains):.3f} std={np.std(gains):.3f}")
                break
        if self.per_seg_gain is None:
            self.log.warning("per_seg_psnr.json not found, using default gain=1.0 for all segments")
            self.per_seg_gain = {}
        return model

    async def measure_latency(self, resolution_set, model_pool, file_name="pacnet_latency.json"):
        """测量 PACNet 三个出口在各分辨率下的延迟，返回 BONES 用的 5×5 表。
        latency_table[4-repr_idx, level] = 每帧延迟(秒)，level 1/2/3 对应 e1/e2/e3。"""
        if getattr(self, "video_dataset", "bbb") == "tos":
            file_name = "pacnet_latency_tos.json"
        if os.path.exists(file_name):
            saved = json.load(open(file_name))
            if isinstance(saved, dict) and "table" in saved:
                self.pacnet_latency = saved.get("per_exit", [0.014, 0.026, 0.039])
                self.latency_table = np.array(saved["table"])
                self.log.info(f"PACNet latency loaded from cache")
                return self.latency_table
        self.log.info("Measuring PACNet three-exit latency per resolution...")
        self.pacnet = model_pool
        latency_set = {}
        per_exit_by_res = {}
        for repr_idx in resolution_set:
            w, h, scale = resolution_set[repr_idx]
            latencies = []
            for exit_idx in range(3):
                data = [torch.rand((1, 3, h, w)).to(self.device) for _ in range(self.warmup_epoch + self.measure_epoch)]
                with torch.no_grad():
                    for i in range(self.warmup_epoch):
                        self.pacnet(data[i], exit_idx=exit_idx)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    for i in range(self.warmup_epoch, self.warmup_epoch + self.measure_epoch):
                        self.pacnet(data[i], exit_idx=exit_idx)
                    torch.cuda.synchronize()
                    elapsed = (time.perf_counter() - t0) / self.measure_epoch
                latencies.append(elapsed)
                latency_set[(repr_idx, exit_idx + 1)] = elapsed  # level 1/2/3 -> e1/e2/e3
            per_exit_by_res[repr_idx] = latencies
            self.log.info(f"  res {w}x{h}: e1={latencies[0]*1000:.1f} e2={latencies[1]*1000:.1f} e3={latencies[2]*1000:.1f} ms/frame")

        # 用最低分辨率的出口延迟作为全局 pacnet_latency（halt 控制器用）
        min_repr = min(resolution_set.keys())
        self.pacnet_latency = per_exit_by_res[min_repr]

        # 构建 5×5 表（与父类格式一致：latency_table[4-repr_idx, level]）
        latency_table = np.zeros((5, 5))
        for (repr_idx, level), lat in latency_set.items():
            latency_table[4 - repr_idx, level] = lat

        json.dump({"table": latency_table.tolist(), "per_exit": self.pacnet_latency},
                  open(file_name, "w"), indent=2)
        self.log.info("PACNet latency measurement completed")
        self.latency_table = latency_table
        return latency_table

    def _halt_decide(self, remaining_time, remaining_frames, segment_index):
        """帧级 halt 控制器：根据剩余时间预算 + 段内容难度决定出口。
        返回 exit_idx (0=e1, 1=e2, 2=e3)。
        策略：预算充足全 e3；预算紧张时内容感知（困难段深算、简单段浅算）；预算极少保命 e1。
        """
        if remaining_frames <= 0:
            return 0
        budget_per_frame = remaining_time / remaining_frames
        l1, l2, l3 = self.pacnet_latency
        gain = self.per_seg_gain.get(segment_index, 1.0)
        is_hard = gain >= self._halt_gain_thresh

        # 预算充足：全 e3（和 greedy 一样）
        if budget_per_frame >= l3 * self._halt_safe_factor:
            return 2
        # 预算够 e3 但不充足：困难段 e3，简单段 e2
        if budget_per_frame >= l3 * 1.2:
            return 2 if is_hard else 1
        # 预算够 e2（带安全因子）：困难段 e2，简单段 e1
        if budget_per_frame >= l2 * 1.3:
            return 1 if is_hard else 0
        # 预算极少：保命 e1
        return 0

    async def enhance_one_frame(self, surf, model, tensor_converter, exit_idx=None):
        """重写：用 PACNet 指定出口增强。exit_idx 由 run 循环里的 halt 控制器设置。"""
        if exit_idx is None:
            exit_idx = self._current_exit_idx
        tensor = tensor_converter.surface_to_tensor(surf)
        tensor = tensor.to(self.device)
        await asyncio_sleep()
        with torch.no_grad():
            tensor = model(tensor, exit_idx=exit_idx)
            await asyncio_sleep()
            tensor = F.interpolate(tensor, size=(self.display_H, self.display_W),
                                   mode="bicubic", align_corners=False)
            await asyncio_sleep()
        tensor = tensor.cpu()
        await asyncio_sleep()
        return tensor

    async def run(self):
        """重写主循环：在逐帧增强前用 halt 控制器决定 exit_idx。"""
        import asyncio
        async with self._accessible:
            await self._accessible.wait()

        while self._is_ready:
            index, segments = await self.enhance_buffer.dequeue()
            self._current_enhancing_index = index
            if self.enhance_buffer.is_empty():
                if self.scheduler.is_end:
                    self._is_ready = False
                    self.log.info("PACNet Enhancer closed")
                    return

            for listener in self.listeners:
                await listener.on_enhancement_start(segments)

            for as_idx in segments:
                segment = segments[as_idx]
                level = segment.enhance_action  # 忽略，PACNet halt 控制器接管
                self.enhance_action = level if level in (1,2,3) else 2
                abort = False
                abort_reason = ""

                enhance_start_to_play_time = (self.renderer.remain_task() +
                    (self._current_enhancing_index - self.download_buffer.get_next_render_segment_index()) * self.seg_time)

                repr_idx = segment.repr_id
                if repr_idx not in self.resolution_set:
                    abort = True
                    abort_reason = "Max resolution"
                if segment.url in self.played_urls:
                    abort = True
                    abort_reason = "Already played before enhance"
                if self.use_fast_complete and enhance_start_to_play_time < self._min_threshold.get(level, 0.3):
                    abort = True
                    abort_reason = "Remaining time less than min threshold"

                if abort:
                    self.abort_total += 1
                    segment.is_enhance = False
                    segment.abort_reason = abort_reason
                    segment.wasted_enhanced_cnt = 0
                    self.task_start = None
                    self._reset_enhancement_state(reset_counters=False)
                    break

                _, _, scale = self.resolution_set[repr_idx]
                segment.enhance_scale = scale
                decoder = self._make_decoder(segment)
                decode_W, decode_H = decoder.resolution()
                tensor_converter = TensorConverter(decode_W, decode_H, self.display_W, self.display_H, gpu_id=0)
                start_time = time.perf_counter()
                self.task_start = start_time
                self.task_total = self.pacnet_latency[1] * self._num_frames  # e2延迟×帧数，段级估计

                result = []
                self.cnt = 0
                self._enhanced_frames = 0
                fast_complete = False
                model = self.model_pool  # PACNet 单模型

                while True:
                    # fast_complete 提前检测（复用父类逻辑）
                    if self.use_fast_complete and not fast_complete:
                        if self.cnt > 0 and self.cnt % self._check_interval_frames == 0:
                            fast_complete = self.check_fast_complete()
                    if fast_complete:
                        from istream_player.modules.decoder.decoder_nvcodec import DecoderWrapper
                        remaining_frames = self._num_frames - self.cnt
                        # headless模式下不存decoder_wrapper，只标记剩余帧数
                        result.extend([None] * remaining_frames)
                        self.cnt = self._num_frames
                        break

                    surf = decoder.decode_one_frame()
                    await asyncio_sleep()
                    if surf is None:
                        break

                    # halt 控制器：根据剩余时间预算 + 段内容难度决定出口
                    elapsed = time.perf_counter() - start_time
                    remaining_time = enhance_start_to_play_time - elapsed
                    remaining_frames = max(self._num_frames - self.cnt, 1)
                    self._current_exit_idx = self._halt_decide(remaining_time, remaining_frames, index)

                    surf_enh = await self.enhance_one_frame(surf, model, tensor_converter)
                    self._enhanced_frames += 1
                    # headless模式下不存完整CPU tensor（渲染器不用，analyzer用离线VMAF），避免内存积压
                    result.append(None)
                    self.cnt += 1
                    await asyncio_sleep()

                    if self.check_abort and segment.url in self.played_urls:
                        self.abort_total += 1
                        abort_reason = "Already played in enhance"
                        abort = True
                        self.check_abort = False
                        break
                    else:
                        self.check_abort = False

                if abort:
                    segment.is_enhance = False
                    segment.abort_reason = abort_reason
                    segment.wasted_enhanced_cnt = self._enhanced_frames
                    self._reset_enhancement_state(reset_counters=False)
                    break

                segment.decode_data = result
                end_time = time.perf_counter()
                enhance_latency = end_time - start_time
                interpolated_cnt = self.cnt - self._enhanced_frames
                enhance_fps = self.cnt / enhance_latency if enhance_latency > 0 else None

                self.segment_stats[segment.url] = {
                    "enhance_start_to_play_time": enhance_start_to_play_time,
                    "enhance_latency": enhance_latency,
                    "enhanced_cnt": self._enhanced_frames,
                    "interpolated_cnt": interpolated_cnt,
                    "enhance_fps": enhance_fps,
                    "exit_used": self._current_exit_idx,
                    "is_enhance": True,
                }
                await self.download_buffer.replace(self._current_enhancing_index, {as_idx: segment})
                self.log.info(f"PACNet enhance seg {index}: lat={enhance_latency:.2f}s, "
                              f"enhanced={self._enhanced_frames}, fps={enhance_fps:.1f}, "
                              f"last_exit=e{self._current_exit_idx+1}, "
                              f"start_to_play={enhance_start_to_play_time:.2f}s")
                self._enhanced_frames = 0
                self._reset_enhancement_state(reset_counters=True)

    def _make_decoder(self, segment):
        from istream_player.modules.decoder import DecoderNvCodec
        return DecoderNvCodec(self.config, segment, resize=False)

    def get_quality_table(self):
        """返回 PAC-Net 三出口校准后的 VMAF 质量表（5x5）。
        用原 IMDN 表 no_enh 列作基准 + PAC-Net e1/e2/e3 相对增益校准。"""
        _player_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        _qname = "pacnet_tos_quality.json" if getattr(self, "video_dataset", "bbb") == "tos" else "pacnet_quality.json"
        qpath = os.path.join(_player_dir, _qname)
        if os.path.exists(qpath):
            table = np.array(json.load(open(qpath)))
            self.log.info(f"PACNet quality table loaded from {qpath}")
            return table
        self.log.warning("pacnet_quality.json not found, falling back to IMDN table")
        return super().get_quality_table()

    # 注：get_latency_table 复用父类实现（self.latency_table 已在 measure_latency 中设置为 PAC-Net 5x5 表）。


def asyncio_sleep():
    import asyncio
    return asyncio.sleep(0)
