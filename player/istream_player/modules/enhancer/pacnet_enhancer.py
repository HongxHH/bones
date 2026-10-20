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
from istream_player.modules.enhancer.compare_view import drop_segment, keep_input, pad_same, start_segment
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
        # 阶段二实验开关：环境变量控制，跑完不留痕
        self._force_exit = int(os.environ.get("PACNET_FORCE_EXIT", "-1"))  # -1=正常halt, 0/1/2=强制出口
        self._halt_e3_thresh = float(os.environ.get("PACNET_E3_THRESH", "1.0"))
        self._halt_e2_thresh = float(os.environ.get("PACNET_E2_THRESH", "1.0"))

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
                # 阶段二：把 gpuburn 额外耗时计入 halt 用的端到端延迟，保证预算判断准确
                # 乘法模式 K：整体放大 k 倍（保持出口延迟比，阈值不倒挂）；加法模式 EXTRA_MS：固定加毫秒
                _burn_k = float(os.environ.get("PACNET_BURN_K", "0") or 0)
                _extra = float(os.environ.get("PACNET_EXTRA_MS", "0")) / 1000
                if _burn_k > 1.0:
                    self.pacnet_latency = [l * _burn_k for l in self.pacnet_latency]
                    self.log.info(f"halt latency adjusted x{_burn_k:.2f}(gpuburn MULT): "
                                  f"{[f'{l*1000:.1f}ms' for l in self.pacnet_latency]}")
                elif _extra > 0:
                    self.pacnet_latency = [l + _extra for l in self.pacnet_latency]
                    self.log.info(f"halt latency adjusted +{_extra*1000:.1f}ms(gpuburn ADD): "
                                  f"{[f'{l*1000:.1f}ms' for l in self.pacnet_latency]}")
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

    def _halt_decide(self, lead_time, remaining_frames, segment_index):
        """帧级 halt（可调度性前瞻，EDF 式可行性检查）：
        lead_time = 增强器当前领先播放的全局库存（秒）；remaining_frames = 本段剩余待增强帧。
        逐级检查"剩余帧若全程跑该档，能否在本段被播放前做完"（need = l_exit × 剩余帧数），
        选满足 lead_time >= need × 安全系数 的最高出口；都不满足则 e1 保命，
        更外层的 fast_complete 负责在彻底来不及时放弃剩余帧。
        阈值 1.0 = 恰好可行；>1 更保守（预留余量）。
        PACNET_FORCE_EXIT>=0 时强制该出口（全 e3 对照用）。"""
        if self._force_exit >= 0:
            return self._force_exit
        rf = max(int(remaining_frames), 1)
        l1, l2, l3 = self.pacnet_latency
        if lead_time >= l3 * rf * self._halt_e3_thresh:
            return 2   # e3
        if lead_time >= l2 * rf * self._halt_e2_thresh:
            return 1   # e2
        return 0       # e1 保命

    async def enhance_one_frame(self, surf, model, tensor_converter, exit_idx=None, compare_segment=None):
        """重写：用 PACNet 指定出口增强。exit_idx 由 run 循环里的 halt 控制器设置。"""
        if exit_idx is None:
            exit_idx = self._current_exit_idx
        tensor = tensor_converter.surface_to_tensor(surf)
        tensor = tensor.to(self.device)
        keep_input(compare_segment, tensor)
        await asyncio_sleep()
        with torch.no_grad():
            import time as _time
            torch.cuda.synchronize()
            _t0 = _time.perf_counter()
            tensor = model(tensor, exit_idx=exit_idx)
            torch.cuda.synchronize()
            fwd_ms = (_time.perf_counter() - _t0) * 1000.0
            await asyncio_sleep()
            tensor = F.interpolate(tensor, size=(self.display_H, self.display_W),
                                   mode="bicubic", align_corners=False)
            await asyncio_sleep()
            from ._gpuburn import burn as _burn; _burn(fwd_ms)
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
            if index is None or segments is None:
                self._is_ready = False
                self.log.info("PACNet Enhancer closed")
                return
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
                start_segment(segment, self.config)
                self.cnt = 0
                self._enhanced_frames = 0
                # 段内三出口实际增强帧计数 [e1,e2,e3]，供 analyzer 按帧占比算真实 VMAF
                self._exit_frame_cnt = [0, 0, 0]
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
                        pad_same(segment, remaining_frames)
                        self.cnt = self._num_frames
                        break

                    surf = decoder.decode_one_frame()
                    await asyncio_sleep()
                    if surf is None:
                        break

                    # halt 控制器（可调度性前瞻）：实时读取增强器领先播放的全局库存，
                    # 逐级判断"剩余帧全程跑该档能否按时做完"，选最高可行出口
                    try:
                        lead_time = (self.renderer.remain_task() +
                            (self._current_enhancing_index -
                             self.download_buffer.get_next_render_segment_index()) * self.seg_time)
                    except Exception:
                        lead_time = enhance_start_to_play_time - (time.perf_counter() - start_time)
                    remaining_frames = max(self._num_frames - self.cnt, 1)
                    self._current_exit_idx = self._halt_decide(lead_time, remaining_frames, index)

                    surf_enh = await self.enhance_one_frame(
                        surf, model, tensor_converter, compare_segment=segment
                    )
                    self._enhanced_frames += 1
                    # 记录该帧实际使用的出口（halt 帧级决策结果）
                    _ei = self._current_exit_idx if self._current_exit_idx in (0, 1, 2) else 2
                    self._exit_frame_cnt[_ei] += 1
                    # 对比打开时留下 uint8 增强帧，关闭时仍不占缓冲。
                    if getattr(segment, "compare_lr", None) is not None:
                        result.append(_uint8_cpu(surf_enh))
                    else:
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
                    drop_segment(segment)
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
                    "exit_frame_counts": list(self._exit_frame_cnt),
                    "fast_complete_triggered": bool(fast_complete),
                    "is_enhance": True,
                }
                await self.download_buffer.replace(self._current_enhancing_index, {as_idx: segment})
                _dist = "/".join(str(c) for c in self._exit_frame_cnt)
                self.log.info(f"PACNet enhance seg {index}: lat={enhance_latency:.2f}s, "
                              f"enhanced={self._enhanced_frames}, fps={enhance_fps:.1f}, "
                              f"last_exit=e{self._current_exit_idx+1}, dist(e1/e2/e3)={_dist}, "
                              f"fast_complete={fast_complete}, "
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


def _uint8_cpu(tensor):
    """Display tensor as uint8 so compare mode does not keep float frames."""
    if tensor is None:
        return None
    sample = tensor.detach()
    if getattr(sample, "dtype", None) is not None and sample.dtype == torch.uint8:
        return sample.cpu().contiguous()
    return (sample * 255).clamp(0, 255).to(dtype=torch.uint8).cpu().contiguous()
