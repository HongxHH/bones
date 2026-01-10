import asyncio

import json
import logging

from istream_player.config.config import PlayerConfig
# from istream_player.core.buffer import BufferManager
from istream_player.core.module import Module, ModuleOption
from istream_player.core.mpd_provider import MPDProvider
from istream_player.core.enhancer import Enhancer, EnhancerEventListener
from istream_player.core.scheduler import Scheduler
from istream_player.modules.decoder.decoder_nvcodec import DecoderWrapper
from istream_player.models import State
from istream_player.utils.async_utils import critical_task

from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.modules.decoder import DecoderNvCodec, TensorConverter
from istream_player.core.player import PlayerEventListener
from istream_player.models.mpd_objects import Segment
from istream_player.core.player import Player
from istream_player.core.downloader import (DownloadManager, DownloadRequest, DownloadType)
from istream_player.core.renderer import Renderer
from .imdn_model import IMDN, IMDN_RTC

import torch
import os
import time
import PyNvCodec as nvc
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
import tempfile
from .fast_complete_policy import RiskAwareFastCompletePolicy, UCBConfig




# IMDN增强器实现类 - 继承自Module、Enhancer和PlayerEventListener接口
# 负责使用IMDN（Information Multi-Distillation Network）模型对低分辨率视频进行超分辨率增强
@ModuleOption("imdn_always", default=True, requires=["model_downloader", DownloadBufferImpl, EnhanceBufferImpl, Scheduler, Renderer, Player])
class IMDNEnhancerAlways(Module, Enhancer, PlayerEventListener):
    # 获取日志记录器实例，用于记录IMDN增强相关的日志信息
    log = logging.getLogger("IMDNEnhancerAlways")

    # 常量定义
    # 支持的缩放比例
    SUPPORTED_SCALES = [2, 3, 4]
    # 支持的增强级别
    SUPPORTED_LEVELS = [1, 2, 3]
    # 模型预热轮数
    WARMUP_EPOCH = 3
    # 模型性能测量轮数
    MEASURE_EPOCH = 5
    # 默认段帧数
    DEFAULT_NUM_FRAMES = 120
    # === 风险约束 fast-complete 默认参数 ===
    # 允许“错过deadline”的概率上界（越小越保守）
    FAST_COMPLETE_DELTA = 0.2
    # 单次检查间隔允许的最大额外浪费计算时间（秒），用于推导自适应检查间隔
    FAST_COMPLETE_WASTE_BUDGET_S = 0.08
    # 自适应检查间隔的上界（帧）
    FAST_COMPLETE_CHECK_MAX = 30
    # 统计窗口大小（帧）
    FAST_COMPLETE_STATS_WINDOW = 30
    # 触发 fast-complete 前保留的保护裕量（秒）：给渲染/调度留余地，避免贴边
    FAST_COMPLETE_GUARD_S = 0.15
    # ====================================================
    # 默认提前检测阈值（秒）
    DEFAULT_FAST_COMPLETE_THRESHOLD = 2.0
    # 最小阈值映射（按增强级别）
    MIN_THRESHOLDS = {1: 0.2, 2: 0.3, 3: 0.4}

    def __init__(self):
        # 调用父类构造函数，初始化模块基础功能
        super().__init__()
        # 异步条件变量，用于控制增强器的访问和同步
        self._accessible = asyncio.Condition()
        # 增强器就绪状态标志，表示增强器是否已初始化完成
        self._is_ready = False

        # GPU设备配置，用于指定模型运行的设备
        self.device = None
        # 模型预热轮数，用于稳定模型性能测量
        self.warmup_epoch = self.WARMUP_EPOCH
        # 模型性能测量轮数，用于计算平均延迟
        self.measure_epoch = self.MEASURE_EPOCH

        # 播放器配置对象，包含所有配置参数
        self.config = None
        # 显示宽度，目标增强后的视频宽度
        self.display_W = None
        # 显示高度，目标增强后的视频高度
        self.display_H = None
        # 增强缓冲区实例，用于存储待增强的视频段
        self.enhance_buffer = None
        # 调度器实例，用于获取MPD信息和段持续时间
        self.scheduler = None
        # 分辨率集合，存储需要增强的视频分辨率信息
        self.resolution_set = None
        # 延迟表，存储不同分辨率和增强级别的延迟信息
        self.latency_table = np.zeros((5, 5))
        # 质量表，存储不同分辨率和增强级别的质量信息
        self.quality_table = None

        # 模型池，存储不同缩放比例和增强级别的预训练模型
        self.model_pool = None
        # 张量转换器，用于GPU内存中的张量格式转换
        self.tensor_converter = None
        # 段持续时间（秒），用于计算增强时间因子
        self.seg_time = 4
        # 帧率，用于计算增强时间因子
        self.frame_rate = 30
        # 时间因子，用于调整增强速度的变化
        self.time_factor = 1.  # enhancement speed variation factor
        # 安全因子
        self.safe_factor = 1  # safety factor
        # 任务开始时间，用于计算剩余任务时间
        self.task_start = None
        # 任务总时间，用于计算剩余任务时间
        self.task_total = None
        # 旧的质量表，用于内容感知模式的质量调整
        self.old_quality = None

        # 检查中止标志，用于控制增强任务的中止
        self.check_abort = False
        # 已播放URL列表，用于跟踪已播放的段，避免重复增强
        self.played_urls = []
        # 记录总的中止增强次数
        self.abort_total = 0

        # ==提前检测相关==
        # 当前正在增强的段索引（用于提前检测）
        self._current_enhancing_index: Optional[int] = None
        # 提前检测的缓冲水位阈值（秒），低于此值且段即将播放时触发快速完成
        self._fast_complete_threshold = self.DEFAULT_FAST_COMPLETE_THRESHOLD  # 默认2秒（将被动态调整）
        # 提前检测的检查间隔（帧数）
        self._check_interval_frames = 10
        # 是否使用提前检测
        self.use_fast_complete = False
        # 已经增强的帧数
        self._enhanced_frames = 0
        # 段的总帧数 默认120帧
        self._num_frames = self.DEFAULT_NUM_FRAMES

        # 风险约束 fast-complete 策略（在 start() 后可用 latency_table 初始化先验）
        self._fast_complete_policy: Optional[RiskAwareFastCompletePolicy] = None
        # 若需要“部分增强后再 fast-complete”，记录切换触发帧号（cnt >= 该值则切换）
        self._fast_complete_switch_at_frame: Optional[int] = None

        self.cnt = 0

        # 最小增强阈值（秒）- 根据增强级别不同，设置不同的最小阈值
        self._min_threshold = self.MIN_THRESHOLDS.copy()

        # 段级增强统计信息缓存（按 URL 索引），供 Analyzer 聚合使用
        self.segment_stats: Dict[str, Dict[str, Any]] = {}


    # 设置方法 - 初始化IMDN增强器的配置参数和依赖组件
    async def setup(self,
                    config: PlayerConfig,
                    model_downloader: DownloadManager,
                    download_buffer: DownloadBufferImpl,
                    enhance_buffer: EnhanceBufferImpl,
                    scheduler: Scheduler,
                    renderer: Renderer,
                    player: Player,
                    **kwargs
    ):
        # 保存播放器配置对象
        self.config = config
        # 从配置中获取显示宽度
        self.display_W = config.display_width
        # 从配置中获取显示高度
        self.display_H = config.display_height
        # 从配置中获取增强器设备（CPU/GPU）
        self.device = config.enhancer_device
        # 从配置中获取内容感知标志，决定是否使用内容感知增强
        self.content_aware = config.content_aware
        # 从配置中获取运行目录，用于临时文件存储
        self.run_dir = config.run_dir
        # 获取超分间隔配置，默认0表示全帧超分
        self.enhance_frame_interval = 15
        # 保存下载缓冲区实例的引用
        self.download_buffer = download_buffer
        # 保存增强缓冲区实例的引用
        self.enhance_buffer = enhance_buffer
        # 保存调度器实例的引用
        self.scheduler = scheduler
        # 保存模型下载管理器实例的引用
        self.download_manager = model_downloader
        # 保存渲染器实例的引用
        self.renderer = renderer
        # 将当前实例注册为播放器的事件监听器
        player.add_listener(self)
        # 是否使用快速完成
        self.use_fast_complete = config.use_fast_complete

        # 获取质量表，用于评估不同增强级别的质量
        self.quality_table = self.get_quality_table()
        # 注释掉的质量表日志输出
        # self.log.info("Quality table: {}".format(self.quality_table))
        
        

    # 启动增强器的方法 - 初始化模型池和性能测量
    async def start(self, adaptation_sets):
        # 使用异步条件变量确保线程安全
        async with self._accessible:
            # 加载预训练模型池
            self.model_pool = await self.load_model()

            # 获取需要增强的分辨率集合
            self.resolution_set = self.get_resolution_set(adaptation_sets)

            self.log.info("Need Enhance Resolution set: {}".format(self.resolution_set))

            # 测量模型延迟，构建延迟表
            self.latency_table = await self.measure_latency(self.resolution_set, self.model_pool)

            # self.log.info("Latency table: {}".format(self.latency_table))

            # 获取视频帧率
            self.frame_rate = self.get_frame_rate(adaptation_sets)
            # 获取段持续时间
            self.seg_time = self.scheduler.mpd_provider.mpd.max_segment_duration

            if self.use_fast_complete:
                # 初始化风险约束 fast-complete 策略：
                default_enh_time_s = None
                if self.latency_table is not None:
                    valid = self.latency_table[np.isfinite(self.latency_table) & (self.latency_table > 0)]
                    default_enh_time_s = float(np.quantile(valid, 0.7))

                if default_enh_time_s is None or not (default_enh_time_s > 0):
                    default_enh_time_s = 0.01

                self._fast_complete_policy = RiskAwareFastCompletePolicy(
                    config=UCBConfig(
                        delta=self.FAST_COMPLETE_DELTA, # DELTA是指错过deadline的概率上界
                        waste_budget_s=self.FAST_COMPLETE_WASTE_BUDGET_S, # 
                        check_interval_min=10,
                        check_interval_max=self.FAST_COMPLETE_CHECK_MAX,
                    ),
                    default_enhance_time_s=default_enh_time_s,
                    stats_window=self.FAST_COMPLETE_STATS_WINDOW,
                )

            # 如果存在旧的质量表，使用它替换当前质量表
            if self.old_quality is not None:
                self.quality_table = self.old_quality

            # 通知所有等待的线程增强器已就绪
            self._accessible.notify_all()
            # 设置增强器就绪状态
            self._is_ready = True
        return

    # 获取视频帧率的方法
    def get_frame_rate(self, adaptation_sets):
        # 遍历所有自适应集，查找视频内容
        for as_idx in adaptation_sets:
            as_obj = adaptation_sets[as_idx]
            # 跳过非视频内容
            if as_obj.content_type != "video":
                continue
            # 提取帧率并转换为浮点数
            frame_rate = np.array(as_obj.frame_rate).astype(float)
            return frame_rate

    # 检查增强器是否就绪的方法
    def is_ready(self):
        return self._is_ready

    # 获取需要增强的分辨率集合的方法
    def get_resolution_set(self, adaptation_sets):
        # 初始化分辨率集合字典
        resolution_set = {}
        # 遍历所有自适应集，查找视频内容
        for as_idx in adaptation_sets:
            as_obj = adaptation_sets[as_idx]
            # 跳过非视频内容
            if as_obj.content_type != "video":
                continue
            # 遍历当前自适应集的所有表示
            for repr_idx in as_obj.representations:
                repr_obj = as_obj.representations[repr_idx]
                # 计算缩放比例，取宽度和高度的最小缩放比例
                scale = min(int(self.display_W / repr_obj.width), int(self.display_H / repr_obj.height))
                # 确保缩放比例至少为1（不缩小）
                if scale < 1:
                    scale = 1
                # 只增强低分辨率视频（缩放比例大于1）
                if scale != 1:
                    resolution_set[repr_idx] = ((repr_obj.width, repr_obj.height, scale))
            break  # 假设只有一个视频轨道
        return resolution_set

    # 增强器主运行方法 - 使用critical_task装饰器确保关键任务执行
    @critical_task()
    async def run(self):
        # 等待增强器就绪
        async with self._accessible:
            await self._accessible.wait()

        # 主增强循环
        while self._is_ready:
            # 从增强缓冲区获取待增强的段
            index, segments = await self.enhance_buffer.dequeue()
            # 设置当前正在增强的段索引（用于提前检测）
            self._current_enhancing_index = index
            # 检查增强缓冲区是否为空
            if self.enhance_buffer.is_empty():
                # 如果调度器已结束，停止增强器
                if self.scheduler.is_end:
                    self._is_ready = False
                    self.log.info("Enhancer closed")
                    return

            # 通知所有监听器增强开始
            for listener in self.listeners:
                await listener.on_enhancement_start(segments)

            # 处理每个自适应集的段
            for as_idx in segments:
                segment = segments[as_idx]
                # 获取增强级别
                level = segment.enhance_action
                self.enhance_action = level
                # 初始化中止标志
                abort = False
                # 初始化中止原因
                abort_reason = ""

                enhance_start_to_play_time = (self.renderer.remain_task() +
                                                    (self._current_enhancing_index - self.download_buffer.get_next_render_segment_index()) * self.seg_time)
                # 记录增强开始信息
                self.log.info(f"Enhancing segment index: {self._current_enhancing_index },"
                              f"download action: {segment.download_action}, "
                              f"enhance action: {level}, "
                              f"enhance start to play time: {enhance_start_to_play_time:.3f}s")

                # 记录增强开始时间戳
                start_time = time.perf_counter()
                self.task_start = start_time
                # 获取该增强级别的预期延迟
                self.task_total = self.get_latency_table()[segment.download_action, segment.enhance_action]

                # 检查是否需要中止增强
                repr_idx = segment.repr_id
                # 最大分辨率（不需要增强）
                if repr_idx not in self.resolution_set:
                    self.abort_total -= 1 # 不计算最大分辨率的中止
                    abort = True
                    abort_reason = "Max resolution"
                # 已播放的段
                if segment.url in self.played_urls:
                    abort = True
                    abort_reason = "Already played before enhance"
                if self.use_fast_complete and enhance_start_to_play_time < self._min_threshold[self.enhance_action ]:
                    abort = True
                    abort_reason = "Remaining time less than min threshold"  # 剩余时长小于阈值，直接中止
                if abort:
                    self.abort_total += 1
                    # 记录段级别的中止原因与统计（此时尚未开始增强帧，浪费帧数为0）
                    segment.is_enhance = False
                    segment.abort_reason = abort_reason
                    segment.wasted_enhanced_cnt = 0
                    segment.enhance_frame_interval = 15
                    segment.fast_complete_threshold = self._fast_complete_threshold
                    segment.fast_complete_triggered = False
                    self.task_start = None
                    # 将统计结果写入全局缓存，供 Analyzer 在播放时按 URL 查询
                    self.segment_stats[segment.url] = {
                        "enhance_start_to_play_time": enhance_start_to_play_time,
                        "enhance_start_time": start_time,
                        "enhance_end_time": None,
                        "is_enhance": segment.is_enhance,
                        "abort_reason": segment.abort_reason,
                        "wasted_enhanced_cnt": segment.wasted_enhanced_cnt,
                        "enhance_frame_interval": segment.enhance_frame_interval,
                        "fast_complete_threshold": segment.fast_complete_threshold,
                        "fast_complete_triggered": segment.fast_complete_triggered,
                    }
                    self.log.info(
                        f"Abort enhancing segment index: {self._current_enhancing_index}, "
                        f"download action: {segment.download_action}, "
                        f"enhance action: {segment.enhance_action}, reason: {abort_reason}, enhance start to play time: {enhance_start_to_play_time:.3f}s"
                    )
                    # 重置增强状态（abort情况下不重置计数器）
                    self._reset_enhancement_state(reset_counters=False)
                    break

                # 获取缩放比例
                _, _, scale = self.resolution_set[repr_idx]
                # 设置增强比例
                segment.enhance_scale = scale

                # 创建解码器实例
                decoder = DecoderNvCodec(self.config, segment, resize=False)
                # 获取解码分辨率
                decode_W, decode_H = decoder.resolution()
                # 获取显示分辨率
                display_W, display_H = self.config.display_width, self.config.display_height
                # 创建张量转换器
                tensor_converter = TensorConverter(decode_W, decode_H, display_W, display_H, gpu_id=0)
                # 获取模型
                model = self.model_pool[(scale, level)]
                # 初始化增强结果列表
                result : List[object] = [] 

                self.cnt = 0 # 初始化帧计数器
                self._enhanced_frames = 0 # 初始化已增强帧计数器
                fast_complete = False # 标记是否需要快速完成（abort时使用）

                # 清理 fast-complete 调试缓存
                self._fast_complete_switch_at_frame = None
                # 逐帧解码和增强循环
                while True:
                    # 提前检测：检查段是否即将被播放（每N帧检查一次）
                    if self.use_fast_complete and not fast_complete:
                        # 若已决定“再增强到某个帧号后切 fast-complete”，在切换点先重规划：
                        # 如果此时 slack 仍充裕、或统计变好，则推迟/取消切换, 只有真正预算不足时才切 fast
                        if self._fast_complete_switch_at_frame is not None and self.cnt >= self._fast_complete_switch_at_frame:
                            # 到达切换点时重规划
                            if self.cnt == self._fast_complete_switch_at_frame:
                                fast_complete = self.check_fast_complete(current_frame_index=self.cnt)
                                # 若重规划后仍然没有立刻触发 fast，则继续跑（可能会更新 switch_at）
                            else:
                                fast_complete = True
                        # 否则按自适应检查间隔进行判定
                        if not fast_complete and self.cnt > 0 and self.cnt % self._check_interval_frames == 0:
                            fast_complete = self.check_fast_complete(current_frame_index=self.cnt)

                    # 如果进入快速完成模式，直接使用解码器进行实时解码
                    if fast_complete:
                        remaining_frames = self._num_frames - self.cnt 
                        self.log.info(f"Fast complete triggered at frame {self.cnt}, using decoder for remaining {remaining_frames} frames")
                        # 创建解码器包装器，将解码器传递给 renderer 进行实时解码
                        decoder_wrapper = DecoderWrapper(decoder, remaining_frames)
                        result.append(decoder_wrapper)
                        break  

                    # 解码一帧
                    surf = decoder.decode_one_frame()
                    # 让出控制权给其他任务
                    await asyncio.sleep(0)
                    # 检查是否解码完成
                    if surf is None:
                        break

                    t0 = time.perf_counter()
                    surf_enh = await self.enhance_one_frame(surf, model, tensor_converter)
                    dt = time.perf_counter() - t0
                    if self._fast_complete_policy is not None:
                        self._fast_complete_policy.update_observation(enhanced_dt_s=dt)
                    self._enhanced_frames += 1

                    result.append(surf_enh) # 将增强结果添加到结果列表
                    self.cnt += 1  # 增加帧计数

                    await asyncio.sleep(0)  # 让出控制权给其他任务

                    # 检查是否需要中止当前增强任务
                    if self.check_abort and segment.url in self.played_urls:
                        self.abort_total += 1
                        abort_reason = "Already played in enhance"
                        abort = True
                        self.check_abort = False
                        break
                    else:
                        self.check_abort = False # 设置已经检查了该段了

                # 如果中止，记录日志并跳出
                if abort:
                    # 记录段级别的中止原因与浪费的增强帧数
                    segment.is_enhance = False
                    segment.abort_reason = abort_reason
                    segment.wasted_enhanced_cnt = self._enhanced_frames
                    segment.enhance_frame_interval = self.enhance_frame_interval
                    segment.fast_complete_threshold = self._fast_complete_threshold
                    segment.fast_complete_triggered = fast_complete
                    # 将统计结果写入全局缓存，供 Analyzer 在播放时按 URL 查询
                    self.segment_stats[segment.url] = {
                        "enhance_start_to_play_time": enhance_start_to_play_time,
                        "enhance_start_time": start_time,
                        "enhance_end_time": time.perf_counter(),
                        "is_enhance": segment.is_enhance,
                        "abort_reason": segment.abort_reason,
                        "wasted_enhanced_cnt": segment.wasted_enhanced_cnt,
                        "enhance_frame_interval": segment.enhance_frame_interval,
                        "fast_complete_threshold": segment.fast_complete_threshold,
                        "fast_complete_triggered": segment.fast_complete_triggered,
                    }
                    self.log.info(
                        f"Abort enhancing segment index: {self._current_enhancing_index}, "
                        f"download action: {segment.download_action}, "
                        f"enhance action: {segment.enhance_action}, reason: {abort_reason}, "
                        f"wasted_enhanced_cnt: {self._enhanced_frames}"
                    )
                    # 重置增强状态（abort情况下不重置计数器）
                    self._reset_enhancement_state(reset_counters=False)
                    break

                # 正常完成：将增强结果保存到段对象
                segment.decode_data = result
                is_enhance = True  # 执行了增强
                end_time = time.perf_counter() # 记录增强结束时间
                enhance_latency = end_time - start_time
                # 若执行了增强动作，增强完成后该增强段距离播放开始的时间 
                if self._current_enhancing_index <= self.renderer.get_current_render_segment_index():
                    enhance_end_to_play_time = (segment.duration - self.renderer.remain_task()) + (self._current_enhancing_index - self.renderer.get_current_render_segment_index()) * segment.duration
                else:
                    enhance_end_to_play_time = (self.renderer.remain_task() +
                                                    abs((self.download_buffer.get_next_render_segment_index() -
                                                            self._current_enhancing_index)) * self.seg_time)

                # 记录与评估相关的统计信息，便于 PlaybackAnalyzer 聚合
                interpolated_cnt = self.cnt - self._enhanced_frames
                enhance_fps = self.cnt / (end_time - start_time) if end_time > start_time else None

                # 将统计结果写入全局缓存，供 Analyzer 按照 URL 查询
                self.segment_stats[segment.url] = {
                    "enhance_start_to_play_time": enhance_start_to_play_time,
                    "enhance_start_time": start_time,
                    "enhance_end_time": end_time,
                    "is_enhance": is_enhance,
                    "abort_reason": segment.abort_reason,
                    "wasted_enhanced_cnt": segment.wasted_enhanced_cnt,
                    "enhance_latency": enhance_latency,
                    "enhance_scale": segment.enhance_scale,
                    "enhanced_cnt": self._enhanced_frames,
                    "interpolated_cnt": interpolated_cnt,
                    "enhance_fps": enhance_fps,
                    "enhance_frame_interval": self.enhance_frame_interval,
                    "fast_complete_threshold": self._fast_complete_threshold,
                    "fast_complete_triggered": fast_complete,
                    "enhance_end_to_play_time": enhance_end_to_play_time,
                }
                # 用增强后的段替换原始段
                await self.download_buffer.replace(self._current_enhancing_index, {as_idx: segment}) 
                # 更新延迟表，计算实际增强时间因子
                if not fast_complete:
                    self.latency_table[segment.download_action, segment.enhance_action] = (end_time - start_time) / (self.seg_time * self.frame_rate)
                
                # 重置增强状态（成功完成后重置所有计数器）
                self._reset_enhancement_state(reset_counters=True)
                # 记录增强完成信息
                self.log.info(f"Complete enhancing segment index: {self._current_enhancing_index},"
                            f"download action: {segment.download_action}, "
                            f"enhance action: {segment.enhance_action}, "
                            f"enhance start to play time: {enhance_start_to_play_time:.3f}s, "
                            f"latency: {enhance_latency:.3f}s, "
                            f"enhanced frames: {self._enhanced_frames}, "
                            f"interpolated frames: {interpolated_cnt}, "
                            f"enhance FPS: {enhance_fps:.3f}fps, "
                            f"enhance frame interval: {self.enhance_frame_interval}, "
                            f"threshold: {self._fast_complete_threshold:.3f}s, "
                            f"enhance end to play time: {enhance_end_to_play_time:.3f}s")
                    

    # 增强单帧的方法
    async def enhance_one_frame(self, surf: nvc.Surface, model: torch.nn.Module, tensor_converter: TensorConverter) -> torch.Tensor:
        # 将NVIDIA Surface转换为PyTorch张量
        tensor = tensor_converter.surface_to_tensor(surf)
        # 将张量移动到指定设备
        tensor = tensor.to(self.device)
        # 让出控制权给其他任务
        await asyncio.sleep(0)
        # 使用模型进行推理（不计算梯度）
        with torch.no_grad():
            # 模型前向传播
            tensor = model(tensor)
            # 让出控制权给其他任务
            await asyncio.sleep(0)
            # 使用双三次插值调整到目标分辨率
            tensor = torch.nn.functional.interpolate(tensor, size=(self.display_H, self.display_W), mode='bicubic', align_corners=False)
            # 让出控制权给其他任务
            await asyncio.sleep(0)
        # 将张量移回CPU以节省GPU内存
        tensor = tensor.cpu()  # save on CPU to save GPU memory
        # 让出控制权给其他任务
        await asyncio.sleep(0)
        return tensor


    def _reset_enhancement_state(self, reset_counters: bool = False):
        """
        重置增强相关的状态字段
        """
        # 清除当前增强段索引
        self._current_enhancing_index = None
        # 重置与增强相关的计数器
        self._enhanced_frames = 0
        self.task_start = None

        # 在成功完成时额外重置的字段
        if reset_counters:
            self.cnt = 0
            self.task_total = None

    # 检查当前增强的段是否即将被播放（提前检测）
    def check_fast_complete(self, current_frame_index: int = 0) -> bool:
        """
        风险约束 fast-complete 判定 + 自适应检查间隔更新：
        - 若本段即将播放（当前增强段 == next to play），计算 slack（播放裕量）
        - 用在线观测得到的剩余时间上置信界 U_delta(T_remain) 与 slack 比较：
          若做完整剩余帧来不及，则在 slack 内先尽量多增强一些帧，然后再切换到 DecoderWrapper（fast 模式）
        - 同时根据 waste_budget 推导新的检查间隔（帧）
        """

        # 获取下一个要播放的段索引
        next_index = self.download_buffer.get_next_render_segment_index()
        # 如果当前增强的段就是下一个要播放的段
        if self._current_enhancing_index == next_index:
            slack_s = self.renderer.remain_task()   # 渲染剩余时长
            rem_enh = self._num_frames - current_frame_index  # 剩余增强帧数

            # 优化：一次性批量计算所有需要的统计量，避免重复计算
            stats = self._fast_complete_policy.compute_all_stats(
                remaining_enhanced_frames=rem_enh,
                slack_s=slack_s,
            )

            # 从批量结果中提取所需值
            u_remain = stats["u_remain"]
            u_enh = stats["u_enh"]
            check_k = int(stats["check_k"])
            triggered = stats["triggered"] > 0.5

            # 更新检查间隔
            self._check_interval_frames = check_k

            if triggered:
                # 计算"在 slack 内最多还能安全增强多少帧"
                guard = self.FAST_COMPLETE_GUARD_S
                budget_s = max(slack_s - guard, 0.0)  # 计算剩余预算

                # 默认所有帧都增强：allowed_frames 直接由 budget/u_enh 推导
                allowed_frames = int(budget_s / u_enh)
                allowed_frames = min(allowed_frames, rem_enh)

                if allowed_frames <= 0:
                    # 没有预算：立刻切 fast-complete（剩余帧交给 DecoderWrapper）
                    self.log.info(
                        f"Fast complete triggered (risk-aware, immediate): seg {self._current_enhancing_index} next to play, "
                        f"current_cnt: {current_frame_index}, slack: {slack_s:.3f}s, guard: {guard:.3f}s, budget: {budget_s:.3f}s, "
                        f"U_delta(remain): {u_remain:.3f}s, "
                        f"rem_enh: {rem_enh}, "
                        f"u_enh: {u_enh:.5f}s, "
                        f"delta: {self._fast_complete_policy.cfg.delta}, check_k: {self._check_interval_frames}"
                    )
                    return True

                # 有预算：继续跑 allowed_frames 帧后再切 fast-complete
                self._fast_complete_switch_at_frame = current_frame_index + allowed_frames
                self.log.info(
                    f"Fast complete scheduled (risk-aware): seg {self._current_enhancing_index} next to play, "
                    f"current_cnt: {current_frame_index}, slack: {slack_s:.3f}s, guard: {guard:.3f}s, budget: {budget_s:.3f}s, "
                    f"U_delta(remain): {u_remain:.3f}s, "
                    f"allow_frames: {allowed_frames} -> switch_at_cnt: {self._fast_complete_switch_at_frame}, "
                    f"rem_enh: {rem_enh}, "
                    f"u_enh: {u_enh:.5f}s, "
                    f"delta: {self._fast_complete_policy.cfg.delta}, check_k: {self._check_interval_frames}"
                )
                return False

        return False

    # 加载模型池的方法
    async def load_model(self):
        # 加载模型路径配置文件
        file_name = "imdn_path.json"
        model_path = json.load(open(file_name))
        # 根据内容感知标志选择不同的模型路径
        if self.content_aware:
            model_path = model_path["aware"]
        else:
            model_path = model_path["agnostic"]

        # 初始化模型池字典
        model_pool = {}
        # 获取当前文件的绝对路径
        ABSOLUTE_PATH = os.path.dirname(os.path.abspath(__file__))
        # 遍历所有缩放比例和增强级别组合
        for scale in [2, 3, 4]:
            for level in [1, 2, 3]:
                # 构建模型文件路径
                url = os.path.join(ABSOLUTE_PATH, model_path[f"scale{scale}_level{level}"])

                # 根据增强级别创建不同的模型
                if level == 1:
                    # 低级别增强：使用轻量级RTC模型
                    model = IMDN_RTC(upscale=scale, num_modules=3, nf=6)  # low
                    model = await self._download_and_load(model, url, download=self.content_aware)
                elif level == 2:
                    # 中级别增强：使用标准RTC模型
                    model = IMDN_RTC(upscale=scale)  # medium
                    model = await self._download_and_load(model, url, download=self.content_aware)
                elif level == 3 and (not self.content_aware):
                    # 高级别增强：使用完整IMDN模型（仅非内容感知模式）
                    model = IMDN(upscale=scale, nf=32)  # high
                    model = await self._download_and_load(model, url, download=self.content_aware)

                # 将模型移动到指定设备并存储到模型池
                model_pool[(scale, level)] = model.to(self.device)

        return model_pool

    # 下载并加载模型的方法
    async def _download_and_load(self, model, url, download=False):
        # 如果需要下载模型
        if download:
            # 下载模型文件
            await self.download_manager.download(DownloadRequest(url, DownloadType.STREAM_INIT))
            # 等待下载完成并获取模型数据
            model_data, _ = await self.download_manager.wait_complete(url)
            # 创建临时文件存储模型数据
            model_file = tempfile.NamedTemporaryFile(dir=self.run_dir, delete=False)
            model_file.write(model_data)
            model_file.close()
            # 更新URL为临时文件路径
            url = model_file.name
            pass

        # 加载模型权重
        model.load_state_dict(torch.load(url))
        return model

    # 测量模型延迟的方法
    async def measure_latency(self, resolution_set, model_pool, file_name="imdn_latency.json"):
        """
        预热增强模型并测量其延迟
        """
        # 如果延迟文件已存在，直接加载
        if os.path.exists(file_name):
            latency_table = json.load(open(file_name))
            return np.array(latency_table)

        # 记录延迟测量开始信息
        self.log.info("Start measuring latency (only for the first time)")
        # 初始化延迟集合字典
        latency_set = {}
        # 遍历所有需要增强的分辨率
        for repr_idx in resolution_set:
            width, height, scale = resolution_set[repr_idx]
            # 遍历所有增强级别
            for level in [1, 2, 3]:
                # 初始化数据池
                data_pool = []
                # 生成测试数据（预热轮数 + 测量轮数）
                for i in range(self.warmup_epoch + self.measure_epoch):
                    data_pool.append(torch.rand((1, 3, height, width)).to(self.device))
                    await asyncio.sleep(0)

                # 检查模型是否存在
                if (scale, level) not in model_pool:
                    continue
                # 获取对应模型
                model = model_pool[(scale, level)]
                # 设置为评估模式
                model.eval()
                # 预热阶段
                with torch.no_grad():
                    for i in range(self.warmup_epoch):
                        tensor = data_pool[i]
                        model(tensor)
                        await asyncio.sleep(0)

                    # 同步CUDA操作
                    torch.cuda.synchronize()
                    # 开始测量
                    start_time = time.perf_counter()
                    for i in range(self.measure_epoch):
                        tensor = data_pool[i + self.warmup_epoch]
                        model(tensor)
                        await asyncio.sleep(0)
                    # 同步CUDA操作
                    torch.cuda.synchronize()
                    # 结束测量
                    end_time = time.perf_counter()
                    # 计算平均延迟
                    latency_set[(scale, level)] = (end_time - start_time) / self.measure_epoch

        # 构建延迟表：比特率(240p, 360p, 480p, 720p, 1080p)，级别(无, 低, 中, 高, 超)
        latency_table = np.zeros((5, 5))
        for setting in latency_set:
            repr_idx, level = setting
            latency_table[4 - repr_idx, level] = latency_set[(repr_idx, level)]
        # 保存延迟表到文件
        json.dump(latency_table.tolist(), open(file_name, "w"), indent=4)
        self.log.info("Latency measurement completed")
        return latency_table

    # 获取延迟表的方法
    def get_latency_table(self):
        try:
            # 返回调整后的延迟表（考虑时间因子、段时间和帧率）
            return self.latency_table * self.time_factor * self.seg_time * self.frame_rate * self.safe_factor
        except:
            return None

    # 获取质量表的方法 - 假设增强质量元数据在本地存在
    def get_quality_table(self):
        # 根据内容感知标志选择不同的质量表文件
        if self.content_aware:
            file_name = "imdn_bbb_quality.json"
        else:
            file_name = "imdn_div2k_quality.json"

        # 如果质量表已存在，直接返回
        if self.quality_table is not None:
            return self.quality_table

        # 加载质量表
        quality_table = json.load(open(file_name))
        quality_table = np.array(quality_table)
        # 将负值设置为负无穷（表示无效增强）
        quality_table[quality_table < 0] = -np.inf  # invalid enhancement

        # 如果是内容感知模式，调整质量表
        if self.content_aware:
            # 保存原始质量表
            self.old_quality = quality_table.copy()
            # 只允许低级别和中级别增强
            self.old_quality[:, -2:] = -np.inf  # only low and medium level
            quality_table[:, 1:] = -np.inf

        # 设置质量表
        self.quality_table = quality_table
        return quality_table

    # 计算剩余任务时间的方法
    def remain_task(self):
        # 如果任务未开始，返回0
        if self.task_start is None:
            return 0
        # 返回剩余任务时间（总时间 - 已用时间）
        return max(self.task_total - (time.perf_counter() - self.task_start), 0)

    # 播放完成后，将延迟信息写回文件
    def write_latency_table(self):
        json.dump(self.latency_table.tolist(), open("latency_table.json", "w"), indent=4)

    # 段播放开始事件处理器
    async def on_segment_playback_start(self, segments: Dict[int, Segment]):
        # 中止正在进行的和未来的任务
        for idx in segments:
            segment = segments[idx]
            # 将已播放的URL添加到列表
            self.played_urls.append(segment.url)
        # 设置中止检查标志
        self.check_abort = True
        return

    # 播放状态变化事件处理器（用于记录卡顿）
    async def on_state_change(self, position: float, old_state: State, new_state: State):
        """
        监听播放状态变化，记录卡顿事件
        """

    # 清理方法
    async def cleanup(self) -> None:
        # 设置增强器为非就绪状态
        self._is_ready = False
        # 将延迟信息写回文件
        self.write_latency_table()
        # 清理增强缓冲区
        await self.enhance_buffer.cleanup()
        return