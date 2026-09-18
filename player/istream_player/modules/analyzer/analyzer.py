from dataclasses import asdict, dataclass, fields as dc_fields
import copy
import datetime
import io
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time
from os.path import join
import traceback
from typing import Any, Dict, List, Optional, TextIO, Tuple
from matplotlib.patches import Rectangle

import matplotlib.pyplot as plt

from istream_player.config.config import PlayerConfig
from istream_player.core.analyzer import Analyzer
from istream_player.core.buffer import BufferEventListener
from istream_player.core.bw_meter import BandwidthMeter, BandwidthUpdateListener, DownloadStats
from istream_player.core.module import Module, ModuleOption
from istream_player.core.mpd_provider import MPDProvider
from istream_player.core.player import Player, PlayerEventListener
from istream_player.core.scheduler import Scheduler, SchedulerEventListener
from istream_player.models import State
from istream_player.models.mpd_objects import Segment
from istream_player.core.enhancer import Enhancer

from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.utils.gpu_monitor import GPUMonitor

# 分析器段数据类 - 存储单个视频段的详细分析信息
@dataclass
class AnalyzerSegment:
    # 段基本信息
    index: int          # 段索引
    url: str            # 段URL
    repr_id: int        # 表示ID
    adap_set_id: int    # 自适应集ID
    bitrate: int        # 比特率

    # 时间信息
    start_time: Optional[float] = None    # 下载开始时间
    stop_time: Optional[float] = None     # 下载结束时间
    first_byte_at: Optional[float] = None # 第一个字节到达时间
    last_byte_at: Optional[float] = None  # 最后一个字节到达时间

    # 质量信息
    quality: Optional[int] = None                    # 质量级别
    segment_throughput: Optional[float] = None       # 段吞吐量
    adaptation_throughput: Optional[float] = None    # 自适应吞吐量

    # 字节统计
    total_bytes: Optional[int] = None      # 总字节数
    received_bytes: Optional[int] = None   # 接收字节数
    stopped_bytes: Optional[int] = None    # 停止字节数

    # 动作信息
    download_action: Optional[int] = 0     # 下载动作
    enhance_action: Optional[int] = 0      # 增强动作

    # BONES / IMDN 相关增强与决策信息（per-segment）
    enhance_start_time: Optional[float] = None          # 段增强开始时间戳
    enhance_end_time: Optional[float] = None            # 段增强结束时间戳  
    decision_time_buffer_level: Optional[float] = None   # ABR 决策时的下载缓冲水位（秒）
    enhance_latency: Optional[float] = None              # 段增强耗时（秒）
    enhance_fps: Optional[float] = None                  # 段增强 FPS
    enhance_scale: Optional[int] = None                  # 增强放大比例
    is_enhance: Optional[bool] = None                    # 是否实际执行了增强（未 abort）
    enhanced_cnt: Optional[int] = None                   # 使用模型增强的帧数
    interpolated_cnt: Optional[int] = None               # 插值 / 复制的帧数
    enhance_frame_interval: Optional[int] = None         # 超分间隔帧数（后续可变）
    fast_complete_threshold: Optional[float] = None      # fast-complete 阈值（秒）
    fast_complete: Optional[bool] = None                 # 是否触发 fast-complete
    enhance_ratio: Optional[float] = None                # 段中模型增强帧数占总帧数的比例
    base_vmaf: Optional[float] = None                    # 段增强前的离线 VMAF
    enhance_vmaf: Optional[float] = None                # 段增强后的理想离线 VMAF
    real_vmaf: Optional[float] = None                    # 段增强后的实际离线 VMAF
    abort_reason: Optional[str] = None                   # 若被中止，记录中止原因
    wasted_enhanced_cnt: Optional[int] = None            # 若被中止，浪费的增强帧数
    enhance_start_to_play_time: Optional[float] = None   #  增强开始到播放开始的时间 (渲染器剩余时长 + 中间隔得段数*段时长)
    enhance_end_to_play_time: Optional[float] = None      # 若执行完成了增强动作，增强完成后该增强段距离播放开始的时间 (渲染器剩余时长 + 中间隔得段数*段时长)
    exit_frame_counts: Optional[list] = None              # PACNet 段内 [e1,e2,e3] 各出口实际增强帧数（帧级 halt 结果）

    # ===== fast_complete 质量退化量化字段（任务2.1）=====
    quality_switches_within_segment: Optional[int] = None  # 段内增强/未增强切换次数（质量突变次数）
    enhanced_frame_vmaf: Optional[float] = None             # 增强帧的 VMAF（= enhance_vmaf）
    non_enhanced_frame_vmaf: Optional[float] = None          # 未增强帧的 VMAF（= base_vmaf）
    quality_drop_due_to_fast: Optional[float] = None        # fast_complete 导致的质量下降 (enhance_vmaf - real_vmaf)

    @property 
    def stop_ratio(self) -> Optional[float]:
        """
        计算停止比例 - 停止字节数占总字节数的比例
        
        Returns:
            Optional[float]: 停止比例，如果数据不完整则返回None
        """
        if self.total_bytes is not None and self.stopped_bytes is not None:
            return self.stopped_bytes / self.total_bytes
        else:
            return None

    @property
    def ratio(self) -> Optional[float]:
        """
        计算接收比例 - 接收字节数占总字节数的比例
        
        Returns:
            Optional[float]: 接收比例，如果数据不完整则返回None
        """
        if self.received_bytes is not None and self.total_bytes is not None:
            return self.received_bytes / self.total_bytes
        else:
            return None


# 缓冲水位数据类 - 存储缓冲水位的时间序列数据
@dataclass
class BufferLevel:
    time: float     # 时间戳
    level: float    # 缓冲水位（秒）


# 卡顿数据类 - 存储播放卡顿的时间信息
@dataclass
class Stall:
    time_start: float   # 卡顿开始时间
    time_end: float     # 卡顿结束时间

# GPU内存使用数据类 - 存储GPU内存使用情况
@dataclass
class GPUUsage:
    time: float   # 时间戳
    usage: float # GPU使用率
    memory: float # GPU内存使用情况

# CPU使用数据类 - 存储CPU使用情况
@dataclass
class CPUUsage:
    time: float   # 时间戳
    usage: float # CPU使用率


# 播放分析器实现类 - 继承多个接口，实现全面的播放数据分析
@ModuleOption(
    "data_collector",
    default=True,
    requires=[MPDProvider, BandwidthMeter, Scheduler, Player, DownloadBufferImpl, EnhanceBufferImpl, Enhancer],
)
class PlaybackAnalyzer(
    Module, Analyzer, PlayerEventListener, SchedulerEventListener, BandwidthUpdateListener, BufferEventListener
):
    # 日志记录器 - 用于记录分析器的操作日志
    log = logging.getLogger("PlaybackAnalyzer")

    def __init__(self, *, plots_dir: Optional[str] = None):
        """
        初始化播放分析器
        
        Args:
            plots_dir: 图表保存目录，如果为None则不生成图表
        """
        # 记录分析开始时间
        self._start_time = datetime.datetime.now().timestamp()
        # 记录 perf_counter 基准值（用于与增强器的 perf_counter 时间戳换算）
        self._start_perf = time.perf_counter()
        # 缓冲水位时间序列数据
        self._buffer_levels: List[BufferLevel] = []
        # 吞吐量时间序列数据 (时间, 吞吐量)
        self._throughputs: List[Tuple[float, int]] = []
        # 连续带宽估计时间序列数据 (时间, 带宽)
        self._cont_bw: List[Tuple[float, int]] = []
        # 播放状态时间序列数据 (时间, 状态, 位置)
        self._states: List[Tuple[float, State, float]] = []
        # 段数据字典，按URL索引
        self._segments_by_url: Dict[str, AnalyzerSegment] = {}
        # 当前播放位置
        self._position = 0
        # 卡顿记录列表
        self._stalls: List[Stall] = []
        # GPU内存使用时间序列数据 (时间, 内存使用情况)
        self._gpu_usage: List[GPUUsage] = []
        # CPU使用时间序列数据 (时间, 使用率)
        self._cpu_usage: List[CPUUsage] = []

        # 图表保存目录
        self.plots_dir = plots_dir

        # 运行期依赖组件引用（用于获取更多统计信号）
        self._download_buffer: Optional[DownloadBufferImpl] = None
        self._enhance_buffer: Optional[EnhanceBufferImpl] = None
        self._enhancer: Optional[Enhancer] = None
        self.quality_table = None

        # ===== 实验环境元数据 =====
        self.env_info: Dict[str, Any] = {
            "timestamp_start": None,
            "timestamp_end": None,
            "duration_seconds": None,
            "python_version": None,
            "torch_version": None,
            "cuda_version": None,
            "gpu_name": None,
            "gpu_count": None,
            "git_commit": None,
            "git_branch": None,
            "config_snapshot": None,
        }
        # GPU 指标（显存峰值等）
        self.gpu_metrics: Dict[str, Any] = {
            "peak_memory_mb": None,
            "model_pool_memory_mb": None,
        }
        # GPU 功耗监控器（可选，pynvml 可用时自动启用）
        self._gpu_monitor: Optional[GPUMonitor] = None
        # 启动延迟指标
        self.startup_metrics: Dict[str, Any] = {
            "time_to_first_download": None,
            "time_to_first_enhance": None,
            "time_to_first_play": None,
        }
        self._first_download_done = False
        self._first_enhance_done = False
        self._first_play_done = False

        # 注释掉的实验记录器（可能在未来版本中使用）
        # if self.config.recorder:
        #     self.config.recorder.write_event(ExpEvent_PlaybackStart(int(self._start_time * 1000)))

    async def setup(
        self,
        config: PlayerConfig,     # 播放器配置对象
        mpd_provider: MPDProvider, # MPD提供器
        bandwidth_meter: BandwidthMeter, # 带宽测量器
        scheduler: Scheduler, # 调度器
        player: Player, # 播放器
        download_buffer: DownloadBufferImpl, # 下载缓冲区
        enhance_buffer: EnhanceBufferImpl, # 增强缓冲区
        enhancer: Enhancer, # 增强器
        **kwargs, # 其他可选参数
    ):
        """
        设置播放分析器 - 初始化所有依赖组件和监听器
        """
        # 存储带宽测量器引用
        self.bandwidth_meter = bandwidth_meter
        # 存储MPD提供器引用
        self._mpd_provider = mpd_provider
        # 设置结果保存路径
        self.dump_results_path = config.metric_output
        os.makedirs(self.dump_results_path, exist_ok=True)
        self.plots_dir = config.plots_dir
        os.makedirs(self.plots_dir, exist_ok=True)

        # 存储配置对象引用（用于生成文件名）
        self.config = config
        # 存储缓冲区与增强器引用，方便采集更多运行期统计
        self._download_buffer = download_buffer
        self._enhance_buffer = enhance_buffer
        self._enhancer = enhancer
        self.quality_table = enhancer.get_quality_table()


        # 添加各种事件监听器
        # segment_downloader.add_listener(self)  # 注释掉的段下载器监听器
        bandwidth_meter.add_listener(self)      # 带宽更新监听器
        scheduler.add_listener(self)            # 调度器事件监听器
        player.add_listener(self)               # 播放器事件监听器
        download_buffer.add_listener(self)      # 下载缓冲区监听器
        enhance_buffer.add_listener(self)       # 增强缓冲区监听器

        # ===== 采集实验环境元数据 =====
        self._collect_env_info(config)

        # ===== 启动 GPU 功耗监控（可选，pynvml 可用时自动启用）=====
        try:
            self._gpu_monitor = GPUMonitor(device_index=0, sample_interval=0.5)
            if self._gpu_monitor.is_available:
                self._gpu_monitor.start()
                self.log.info("GPU power monitor started")
        except Exception as e:
            self.log.debug(f"Failed to start GPU monitor: {e}")

    def _collect_env_info(self, config: PlayerConfig):
        """采集实验环境信息：Python/PyTorch/CUDA/GPU/git/config快照"""
        self.env_info["timestamp_start"] = datetime.datetime.now().isoformat()
        self.env_info["python_version"] = sys.version
        try:
            import torch
            self.env_info["torch_version"] = torch.__version__
            self.env_info["cuda_version"] = torch.version.cuda
            if torch.cuda.is_available():
                self.env_info["gpu_name"] = torch.cuda.get_device_name(0)
                self.env_info["gpu_count"] = torch.cuda.device_count()
                # 重置显存峰值统计，以便在实验结束时获取准确的峰值
                torch.cuda.reset_peak_memory_stats()
        except Exception as e:
            self.log.warning(f"Failed to collect torch/cuda info: {e}")

        # 采集 git 信息（try-except，非 git 环境不报错）
        try:
            repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root, capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                self.env_info["git_commit"] = result.stdout.strip()
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=repo_root, capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                self.env_info["git_branch"] = result.stdout.strip()
        except Exception as e:
            self.log.debug(f"Git info not available: {e}")

        # 保存配置快照（将 dataclass 转为字典，排除不可序列化的 static 类引用）
        try:
            config_dict = {}
            for f in dc_fields(config):
                val = getattr(config, f.name)
                # 跳过不可 JSON 序列化的对象（如 StaticConfig 类引用、函数等）
                if isinstance(val, (str, int, float, bool, list, dict, type(None))):
                    config_dict[f.name] = val
                elif isinstance(val, tuple):
                    config_dict[f.name] = list(val)
                else:
                    config_dict[f.name] = str(val)
            # 手动补充非 dataclass 字段（类属性）
            for extra_field in ["use_fast_complete", "enhance_frame_interval", "enhancer_device", "renderer_device",
                                 "display_width", "display_height", "display_fps", "content_aware", "enhance_metadata"]:
                if hasattr(config, extra_field):
                    config_dict[extra_field] = getattr(config, extra_field)
            self.env_info["config_snapshot"] = config_dict
        except Exception as e:
            self.log.warning(f"Failed to snapshot config: {e}")

    async def cleanup(self) -> None:
        """
        清理分析器 - 保存分析结果
        """
        try:
            # 记录实验结束时间和总时长
            self.env_info["timestamp_end"] = datetime.datetime.now().isoformat()
            if self.env_info["timestamp_start"] is not None:
                start = datetime.datetime.fromisoformat(self.env_info["timestamp_start"])
                end = datetime.datetime.fromisoformat(self.env_info["timestamp_end"])
                self.env_info["duration_seconds"] = (end - start).total_seconds()

            # 采集 GPU 峰值显存
            try:
                import torch
                if torch.cuda.is_available():
                    self.gpu_metrics["peak_memory_mb"] = round(
                        torch.cuda.max_memory_allocated(0) / 1024 / 1024, 2
                    )
                    # 估算模型池显存占用
                    if self._enhancer is not None and hasattr(self._enhancer, "model_pool") and self._enhancer.model_pool:
                        model_mem = 0
                        for model in self._enhancer.model_pool.values():
                            if hasattr(model, "parameters"):
                                model_mem += sum(p.numel() * p.element_size() for p in model.parameters())
                        self.gpu_metrics["model_pool_memory_mb"] = round(model_mem / 1024 / 1024, 2)
            except Exception as e:
                self.log.debug(f"Failed to collect GPU metrics: {e}")

            # 停止 GPU 功耗监控并记录结果
            try:
                if self._gpu_monitor is not None and self._gpu_monitor.is_available:
                    power_stats = self._gpu_monitor.stop()
                    if power_stats:
                        self.gpu_metrics.update(power_stats)
            except Exception as e:
                self.log.debug(f"Failed to stop GPU monitor: {e}")

            # 保存分析结果到标准输出
            self.save(sys.stdout)
        except Exception as e:
            # 打印异常堆栈并记录错误
            traceback.print_exc()
            self.log.error(f"Failed to save analysis : {e}")

    @staticmethod
    def _seconds_since(start_time: float):
        """
        计算自给定时间以来的秒数
        """
        return datetime.datetime.now().timestamp() - start_time

    async def on_position_change(self, position):
        """
        处理播放位置变化事件
        
        Args:
            position: 新的播放位置
        """
        self._position = position

    async def on_state_change(self, position: float, old_state: State, new_state: State):
        """
        处理播放状态变化事件
        
        Args:
            position: 播放位置
            old_state: 旧状态
            new_state: 新状态
        """
        # 记录状态变化时间、新状态和播放位置
        self._states.append((self._seconds_since(self._start_time), new_state, position))

    async def on_buffer_level_change(self, buffer_level):
        """
        处理缓冲水位变化事件
        
        Args:
            buffer_level: 新的缓冲水位
        """
        # 记录缓冲水位变化时间和水位值
        self._buffer_levels.append(BufferLevel(self._seconds_since(self._start_time), buffer_level))

    async def on_segment_download_start(self, index, adap_bw: Dict[int, float], segments: Dict[int, Segment]):
        """
        处理段下载开始事件
        
        Args:
            index: 段索引
            adap_bw: 自适应集带宽字典
            segments: 段字典
        """
        # 确保MPD已加载
        assert self._mpd_provider.mpd is not None

        # 为每个段创建分析器段对象
        for as_id, segment in segments.items():
            # 获取自适应集的表示信息
            as_reprs = self._mpd_provider.mpd.adaptation_sets[int(as_id)].representations
            # 计算质量级别（相对于最低质量表示）
            quality = segment.repr_id - min(as_reprs.keys())
            # 获取表示对象
            repr = as_reprs[segment.repr_id]
            # 创建分析器段对象并存储
            self._segments_by_url[segment.url] = AnalyzerSegment(
                index=index,
                url=segment.url,
                repr_id=segment.repr_id,
                adap_set_id=as_id,
                adaptation_throughput= adap_bw[as_id],  # 自适应集带宽
                quality=quality,
                bitrate=repr.bandwidth,
                decision_time_buffer_level=segment.decision_time_buffer_level,
            )

    async def on_segment_download_complete(self, index: int, segments: Dict[int, Segment], stats: Dict[int, DownloadStats]):
        """
        处理段下载完成事件
        
        Args:
            index: 段索引
            segments: 段字典
            stats: 下载统计信息字典
        """
        # 记录第一个段下载完成时间（启动延迟指标）
        if not self._first_download_done:
            self.startup_metrics["time_to_first_download"] = round(
                self._seconds_since(self._start_time), 4
            )
            self._first_download_done = True

        # 更新每个段的下载统计信息
        for segment, stat in zip(segments.values(), stats.values()):
            # 确保统计信息完整
            assert stat.stop_time is not None and stat.start_time is not None
            # 获取对应的分析器段对象
            analyzer_segment = self._segments_by_url[segment.url]

            # 更新时间信息
            analyzer_segment.stop_time = stat.stop_time
            analyzer_segment.start_time = stat.start_time
            analyzer_segment.first_byte_at = stat.first_byte_at
            analyzer_segment.last_byte_at = stat.last_byte_at

            # 更新字节统计信息
            analyzer_segment.received_bytes = stat.received_bytes
            analyzer_segment.total_bytes = stat.total_bytes
            analyzer_segment.stopped_bytes = stat.stopped_bytes

            # 计算段吞吐量（比特每秒）
            analyzer_segment.segment_throughput = (stat.received_bytes * 8) / (stat.stop_time - stat.start_time)

    async def on_segment_playback_start(self, segments: Dict[int, Segment]):
        """
        处理段播放开始事件
        
        Args:
            segments: 段字典
        """
        # 记录第一个段开始播放时间（启动延迟指标）
        if not self._first_play_done:
            self.startup_metrics["time_to_first_play"] = round(
                self._seconds_since(self._start_time), 4
            )
            self._first_play_done = True

        # 更新每个段的动作与增强信息
        for segment in segments.values():
            analyzer_segment = self._segments_by_url.get(segment.url)
            if analyzer_segment is None:
                continue
            # 记录下载 / 增强动作
            analyzer_segment.download_action = segment.download_action
            analyzer_segment.enhance_action = segment.enhance_action
            
    async def on_bandwidth_update(self, bw: int) -> None:
        """
        处理带宽更新事件
        Args:
            bw: 新的带宽估计值（比特每秒）
        """
        # 记录带宽更新时间序列
        self._throughputs.append((self._seconds_since(self._start_time), bw))

    # TODO: 添加GPU使用率统计监听器
    # async def on_gpu_usage_update(self, usage: float) -> None:


    def save(self, output: io.TextIOBase | TextIO) -> None:
        """
        保存分析结果到输出流
        
        Args:
            output: 输出流对象
        """
        # 检查MPD是否可用
        if self._mpd_provider.mpd is None:
            self.log.error("MPD not found. Aborting analysis")
            return
        # 初始化比特率列表
        bitrates = []

        # 初始化质量切换统计
        last_quality = None
        quality_switches = 0

        # 初始化卡顿统计
        total_stall_duration = 0
        total_stall_num = 0

        # 如果播放未正常结束，添加结束状态
        if len(self._states) > 0 and self._states[-1][1] != State.END:
            self._states.append((self._seconds_since(self._start_time), State.END, self._position))


        # ===== 统计质量切换次数 =====
        for segment in sorted(self._segments_by_url.values(), key=lambda s: s.index):
            if last_quality is None:
                # 第一个段
                last_quality = segment.quality
            else:
                if last_quality != segment.quality:
                    last_quality = segment.quality
                    quality_switches += 1

        # ===== 统计卡顿次数和时长 =====
        buffering_start = None
        self._stalls = []
        for time, state, position in self._states:
            if state == State.BUFFERING and buffering_start is None:
                # 开始缓冲（卡顿开始）
                buffering_start = time
            elif state == State.READY and buffering_start is not None:
                # 结束缓冲（卡顿结束）
                duration = time - buffering_start
                # 记录卡顿信息
                self._stalls.append(Stall(buffering_start, time))
                total_stall_num += 1
                total_stall_duration += duration
                buffering_start = None

        # ===== 计算平均比特率 =====
        average_bitrate = sum(bitrates) / len(bitrates) if len(bitrates) > 0 else 0
        
        # ===== 统计所有段的增强信息和计算VMAF值 =====
        seg_stats = self._enhancer.segment_stats
        for analyzer_segment in self._segments_by_url.values():
            stats = seg_stats.get(analyzer_segment.url)
            if stats is not None:
                analyzer_segment.enhance_start_time = stats.get("enhance_start_time")
                analyzer_segment.enhance_end_time = stats.get("enhance_end_time")
                analyzer_segment.is_enhance = stats.get("is_enhance")
                analyzer_segment.abort_reason = stats.get("abort_reason")
                analyzer_segment.wasted_enhanced_cnt = stats.get("wasted_enhanced_cnt")
                analyzer_segment.enhance_latency = stats.get("enhance_latency") 
                analyzer_segment.enhance_scale = stats.get("enhance_scale")
                analyzer_segment.enhanced_cnt = stats.get("enhanced_cnt")
                analyzer_segment.interpolated_cnt = stats.get("interpolated_cnt")
                analyzer_segment.enhance_fps = stats.get("enhance_fps")
                analyzer_segment.enhance_frame_interval = stats.get("enhance_frame_interval")
                analyzer_segment.fast_complete_threshold = stats.get("fast_complete_threshold")
                analyzer_segment.fast_complete = stats.get("fast_complete_triggered")
                analyzer_segment.exit_frame_counts = stats.get("exit_frame_counts")
                analyzer_segment.enhance_start_to_play_time = stats.get("enhance_start_to_play_time")
                analyzer_segment.enhance_end_to_play_time = stats.get("enhance_end_to_play_time")

        # ===== 任务3.1：计算首段增强完成时间（启动延迟）=====
        if not self._first_enhance_done:
            # 找到第一个实际完成增强的段（enhance_end_time 不为 None）
            for seg in sorted(self._segments_by_url.values(), key=lambda s: s.index):
                if seg.enhance_end_time is not None:
                    # enhance_end_time 是 time.perf_counter() 值，换算为相对于实验开始的秒数
                    self.startup_metrics["time_to_first_enhance"] = round(
                        seg.enhance_end_time - self._start_perf, 4
                    )
                    self._first_enhance_done = True
                    break

        # ===== 基于质量表推算所有段增强后的离线 VMAF =====
        for analyzer_segment in self._segments_by_url.values():
            if analyzer_segment.quality is not None:
                q_idx = int(analyzer_segment.quality)
                analyzer_segment.base_vmaf = self.quality_table[q_idx, 0]
                # 如果该段没有执行增强，则增强索引为0
                if analyzer_segment.is_enhance:
                    e_idx = int(analyzer_segment.enhance_action)
                    # PACNet 帧级 halt：按段内 [e1,e2,e3] 实际增强帧数精确计算 VMAF，
                    # 未增强帧（fast-complete 截断部分）按 base_vmaf 计，避免用下发动作 e_idx 虚高打分。
                    _efc = analyzer_segment.exit_frame_counts
                    if _efc is not None and sum(int(c) for c in _efc) > 0:
                        _cnts = [int(c) for c in _efc]
                        _n_enh = sum(_cnts)
                        _n_non = int(analyzer_segment.interpolated_cnt) if analyzer_segment.interpolated_cnt else 0
                        _vmaf_enh = sum(_cnts[i] * float(self.quality_table[q_idx, i + 1])
                                        for i in range(3)) / _n_enh
                        _tot = _n_enh + _n_non
                        analyzer_segment.enhance_vmaf = _vmaf_enh
                        analyzer_segment.enhance_ratio = _n_enh / _tot if _tot > 0 else 1.0
                        analyzer_segment.real_vmaf = (
                            (_vmaf_enh * _n_enh + analyzer_segment.base_vmaf * _n_non) / _tot
                            if _tot > 0 else _vmaf_enh)
                        analyzer_segment.enhanced_frame_vmaf = _vmaf_enh
                        analyzer_segment.non_enhanced_frame_vmaf = analyzer_segment.base_vmaf
                        analyzer_segment.quality_drop_due_to_fast = _vmaf_enh - analyzer_segment.real_vmaf
                        analyzer_segment.quality_switches_within_segment = 1 if _n_non > 0 else 0
                    # 这里我们需要更加精细化的QOE,因为即使该段进行了增强，但是也可能触发了Fast模式，仅增强了部分，因此还要扣除没有增强的部分
                    elif analyzer_segment.fast_complete:
                        # 安全处理 enhanced_cnt / interpolated_cnt 为 None 的情况
                        _enhanced = int(analyzer_segment.enhanced_cnt) if analyzer_segment.enhanced_cnt is not None else 0
                        _interpolated = int(analyzer_segment.interpolated_cnt) if analyzer_segment.interpolated_cnt is not None else 0
                        _total = _enhanced + _interpolated
                        ratio = _enhanced / _total if _total > 0 else 0.0
                        analyzer_segment.enhance_vmaf = self.quality_table[q_idx, e_idx]
                        analyzer_segment.enhance_ratio = ratio
                        analyzer_segment.real_vmaf = analyzer_segment.base_vmaf * (1 - ratio) + analyzer_segment.enhance_vmaf * ratio
                        # ===== 任务2.1：fast_complete 质量退化量化 =====
                        analyzer_segment.enhanced_frame_vmaf = analyzer_segment.enhance_vmaf
                        analyzer_segment.non_enhanced_frame_vmaf = analyzer_segment.base_vmaf
                        analyzer_segment.quality_drop_due_to_fast = analyzer_segment.enhance_vmaf - analyzer_segment.real_vmaf
                        # 段内质量切换次数：fast_complete 截断型（前半增强后半未增强）= 1 次切换；
                        # 间隔增强型（增强/未增强交替）≈ 2 * enhanced_cnt 次
                        _interval = int(analyzer_segment.enhance_frame_interval) if analyzer_segment.enhance_frame_interval is not None else 0
                        if _interval > 0:
                            analyzer_segment.quality_switches_within_segment = 2 * _enhanced
                        else:
                            analyzer_segment.quality_switches_within_segment = 1 if 0 < ratio < 1 else 0
                    else:
                        analyzer_segment.real_vmaf = float(self.quality_table[q_idx, e_idx])
                else:
                    analyzer_segment.real_vmaf = float(self.quality_table[q_idx, 0])
        
        # 如果指定了图表目录，生成图表
        if self.plots_dir is not None:
            self.save_plots()

        # ===== 统计增强等级、中止原因和 fast_complete 节省的帧数 =====
        # 统计各增强等级的使用次数
        enhance_level_count: Dict[int, int] = {}
        # 统计各增强等级的实际完成增强的段数
        enhance_level_completed_count: Dict[int, int] = {}
        # 统计被中止或未执行增强的段的 abort 情况与浪费帧数
        abort_reason_count: Dict[str, int] = {}
        abort_wasted_frames_total: int = 0
        abort_wasted_frames_by_level: Dict[int, int] = {}
        # 统计 fast_complete 节省的模型增强帧数（按增强等级划分）
        fast_complete_saved_frames_total: int = 0
        fast_complete_saved_frames_by_level: Dict[int, int] = {}

        for seg in self._segments_by_url.values():

            # 统计各增强等级的使用次数
            if seg.enhance_action is not None:
                lvl = int(seg.enhance_action)
                enhance_level_count[lvl] = enhance_level_count.get(lvl, 0) + 1
                # 实际完成增强的段（is_enhance=True）
                if seg.is_enhance:
                    enhance_level_completed_count[lvl] = enhance_level_completed_count.get(lvl, 0) + 1

            # 若增强动作 > 0 但最终未完成增强且没有记录 abort_reason，则视为"未开始或被跳过的增强"
            if (
                (not seg.is_enhance) 
                and seg.enhance_action > 0
                and seg.abort_reason is None
            ):
                seg.abort_reason = "UnknownReason"
                if seg.wasted_enhanced_cnt is None:
                    seg.wasted_enhanced_cnt = 0

            # 被中止/未完成的段统计（此时 abort_reason 一定非空）
            if seg.abort_reason is not None:
                abort_reason_count[seg.abort_reason] = abort_reason_count.get(seg.abort_reason, 0) + 1
                if seg.wasted_enhanced_cnt is not None:
                    abort_wasted_frames_total += int(seg.wasted_enhanced_cnt)
                    lvl = int(seg.enhance_action)
                    abort_wasted_frames_by_level[lvl] = abort_wasted_frames_by_level.get(lvl, 0) + int(seg.wasted_enhanced_cnt)

            # 统计 fast_complete 节省的帧数（fast_complete=True 且 is_enhance=True，且 enhance_end_to_play_time < fast_complete_threshold）
            # fast_complete 后模型增强的帧数就是节省的模型增强帧数
            if (
                seg.fast_complete
                and seg.is_enhance
                and seg.enhanced_cnt is not None
                and seg.enhance_end_to_play_time is not None
                and seg.fast_complete_threshold is not None
                and seg.enhance_end_to_play_time < seg.fast_complete_threshold
            ):
                
                saved_frames = int(seg.enhanced_cnt)
                fast_complete_saved_frames_total += saved_frames
                lvl = int(seg.enhance_action)
                fast_complete_saved_frames_by_level[lvl] = fast_complete_saved_frames_by_level.get(lvl, 0) + saved_frames

        # ===== 任务2.1：fast_complete 质量退化聚合统计 =====
        fast_complete_segments_count = 0
        fast_complete_quality_drop_total = 0.0
        fast_complete_quality_switches_total = 0
        for seg in self._segments_by_url.values():
            if seg.fast_complete and seg.is_enhance:
                fast_complete_segments_count += 1
                if seg.quality_drop_due_to_fast is not None:
                    fast_complete_quality_drop_total += float(seg.quality_drop_due_to_fast)
                if seg.quality_switches_within_segment is not None:
                    fast_complete_quality_switches_total += int(seg.quality_switches_within_segment)
        fast_complete_quality_drop_avg = (
            fast_complete_quality_drop_total / fast_complete_segments_count
            if fast_complete_segments_count > 0 else None
        )

        # ===== 任务2.2：增强级别使用分布与完成率统计 =====
        enhance_level_stats: Dict[str, Dict[str, Any]] = {}
        for seg in self._segments_by_url.values():
            if seg.enhance_action is None:
                continue
            lvl_key = str(int(seg.enhance_action))
            if lvl_key not in enhance_level_stats:
                enhance_level_stats[lvl_key] = {
                    "count": 0, "completed": 0, "completion_rate": None,
                    "avg_latency": None, "avg_fps": None,
                    "_latency_sum": 0.0, "_latency_cnt": 0,
                    "_fps_sum": 0.0, "_fps_cnt": 0,
                }
            stats = enhance_level_stats[lvl_key]
            stats["count"] += 1
            if seg.is_enhance:
                stats["completed"] += 1
                if seg.enhance_latency is not None:
                    stats["_latency_sum"] += float(seg.enhance_latency)
                    stats["_latency_cnt"] += 1
                if seg.enhance_fps is not None:
                    stats["_fps_sum"] += float(seg.enhance_fps)
                    stats["_fps_cnt"] += 1
        # 计算完成率和平均值，清理内部字段
        for lvl_key, stats in enhance_level_stats.items():
            stats["completion_rate"] = round(stats["completed"] / stats["count"], 4) if stats["count"] > 0 else None
            stats["avg_latency"] = round(stats["_latency_sum"] / stats["_latency_cnt"], 4) if stats["_latency_cnt"] > 0 else None
            stats["avg_fps"] = round(stats["_fps_sum"] / stats["_fps_cnt"], 2) if stats["_fps_cnt"] > 0 else None
            del stats["_latency_sum"], stats["_latency_cnt"], stats["_fps_sum"], stats["_fps_cnt"]

        # 从增强器读取全局统计
        abort_total = self._enhancer.abort_total
        use_fast_complete = self._enhancer.use_fast_complete

        # ===== 质量与 QoE 相关指标计算 =====
        # 1) 计算整体评价 VMAF（对所有有 vmaf 的段取简单平均）
        sorted_segments = sorted(self._segments_by_url.values(), key=lambda s: s.index)
        vmafs = [seg.real_vmaf for seg in sorted_segments if seg.real_vmaf is not None]
        avg_vmaf = float(sum(vmafs) / len(vmafs)) if len(vmafs) > 0 else None

        # 2) 计算平均 VMAF 振荡（相邻分段 VMAF 绝对差的平均值）
        oscillation_vmaf = None
        if len(vmafs) > 1:
            diff_sum = 0.0
            for i in range(1, len(vmafs)):
                diff_sum += abs(vmafs[i] - vmafs[i - 1])
            oscillation_vmaf = float(diff_sum / (len(vmafs) - 1))

        # 3) 计算平均卡顿时长（对应 simulator 中的 avg_freeze）
        num_seg = len(sorted_segments)
        avg_freeze = float(total_stall_duration / num_seg) if num_seg > 0 else 0.0

        # 4) 计算 QoE（安全处理 None 值）
        _avg = avg_vmaf if avg_vmaf is not None else 0.0
        _osc = oscillation_vmaf if oscillation_vmaf is not None else 0.0
        qoe = float(_avg - _osc - avg_freeze / 10.0)

        # ===== 构建结果数据字典 =====
        data = {
            "num_stall": total_stall_num,                            # 卡顿次数
            "dur_stall": total_stall_duration,                            # 卡顿总时长
            "avg_bitrate": average_bitrate,                        # 平均比特率
            "num_quality_switches": quality_switches,      # 质量切换次数
            "bandwidth_estimate": [{"time": bw[0], "bandwidth": bw[1]} for bw in self._cont_bw],  # 带宽估计数据
            "enhancer_abort_total": abort_total,  # 中止总数
            "enhancer_use_fast_complete": use_fast_complete,  # 是否使用 fast-complete
            "enhance_level_count": enhance_level_count,  # 增强等级使用次数
            "enhance_level_completed_count": enhance_level_completed_count,  # 增强等级实际完成增强的段数
            "abort_reason_count": abort_reason_count,  # 中止/跳过原因使用次数
            "abort_wasted_frames_total": abort_wasted_frames_total,  # 浪费的增强帧数总数
            "abort_wasted_frames_by_level": abort_wasted_frames_by_level,  # 浪费的增强帧数按等级统计
            "fast_complete_saved_frames_total": fast_complete_saved_frames_total,  # fast_complete 节省的模型增强帧数总数
            "fast_complete_saved_frames_by_level": fast_complete_saved_frames_by_level,  # fast_complete 节省的帧数按等级统计
            # ===== 任务2.1：fast_complete 质量退化统计 =====
            "fast_complete_segments_count": fast_complete_segments_count,  # 触发 fast_complete 的段数
            "fast_complete_quality_drop_total": round(fast_complete_quality_drop_total, 4),  # fast_complete 导致的质量下降总和
            "fast_complete_quality_drop_avg": round(fast_complete_quality_drop_avg, 4) if fast_complete_quality_drop_avg is not None else None,  # 平均每次 fast_complete 的质量下降
            "fast_complete_quality_switches_total": fast_complete_quality_switches_total,  # fast_complete 段内质量切换总次数
            # ===== 任务2.2：增强级别分布与完成率统计 =====
            "enhance_level_stats": enhance_level_stats,  # 各增强级别的使用次数、完成率、平均延迟、平均FPS
            "avg_vmaf": avg_vmaf,
            "oscillation_vmaf": oscillation_vmaf, # VMAF振荡
            "avg_freeze": avg_freeze, # 平均卡顿时长
            "QOE": qoe,
            "environment": self.env_info,           # 实验环境元数据
            "gpu_metrics": self.gpu_metrics,        # GPU 指标（显存峰值等）
            "startup_metrics": self.startup_metrics, # 启动延迟指标
            "segments": list(map(asdict, self._segments_by_url.values())),  # 段数据
            "stalls": list(map(asdict, self._stalls)),         # 卡顿数据
            "states": [{"time": time, "state": str(state), "position": pos} for time, state, pos in self._states],  # 状态数据
            "buffer_level": list(map(asdict, self._buffer_levels)),  # 缓冲水位数据
        }

        # ===== 保存到文件或输出到标准输出 =====
        if self.dump_results_path is not None:
            PlaybackAnalyzer.save_file(self.dump_results_path, data, self.config)  # type: ignore
        else:
            json.dump(data["segments"], sys.stdout, indent=4)

    @staticmethod
    def save_file(path: str, data: dict[str, Any], config):
        """
        保存数据到文件，自动处理文件名冲突
        
        Args:
            path: 基础文件路径（目录）
            data: 要保存的数据
            config: 播放器配置对象，用于生成文件名
        """
        # 确保路径是目录
        os.makedirs(path, exist_ok=True)
        
        # 生成文件名：{nes}_{enhancer}_fast-{fast}_video-{video}_trace-{trace}_{timestamp}.json
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        # 从 input 路径提取视频名（不含扩展名）
        video_name = os.path.splitext(os.path.basename(config.input))[0] if config.input else "unknown"
        # 从 trace 路径提取 trace 名（不含扩展名）
        trace_name = os.path.splitext(os.path.basename(config.trace))[0] if config.trace else "unknown"
        base_filename = (
            f"{config.mod_nes}_{config.mod_enhancer}_fast-{config.use_fast_complete}"
            f"_video-{video_name}_trace-{trace_name}_{timestamp}"
        )
        file_path = os.path.join(path, f"{base_filename}.json")
        
        # 处理文件名冲突，自动添加数字后缀
        extra_index = 1
        final_path = file_path
        while os.path.exists(final_path):
            final_path = os.path.join(path, f"{base_filename}-{extra_index}.json")
            extra_index += 1

        # 输出文件保存信息
        print(f"Writing results in file {final_path}")
        # 以缩进格式保存 JSON，提升可读性
        with open(final_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
            f.write("\n")

    def save_plots(self):
        """
        生成并保存分析图表
        创建带宽和缓冲水位的可视化图表
        """
        def plot_bws(ax: plt.Axes):
            """
            绘制带宽图表
            
            Args:
                ax: matplotlib坐标轴对象
                
            Returns:
                tuple: 绘制的线条对象
            """
            # 提取时间和带宽数据
            xs = [i[0] for i in self._throughputs]
            ys = [i[1] / 1000 for i in self._throughputs]  # 转换为kbps
            # 绘制带宽曲线
            lines1 = ax.plot(xs, ys, color="red", label="Throughput")
            # 设置坐标轴范围
            ax.set_xlim(0)
            ax.set_ylim(0)
            # 设置坐标轴标签
            ax.set_xlabel("Time (second)")
            ax.set_ylabel("Bandwidth (kbps)", color="red")
            return (*lines1,)

        def plot_bufs(ax: plt.Axes):
            """
            绘制缓冲水位图表
            
            Args:
                ax: matplotlib坐标轴对象
                
            Returns:
                tuple: 绘制的线条对象
            """
            # 提取时间和缓冲水位数据
            xs = [i.time for i in self._buffer_levels]
            ys = [i.level for i in self._buffer_levels]
            # 绘制缓冲水位阶梯图
            line1 = ax.step(xs, ys, color="blue", label="Buffer", where="post")
            # 计算Y轴范围
            y_top = max(ys)
            y_bottom = min(ys)
            # 设置坐标轴范围
            ax.set_xlim(0)
            ax.set_ylim(y_bottom, y_top)
            # 设置Y轴标签
            ax.set_ylabel("Buffer (second)", color="blue")
            # 在卡顿期间添加红色矩形标记
            for stall in self._stalls:
                ax.add_patch(
                    Rectangle(
                        (stall.time_start, y_bottom),
                        stall.time_end - stall.time_start,
                        y_top - y_bottom,
                        facecolor=(1, 0, 0, 0.3),  # 半透明红色
                    )
                )
            # 注释掉的恐慌缓冲线（备用功能）
            # line2 = ax.hlines(1.5, 0, 20, linestyles="dashed", label="Panic buffer")
            return (*line1,)

        # 创建图表和坐标轴
        fig, ax1 = plt.subplots()
        # ax2: plt.Axes = ax1.twinx()  # 注释掉的双Y轴（备用功能）
        # 绘制缓冲水位图表
        lines = plot_bufs(ax1)
        # 获取线条标签
        labels = [line.get_label() for line in lines]
        # 添加图例
        fig.legend(lines, labels)
        # 如果指定了图表目录，保存图表
        if self.plots_dir is not None:
            # 创建目录（如果不存在）
            Path(self.plots_dir).mkdir(parents=True, exist_ok=True)
            # 设置输出文件路径
            output_file = os.path.join(self.plots_dir, "status.pdf")
            # 保存图表为PDF文件
            fig.savefig(output_file)
