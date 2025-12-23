import asyncio
import copy
import itertools
import logging
import os.path
import tempfile
import shutil
from asyncio import Task
from typing import Dict, Optional, Set, Tuple, List

from istream_player.config.config import PlayerConfig
# from istream_player.core.abr import ABRController
from istream_player.core.nes import NESController
# from istream_player.core.buffer import BufferManager
from istream_player.core.bw_meter import BandwidthMeter
from istream_player.core.downloader import (DownloadManager, DownloadRequest,
                                            DownloadType)
from istream_player.core.module import Module, ModuleOption
from istream_player.core.mpd_provider import MPDProvider
from istream_player.core.scheduler import Scheduler, SchedulerEventListener
from istream_player.models import AdaptationSet
from istream_player.utils import critical_task

from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.core.enhancer import Enhancer


# 调度器实现类 - 继承自Module和Scheduler接口
# 负责管理视频段的下载调度、缓冲管理和增强处理
@ModuleOption(
    "scheduler", default=True, requires=["segment_downloader", BandwidthMeter, DownloadBufferImpl, EnhanceBufferImpl, MPDProvider, NESController, Enhancer]
)
class SchedulerImpl(Module, Scheduler):
    # 日志记录器 - 用于记录调度器的操作日志
    log = logging.getLogger("SchedulerImpl")

    def __init__(self):
        """
        初始化调度器实现
        设置所有必要的状态变量和配置参数
        """
        # 调用父类构造函数
        super().__init__()

        # 自适应集字典 - 存储选中的自适应集，键为自适应集ID，值为AdaptationSet对象
        self.adaptation_sets: Optional[Dict[int, AdaptationSet]] = None
        # 调度器启动状态标志
        self.started = False

        # 异步任务对象 - 存储调度器的主运行任务
        self._task: Optional[Task] = None
        # 当前处理的段索引 - 指向正在下载的段
        self._index = 0
        # 已初始化的表示集合 - 存储已完成流初始化的表示ID
        self._representation_initialized: Set[str] = set()
        # 当前下载动作字典 - 存储每个自适应集的下载选择（表示ID）
        self._current_download_actions: Optional[Dict[int, int]] = None
        # 当前增强动作字典 - 存储每个自适应集的增强选择（增强级别）
        self._current_enhance_actions: Optional[Dict[int, int]] = None

        # 播放结束标志 - 表示是否已到达播放序列的末尾
        self._end = False
        # 被丢弃的段索引 - 记录因网络问题被丢弃的段
        self._dropped_index = None

    async def setup(
        self,
        config: PlayerConfig,
        segment_downloader: DownloadManager,
        bandwidth_meter: BandwidthMeter,
        download_buffer: DownloadBufferImpl,
        enhance_buffer: EnhanceBufferImpl,
        mpd_provider: MPDProvider,
        nes_controller: NESController,
        enhancer: Enhancer,
    ):
        """
        设置调度器 - 初始化所有依赖组件和配置参数
        
        Args:
            config: 播放器配置对象
            segment_downloader: 段下载管理器
            bandwidth_meter: 带宽测量器
            download_buffer: 下载缓冲区
            enhance_buffer: 增强缓冲区
            mpd_provider: MPD提供器
            nes_controller: 网络感知调度控制器
            enhancer: 视频增强器
        """
        # 从配置中提取缓冲相关参数
        self.max_buffer_duration = config.buffer_duration  # 最大缓冲持续时间
        self.update_interval = config.static.update_interval  # 更新间隔
        self.time_factor = config.time_factor  # 时间因子（播放速度倍数）
        self.run_dir = config.run_dir  # 运行目录
        self.content_aware = config.content_aware  # 内容感知标志

        # 存储所有依赖组件的引用
        self.download_manager = segment_downloader  # 段下载管理器
        self.bandwidth_meter = bandwidth_meter  # 带宽测量器
        self.download_buffer = download_buffer  # 下载缓冲区
        self.enhance_buffer = enhance_buffer  # 增强缓冲区
        self.nes_controller = nes_controller  # 网络感知调度控制器
        self.mpd_provider = mpd_provider  # MPD提供器
        self.enhancer = enhancer  # 视频增强器

        # 解析自适应集选择配置
        # 格式支持："start-end" 或 "index"
        select_as = config.select_as.split("-")
        if len(select_as) == 1 and select_as[0].isdecimal():
            # 单个索引格式："5" -> 只选择第5个自适应集
            self.selected_as_start = int(select_as[0])
            self.selected_as_end = int(select_as[0])
        elif (
            len(select_as) == 2
            and (select_as[0].isdecimal() or select_as[0] == "")
            and (select_as[1].isdecimal() or select_as[1] == "")
        ):
            # 范围格式："2-5" 或 "2-" 或 "-5"
            self.selected_as_start = int(select_as[0]) if select_as[0] != "" else None
            self.selected_as_end = int(select_as[1]) if select_as[1] != "" else None
        else:
            # 格式错误，抛出异常
            raise Exception("select_as should be of the format '<uint>-<uint>' or '<uint>'.")

        # 设置运行目录
        if self.run_dir is not None:
            # 如果目录已存在，先删除
            if os.path.exists(self.run_dir):
                shutil.rmtree(self.run_dir)
            # 创建新的运行目录
            os.mkdir(self.run_dir)

    def segment_limits(self, adap_sets: Dict[int, AdaptationSet]) -> tuple[int, int]:
        """
        计算段索引范围 - 获取所有自适应集中段索引的最小值和最大值
        
        Args:
            adap_sets: 自适应集字典
            
        Returns:
            (min_index, max_index): 段索引的最小值和最大值
        """
        # 提取所有自适应集中所有表示的所有段ID
        ids = [
            [[seg_id for seg_id in repr.segments.keys()] for repr in as_val.representations.values()]
            for as_val in adap_sets.values()
        ]
        # 展平嵌套列表结构
        ids = itertools.chain(*ids)
        ids = list(itertools.chain(*ids))
        # print(adap_sets, ids)  # 调试输出
        # 返回最小和最大段索引
        return min(ids), max(ids)

    @critical_task()
    async def run(self):
        """
        调度器主运行循环 - 核心调度逻辑
        负责协调段下载、缓冲管理和增强处理
        """
        # 等待MPD可用
        await self.mpd_provider.available()
        # 确保MPD已加载
        assert self.mpd_provider.mpd is not None
        # 选择要处理的自适应集
        self.adaptation_sets = self.select_adaptation_sets(self.mpd_provider.mpd.adaptation_sets)

        # 快速启动 - 如果不是内容感知模式，立即启动增强器
        if not self.content_aware:
            await self.enhancer.start(self.adaptation_sets)
            self.log.info("Enhancer started")

        # 从最小段索引开始处理
        self.first_segment, self.last_segment = self.segment_limits(self.adaptation_sets)
        self.log.info(f"{self.first_segment=}, {self.last_segment=}")
        self._index = self.first_segment
        
        # 主调度循环
        while True:
            # 检查缓冲水位 - 如果缓冲已满，等待一段时间
            if self.download_buffer.buffer_level() > self.max_buffer_duration:
                await asyncio.sleep(self.time_factor * self.update_interval)
                continue

            # 确保MPD仍然可用
            assert self.mpd_provider.mpd is not None

            # 处理动态MPD - 需要定期更新
            if self.mpd_provider.mpd.type == "dynamic":
                # 更新MPD内容
                await self.mpd_provider.update()
                # 重新选择自适应集
                self.adaptation_sets = self.select_adaptation_sets(self.mpd_provider.mpd.adaptation_sets)
                # 重新计算段索引范围
                self.first_segment, self.last_segment = self.segment_limits(self.adaptation_sets)
                self.log.info(f"{self.first_segment=}, {self.last_segment=}")

            # 检查当前段是否在MPD范围内
            if self._index < self.first_segment:
                self.log.info(f"Segment {self._index} not in mpd, Moving to next segment")
                self._index += 1
                continue

            # 检查是否已到达动态MPD的末尾
            if self.mpd_provider.mpd.type == "dynamic" and self._index > self.last_segment:
                self.log.info(f"Waiting for more segments in mpd : {self.mpd_provider.mpd.type}")
                await asyncio.sleep(self.time_factor * self.update_interval)
                continue

            # 内容感知模式下的增强器启动条件
            if self.content_aware and (not self.enhancer.is_ready()) and self.download_buffer.buffer_level(continuous=True) > 10:
                await self.enhancer.start(self.adaptation_sets)
                self.log.info("Enhancer started ")

            # 从每个自适应集下载一个段
            # 根据是否是被丢弃的段选择不同的调度策略
            if self._index == self._dropped_index:
                # 被丢弃的段使用最低质量策略
                download_actions, enhance_actions = self.nes_controller.update_selection_lowest(self.adaptation_sets)
            else:
                # 正常段使用智能调度策略
                download_actions, enhance_actions = self.nes_controller.update_selection(self.adaptation_sets, self._index)
            self.log.info(f"Index {self._index}, download actions {download_actions}, enhance actions {enhance_actions}")
            # 保存当前的动作选择
            self._current_download_actions = download_actions
            self._current_enhance_actions = enhance_actions

            # 所有自适应集使用当前带宽进行下载
            adap_bw = {as_id: self.bandwidth_meter.bandwidth for as_id in download_actions.keys()}

            # 获取每个自适应集要下载的段
            try:
                segments = {}
                for adaptation_set_id in download_actions.keys():
                    download_action = download_actions[adaptation_set_id]
                    enhance_action = enhance_actions[adaptation_set_id]
                    adapt_set = self.adaptation_sets[adaptation_set_id]

                    # 只处理视频轨道
                    if adapt_set.content_type != "video":
                        continue

                    # 获取对应的段对象
                    segment = adapt_set.representations[download_action].segments[self._index]
                    # 设置段的下载和增强动作
                    segment.download_action = download_action
                    segment.enhance_action = enhance_action
                    segment.decision_time_buffer_level = self.download_buffer.buffer_level(continuous=True)
                    segments[adaptation_set_id] = segment

            except KeyError:
                # 没有更多段可下载
                self.log.info("No more segments left")
                self._end = True
                return

            # 通知所有监听器段下载开始
            for listener in self.listeners:
                await listener.on_segment_download_start(self._index, adap_bw, segments)

            # duration = 0  # 注释掉的持续时间变量
            urls = {}  # 键：URL，值：[段对象，下载结果]
            # 处理每个自适应集的下载
            for adaptation_set_id, download_action in download_actions.items():
                adaptation_set = self.adaptation_sets[adaptation_set_id]
                representation = adaptation_set.representations[download_action]
                representation_str = "%d-%d" % (adaptation_set_id, representation.id)
                
                # 检查流是否已初始化
                if representation_str not in self._representation_initialized:
                    # 下载流初始化段
                    await self.download_manager.download(DownloadRequest(representation.initialization, DownloadType.STREAM_INIT))
                    init_data, _ = await self.download_manager.wait_complete(representation.initialization)
                    # 保存流初始化段到本地文件
                    init_file = tempfile.NamedTemporaryFile(dir=self.run_dir, delete=False)
                    init_file.write(init_data)
                    init_file.close()
                    # 标记该表示已初始化
                    self._representation_initialized.add(representation_str)
                    # 为所有段设置初始化文件路径
                    for segment in representation.segments.values():
                        segment.init_path = init_file.name

                    self.log.info(f"Stream {representation_str} initialized")
                
                try:
                    # 获取当前索引的段
                    segment = representation.segments[self._index]
                except IndexError:
                    # 段已结束
                    self.log.info("Segments ended")
                    self._end = True
                    return
                
                # 记录URL和段对象
                urls[segment.url] = [segment, None]
                # 开始下载段
                await self.download_manager.download(DownloadRequest(segment.url, DownloadType.SEGMENT))
                # duration = segment.duration  # 注释掉的持续时间设置
                
            self.log.info(f"Waiting for completion urls {urls.keys()}")

            # 等待所有下载完成
            for url in urls.keys():
                urls[url][1], _ = await self.download_manager.wait_complete(url)

            self.log.info(f"Completed downloading from urls {urls.keys()}")

            # 处理下载完成的段
            for url in urls.keys():
                segment_data = urls[url][1]
                if segment_data is None:
                    # 下载失败，标记为丢弃
                    self._dropped_index = self._index
                    continue
                # 保存段数据到本地文件
                segment_file = tempfile.NamedTemporaryFile(dir=self.run_dir, delete=False)
                segment_file.write(segment_data)
                segment_file.close()
                # 设置段的本地文件路径
                urls[url][0].path = segment_file.name

            # 收集下载统计信息
            download_stats = {as_id: self.bandwidth_meter.get_stats(segment.url) for as_id, segment in segments.items()}
            # 通知所有监听器段下载完成
            for listener in self.listeners:
                await listener.on_segment_download_complete(self._index, segments, download_stats)

            # 将段加入下载缓冲区
            await self.download_buffer.enqueue(self._index, segments)
            # 如果需要增强处理，将段加入增强缓冲区
            if self.has_enhance(self._current_enhance_actions):
                await self.enhance_buffer.enqueue(self._index, copy.deepcopy(segments))

            # 移动到下一个段
            self._index += 1

    def select_adaptation_sets(self, adaptation_sets: Dict[int, AdaptationSet]):
        """
        选择自适应集 - 根据配置选择要处理的自适应集范围
        
        Args:
            adaptation_sets: 所有可用的自适应集
            
        Returns:
            选中的自适应集字典
        """
        # 获取所有自适应集ID
        as_ids = adaptation_sets.keys()
        # 确定选择范围的起始和结束ID
        start = self.selected_as_start or min(as_ids)
        end = self.selected_as_end or max(as_ids)
        print(f"{start=}, {end=}")
        # 返回在指定范围内的自适应集
        return {as_id: as_val for as_id, as_val in adaptation_sets.items() if as_id >= start and as_id <= end}

    async def stop(self):
        """
        停止调度器 - 清理资源并取消任务
        """
        # 关闭下载管理器
        await self.download_manager.close()
        # 取消主任务
        if self._task is not None:
            self._task.cancel()

    @property
    def is_end(self):
        """
        检查是否已到达播放序列末尾
        
        Returns:
            bool: 是否已结束
        """
        return self._end

    def add_listener(self, listener: SchedulerEventListener):
        """
        添加调度器事件监听器
        
        Args:
            listener: 要添加的监听器对象
        """
        if listener not in self.listeners:
            self.listeners.append(listener)

    async def cancel_task(self, index: int):
        """
        取消当前下载任务并移动到下一个任务
        
        当网络条件恶化或需要快速跳转时调用此方法
        
        Parameters
        ----------
        index: int
            要取消的段索引
        """

        # 如果索引不是当前正在下载的段，忽略请求
        if self._index != index or self._current_download_actions is None:
            return

        # 不取消第一个索引的任务（确保有内容可播放）
        if index == 0:
            return

        # 确保自适应集已初始化
        assert self.adaptation_sets is not None
        # 停止当前正在下载的所有段
        for adaptation_set_id, selection in self._current_download_actions.items():
            segment = self.adaptation_sets[adaptation_set_id].representations[selection].segments[self._index]
            self.log.debug(f"Stop current downloading URL: {segment.url}")
            await self.download_manager.stop(segment.url)

    async def drop_index(self, index):
        """
        标记段索引为丢弃状态
        当段下载失败或质量过低时调用
        
        Args:
            index: 要丢弃的段索引
        """
        self._dropped_index = index

    def has_enhance(self, enhance_actions):
        """
        检查是否有段需要进行增强处理
        
        Args:
            enhance_actions: 增强动作字典
            
        Returns:
            bool: 是否有段需要增强
        """
        # 检查是否有任何自适应集的增强级别不为0
        for as_id in enhance_actions:
            if enhance_actions[as_id] != 0:
                return True
        return False

