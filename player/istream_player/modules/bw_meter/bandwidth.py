import logging
import time
from typing import Dict

from istream_player.config.config import PlayerConfig
from istream_player.core.bw_meter import BandwidthMeter, DownloadStats
from istream_player.core.downloader import DownloadEventListener, DownloadManager
from istream_player.core.module import Module, ModuleOption
from istream_player.core.scheduler import Scheduler, SchedulerEventListener
from istream_player.models.mpd_objects import Segment


# 标准带宽测量器实现类 - 继承自Module、BandwidthMeter、DownloadEventListener和SchedulerEventListener接口
# 负责基于段级下载完成事件进行带宽测量，提供段级带宽估计功能
@ModuleOption("bw_meter", default=True, requires=["segment_downloader", Scheduler])
class BandwidthMeterImpl(Module, BandwidthMeter, DownloadEventListener, SchedulerEventListener):
    # 获取日志记录器实例，用于记录带宽测量相关的日志信息
    log = logging.getLogger("BandwidthMeterImpl")

    def __init__(self):
        # 调用父类构造函数，初始化模块基础功能
        super().__init__()
        # 下载统计信息字典，以URL为键存储每个下载任务的统计信息
        # 用于跟踪每个视频段的下载进度和性能数据
        self.stats: Dict[str, DownloadStats] = {}
        # 总下载字节数，用于计算整体带宽
        # 累计所有下载任务的字节数，用于段级带宽计算
        self.total_bytes = 0
        # 开始时间戳，用于计算总下载时间
        # 记录第一个下载任务开始的时间，用于整体带宽计算
        self.start_time = 0

    # 设置方法 - 初始化带宽测量器的配置参数和依赖组件
    async def setup(self, config: PlayerConfig, segment_downloader: DownloadManager, scheduler: Scheduler):
        # 从配置中获取初始带宽估计值，作为带宽测量的起始点
        # 这个值通常设置为一个保守的估计值，避免初始带宽过高
        self._bw = config.static.max_initial_bitrate
        # 从配置中获取平滑因子，用于带宽估计的指数移动平均
        # 平滑因子控制历史带宽和当前测量值的权重平衡
        self.smooth_factor = config.static.smoothing_factor
        # 将当前实例注册为下载管理器的事件监听器
        # 这样就能接收下载过程中的各种事件通知
        segment_downloader.add_listener(self)
        # 将当前实例注册为调度器的事件监听器
        # 这样就能接收段下载完成的事件通知，触发带宽更新
        scheduler.add_listener(self)

    # 带宽属性访问器 - 返回当前的带宽估计值
    @property
    def bandwidth(self) -> float:
        return self._bw

    # 传输开始事件处理器 - 当开始下载新的视频段时调用
    async def on_transfer_start(self, url) -> None:
        # 检查是否为第一个下载任务，如果是则记录开始时间
        # 这个时间用于计算整体下载时间，用于段级带宽计算
        if self.start_time == 0:
            self.start_time = time.time()
        # 为当前URL创建下载统计信息对象，记录传输开始时间
        # 使用DownloadStats数据类存储详细的下载统计信息
        self.stats[url] = DownloadStats(start_time=time.time())

    # 传输结束事件处理器 - 当视频段下载完成时调用
    async def on_transfer_end(self, size: int, url: str) -> None:
        # 获取当前URL对应的下载统计信息
        stats = self.stats.get(url)
        # 如果统计信息不存在，直接返回（避免异常情况）
        if stats is None:
            return
        # 记录传输结束时间戳，用于计算该段的下载时间
        stats.stop_time = time.time()
        # 如果stopped_bytes字段不为None，则更新停止时的字节数
        # 这个字段用于记录传输被中断时的字节数
        if stats.stopped_bytes is not None:
            stats.stopped_bytes = size

    # 字节传输事件处理器 - 当接收到新的数据字节时调用
    # 这个方法用于实时跟踪下载进度和更新统计信息
    async def on_bytes_transferred(self, length: int, url: str, position: int, size: int, content: bytes) -> None:
        # 获取当前URL对应的下载统计信息
        stats = self.stats.get(url)
        # 如果统计信息不存在，直接返回（避免异常情况）
        if stats is None:
            return
        # 累加总下载字节数，用于整体带宽计算
        self.total_bytes += length
        # 累加该段已接收的字节数，用于段级统计
        stats.received_bytes += length
        # 更新该段的总字节数，用于计算下载完成度
        stats.total_bytes = size
        # 检查是否为该段的第一个字节
        if stats.first_byte_at is None:
            # 如果是第一个字节，记录第一个字节的接收时间
            stats.first_byte_at = time.time()
            # 同时将最后字节时间设置为第一个字节时间
            stats.last_byte_at = stats.first_byte_at
        else:
            # 如果不是第一个字节，更新最后字节的接收时间
            stats.last_byte_at = time.time()

    # 传输取消事件处理器 - 当视频段下载被取消时调用
    async def on_transfer_canceled(self, url: str, position: int, size: int) -> None:
        # 获取当前URL对应的下载统计信息
        stats = self.stats.get(url)
        # 如果统计信息不存在，直接返回（避免异常情况）
        if stats is None:
            return
        # 将已接收的字节数设置为停止字节数
        # 这表示传输在position位置被取消
        stats.stopped_bytes = stats.received_bytes
        # 记录传输停止时间戳，用于计算实际下载时间
        stats.stop_time = time.time()

    # 获取统计信息方法 - 返回指定URL的下载统计信息
    def get_stats(self, url: str) -> DownloadStats:
        return self.stats[url]

    # 段下载完成事件处理器 - 当调度器完成一个段的下载时调用
    # 这是标准带宽测量器的核心方法，基于段级完成事件进行带宽计算
    async def on_segment_download_complete(self, index: int, segments: Dict[int, Segment], stats: Dict[int, DownloadStats]):
        # 计算当前段的带宽估计值
        # 公式：curr_bw = 8 * total_bytes / total_time
        # 乘以8将字节转换为位，除以总时间得到每秒位数
        curr_bw = 8 * self.total_bytes / (time.time() - self.start_time)
        # 使用指数移动平均更新带宽估计
        # 公式：new_bw = old_bw * smooth_factor + curr_bw * (1 - smooth_factor)
        # 平滑因子控制历史带宽和当前测量值的权重平衡
        self._bw = self._bw * self.smooth_factor + curr_bw * (1 - self.smooth_factor)
        # 通知所有带宽更新监听器，传递最新的带宽估计值
        for listener in self.listeners:
            await listener.on_bandwidth_update(self._bw)

        # 清理上一个段的统计信息，为下一个段的下载做准备
        # 清空下载统计信息字典，释放内存
        self.stats.clear()
        # 重置总下载字节数，为下一个段重新开始计数
        self.total_bytes = 0
        # 重置开始时间，为下一个段重新开始计时
        self.start_time = 0
