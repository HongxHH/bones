import logging
import time
from collections import deque

from istream_player.config.config import PlayerConfig
from istream_player.core.bw_meter import (BandwidthMeter,
                                          BandwidthUpdateListener)
from istream_player.core.downloader import (DownloadEventListener,
                                            DownloadManager)
from istream_player.core.module import Module, ModuleOption


# 连续带宽测量器实现类 - 继承自Module、BandwidthMeter和DownloadEventListener接口
# 负责实时测量和更新网络带宽，提供连续带宽更新功能
@ModuleOption("bw_cont", requires=["segment_downloader"])
class ContinuousBandwidthMeter(Module, BandwidthMeter, DownloadEventListener):
    # 获取日志记录器实例，用于记录带宽测量相关的日志信息
    log = logging.getLogger("ContinuousBandwidthMeter")

    def __init__(self):
        # 调用父类构造函数，初始化模块基础功能
        super().__init__()

        # 上次接收字节的时间戳，用于计算连续带宽
        self.last_byte_at = 0
        # 当前传输会话中已传输的字节总数
        self.bytes_transferred = 0
        # 传输开始时间戳，用于计算总传输时间
        self.transmission_start_time = None
        # 传输结束时间戳，用于计算总传输时间
        self.transmission_end_time = None
        # 额外的统计信息字典，用于存储带宽计算的中间结果
        self.extra_stats = {}
        # 标志位：是否为当前段中的第一个字节
        # 用于区分段开始和段内连续传输
        self.first_byte_in_segment = True
        # 滑动窗口样本：(开始时间, 结束时间, 字节数)。窗口以外的旧样本会丢掉。
        self._cont_bw: deque = deque()
        # 最新的连续带宽估计值（bps），用于实时带宽更新
        self.last_cont_bw = None
        # 当前正在下载的URL，用于跟踪下载状态
        self.downloading_url = None

    # 设置方法 - 初始化带宽测量器的配置参数和依赖组件
    async def setup(self, config: PlayerConfig, segment_downloader: DownloadManager, **kwargs):
        # 从配置中获取初始带宽估计值，作为带宽测量的起始点
        self._bw = config.static.max_initial_bitrate
        # 从配置中获取平滑因子，用于带宽估计的指数移动平均
        # 平滑因子越小，对最新测量值的权重越大
        self.smooth_factor = config.static.smoothing_factor
        # 从配置中获取最大数据包延迟阈值，用于过滤异常测量值
        self.max_packet_delay = config.static.max_packet_delay
        # 从配置中获取连续带宽测量窗口大小（秒），用于滑动窗口计算
        self.cont_bw_window = config.static.cont_bw_window

        # 将当前实例注册为下载管理器的事件监听器
        # 这样就能接收下载过程中的各种事件通知
        segment_downloader.add_listener(self)

    # 传输开始事件处理器 - 当开始下载新的视频段时调用
    async def on_transfer_start(self, url) -> None:
        # 记录传输开始时间戳，用于后续计算总传输时间
        self.transmission_start_time = time.time()
        # 重置当前传输会话的字节计数器
        self.bytes_transferred = 0
        # 标记为段内第一个字节，用于连续带宽计算的初始化
        self.first_byte_in_segment = True
        # 记录当前下载的URL，用于状态跟踪
        self.downloading_url = url
        # 记录传输开始的日志信息，包含URL用于调试
        self.log.info("Transmission starts. URL: " + url)

    # 字节传输事件处理器 - 当接收到新的数据字节时调用
    # 这是连续带宽测量的核心方法，每次接收数据都会触发
    async def on_bytes_transferred(self, length: int, url: str, position: int, size: int, content) -> None:
        # 注释掉的URL检查：原本只处理当前下载URL的数据
        # 现在处理所有URL的数据，以支持多流并发下载
        # if url == self.downloading_url:
        # 累加已传输的字节数，用于计算总传输量
        self.bytes_transferred += length
        # 获取当前时间戳，用于连续带宽计算
        t = time.time()
        # 调用连续带宽更新方法，传入字节数和时间戳
        await self.update_cont_bw(length, t)

    # 传输结束事件处理器 - 当视频段下载完成时调用
    async def on_transfer_end(self, size: int, url: str) -> None:
        # 记录传输结束时间戳，用于计算总传输时间
        self.transmission_end_time = time.time()
        # 调用带宽更新方法，基于完整传输计算最终带宽
        self.update_bandwidth()
        # 重置字节计数器，为下次传输做准备
        self.bytes_transferred = 0

        # 通知所有带宽更新监听器，传递最新的带宽估计值
        for listener in self.listeners:
            await listener.on_bandwidth_update(self._bw)

    # 注释掉的传输取消事件处理器 - 当前未实现
    # 如果实现，应该在传输被取消时调用传输结束处理
    # async def on_transfer_canceled(self, url: str, position: int, size: int) -> None:
    #     return await self.on_transfer_end(position, url)

    # 带宽属性访问器 - 返回当前的带宽估计值
    @property
    def bandwidth(self) -> float:
        return self._bw

    # 连续带宽更新方法 - 实时计算和更新带宽估计
    # 这是连续带宽测量的核心算法，使用滑动窗口方法
    async def update_cont_bw(self, bytes_transferred: int, time_at: float):
        # 设置最小测量值数量，确保带宽计算的可靠性
        min_values = 2
        # 检查是否为段内第一个字节
        if self.first_byte_in_segment:
            # 如果是第一个字节，标记为非第一个字节，但不进行带宽计算
            # 因为需要至少两个时间点才能计算带宽
            self.first_byte_in_segment = False
        else:
            self._cont_bw.append((self.last_byte_at, time_at, bytes_transferred))
            if len(self._cont_bw) >= min_values:
                window_start = time_at - self.cont_bw_window
                # Drop samples that fell out of the window. Keep at least
                # min_values so a sparse download still has a mean.
                while len(self._cont_bw) > min_values and self._cont_bw[0][1] < window_start:
                    self._cont_bw.popleft()
                total_bytes = 0
                total_time = 0.0
                for start, end, nbytes in self._cont_bw:
                    total_bytes += nbytes
                    total_time += end - start
                if total_time > 0:
                    self.last_cont_bw = int(8 * total_bytes / total_time)
        # 如果存在连续带宽估计值，通知所有监听器
        if self.last_cont_bw is not None:
            for listener in self.listeners:
                # 调用监听器的连续带宽更新方法
                # 注意：这个方法在基类中被注释掉了，需要子类实现
                await listener.on_continuous_bw_update(self.last_cont_bw)
        # 更新上次接收字节的时间戳，为下次计算做准备
        self.last_byte_at = time_at

    # 带宽更新方法 - 基于完整传输计算最终带宽估计
    def update_bandwidth(self):
        # 断言确保传输开始和结束时间都已设置
        assert self.transmission_end_time is not None and self.transmission_start_time is not None
        # 使用指数移动平均更新带宽估计
        # 公式：new_bw = old_bw * smooth_factor + measured_bw * (1 - smooth_factor)
        # 其中measured_bw = 8 * bytes / time，将字节转换为位并除以时间
        self._bw = self._bw * self.smooth_factor + (8 * self.bytes_transferred) / (
            self.transmission_end_time - self.transmission_start_time
        ) * (1 - self.smooth_factor)
        # 注释掉的调试输出 - 用于开发时的带宽更新日志
        # print(f"Bandwith updated : {self._bw}")
        # 注释掉的连续带宽融合方法 - 将连续带宽和段带宽进行平均
        # if self.last_cont_bw is not None:
        #     self._bw = (self._bw + self.last_cont_bw)/2
        # 注释掉的纯连续带宽方法 - 直接使用连续带宽估计
        # self._bw = self.last_cont_bw
        # 构建额外的统计信息字典，用于调试和分析
        self.extra_stats = {
            "_bw": self._bw,  # 当前带宽估计值
            "smooth_factor": self.smooth_factor,  # 平滑因子
            "bytes_transferred": self.bytes_transferred,  # 传输字节数
            "transmission_end_time": self.transmission_end_time,  # 传输结束时间
            "transmission_start_time": self.transmission_start_time,  # 传输开始时间
        }
        # 注释掉的详细统计日志 - 用于开发时的详细调试信息
        # self.log.info(f"************* Updated stats : {self.extra_stats}")

    # 添加监听器方法 - 将带宽更新监听器添加到监听器列表
    def add_listener(self, listener: BandwidthUpdateListener):
        # 检查监听器是否已存在，避免重复添加
        if listener not in self.listeners:
            # 将监听器添加到监听器列表
            self.listeners.append(listener)
