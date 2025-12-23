import asyncio
import logging

from istream_player.config.config import PlayerConfig
# from istream_player.core.buffer import BufferManager
from istream_player.core.module import Module, ModuleOption
from istream_player.core.mpd_provider import MPDProvider
from istream_player.core.player import Player, PlayerEventListener
from istream_player.core.scheduler import Scheduler
from istream_player.models import State
from istream_player.utils.async_utils import critical_task

from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.core.enhancer import Enhancer

# DASH播放器实现类 - 继承自Module和Player接口
# 负责协调各个组件，实现DASH视频流的播放控制
@ModuleOption("dash", default=True, requires=[DownloadBufferImpl, EnhanceBufferImpl, Enhancer, Scheduler, MPDProvider])
class DASHPlayer(Module, Player):
    # 日志记录器 - 用于记录播放器的操作日志
    log = logging.getLogger("DASHPlayer")

    def __init__(self):
        """
        初始化DASH播放器
        设置播放器状态和播放相关的状态变量
        """
        # 调用父类构造函数
        super().__init__()

        # 状态相关变量
        # 当前播放器状态 - 初始为IDLE（空闲状态）
        self._state = State.IDLE

        # 播放相关变量
        # 播放是否已开始的标志
        self._playback_started = False
        # 当前播放位置（时间戳，单位：秒）
        self._position = 0.0

    async def setup(self,
                    config: PlayerConfig,
                    download_buffer: DownloadBufferImpl,
                    enhance_buffer:EnhanceBufferImpl,
                    enhancer: Enhancer,
                    scheduler: Scheduler,
                    mpd_provider: MPDProvider, **kwargs
    ):
        """
        设置播放器 - 初始化所有依赖组件和配置参数
        
        Args:
            config: 播放器配置对象
            download_buffer: 下载缓冲区实现
            enhance_buffer: 增强缓冲区实现
            enhancer: 视频增强器
            scheduler: 调度器
            mpd_provider: MPD提供器
            **kwargs: 其他可选参数
        """
        # 注释掉的缓冲相关配置（可能在未来版本中使用）
        # self.min_start_buffer_duration = config.min_start_duration
        # self.min_rebuffer_duration = config.min_rebuffer_duration
        
        # 从配置中提取时间因子（播放速度倍数）
        self.time_factor = config.time_factor

        # 存储所有依赖组件的引用
        self.download_buffer = download_buffer  # 下载缓冲区
        self.enhance_buffer = enhance_buffer    # 增强缓冲区
        self.enhancer = enhancer                # 视频增强器
        self.scheduler = scheduler              # 调度器
        self.mpd_provider = mpd_provider        # MPD提供器

    @property
    def state(self) -> State:
        """
        获取当前播放器状态
        
        Returns:
            State: 当前播放器状态
        """
        return self._state

    async def _switch_state(self, old_state: State, new_state: State):
        """
        切换播放器状态 - 内部状态切换方法
        通知所有监听器状态变化事件
        
        Args:
            old_state: 旧状态
            new_state: 新状态
        """
        # 通知所有监听器状态变化
        for listener in self.listeners:
            await listener.on_state_change(self._position, old_state, new_state)

    def stop(self) -> None:
        """
        停止播放器 - 停止播放并重置所有状态
        注意：此方法在当前实现中未完成，抛出NotImplementedError
        """
        raise NotImplementedError

    def pause(self) -> None:
        """
        暂停播放器 - 暂停播放
        注意：此方法在当前实现中未完成，抛出NotImplementedError
        """
        raise NotImplementedError

    def add_listener(self, listener: PlayerEventListener):
        """
        添加播放器事件监听器
        
        Args:
            listener: 要添加的监听器对象
        """
        if listener not in self.listeners:
            self.listeners.append(listener)

    @critical_task()
    async def run(self):
        """
        播放器主运行循环
        此方法协调不同组件之间的工作，实现完整的播放流程
        """
        # 等待MPD可用
        await self.mpd_provider.available()
        # 启动调度器
        self._state = State.BUFFERING  # 切换到缓冲状态
        # 确保MPD已加载
        assert self.mpd_provider.mpd is not None
        # 记录首次开始时间（用于计算播放位置）
        first_start_time = None
        # 通知状态变化：进入缓冲状态
        await self._switch_state(self._state, State.BUFFERING)

        # 主播放循环 - 持续运行直到播放结束
        while self._state != State.END:
            # 从下载缓冲区获取下一个要播放的段
            index, segments = await self.download_buffer.dequeue()
            
            # 如果当前状态是缓冲，切换到就绪状态
            if self.state == State.BUFFERING:
                await self._switch_state(self._state, State.READY)

            # 记录正在播放的段索引
            self.log.info(f"Playing index: {index}")

            # 计算首次开始时间（用于确定播放起始点）
            if first_start_time is None:
                first_start_time = min(map(lambda s: s.start_time, segments.values()))

            # 更新当前播放位置为当前段组的开始时间
            self._position = min(map(lambda s: s.start_time, segments.values()))

            # 通知所有监听器播放位置变化
            for listener in self.listeners:
                await listener.on_position_change(self._position)
            # 确保状态为就绪
            await self._switch_state(self._state, State.READY)

            # 通知所有监听器段播放开始
            for listener in self.listeners:
                await listener.on_segment_playback_start(segments)

            # 计算当前段组的最大持续时间
            duration = max(map(lambda s: s.duration, segments.values()))
            # 更新播放位置（加上段持续时间）
            self._position += duration
            # 再次通知播放位置变化
            for listener in self.listeners:
                await listener.on_position_change(self._position)

            # 为下一轮更新做准备
            # 检查下载缓冲区是否为空
            if self.download_buffer.is_empty():
                # 如果调度器已结束，则播放结束
                if self.scheduler.is_end:
                    # 切换到结束状态
                    await self._switch_state(self._state, State.END)
                    self._state = State.END
                    # 清理所有缓冲区
                    await self.download_buffer.cleanup()
                    await self.enhance_buffer.cleanup()
                    await self.enhancer.cleanup()
                    # 记录播放器关闭日志
                    self.log.info("Player closed")
                    return
                else:
                    # 如果调度器未结束但缓冲区为空，进入缓冲状态
                    await self._switch_state(self._state, State.BUFFERING)
                    self._state = State.BUFFERING
            else:
                # 如果缓冲区不为空，保持就绪状态
                await self._switch_state(self._state, State.READY)