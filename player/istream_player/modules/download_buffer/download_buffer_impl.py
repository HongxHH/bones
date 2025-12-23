import asyncio
import logging

from typing import Dict, Tuple, Optional

from istream_player.config.config import PlayerConfig
from istream_player.core.buffer import BufferManager
from istream_player.core.module import Module, ModuleOption
from istream_player.models.mpd_objects import Segment

from queue import PriorityQueue
import math
from istream_player.core.renderer import Renderer


# 下载缓冲区实现类 - 继承自Module和BufferManager接口
# 负责存储已下载的视频段，并提供队列操作和段替换功能
@ModuleOption("download_buffer", requires=[Renderer], default=True)
class DownloadBufferImpl(Module, BufferManager):
    """
    存储已下载段的缓冲区
    除了队列操作外，此缓冲区还允许用增强版本替换已下载的段
    """
    def __init__(self) -> None:
        """
        初始化下载缓冲区
        设置所有必要的状态变量和数据结构
        """
        # 调用父类构造函数
        super().__init__()
        # 日志记录器 - 用于记录缓冲区的操作日志
        self.log = logging.getLogger("DownloadBuffer")

        # 当前缓冲水位（秒）- 表示缓冲区中存储的视频时长
        self._buffer_level: float = 0
        # 段索引优先级队列 - 按段索引顺序管理待播放的段
        self._indices = PriorityQueue()  # priority queue of segment indices
        # 段字典 - 存储段数据，键为段索引，值为自适应集ID到段对象的映射
        self._segments: Dict[int, Dict[int, Segment]] = dict()  # dictionary of (segment index, segments)
        # 异步条件变量 - 用于线程安全的等待和通知机制
        self._accessible: asyncio.Condition = asyncio.Condition()
        # 最大缓冲水位 - 防止缓冲区溢出，初始设为无穷大
        self._max_buffer_level = math.inf
        # 当前正在播放的段索引
        self._current_segment_index: Optional[int] = None

        # 缓冲区结束标志 - 表示缓冲区是否已关闭
        self._is_end = False
        # 下一个要播放的段索引 - 用于快速查询，避免频繁操作队列
        self._next_render_segment_index: Optional[int] = None

    async def publish_buffer_level(self):
        """
        发布缓冲水位变化事件
        通知所有监听器当前缓冲水位的变化
        """
        # 通知所有监听器缓冲水位变化
        for listener in self.listeners:
            await listener.on_buffer_level_change(self._buffer_level)

    async def setup(self, config: PlayerConfig, renderer: Renderer, **kwargs):
        """
        设置下载缓冲区 - 初始化配置和依赖组件
        
        Args:
            config: 播放器配置对象
            renderer: 渲染器组件
            **kwargs: 其他可选参数
        """
        # 从配置中设置最大缓冲持续时间
        self._max_buffer_level = config.buffer_duration

        self._segment_duration = 4.0
        # 存储渲染器引用
        self.renderer = renderer

    async def run(self) -> None:
        """
        运行缓冲区 - 发布初始缓冲水位
        """
        # 发布当前缓冲水位
        await self.publish_buffer_level()

    async def enqueue(self, index:int, segments: Dict[int, Segment]) -> None:
        """
        将段加入缓冲区队列
        在缓冲区有足够空间时添加新的段，否则等待空间释放
        
        Args:
            index: 段索引
            segments: 段字典，键为自适应集ID，值为段对象
        """
        # 使用异步条件变量确保线程安全
        async with self._accessible:
            # 计算当前段组的最大持续时间
            max_duration = max(map(lambda s: s[1].duration, segments.items()))

            # 防止缓冲区溢出 - 如果添加新段会超过最大缓冲水位，则等待
            while self._buffer_level + max_duration > self._max_buffer_level:
                await self._accessible.wait()

            # 将段索引加入优先级队列
            self._indices.put(index)
            # 存储段数据
            self._segments[index] = segments
            

            
            # 更新缓冲水位
            self._buffer_level += max_duration
            # 发布缓冲水位变化事件
            await self.publish_buffer_level()
            # 通知等待的线程
            self._accessible.notify_all()

    async def dequeue(self) -> Tuple[int, Dict[int, Segment]]:
        """
        从缓冲区队列中取出段
        按索引顺序取出段，如果缓冲区为空则等待
        
        Returns:
            Tuple[int, Dict[int, Segment]]: (段索引, 段字典) 或 (None, None) 如果缓冲区已关闭
        """
        # 使用异步条件变量确保线程安全
        async with self._accessible:
            # 防止缓冲区下溢 - 如果缓冲区为空，则等待
            while self._indices.empty():
                await self._accessible.wait()

                # 如果缓冲区已关闭，返回None
                if self._is_end:
                    self.log.info("Download buffer closed.")
                    return None, None

            # 从优先级队列中取出段索引
            index = self._indices.get()

            self._current_segment_index = index

            # 获取对应的段数据
            segments = self._segments[index]
            # 从字典中删除该段
            del self._segments[index]
            
            # 更新下一个要渲染的段索引
            self._next_render_segment_index = index + 1
                
            # 计算段组的最大持续时间
            max_duration = max(map(lambda s: s[1].duration, segments.items()))
            # 更新缓冲水位
            self._buffer_level -= max_duration
            # 发布缓冲水位变化事件
            await self.publish_buffer_level()
            # 通知等待的线程
            self._accessible.notify_all()
            # 返回段索引和段数据
            return index, segments

    async def replace(self, index: int, segments: Dict[int, Segment]) -> None:
        """
        替换缓冲区中的段
        用增强版本的段替换已存在的段
        
        Args:
            index: 要替换的段索引
            segments: 新的段数据
        """
        # 使用异步条件变量确保线程安全
        async with self._accessible:
            # 检查段是否存在于缓冲区中
            if index not in self._segments:
                # 如果段不存在，中止替换操作
                self.log.info(f"Replacement failure. Segment {index} not found in buffer")
                return
            # 替换段数据
            self._segments[index] = segments
            # 通知等待的线程
            self._accessible.notify_all()

    def buffer_level(self, continuous: bool = False) -> float:
        """
        获取当前缓冲水位
        
        Args:
            continuous: 是否包含渲染器中的剩余任务
            
        Returns:
            float: 当前缓冲水位（秒）
        """
        if continuous:
            # 如果请求连续缓冲水位，包含渲染器中的剩余任务
            return self._buffer_level + self.renderer.remain_task()
        # 返回基本缓冲水位
        return self._buffer_level

    def renderer_remain_task_level(self) -> float:
        """
        获取渲染器剩余任务缓冲水位

        Returns:
            float: 渲染器剩余任务缓冲水位（秒）
        """
        return self.renderer.remain_task()

    # 传入一个段的index，获取该段到和目前正在播放片段之间的剩余缓冲
    def get_remaining_buffer(self, index: int) -> float:
        """
        获取该段到和目前正在播放片段之间的剩余缓冲
        """
        return self._buffer_level + self.renderer.remain_task() - (index - self._current_segment_index) * self._segment_duration

    def is_empty(self) -> bool:
        """
        检查缓冲区是否为空
        
        Returns:
            bool: 如果缓冲区为空返回True，否则返回False
        """
        return self._indices.empty()

    def get_next_render_segment_index(self) -> Optional[int]:
        """
        获取下一个要播放的段索引

        Returns:
            Optional[int]: 下一个段索引
        """

        return self._next_render_segment_index

    async def cleanup(self) -> None:
        """
        清理缓冲区 - 关闭缓冲区并通知所有等待的线程
        """
        # 使用异步条件变量确保线程安全
        async with self._accessible:
            # 设置结束标志
            self._is_end = True
            # 通知所有等待的线程
            self._accessible.notify_all()
        return
