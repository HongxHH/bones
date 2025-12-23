import logging
import time
from asyncio import Task
from typing import Dict, Optional

from istream_player.config.config import PlayerConfig
from istream_player.core.downloader import (DownloadManager, DownloadRequest,
                                            DownloadType)
from istream_player.core.module import Module, ModuleOption
from istream_player.core.mpd_provider import MPDProvider
from istream_player.models.mpd_objects import MPD, Segment
from istream_player.modules.mpd.parser import DefaultMPDParser
from istream_player.utils.async_utils import AsyncResource, critical_task


# MPD提供器实现类 - 继承自Module和MPDProvider接口
# 负责下载、解析和管理DASH媒体演示描述（MPD）文件，为播放器提供媒体流信息
@ModuleOption("mpd", default=True, requires=["mpd_downloader"])
class MPDProviderImpl(Module, MPDProvider):
    # 获取日志记录器实例，用于记录MPD提供器相关的日志信息
    log = logging.getLogger("MPDProviderImpl")

    def __init__(self):
        # 调用父类构造函数，初始化模块基础功能
        super().__init__()
        # MPD解析器实例，用于解析下载的MPD文件内容
        self.parser = DefaultMPDParser()
        # 上次更新时间戳，用于控制MPD文件的更新频率
        self.last_updated = 0

        # 异步资源对象，用于管理MPD对象的异步访问
        # 使用AsyncResource确保MPD对象在可用时能被正确获取
        self._mpd_res: AsyncResource[Optional[MPD]] = AsyncResource(None)
        # 段URL到段对象的映射字典，用于快速查找段信息
        # 键为段URL，值为对应的段对象（初始化段为None）
        self._segments_by_url: Dict[str, Optional[Segment]] = {}
        # 后台任务对象，用于管理MPD更新任务
        self._task: Optional[Task] = None
        # 注释掉的表示质量映射字典 - 用于将表示ID映射到质量级别
        # self._repr_quality: Dict[int, int] = {}

    # 设置方法 - 初始化MPD提供器的配置参数和依赖组件
    async def setup(self, config: PlayerConfig, mpd_downloader: DownloadManager, **kwargs):
        # 从配置中获取MPD更新间隔（秒），用于控制动态MPD的更新频率
        self.update_interval = config.static.update_interval
        # 保存MPD下载管理器实例的引用，用于下载MPD文件
        self.download_manager = mpd_downloader
        # 从配置中获取MPD文件的URL，这是播放器的主要输入源
        self.mpd_url = config.input

    # MPD属性访问器 - 返回当前的MPD对象
    @property
    def mpd(self) -> Optional[MPD]:
        return self._mpd_res.value

    # 注释掉的表示质量映射方法 - 用于将表示ID转换为质量级别
    # def repr_to_quality(self, repr: int):
    #     return self._repr_quality[repr]

    # 等待MPD可用的方法 - 异步等待MPD对象可用并返回
    async def available(self) -> MPD:
        # 异步等待MPD对象变为非None状态
        value = await self._mpd_res.value_non_none()
        # 断言确保MPD对象不为None
        assert value is not None
        return value

    # 根据URL获取段对象的方法 - 用于快速查找段信息
    def segment_by_url(self, url: str) -> Optional[Segment]:
        return self._segments_by_url[url]

    # 更新MPD文件的方法 - 使用critical_task装饰器确保关键任务执行
    @critical_task()
    async def update(self):
        # 检查是否需要更新MPD文件
        # 如果MPD已存在且距离上次更新时间小于更新间隔，则跳过更新
        if self.mpd is not None and (time.time() - self.last_updated) < self.update_interval:
            return
        
        # 下载MPD文件
        # 使用MPD下载类型，并设置save=True保存到本地
        await self.download_manager.download(DownloadRequest(self.mpd_url, DownloadType.MPD), save=True)
        # 等待下载完成并获取文件内容
        content, size = await self.download_manager.wait_complete(self.mpd_url)
        # 将字节内容解码为UTF-8文本
        text = content.decode("utf-8")

        # 打印MPD URL用于调试
        print("mpd url: ", self.mpd_url)

        # 解析MPD文件内容
        # 使用解析器将XML文本转换为MPD对象
        mpd = self.parser.parse(text, url=self.mpd_url)
        # 更新异步资源中的MPD对象
        self._mpd_res.value = mpd
        
        # 构建段URL到段对象的映射
        # 遍历所有自适应集
        for adap_set in mpd.adaptation_sets.values():
            # 遍历当前自适应集的所有表示
            for repr in adap_set.representations.values():
                # 遍历当前表示的所有段
                for seg in repr.segments.values():
                    # 将段URL映射到段对象
                    self._segments_by_url[seg.url] = seg
                    # 将初始化段URL映射到None（初始化段不需要段对象）
                    self._segments_by_url[seg.init_url] = None

        # 更新最后更新时间戳
        self.last_updated = time.time()

    # 注释掉的重复更新方法 - 用于动态MPD的持续更新
    # @critical_task()
    # async def update_repeatedly(self):
    #     # 断言确保MPD对象已存在
    #     assert self._mpd is not None
    #     # 如果MPD类型为动态，持续更新
    #     while self._mpd.type == "dynamic":
    #         await self.update()
    #         await asyncio.sleep(self.update_interval)
    #     # 记录MPD类型变化
    #     self.log.info(f"MPD file changed from dynamic to {self._mpd.type}")

    # 运行方法 - 启动MPD提供器并下载初始MPD文件
    async def run(self):
        # 断言确保MPD URL已设置
        assert self.mpd_url is not None
        # 执行初始MPD更新
        await self.update()
        # 断言确保MPD对象已成功获取
        assert self.mpd is not None

    # 停止方法 - 清理MPD提供器资源
    async def stop(self):
        # 记录停止信息到日志
        self.log.info("Stopping MPD Provider")
        # 如果存在后台任务，取消它
        if self._task is not None:
            self._task.cancel()
        # 关闭下载管理器，释放相关资源
        await self.download_manager.close()
