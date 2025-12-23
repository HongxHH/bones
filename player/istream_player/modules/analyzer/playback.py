import asyncio
import logging
import sys
from asyncio import create_subprocess_exec
from asyncio.subprocess import PIPE
from typing import Dict

from istream_player.config.config import PlayerConfig
from istream_player.core.analyzer import Analyzer
from istream_player.core.downloader import (DownloadEventListener,
                                            DownloadManager)
from istream_player.core.module import Module, ModuleOption
from istream_player.core.mpd_provider import MPDProvider
from istream_player.core.player import Player, PlayerEventListener
from istream_player.models import Segment, State


class Decoder:
    """
    FFmpeg解码器类
    使用FFmpeg子进程解码MP4视频段，将编码视频转换为原始RGB帧数据
    支持异步解码和帧数据读取
    """
    # 日志记录器 - 用于记录解码器的操作日志
    log = logging.getLogger("FfmpegDecoder")

    def __init__(self, decoded: Dict[str, bytearray]) -> None:
        """
        初始化FFmpeg解码器
        
        Args:
            decoded: 解码数据字典，键为URL，值为解码后的字节数组
        """
        # 存储解码数据字典的引用
        self.decoded = decoded
        # 创建异步队列用于调度解码任务
        self.decode_queue = asyncio.Queue()
        pass

    async def start(self):
        """
        启动FFmpeg解码子进程
        配置FFmpeg参数以从标准输入读取MP4数据并输出RGB32原始视频
        """
        # 创建FFmpeg子进程
        self._proc = await create_subprocess_exec(
            # 注释掉的ffplay命令（备用播放方案）
            # "ffplay", "-", "-loglevel", "quiet", stdin=PIPE
            "ffmpeg",                    # FFmpeg可执行文件
            "-f", "mp4",                 # 输入格式：MP4
            "-i", "-",                   # 输入源：标准输入
            "-pix_fmt", "rgb32",         # 像素格式：RGB32
            "-f", "rawvideo",            # 输出格式：原始视频
            "-loglevel", "error",        # 日志级别：仅错误
            "-",                         # 输出到标准输出
            # "-y",                      # 注释掉的覆盖选项
            # "/tmp/test.y4m",           # 注释掉的测试输出文件
            stdin=PIPE,                  # 标准输入管道
            stdout=PIPE,                 # 标准输出管道
            stderr=sys.stderr,           # 错误输出到标准错误
        )
        # 创建异步读取任务
        self._reader = asyncio.Task(self.read())
        # 初始化已发送字节计数器
        self.total_sent = 0

    # 注释掉的同步读取方法（备用实现）
    # async def stdout_read(self, n: int, buff: bytearray):
    #     count = 0
    #     while count < n:
    #         b = self._proc.stdout.read(n)
    #         bytearray.append()

    async def read(self):
        """
        异步读取解码后的帧数据
        从FFmpeg标准输出读取原始RGB帧数据并存储到解码缓冲区
        """
        # 确保标准输出可用
        assert self._proc.stdout is not None
        # 持续读取解码数据
        while True:
            # 从队列中获取要解码的URL
            url = await self.decode_queue.get()
            self.log.debug(f"Reading decoded bytes for url : {url}")
            # 计算单帧大小（1024x576像素，每像素4字节RGB32）
            frame_size = 1024 * 576 * 4
            # 初始化帧索引
            frame_index = 0
            # 注释掉的缓冲区引用（备用实现）
            # buff = self.decoded[url]
            # 持续读取帧数据
            while True:
                count = 0
                # 读取完整的一帧数据
                while count < frame_size:
                    # 从FFmpeg标准输出读取数据
                    b = await self._proc.stdout.read(frame_size)
                    if not b:
                        # 如果没有数据，说明进程已关闭
                        raise Exception("Debugger process stdout closed")
                    # 注释掉的缓冲区扩展（备用实现）
                    # buff.extend(b)
                    count += len(b)
                # 增加帧计数
                frame_index += 1
                # 记录解码帧信息
                self.log.info(f"***************** Frame decoded: {frame_index}")

    def stop(self):
        """
        停止FFmpeg解码进程
        取消读取任务并关闭进程
        """
        # 确保标准输入可用
        assert self._proc.stdin is not None
        try:
            # 取消读取任务
            self._reader.cancel()
            # 关闭标准输入
            self._proc.stdin.close()
            # 终止进程
            self._proc.terminate()
        except ProcessLookupError:
            # 如果进程已不存在，忽略错误
            pass

    def send(self, buff):
        """
        向FFmpeg进程发送数据
        
        Args:
            buff: 要发送的字节数据
        """
        # 向FFmpeg标准输入写入数据
        self._proc.stdin.write(buff)  # type: ignore
        # 更新已发送字节计数
        self.total_sent += len(buff)
        # 记录发送字节数
        self.log.debug(f"Sent bytes: {self.total_sent=}")

    async def schedule_decode(self, url):
        """
        调度URL的解码任务
        
        Args:
            url: 要解码的段URL
        """
        self.log.debug("--------------- Schedulign deode")
        # 为URL创建解码缓冲区
        self.decoded[url] = bytearray()
        # 将URL加入解码队列
        await self.decode_queue.put(url)


# 播放分析器实现类 - 继承多个接口，实现视频播放和解码分析
@ModuleOption("playback", requires=[Player, "segment_downloader", MPDProvider])
class Playback(Module, Analyzer, PlayerEventListener, DownloadEventListener):
    # 日志记录器 - 用于记录播放分析器的操作日志
    log = logging.getLogger("Playback")

    def __init__(self) -> None:
        """
        初始化播放分析器
        设置文件内容存储和解码器管理
        """
        # 调用父类构造函数
        super().__init__()
        # 文件内容字典 - 存储下载的文件数据，键为URL，值为字节数组
        self.file_content: dict[str, bytearray] = {}
        # 注释掉的ffplay进程（备用播放方案）
        # self.ffplay = Popen(['ffplay'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
        # self.ffplay = None
        # 解码器字典 - 存储每个流的解码器，键为URL，值为Decoder对象
        self.decoders: Dict[str, Decoder] = {}
        # 编码缓冲区 - 存储编码视频数据
        self.encoded_buffer = {}
        # 解码缓冲区 - 存储解码后的视频数据
        self.decoded_buffer = {}

    async def setup(self, config: PlayerConfig, player: Player, segment_downloader: DownloadManager, mpd_provider: MPDProvider):
        """
        设置播放分析器 - 初始化依赖组件和事件监听器
        
        Args:
            config: 播放器配置对象
            player: 播放器实例
            segment_downloader: 段下载管理器
            mpd_provider: MPD提供器
        """
        # 存储MPD提供器引用
        self.mpd_provider = mpd_provider
        # 添加播放器事件监听器
        player.add_listener(self)
        # 添加段下载器事件监听器
        segment_downloader.add_listener(self)

    async def on_transfer_start(self, url) -> None:
        """
        处理传输开始事件
        当开始下载新的URL时，检查是否需要创建新的解码器
        
        Args:
            url: 开始传输的URL
        """
        # 通过URL获取段信息
        segment = self.mpd_provider.segment_by_url(url)
        if segment is None:
            # 初始化URL，需要启动新的解码器进程
            if url not in self.decoders:
                self.log.debug(f"Opening subprocess for stream - {url}")
                # 创建新的解码器实例
                decoder = Decoder(self.decoded_buffer)
                # 存储解码器引用
                self.decoders[url] = decoder
                # 启动解码器
                await decoder.start()
        else:
            # 段URL，不需要特殊处理
            pass

    async def on_bytes_transferred(self, length: int, url: str, position: int, size: int, content) -> None:
        """
        处理字节传输事件
        当接收到下载数据时，将其发送到相应的解码器
        
        Args:
            length: 本次传输的字节数
            url: 传输的URL
            position: 当前流位置
            size: 内容总大小
            content: 传输的内容数据
        """
        # 通过URL获取段信息
        segment = self.mpd_provider.segment_by_url(url)
        # 注释掉的调试输出
        # self.log.debug(f"{self.decoders=}")
        if segment is None:
            # 初始化URL，直接发送到对应的解码器
            self.decoders[url].send(content)
        else:
            # 段URL，发送到对应的初始化流解码器
            decoder = self.decoders[segment.init_url]
            decoder.send(content)
            # 如果段URL尚未在解码缓冲区中，调度解码
            if url not in self.decoded_buffer:
                await decoder.schedule_decode(url)
        pass

    async def on_state_change(self, position: float, old_state: State, new_state: State):
        """
        处理播放状态变化事件
        当播放结束时，停止所有解码器
        
        Args:
            position: 播放位置
            old_state: 旧状态
            new_state: 新状态
        """
        if new_state == State.END:
            # 播放结束，停止所有解码器
            for decoder in self.decoders.values():
                decoder.stop()

    async def on_segment_playback_start(self, segments: Dict[int, Segment]):
        """
        处理段播放开始事件
        当段开始播放时执行的回调函数
        
        Args:
            segments: 播放的段字典
        """
        # 注释掉的段处理逻辑（备用实现）
        # seg_name = segment.url.split("/")[-1]
        # init_name = 'init-' + seg_name.split('-')[1] + ".m4s"
        # seg_path = join(self.run_dir, "downloaded", seg_name)
        # init_path = join(self.run_dir, "downloaded", init_name)
        # self.log.info(f"{init_path} {seg_path}")
        #
        # cat_ps = Popen(['cat', init_path, seg_path], stdout=PIPE)
        # assert self.ffplay is not None and self.ffplay.stdin is not None
        # self.log.info(segment.init_url, len(self.file_content[segment.init_url]))
        # self.log.info(f"{segment.url}, {list(self.file_content.keys())}")
        # self.ffplay.stdin.write(self.file_content[segment.init_url])
        # self.log.info(f"{len(self.file_content[segment.url])=}")
        # Thread(target=self.ffplay.stdin.write, args=[self.file_content[segment.url]]).start()
        # del


