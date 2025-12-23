import asyncio
import logging
import os
import time
import threading
from datetime import datetime
from fileinput import filename
from typing import Dict, Optional
import queue
import cv2
import numpy as np
import PyNvCodec as nvc


from istream_player.config.config import PlayerConfig
from istream_player.core.renderer import Renderer
from istream_player.core.module import Module, ModuleOption
from istream_player.core.player import Player, PlayerEventListener
from istream_player.models import Segment, State
from istream_player.modules.decoder import DecoderNvCodec
from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.core.bw_meter import BandwidthMeter


class FPSLogger:
    def __init__(self, interval=1):
        # 日志记录间隔（秒），默认每秒记录一次
        self.interval = interval
        # 帧计数器，用于统计在间隔时间内的帧数
        self.framecount = 0
        # 上次记录时间戳，用于计算时间间隔
        self.seconds = time.time()

        self.fps = 0

    def log(self, titlebar=True, fmt="fps : {0}"):
        # 增加帧计数
        self.framecount += 1
        # 检查是否到达记录间隔
        if self.seconds + self.interval < time.time():
            # 计算FPS（帧数除以时间间隔）
            self.fps = self.framecount / self.interval
            # 重置帧计数
            self.framecount = 0
            # 更新记录时间戳
            self.seconds = time.time()
            # 根据参数决定输出方式
            if titlebar:
                return fmt.format(self.fps)
            else:
                return fmt.format(self.fps)
        return None


class FrameRecorder:
    log = logging.getLogger("FrameRecorder")

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        output_path: str,
        file_name: str,
        codec: str = "mp4v",
        max_queue: int = 120,
    ) -> None:
        self.width = width
        self.height = height
        self.fps = fps
        self.output_path = output_path
        self.output_file_name = file_name 
        self.codec = codec
        self.queue: queue.Queue = queue.Queue(max_queue)
        self.stop_event = threading.Event()
        self.writer = None
        self.thread: Optional[threading.Thread] = None

        self._init_writer()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()
        self.log.info(
            "Frame recorder started: %s (%dx%d @ %.2ffps)",
            self.output_path,
            self.width,
            self.height,
            self.fps,
        )

    def _init_writer(self) -> None:
        os.makedirs(self.output_path, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        output_file_name = os.path.join(self.output_path, self.output_file_name)
        self.writer = cv2.VideoWriter(
            output_file_name,
            fourcc,
            self.fps,
            (self.width, self.height),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to create video writer at {self.output_path}")

    def submit(self, frame: np.ndarray) -> None:
        if self.stop_event.is_set():
            return
        try:
            self.queue.put_nowait(frame.copy())
        except queue.Full:
            self.log.warning("Frame recorder queue full, dropping a frame")

    def _worker(self) -> None:
        while not self.stop_event.is_set() or not self.queue.empty():
            try:
                frame = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self.writer.write(frame)
            except Exception as exc:
                self.log.error("Failed to write frame: %s", exc)
            finally:
                self.queue.task_done()

    def close(self) -> None:
        if self.stop_event.is_set():
            return
        self.stop_event.set()
        self.queue.join()
        if self.thread is not None:
            self.thread.join()
        if self.writer is not None:
            self.writer.release()
        self.log.info("Frame recorder stopped: %s", self.output_path)


@ModuleOption("opencv", requires=[Player,DownloadBufferImpl, EnhanceBufferImpl,BandwidthMeter])
class OpenCVRenderer(Module, Renderer, PlayerEventListener):
    log = logging.getLogger("OpenCV Renderer")

    def __init__(self) -> None:
        super().__init__()
        self.config = None # 播放器配置对象，包含所有配置参数
        self.width = None   # 显示宽度，渲染窗口的宽度
        self.height = None  # 显示高度，渲染窗口的高度
        self.fps = None     # 目标帧率，用于控制渲染速度
        self.device = None  # 渲染设备类型（CPU或GPU），决定渲染路径
        self.frame_recorder: Optional[FrameRecorder] = None

        # OpenCV相关
        self.window_name = "IStream Player - OpenCV"
        self.nv_down = None # NVIDIA Surface下载器，用于将GPU表面数据下载到CPU
        self.data = None # 数据缓冲区，用于存储从GPU下载的帧数据
        # self.surface_resizer = None  # 低分辨率表面放大器，用于将低分辨率表面放大到目标分辨率

        # 播放控制
        self.last_render_time = None # 上次渲染时间戳，用于帧率控制
        self.task_start = None # 任务开始时间，用于计算剩余渲染时间
        self.task_total = None # 任务总时间，用于计算剩余渲染时间
        self.fps_logger = FPSLogger() # FPS日志记录器实例，用于监控渲染性能

        # 线程控制
        self.render_thread = None # 渲染线程，用于处理渲染任务
        self.stop_flag = False # 停止标志，用于停止渲染线程
        self.frame_queue = queue.Queue(maxsize=10) # 帧队列，用于存储渲染帧数据

        # 播放状态
        self.is_playing = False # 播放状态标志，用于控制渲染
        self.is_paused = False  # 暂停状态标志，用于控制渲染画面

    async def setup(self, 
                    config: PlayerConfig, 
                    player: Player,
                    download_buffer: DownloadBufferImpl,
                    enhance_buffer: EnhanceBufferImpl,
                    bw_meter: BandwidthMeter,
                    ):
        player.add_listener(self)
        self.config = config
        self.width = config.display_width
        self.height = config.display_height
        self.fps = config.display_fps
        self.device = config.renderer_device
        self.download_buffer = download_buffer
        self.enhance_buffer = enhance_buffer
        self.bw_meter = bw_meter

        self.current_fps = 0

        # 初始化OpenCV窗口
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.setWindowTitle(self.window_name, "IStream Player - OpenCV")

        # 初始化NVIDIA解码器下载器
        self.nv_down = nvc.PySurfaceDownloader(self.width, self.height, nvc.PixelFormat.RGB, 0)
        self.data = np.zeros((self.width * self.height, 3), np.uint8) # 初始化数据缓冲区，用于存储RGB像素数据
        self.resizer = nvc.PySurfaceResizer(self.width, self.height, nvc.PixelFormat.RGB, 0)

        self.log.info(f"OpenCV Renderer initialized: {self.width}x{self.height} @ {self.fps}fps")

        self._init_recorder()  # 初始化帧记录器

    def remain_task(self):
        if self.task_start is None: # 如果任务未开始，返回0
            return 0
        return max(self.task_total - (time.time() - self.task_start), 0) # 返回剩余任务时间（总时间 - 已用时间）

    async def on_segment_playback_start(self, segments: Dict[int, Segment]):
        """当片段开始播放时调用"""
        self.is_playing = True # 设置播放状态标志为True

        for as_idx in segments:
            segment = segments[as_idx] # 获取当前片段
            # 重置帧率节拍参考，避免跨片段累计导致加速或减速
            self.current_segment_url = segment.url
            self.current_segment_download_action = segment.download_action
            self.current_segment_enhance_action = segment.enhance_action
            self.is_enhance = segment.is_enhance # 是否真正执行了增强，而没有abort
            self.enhance_latency = segment.enhance_latency # 增强延迟时间
            self.enhance_scale = segment.enhance_scale # 增强比例
            self.last_render_time = None
            self.task_start = time.time() # 记录任务开始时间
            self.task_total = segment.duration # 记录任务总时间
            self.decision_time_buffer_level = segment.decision_time_buffer_level #  abr算法做出决策时的缓冲区剩余时长

            self.log.info(f"Rendering segment {segment.url}") # 记录渲染片段的URL

            if segment.decode_data is None:  # 预解码的数据来自增强缓冲区替换回来的段，若没有则表明该段未被增强
                # 实时解码
                await self._render_segment_realtime(segment) # 渲染实时解码的片段
            else:
                # 使用预解码数据
                await self._render_segment_predecoded(segment, segment.decode_data) # 渲染预解码的片段

        self.is_playing = False # 设置播放状态标志为False

    async def _render_segment_realtime(self, segment: Segment):
        """实时解码并渲染片段"""
        decoder = DecoderNvCodec(self.config, segment, resize=True)

        while True:
            surf = decoder.decode_one_frame()
            if surf is None:
                break

            if self.last_render_time is None:
                self.last_render_time = time.time()

            # 渲染帧
            await self._render_one_frame(surf)

            # 控制帧率
            next_render_time = self.last_render_time + 1 / self.fps
            sleep_time = max(next_render_time - time.time(), 0)
            await asyncio.sleep(sleep_time)
            self.last_render_time = next_render_time

    async def _render_segment_predecoded(self, segment: Segment, decode_data):
        """渲染预解码的片段数据"""
        for surf in decode_data: 
            if self.last_render_time is None: # 如果上次渲染时间未记录，记录当前时间
                self.last_render_time = time.time() # 记录上次渲染时间

            # 渲染帧
            await self._render_one_frame(surf)

            # 控制帧率
            next_render_time = self.last_render_time + 1 / self.fps
            sleep_time = max(next_render_time - time.time(), 0)
            await asyncio.sleep(sleep_time)
            self.last_render_time = next_render_time

    async def _render_one_frame(self, surf):
        """渲染单帧"""
        try:
            await self._wait_if_paused()

            # 将GPU表面数据下载到CPU
            if isinstance(surf, nvc.Surface):
                surf = self._resize_surface_if_needed(surf) # 调整表面大小
                self.nv_down.DownloadSingleSurface(surf, self.data) # 下载数据


                # 转换为OpenCV格式 (BGR)
                frame = self.data.reshape((self.height, self.width, 3))
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            elif hasattr(surf, 'cpu'):  # PyTorch tensor
                # 处理增强后的帧
                if surf.is_cuda:
                    surf = surf.cpu()

                # 转换为numpy数组
                frame = surf.numpy()
                if len(frame.shape) == 4:  # (1, C, H, W)
                    frame = frame.squeeze(0)

                # 调整维度顺序 (C, H, W) -> (H, W, C)
                frame = np.transpose(frame, (1, 2, 0))

                # 确保数据类型和范围正确
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)

                # 转换为BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                self.log.warning(f"Unknown surface type: {type(surf)}")
                return

            # if self.frame_recorder:
            #     self.frame_recorder.submit(frame)
            # 添加信息覆盖层
            frame = self._add_overlay(frame)
            if self.frame_recorder:
                self.frame_recorder.submit(frame)

            # 显示帧
            cv2.imshow(self.window_name, frame)

            # 检查退出条件
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:  # 'q' 或 ESC 键退出
                self.log.info("User requested exit")
                self.is_playing = False
                return
            elif key == ord(' '):  # 空格键暂停/继续
                self._toggle_pause()

            # 更新FPS显示
            fps_text = self.fps_logger.log()
            if fps_text:
                self.current_fps = fps_text
                self.log.info(f"Current FPS: {fps_text}")

        except Exception as e:
            self.log.error(f"Error rendering frame: {e}")

    async def _wait_if_paused(self):
        """如果处于暂停状态，则阻塞渲染直到恢复"""
        while self.is_paused and self.is_playing:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                self.log.info("User requested exit while paused")
                self.is_playing = False
                break
            elif key == ord(' '):
                self._toggle_pause()
                break
            await asyncio.sleep(0.01)

    def _add_overlay(self, frame):
        """在帧上添加信息覆盖层"""
        overlay = frame.copy()

        # 通过current_segment_url解析当前段的index 提取'segment_'后的数字
        index = self.current_segment_url.split('segment_')[1].split('.')[0]
        # 添加播放信息
        info_text = [
            f"Resolution: {self.width}x{self.height}",
            f"Target FPS: {self.fps}",
            f"Current FPS: {self.current_fps}",
            f"Device: {self.device}",
            f"Playing: {'Yes' if self.is_playing else 'No'}",
            f"Segment Index: {index}",
            f"Download Action: {self.current_segment_download_action}",
            f"Enhance Action: {self.current_segment_enhance_action}",
            f"Is Enhance: {self.is_enhance}",
            f"SR Scale: {self.enhance_scale}",
            f"Download Buffer: {self.download_buffer.buffer_level(continuous=True):.1f}s",
            f"Enhance Buffer: {self.enhance_buffer.buffer_level(continuous=True):.1f}s",
            f"decision time buffer level: {self.decision_time_buffer_level:.1f}s" ,
        ]

        if self.enhance_latency is not None:
            info_text.append(f"Enhance Latency: {self.enhance_latency:.3f}s")

        # 添加剩余任务时间
        if self.task_start is not None:
            remaining = self.remain_task()
            info_text.append(f"This Segment Remaining: {remaining:.1f}s")

        # 绘制文本
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(overlay, text, (10, y_offset + i * 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 添加控制提示
        control_text = "Press 'q' to quit, SPACE to pause/resume"
        cv2.putText(overlay, control_text, (10, self.height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        return overlay

    def _resize_surface_if_needed(self, surf: nvc.Surface) -> nvc.Surface:
        width = surf.Width()
        height = surf.Height()
        if width == self.width and height == self.height:
            return surf
        resized = self.resizer.Execute(surf)
        return resized or surf

    def _init_recorder(self):
        if self.config is None or not self.config.save_rendered_video:
            return
        timestamp = datetime.now().strftime("%Y%m%d_%H%HS")
        output_path = getattr(self.config, "save_video_path", r"E:\dev\Code\Python\bones\player\output\video")
        file_name = getattr(self.config, "save_file_name", None)
        if file_name is None:
            file_name = f"render_{timestamp}.mp4"
        else:
            file_name = f"{file_name}_{timestamp}.mp4"

        codec = getattr(self.config, "save_video_codec", "mp4v")
        max_queue = getattr(self.config, "save_video_max_queue", 120)
        self.frame_recorder = FrameRecorder(
            self.width,
            self.height,
            self.fps,
            output_path,
            file_name,
            codec,
            max_queue,
        )
        self.log.info("Recording rendered output to %s", output_path)


    def _toggle_pause(self):
        """切换暂停状态"""
        self.is_paused = not self.is_paused
        state = "paused" if self.is_paused else "resumed"
        self.log.info(f"Playback {state}")

    async def cleanup(self):
        """清理资源"""
        self._cleanup_resources()
        self.log.info("OpenCV Renderer cleaned up")

    def _cleanup_resources(self):
        if self.frame_recorder:
            self.frame_recorder.close()
            self.frame_recorder = None
        cv2.destroyAllWindows()

    def __del__(self):
        """析构函数"""
        self._cleanup_resources()
