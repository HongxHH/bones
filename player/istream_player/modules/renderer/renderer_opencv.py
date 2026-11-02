import asyncio
import logging
import os
import time
import threading
import urllib.request
from datetime import datetime
from fileinput import filename
from typing import Dict, Optional
import queue
import cv2
import numpy as np
import PyNvCodec as nvc

import torch

from istream_player.config.config import PlayerConfig
from istream_player.core.renderer import Renderer
from istream_player.core.module import Module, ModuleOption
from istream_player.core.player import Player, PlayerEventListener
from istream_player.models import Segment, State
from istream_player.modules.decoder import DecoderNvCodec
from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.modules.enhancer.compare_view import drop_segment, enabled as compare_enabled, take as take_baseline, to_display_bgr
from istream_player.core.bw_meter import BandwidthMeter


class FPSLogger:
    def __init__(self, interval=1):
        # 日志记录间隔（秒），默认每秒记录一次
        self.interval = interval
        # 帧计数器，用于统计在间隔时间内的帧数
        self.framecount = 0
        # 上次记录时间戳，用于计算时间间隔
        self.seconds = time.perf_counter()

        self.fps = 0.0

    def tick(self) -> float:
        """Count one presented frame; return the latest windowed FPS."""
        self.framecount += 1
        now = time.perf_counter()
        elapsed = now - self.seconds
        if elapsed >= self.interval:
            self.fps = self.framecount / elapsed if elapsed > 0 else 0.0
            self.framecount = 0
            self.seconds = now
        return float(self.fps)

    def log(self, titlebar=True, fmt="fps : {0}"):
        before = self.seconds
        fps = self.tick()
        if self.seconds != before:
            return fmt.format(fps)
        return None


class PreviewPusher:
    """Push a downscaled JPEG of the enhanced frame to the Lab preview."""

    def __init__(self, url: str, run_id: str, arm_id: str) -> None:
        self.url = url
        self.run_id = run_id
        self.arm_id = arm_id
        self._latest: Optional[bytes] = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last_encode = 0.0
        self._thread = threading.Thread(target=self._loop, name="prism-preview", daemon=True)
        self._thread.start()

    def submit(self, frame_bgr: np.ndarray) -> None:
        now = time.perf_counter()
        if now - self._last_encode < 0.12:
            return
        self._last_encode = now
        try:
            h, w = frame_bgr.shape[:2]
            if w > 720:
                scale = 720 / float(w)
                frame_bgr = cv2.resize(frame_bgr, (720, max(2, int(h * scale))))
            ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 55])
            if not ok:
                return
            with self._lock:
                self._latest = buf.tobytes()
        except Exception:
            return

    def close(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                payload = self._latest
                self._latest = None
            if not payload:
                self._stop.wait(0.05)
                continue
            req = urllib.request.Request(
                self.url,
                data=payload,
                method="POST",
                headers={
                    "Content-Type": "image/jpeg",
                    "X-Prism-Run": self.run_id,
                    "X-Prism-Arm": self.arm_id,
                },
            )
            try:
                with urllib.request.urlopen(req, timeout=1.5) as resp:
                    resp.read()
            except Exception:
                pass


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


class _BgrSink:
    def __init__(self, sink) -> None:
        self._sink = sink

    def submit(self, frame_bgr: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._sink.write(rgb)

    def close(self) -> None:
        self._sink.close()


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
        self.preview: Optional[PreviewPusher] = None
        self.frame_sink = None
        self.base_sink = None
        self.frame_archive = None
        self.frame_clock = None
        self._clock_prev = None
        self._frame_in_segment = 0
        self._compare_lr = None
        self._compare_segment = None
        self._compare_dec = None
        self._compare_dec_index = 0
        self._compare_dec_failed = False
        self._compare_down = None
        self._compare_buf = None
        self._compare_native = (0, 0)
        self.window_ok = False

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
        self.player = player
        self.config = config
        self.width = config.display_width
        self.height = config.display_height
        self.fps = config.display_fps
        self.device = config.renderer_device
        self.download_buffer = download_buffer
        self.enhance_buffer = enhance_buffer
        self.bw_meter = bw_meter

        self.current_fps = 0

        self._init_frame_sink()
        if self.frame_sink is None:
            # 有的 OpenCV 构建没有 highgui；窗口失败时仍把帧推到 Lab 预览
            try:
                cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
                cv2.setWindowTitle(self.window_name, "IStream Player - OpenCV")
                self.window_ok = True
            except cv2.error as exc:
                self.window_ok = False
                self.log.warning("OpenCV window unavailable, preview only: %s", exc)

        # 初始化NVIDIA解码器下载器
        self.nv_down = nvc.PySurfaceDownloader(self.width, self.height, nvc.PixelFormat.RGB, 0)
        self.data = np.zeros((self.width * self.height, 3), np.uint8) # 初始化数据缓冲区，用于存储RGB像素数据
        self.resizer = nvc.PySurfaceResizer(self.width, self.height, nvc.PixelFormat.RGB, 0)

        self.log.info(f"OpenCV Renderer initialized: {self.width}x{self.height} @ {self.fps}fps")

        self._init_recorder()  # 初始化帧记录器
        self._init_archive()
        self._init_preview()

    def remain_task(self):
        if self.task_start is None: # 如果任务未开始，返回0
            return 0
        return max(self.task_total - (time.perf_counter() - self.task_start), 0) # 返回剩余任务时间（总时间 - 已用时间）

    def render_fps(self) -> float:
        """Latest ~1s window of presented frames per second."""
        return float(getattr(self.fps_logger, "fps", 0.0) or 0.0)
    
    # 获取当前正在渲染的片段的索引
    def get_current_render_segment_index(self):
        return self.current_segment_index


    async def on_segment_playback_start(self, segments: Dict[int, Segment]):
        """当片段开始播放时调用"""
        self.is_playing = True # 设置播放状态标志为True

        for as_idx in segments:
            segment = segments[as_idx] # 获取当前片段
            # 重置帧率节拍参考，避免跨片段累计导致加速或减速
            self.current_segment_index = (int)(segment.url.split('segment_')[1].split('.')[0])
            self._frame_in_segment = 0
            self.current_segment_url = segment.url
            self.current_segment_download_action = segment.download_action
            self.current_segment_enhance_action = segment.enhance_action
            self.is_enhance = segment.is_enhance # 是否真正执行了增强，而没有abort
            self.enhance_latency = segment.enhance_latency # 增强延迟时间
            self.enhance_scale = segment.enhance_scale # 增强比例
            self.last_render_time = None
            self.task_start = time.perf_counter() # 记录任务开始时间
            self.task_total = segment.duration # 记录任务总时间
            self.decision_time_buffer_level = segment.decision_time_buffer_level #  abr算法做出决策时的缓冲区剩余时长

            self.log.info(f"Rendering segment {segment.url}") # 记录渲染片段的URL
            self._compare_segment = segment
            self._reset_compare_decoder()

            if segment.decode_data is None:  # 预解码的数据来自增强缓冲区替换回来的段，若没有则表明该段未被增强
                self._frame_enhanced = []
                self._compare_lr = None
                drop_segment(segment)
                # 实时解码
                await self._render_segment_realtime(segment) # 渲染实时解码的片段
            else:
                self._frame_enhanced = getattr(segment, "frame_enhanced", None)
                self._compare_lr = getattr(segment, "compare_lr", None) if compare_enabled(self.config) else None
                if self._compare_lr is None:
                    drop_segment(segment)
                # 使用预解码数据
                await self._render_segment_predecoded(segment, segment.decode_data) # 渲染预解码的片段
                # 播放完成后释放内存
                # segment.decode_data = None
                del segment.decode_data
                segment.decode_data = None
                drop_segment(segment)
                self._compare_lr = None


        self.is_playing = False # 设置播放状态标志为False

    async def _render_segment_realtime(self, segment: Segment):
        """实时解码并渲染片段"""
        decoder = DecoderNvCodec(self.config, segment, resize=True)

        while True:
            surf = decoder.decode_one_frame()
            if surf is None:
                break

            if self.last_render_time is None:
                self.last_render_time = time.perf_counter()

            # 渲染帧
            await self._render_one_frame(surf)

            # 控制帧率
            next_render_time = self.last_render_time + 1 / self.fps
            sleep_time = max(next_render_time - time.perf_counter(), 0)
            await asyncio.sleep(sleep_time)
            self.last_render_time = next_render_time

    async def _render_segment_predecoded(self, segment: Segment, decode_data):
        """渲染预解码的片段数据"""
        for item in decode_data:
            # 检查是否是 DecoderWrapper（来自 fast_complete 模式）
            if hasattr(item, 'decode_next_frame'):
                # 处理 DecoderWrapper：实时解码剩余帧
                await self._render_decoder_wrapper(item)
            else:
                # 处理预解码的 surface
                if self.last_render_time is None: # 如果上次渲染时间未记录，记录当前时间
                    self.last_render_time = time.perf_counter() # 记录上次渲染时间

                # 渲染帧
                await self._render_one_frame(item)

                # 控制帧率
                next_render_time = self.last_render_time + 1 / self.fps
                sleep_time = max(next_render_time - time.perf_counter(), 0)
                await asyncio.sleep(sleep_time)
                self.last_render_time = next_render_time

    async def _render_decoder_wrapper(self, decoder_wrapper):
        """渲染 DecoderWrapper 中的帧（用于 fast_complete 模式）"""
        self.log.debug(f"Rendering decoder wrapper with {decoder_wrapper.remaining_frames} remaining frames")

        while decoder_wrapper.has_more_frames():
            if self.last_render_time is None: # 如果上次渲染时间未记录，记录当前时间
                self.last_render_time = time.perf_counter() # 记录上次渲染时间

            # 从解码器获取下一帧
            surf = decoder_wrapper.decode_next_frame()
            if surf is None:
                break

            # 让出控制权给其他任务
            await asyncio.sleep(0)

            # 渲染帧
            await self._render_one_frame(surf)

            # 控制帧率
            next_render_time = self.last_render_time + 1 / self.fps
            sleep_time = max(next_render_time - time.perf_counter(), 0)
            await asyncio.sleep(sleep_time)
            self.last_render_time = next_render_time

    def note_loop_block(self, row: dict) -> None:
        clock = self.frame_clock
        if clock is not None:
            clock.write(row)

    async def _render_one_frame(self, surf):
        """渲染单帧"""
        t_enter = time.perf_counter()
        gap_ms = 0.0 if self._clock_prev is None else (t_enter - self._clock_prev) * 1000.0
        t_wake = t_enter
        t_conv = t_enter
        t_base = t_enter
        t_arch = t_enter
        t_sink = t_enter
        frame = None
        shown_i = -1
        try:
            await self._wait_if_paused()
            t_wake = time.perf_counter()

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

            t_conv = time.perf_counter()
            # if self.frame_recorder:
            #     self.frame_recorder.submit(frame)
            index = self._frame_in_segment
            shown_i = index
            baseline = self._baseline_bgr(index)
            t_base = time.perf_counter()
            self._archive_frame(frame, baseline)
            t_arch = time.perf_counter()
            if self.frame_sink is not None:
                if compare_enabled(self.config) and self.base_sink is not None and baseline is not None:
                    # Only a real bicubic original goes to the left side.
                    # Repeating this frame made the wipe look unchanged.
                    self.base_sink.submit(baseline)
                self.frame_sink.submit(frame)
            t_sink = time.perf_counter()
            if self.frame_sink is None and getattr(self.config, "display_overlay", True):
                frame = self._add_overlay(frame)
            # if self.frame_recorder:
            #     self.frame_recorder.submit(frame)

            if self.preview is not None and self.frame_sink is None:
                self.preview.submit(frame)

            if self.window_ok:
                cv2.imshow(self.window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    self.log.info("User requested exit")
                    self.is_playing = False
                    return
                elif key == ord(' '):
                    self._toggle_pause()

            # 更新出画 FPS（滑窗）；日志仍按间隔打
            fps_text = self.fps_logger.log()
            self.current_fps = float(self.fps_logger.fps)
            if fps_text:
                self.log.info("Current FPS: %s", fps_text)

        except Exception as e:
            self.log.error(f"Error rendering frame: {e}")
        finally:
            t_end = time.perf_counter()
            self._clock_prev = t_end
            self._write_frame_time(
                shown_i,
                gap_ms=gap_ms,
                wait_ms=(t_wake - t_enter) * 1000.0,
                convert_ms=(t_conv - t_wake) * 1000.0,
                baseline_ms=(t_base - t_conv) * 1000.0,
                archive_ms=(t_arch - t_base) * 1000.0,
                sink_ms=(t_sink - t_arch) * 1000.0,
                tail_ms=(t_end - t_sink) * 1000.0,
            )
            # 清理临时变量，释放内存
            if frame is not None:
                del frame

    # 添加监听器方法 - 将监听器添加到监听器列表
    def add_listener(self, listener: PlayerEventListener):
        pass

    async def _wait_if_paused(self):
        """如果处于暂停状态，则阻塞渲染直到恢复"""
        while self.is_paused and self.is_playing:
            key = cv2.waitKey(1) & 0xFF if self.window_ok else 255
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
        index = self.current_segment_index
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

    def _init_frame_sink(self) -> None:
        name = (getattr(self.config, "frame_shm", "") or "").strip()
        if not name or self.config is None:
            return
        try:
            from pathlib import Path
            import sys as _sys

            repo = Path(__file__).resolve().parents[4]
            if str(repo) not in _sys.path:
                _sys.path.insert(0, str(repo))
            from prism.frame_sink import FrameSink

            sink = FrameSink(name, create=False)
            sink.open()
            self.frame_sink = _BgrSink(sink)
            try:
                base = FrameSink(name + "Base", create=False)
                base.open()
                self.base_sink = _BgrSink(base)
            except Exception as exc:
                self.base_sink = None
                self.log.warning("compare sink unavailable: %s", exc)
            self.window_ok = False
            self.log.info("Prism picture pane <- shared frame %s", name)
        except Exception as exc:
            self.log.warning("frame sink unavailable: %s", exc)

    def _buffer_seconds(self, buffer, *, continuous: bool = True) -> float | None:
        if buffer is None:
            return None
        try:
            return float(buffer.buffer_level(continuous=continuous))
        except Exception:
            return None

    def _enhanced_ready_seconds(self) -> float | None:
        buffer = self.download_buffer
        if buffer is None:
            return None
        total = 0.0
        try:
            for group in (getattr(buffer, "_segments", {}) or {}).values():
                for segment in group.values():
                    has_frames = getattr(segment, "decode_data", None) is not None
                    if has_frames or bool(getattr(segment, "is_enhance", False)):
                        total += float(getattr(segment, "duration", 0.0) or 0.0)
                    break
        except Exception:
            return None
        return total

    def _plain(self, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(key): self._plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._plain(item) for item in value]
        return str(value)

    def _frame_was_enhanced(self, index: int) -> bool:
        flags = getattr(self, "_frame_enhanced", None)
        if flags is None:
            return bool(getattr(self, "is_enhance", False))
        if 0 <= index < len(flags):
            return bool(flags[index])
        return False

    def _baseline_bgr(self, index: int):
        """Bicubic original for an enhanced frame. None means 'no distinct pair'."""
        if not compare_enabled(self.config):
            self._close_compare_decoder()
            return None
        if not self._frame_was_enhanced(index):
            return None
        item = take_baseline(self._compare_lr, index)
        if item is not None:
            try:
                return to_display_bgr(item, int(self.width), int(self.height))
            except Exception as exc:
                self.log.warning("baseline upscale failed: %s", exc)
        # Segments enhanced before compare was turned on have no stored LR.
        # Re-decode that same segment and bicubic it up to the display size.
        return self._baseline_from_file(index)

    def _reset_compare_decoder(self) -> None:
        self._close_compare_decoder()
        self._compare_dec_failed = False

    def _close_compare_decoder(self) -> None:
        dec = getattr(self, "_compare_dec", None)
        self._compare_dec = None
        self._compare_down = None
        self._compare_buf = None
        self._compare_dec_index = 0
        self._compare_native = (0, 0)
        if dec is None:
            return
        path = getattr(dec, "mp4_path", None)
        del dec
        if path:
            try:
                os.remove(path)
            except OSError:
                pass

    def _baseline_from_file(self, index: int):
        if self._compare_dec_failed:
            return None
        if index < self._compare_dec_index - 1:
            return None
        if self._compare_dec is None and not self._open_compare_decoder():
            return None
        dec = self._compare_dec
        try:
            while self._compare_dec_index <= index:
                surf = dec.decode_one_frame()
                self._compare_dec_index += 1
                if surf is None:
                    self._compare_dec_failed = True
                    self.log.warning("compare decoder ran out of frames at %s", index)
                    return None
                if self._compare_dec_index - 1 == index:
                    return self._surface_bicubic_bgr(surf)
        except Exception as exc:
            self._compare_dec_failed = True
            self.log.warning("compare baseline decode failed: %s", exc)
        return None

    def _open_compare_decoder(self) -> bool:
        segment = self._compare_segment
        if segment is None or not getattr(segment, "path", None) or not getattr(segment, "init_path", None):
            self._compare_dec_failed = True
            return False
        try:
            dec = DecoderNvCodec(self.config, segment, resize=False)
            width, height = dec.resolution()
            self._compare_dec = dec
            self._compare_dec_index = 0
            self._compare_native = (int(width), int(height))
            self._compare_down = nvc.PySurfaceDownloader(int(width), int(height), nvc.PixelFormat.RGB, 0)
            self._compare_buf = np.zeros((int(width) * int(height), 3), np.uint8)
            self.log.info("Compare baseline decoder %sx%s", width, height)
            return True
        except Exception as exc:
            self._compare_dec_failed = True
            self.log.warning("compare decoder unavailable: %s", exc)
            return False

    def _surface_bicubic_bgr(self, surf):
        width, height = self._compare_native
        self._compare_down.DownloadSingleSurface(surf, self._compare_buf)
        rgb = self._compare_buf.reshape((height, width, 3))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        display_w, display_h = int(self.width), int(self.height)
        if (width, height) != (display_w, display_h):
            bgr = cv2.resize(bgr, (display_w, display_h), interpolation=cv2.INTER_CUBIC)
        return np.ascontiguousarray(bgr)

    def _archive_frame(self, frame, baseline=None) -> None:
        archive = self.frame_archive
        if archive is None:
            return
        segment = int(getattr(self, "current_segment_index", 0) or 0)
        index = self._frame_in_segment
        self._frame_in_segment += 1
        fps = float(self.fps or 30)
        duration = float(self.task_total or 0)
        pts = max(segment - 1, 0) * duration + index / fps
        scale = self._plain(getattr(self, "enhance_scale", None))
        scale_label = f"x{int(scale)}" if isinstance(scale, (int, float)) and not isinstance(scale, bool) and scale > 0 else ""
        model = ""
        if self.config is not None:
            model = str(getattr(self.config, "mod_enhancer", "") or "")
        archive.submit(
            frame,
            {
                "segment": segment,
                "frame_in_segment": index,
                "pts": round(pts, 4),
                "representation": self._plain(getattr(self, "current_segment_download_action", None)),
                "enhance_level": self._plain(getattr(self, "current_segment_enhance_action", None)),
                "enhanced": self._frame_was_enhanced(index),
                "model": model,
                "scale": scale_label,
                "enhance_scale": scale if isinstance(scale, (int, float)) and not isinstance(scale, bool) else 0,
                "has_baseline": baseline is not None,
                "download_buffer_s": self._buffer_seconds(self.download_buffer),
                "enhanced_ready_s": self._enhanced_ready_seconds(),
                "enhance_pending_s": self._buffer_seconds(self.enhance_buffer, continuous=True),
                "enhance_buffer_s": self._buffer_seconds(self.enhance_buffer, continuous=True),
                "enhance_latency_s": self._plain(getattr(self, "enhance_latency", None)),
            },
            baseline,
        )

    def _init_archive(self) -> None:
        directory = (getattr(self.config, "frame_archive", "") or "").strip()
        if not directory:
            return
        try:
            from .frame_archive import FrameArchive

            self.frame_archive = FrameArchive(directory)
            self.log.info("Frame archive -> %s", directory)
        except Exception as exc:
            self.log.warning("frame archive unavailable: %s", exc)
        self._init_frame_clock(directory)

    def _init_frame_clock(self, directory: str) -> None:
        try:
            from pathlib import Path

            from .frame_clock import FrameClock

            path = Path(directory) / "frame_times.jsonl"
            self.frame_clock = FrameClock(path)
            self.log.info("Frame clock -> %s", path)
        except Exception as exc:
            self.log.warning("frame clock unavailable: %s", exc)

    def _write_frame_time(self, shown_i: int, **fields: float) -> None:
        clock = self.frame_clock
        if clock is None:
            return
        row = {
            "kind": "frame",
            "seg": int(getattr(self, "current_segment_index", 0) or 0),
            "i": int(shown_i),
            "buf_s": self._buffer_seconds(self.download_buffer),
        }
        for key, value in fields.items():
            row[key] = round(float(value), 2)
        clock.write(row)

    def _init_preview(self) -> None:
        if self.config is None or self.frame_sink is not None:
            return
        run_id = getattr(self.config, "telemetry_run_id", "") or ""
        arm_id = getattr(self.config, "telemetry_arm_id", "") or ""
        if not run_id or not arm_id:
            return
        base = (getattr(self.config, "status_server_url", "") or "http://127.0.0.1:8080").rstrip("/")
        self.preview = PreviewPusher(f"{base}/v1/preview", run_id, arm_id)
        self.log.info("Lab preview -> %s arm %s", base, arm_id)

    def _toggle_pause(self):
        """切换暂停状态（同步到 Player 控制面）"""
        if getattr(self, "player", None) is not None:
            if self.player.state == State.PAUSED or getattr(self.player, "_paused", False):
                self.player.resume()
            else:
                self.player.pause()
            return
        self.is_paused = not self.is_paused
        state = "paused" if self.is_paused else "resumed"
        self.log.info(f"Playback {state}")

    async def on_state_change(self, position, old_state: State, new_state: State):
        self.is_paused = new_state == State.PAUSED

    async def cleanup(self):
        """清理资源"""
        self._cleanup_resources()
        self.log.info("OpenCV Renderer cleaned up")

    def _cleanup_resources(self):
        self._close_compare_decoder()
        if self.frame_recorder:
            self.frame_recorder.close()
            self.frame_recorder = None
        if self.preview:
            self.preview.close()
            self.preview = None
        if self.frame_sink is not None:
            closer = getattr(self.frame_sink, "close", None)
            if closer:
                closer()
            self.frame_sink = None
        if self.base_sink is not None:
            closer = getattr(self.base_sink, "close", None)
            if closer:
                closer()
            self.base_sink = None
        if self.frame_archive is not None:
            self.frame_archive.close()
            self.frame_archive = None
        if self.frame_clock is not None:
            self.frame_clock.close()
            self.frame_clock = None
        if self.window_ok:
            cv2.destroyAllWindows()

    def __del__(self):
        """析构函数"""
        self._cleanup_resources()
