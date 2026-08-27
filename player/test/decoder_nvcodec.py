import sys
import os
import tqdm
import torch
import torchvision.transforms as T
import PyNvCodec as nvc
import PytorchNvCodec as pnvc

import numpy as np
import tempfile
import time
from typing import List, Optional, Tuple


class Converter:
    """
    视频格式转换器类
    负责将NVIDIA解码器输出的视频表面进行格式转换和尺寸调整
    支持NV12到YUV420再到RGB的转换链，以及可选的尺寸调整
    """
    def __init__(self, width: int, height: int, gpu_id: int = 0, resize: bool = False):
        """
        初始化视频格式转换器
        
        Args:
            width: 目标宽度
            height: 目标高度
            gpu_id: GPU设备ID
            resize: 是否进行尺寸调整
        """
        # 存储GPU设备ID
        self.gpu_id = gpu_id
        # 是否进行尺寸调整的标志
        self.resize = resize

        # 创建尺寸调整器 - 将输入调整到指定尺寸，输出NV12格式
        self.resizer = nvc.PySurfaceResizer(width, height, nvc.PixelFormat.NV12, gpu_id)
        # 创建NV12到YUV420的转换器
        self.to_yuv = nvc.PySurfaceConverter(width, height, nvc.PixelFormat.NV12, nvc.PixelFormat.YUV420, gpu_id)
        # 创建YUV420到RGB的转换器
        self.to_rgb = nvc.PySurfaceConverter(width, height, nvc.PixelFormat.YUV420, nvc.PixelFormat.RGB, gpu_id)

        # 注释掉的RGB尺寸调整器（备用方案）
        # self.resizer = nvc.PySurfaceResizer(display_H, display_W, nvc.PixelFormat.RGB, gpu_id)
        # 创建颜色空间转换上下文 - 使用BT.601标准和MPEG颜色范围
        self.context = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG)

    def run(self, src_surface: nvc.Surface) -> nvc.Surface:
        """
        执行视频表面转换
        按照预定义的转换链处理输入的视频表面
        
        Args:
            src_surface: 输入的视频表面
            
        Returns:
            nvc.Surface: 转换后的RGB视频表面
        """
        # 从输入表面开始
        surf = src_surface
        # 如果需要调整尺寸，先进行尺寸调整
        if self.resize:
            surf = self.resizer.Execute(surf)
        # 将NV12格式转换为YUV420格式
        surf = self.to_yuv.Execute(surf, self.context)
        # 将YUV420格式转换为RGB格式
        surf = self.to_rgb.Execute(surf, self.context)
        # 返回最终的RGB表面
        return surf


class DecoderNvCodec():
    """
    NVIDIA硬件解码器类
    使用NVIDIA GPU的硬件解码能力解码H.264/H.265视频段
    支持从M4S段文件创建完整的MP4文件进行解码
    """
    def __init__(self, mp4_path, gpu_id: int = 0, resize: bool = False):
        """
        初始化NVIDIA解码器
        
        Args:
            config: 播放器配置对象
            segment: 要解码的视频段
            gpu_id: GPU设备ID
            resize: 是否调整解码尺寸到显示尺寸
        """
        # 存储配置和段信息

        self.gpu_id = gpu_id

        # 创建NVIDIA解码器实例
        self.nv_dec = nvc.PyNvDecoder(mp4_path, gpu_id)

        # 根据是否需要调整尺寸确定转换器参数
        # 使用原始解码尺寸
        width = self.nv_dec.Width()
        height = self.nv_dec.Height()
        # 创建视频格式转换器
        self.converter = Converter(width, height, gpu_id, resize=resize)
        return

    def resolution(self) -> Tuple[int, int]:
        """
        获取视频分辨率
        
        Returns:
            Tuple[int, int]: (宽度, 高度)
        """
        return self.nv_dec.Width(), self.nv_dec.Height()

    def num_frames(self) -> int:
        """
        获取视频段的总帧数

        TODO：这里有问题，返回的总帧数为0
        
        Returns:
            int: 总帧数
        """
        return self.nv_dec.Numframes()

    def decode_one_frame(self) -> Optional[nvc.Surface]:
        """
        解码一帧视频
        从视频段中解码下一帧并转换为RGB格式
        
        Returns:
            Optional[nvc.Surface]: 解码后的RGB视频表面，如果没有更多帧则返回None
        """
        # 解码单个视频表面
        surf = self.nv_dec.DecodeSingleSurface()
        # 检查是否为空（可能需要冲刷滞后的参考帧）
        if surf.Empty():
            # 尝试冲刷解码器，获取滞后的帧（处理B帧重排序等情况）
            try:
                surf = self.nv_dec.FlushSingleSurface()
            except Exception:
                # 某些版本接口可能抛错或不可用，视为无剩余帧
                return None
            if surf.Empty():
                return None
        # 通过转换器处理表面（格式转换和可能的尺寸调整）
        surf = self.converter.run(surf)
        # 注释掉的表面克隆操作（备用方案）
        # return surf.Clone(self.gpu_id)
        return surf

    def decode_all_frames(self, max_frames: Optional[int] = None) -> List[nvc.Surface]:
        """
        快速解码当前视频段的所有帧
        该方法在GPU侧完成解码与格式转换，可选限制最大帧数

        Args:
            max_frames: 最多解码的帧数，None 表示解码至结尾

        Returns:
            List[nvc.Surface]: 已转换为RGB的表面列表
        """
        frames: List[nvc.Surface] = []

        def _append(surface: nvc.Surface):
            frames.append(self.converter.run(surface))

        while True:
            surf = self.nv_dec.DecodeSingleSurface()
            if surf.Empty():
                # 尝试冲刷剩余参考帧
                while True:
                    try:
                        surf = self.nv_dec.FlushSingleSurface()
                    except Exception:
                        surf = None
                    if surf is None or surf.Empty():
                        return frames
                    _append(surf)
                    if max_frames is not None and len(frames) >= max_frames:
                        return frames
            else:
                _append(surf)
                if max_frames is not None and len(frames) >= max_frames:
                    return frames


    def bench_decode_one_vs_all(self, video_path: str) -> None:
        """
        对比 decode_one_frame 与 decode_all_frames 的耗时表现
        使用单个 MP4 文件（已含头），逐帧与批量两种方式计时并打印结果
        """
        # 逐帧解码计时
        decoder_one = DecoderNvCodec(video_path, gpu_id=self.gpu_id)
        start = time.perf_counter()
        count = 0
        while True:
            frame = decoder_one.decode_one_frame()
            if frame is None:
                break
            count += 1
        elapsed_one = time.perf_counter() - start

        # 批量解码计时
        decoder_all = DecoderNvCodec(video_path, gpu_id=self.gpu_id)
        start = time.perf_counter()
        frames = decoder_all.decode_all_frames()
        elapsed_all = time.perf_counter() - start

        print(f"视频: {video_path}")
        print(f"逐帧解码: {count} 帧, 总耗时 {elapsed_one:.3f}s, 平均 {elapsed_one / max(count, 1):.6f}s/帧")
        print(f"批量解码: {len(frames)} 帧, 总耗时 {elapsed_all:.3f}s, 平均 {elapsed_all / max(len(frames), 1):.6f}s/帧")


if __name__ == "__main__":
    # 指定 MP4 文件路径与参数信息
    mp4_path = r"E:\dev\Code\Python\bones\player\istream_run_dir\tmp7fapr3h_.mp4"
    # 该文件时长 4 秒，帧率 30，总帧数约 120
    tester = DecoderNvCodec(mp4_path, gpu_id=0)
    tester.bench_decode_one_vs_all(mp4_path)