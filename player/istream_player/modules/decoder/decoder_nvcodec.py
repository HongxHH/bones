import sys
import os
import tqdm
import torch
import torchvision.transforms as T
import PyNvCodec as nvc
import PytorchNvCodec as pnvc

from istream_player.models import Segment
from istream_player.config.config import PlayerConfig
import numpy as np
import tempfile
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
    def __init__(self, config: PlayerConfig, segment: Segment, gpu_id: int = 0, resize: bool = False):
        """
        初始化NVIDIA解码器
        
        Args:
            config: 播放器配置对象
            segment: 要解码的视频段
            gpu_id: GPU设备ID
            resize: 是否调整解码尺寸到显示尺寸
        """
        # 存储配置和段信息
        self.config = config
        self.segment = segment
        self.gpu_id = gpu_id

        # 将M4S段文件转换为完整的MP4文件
        mp4_path = self.m4s_to_mp4()
        # 创建NVIDIA解码器实例
        self.nv_dec = nvc.PyNvDecoder(mp4_path, gpu_id)

        # 根据是否需要调整尺寸确定转换器参数
        if resize:
            # 使用显示尺寸
            width = config.display_width
            height = config.display_height
        else:
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

    def decode_all_frames(self) -> List[nvc.Surface]:
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
            else:
                _append(surf)


    def m4s_to_mp4(self):
        """
        将流初始化段和M4S视频块连接成完整的MP4文件
        DASH流媒体需要将初始化段和媒体段合并才能正确解码
        
        Returns:
            str: 生成的MP4文件路径
        """
        def cat(input_file, output_file):
            """
            文件连接函数 - 将输入文件内容追加到输出文件
            
            Args:
                input_file: 输入文件路径
                output_file: 输出文件路径
            """
            with open(input_file, 'rb') as infile, open(output_file, 'ab') as outfile:
                # 读取输入文件的内容
                content = infile.read()
                # 将内容追加到输出文件
                outfile.write(content)

        # 创建临时MP4文件
        mp4_path = tempfile.NamedTemporaryFile(dir=self.config.run_dir, delete=False, suffix=".mp4").name
        # 获取初始化段和媒体段的路径
        init_path = self.segment.init_path
        seg_path = self.segment.path
        # 先连接初始化段
        cat(init_path, mp4_path)
        # 再连接媒体段
        cat(seg_path, mp4_path)
        # 返回完整的MP4文件路径
        return mp4_path


class TensorConverter:
    """
    张量转换器类
    负责在NVIDIA视频表面和PyTorch张量之间进行转换
    支持GPU内存的直接操作，避免CPU-GPU数据传输开销
    """
    def __init__(self, decode_W: int, decode_H: int, display_W: int, display_H: int, gpu_id: int = 0):
        """
        初始化张量转换器
        
        Args:
            decode_W: 解码宽度
            decode_H: 解码高度
            display_W: 显示宽度
            display_H: 显示高度
            gpu_id: GPU设备ID
        """
        # 存储尺寸参数
        self.decode_W = decode_W
        self.decode_H = decode_H
        self.display_W = display_W
        self.display_H = display_H

        # 存储GPU设备ID
        self.gpu_id = gpu_id
        # 创建RGB到平面RGB的转换器（用于张量转换）
        self.to_planar = nvc.PySurfaceConverter(decode_W, decode_H, nvc.PixelFormat.RGB, nvc.PixelFormat.RGB_PLANAR, self.gpu_id)
        # 创建平面RGB到RGB的转换器（用于张量转换回表面）
        self.to_rgb = nvc.PySurfaceConverter(display_W, display_H, nvc.PixelFormat.RGB_PLANAR, nvc.PixelFormat.RGB, self.gpu_id)
        # 创建颜色空间转换上下文
        self.context = nvc.ColorspaceConversionContext(nvc.ColorSpace.BT_601, nvc.ColorRange.MPEG)

    def surface_to_tensor(self, surface: nvc.Surface) -> torch.Tensor:
        """
        将平面RGB表面转换为CUDA浮点张量
        在GPU内存中直接操作，避免CPU-GPU数据传输
        
        Args:
            surface: 平面RGB视频表面
            
        Returns:
            torch.Tensor: 形状为(1, 3, height, width)的CUDA浮点张量
        """
        # 将RGB表面转换为平面RGB格式
        surface = self.to_planar.Execute(surface, self.context)
        # 克隆表面以避免内存问题
        surface = surface.Clone()

        # 获取表面平面指针
        surf_plane = surface.PlanePtr()
        # 将GPU内存直接转换为PyTorch张量
        img_tensor = pnvc.DptrToTensor(
            surf_plane.GpuMem(),      # GPU内存指针
            surf_plane.Width(),       # 宽度
            surf_plane.Height(),      # 高度
            surf_plane.Pitch(),       # 内存间距
            surf_plane.ElemSize(),    # 元素大小
        )
        # 检查转换是否成功
        if img_tensor is None:
            raise RuntimeError("Can not export to tensor.")

        # 调整张量形状为(3, height, width)
        img_tensor.resize_(3, int(surf_plane.Height() / 3), surf_plane.Width())
        # 转换为CUDA浮点张量
        img_tensor = img_tensor.type(dtype=torch.cuda.FloatTensor)
        # 归一化到[0, 1]范围
        img_tensor = torch.divide(img_tensor, 255.0)
        # 确保值在有效范围内
        img_tensor = torch.clamp(img_tensor, 0.0, 1.0)

        # 添加批次维度，形状变为(1, 3, height, width)
        img_tensor = torch.unsqueeze(img_tensor, 0)
        # 注释掉的张量克隆操作（备用方案）
        # return img_tensor.clone()
        return img_tensor

    def tensor_to_surface(self, img_tensor: torch.tensor, gpu_id: int = 0) -> nvc.Surface:
        """
        将CUDA浮点张量转换为平面RGB表面
        在GPU内存中直接操作，避免CPU-GPU数据传输
        
        Args:
            img_tensor: 形状为(1, 3, height, width)的CUDA浮点张量
            gpu_id: 分配表面的GPU ID
            
        Returns:
            nvc.Surface: 平面RGB视频表面
        """
        # 移除批次维度
        img_tensor = torch.squeeze(img_tensor, 0)
        # 确保值在[0, 1]范围内
        img = torch.clamp(img_tensor, 0.0, 1.0)
        # 反归一化到[0, 255]范围并确保内存连续
        img = torch.multiply(img, 255.0).contiguous()
        # 转换为CUDA字节张量并克隆
        img = img.type(dtype=torch.cuda.ByteTensor).clone()

        # 创建平面RGB表面
        surface = nvc.Surface.Make(nvc.PixelFormat.RGB_PLANAR, self.display_W, self.display_H, gpu_id)
        # 获取表面平面指针
        surf_plane = surface.PlanePtr()
        # 将张量数据直接复制到GPU内存
        pnvc.TensorToDptr(
            img,                      # 输入张量
            surf_plane.GpuMem(),      # 目标GPU内存
            surf_plane.Width(),       # 宽度
            surf_plane.Height(),      # 高度
            surf_plane.Pitch(),       # 内存间距
            surf_plane.ElemSize(),    # 元素大小
        )
        # 将平面RGB转换为标准RGB格式
        surface = self.to_rgb.Execute(surface, self.context)
        # 注释掉的表面克隆操作（备用方案）
        # return surface.Clone(self.gpu_id)
        return surface