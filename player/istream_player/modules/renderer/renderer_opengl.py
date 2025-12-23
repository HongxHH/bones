import asyncio
import logging
import sys
from asyncio import create_subprocess_exec
from asyncio.subprocess import PIPE
from typing import Dict, Union

from istream_player.config.config import PlayerConfig
from istream_player.core.renderer import Renderer
from istream_player.core.downloader import (DownloadEventListener,
                                            DownloadManager)
from istream_player.core.module import Module, ModuleOption
from istream_player.core.mpd_provider import MPDProvider
from istream_player.core.player import Player, PlayerEventListener
from istream_player.models import Segment, State

import pycuda
from OpenGL.GL import *
from OpenGL.GLUT import *
from istream_player.modules.decoder import DecoderNvCodec, TensorConverter
import numpy as np
import PyNvCodec as nvc
import time
import torch


# FPS日志记录器类 - 用于监控和记录渲染帧率
class FPSLogger:
    def __init__(self, interval=1):
        # 日志记录间隔（秒），默认每秒记录一次
        self.interval = interval
        # 帧计数器，用于统计在间隔时间内的帧数
        self.framecount = 0
        # 上次记录时间戳，用于计算时间间隔
        self.seconds = time.time()

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
                # 在窗口标题栏显示FPS
                glutSetWindowTitle(fmt.format(self.fps))
            else:
                # 返回FPS字符串
                return fmt.format(self.fps)


# OpenGL渲染器实现类 - 继承自Module、Renderer和PlayerEventListener接口
# 负责使用OpenGL和CUDA进行硬件加速的视频渲染，支持原始帧和增强帧的显示
@ModuleOption("opengl", requires=[Player])
class OpenGLRenderer(Module, Renderer, PlayerEventListener):
    # 获取日志记录器实例，用于记录OpenGL渲染相关的日志信息
    log = logging.getLogger("OpenGL Renderer")

    def __init__(self) -> None:
        # 调用父类构造函数，初始化模块基础功能
        super().__init__()
        # 播放器配置对象，包含所有配置参数
        self.config = None
        # 显示宽度，渲染窗口的宽度
        self.width = None
        # 显示高度，渲染窗口的高度
        self.height = None
        # 目标帧率，用于控制渲染速度
        self.fps = None
        # 数据缓冲区，用于存储从GPU下载的帧数据
        self.data = None
        # 渲染设备类型（CPU或GPU），决定渲染路径
        self.device = None

        # NVIDIA Surface下载器，用于将GPU表面数据下载到CPU
        self.nv_down = None
        # NVIDIA Frame上传器，用于将CPU数据上传到GPU
        self.nv_up = None
        # 上次渲染时间戳，用于帧率控制
        self.last_render_time = None

        # FPS日志记录器实例，用于监控渲染性能
        self.fps_logger = FPSLogger()

        # 任务开始时间，用于计算剩余渲染时间
        self.task_start = None
        # 任务总时间，用于计算剩余渲染时间
        self.task_total = None

        # 张量转换器，用于GPU张量和Surface之间的转换
        self.tensor_converter = None

    # 设置方法 - 初始化OpenGL渲染器的配置参数和依赖组件
    async def setup(self, config: PlayerConfig, player: Player):
        # 将当前实例注册为播放器的事件监听器
        player.add_listener(self)
        # 保存播放器配置对象
        self.config = config
        # 从配置中获取显示宽度和高度
        self.width, self.height = config.display_width, config.display_height
        # 从配置中获取目标帧率
        self.fps = config.display_fps
        # 从配置中获取渲染设备类型
        self.device = config.renderer_device

        # 创建NVIDIA Surface下载器，用于CPU渲染路径
        self.nv_down = nvc.PySurfaceDownloader(self.width, self.height, nvc.PixelFormat.RGB, 0)
        # 创建NVIDIA Frame上传器，用于CPU渲染路径
        self.nv_up = nvc.PyFrameUploader(self.width, self.height, nvc.PixelFormat.RGB, 0)
        # 初始化数据缓冲区，用于存储RGB像素数据
        self.data = np.zeros((self.width * self.height, 3), np.uint8)

        # 创建张量转换器，用于GPU张量转换
        self.tensor_converter = TensorConverter(self.width, self.height, self.width, self.height, 0)

    # 计算剩余任务时间的方法
    def remain_task(self):
        # 如果任务未开始，返回0
        if self.task_start is None:
            return 0
        # 返回剩余任务时间（总时间 - 已用时间）
        return max(self.task_total - (time.time() - self.task_start), 0)

    # 段播放开始事件处理器 - 渲染器的核心方法
    async def on_segment_playback_start(self, segments: Dict[int, Segment]):
        # 首次渲染时的初始化
        if self.last_render_time is None:
            # 设置显示窗口
            self.setup_display(self.width, self.height)
            # 设置OpenGL环境
            self.setup_opengl()
            # 创建纹理对象
            self.create_textures()
            # 注释掉的CUDA-GL握手，在需要时调用
            # self.cuda_gl_handshake()

        # 处理每个自适应集的段
        for as_idx in segments:
            segment = segments[as_idx]
            # 记录任务开始时间
            self.task_start = time.time()
            # 设置任务总时间为段持续时间
            self.task_total = segment.duration
            # 记录渲染开始信息
            self.log.info(f"Rendering segment {segment.url}")
            
            # 检查段是否有预解码数据
            if segment.decode_data is None:
                # 如果没有预解码数据，创建解码器进行实时解码
                decoder = DecoderNvCodec(self.config, segment, resize=True)
                # 执行CUDA-GL握手，建立GPU内存共享
                self.cuda_gl_handshake()

                # 逐帧解码和渲染循环
                while True:
                    # 解码一帧
                    surf = decoder.decode_one_frame()

                    # 检查是否解码完成
                    if surf is None:
                        break

                    # 如果是首次渲染，记录开始时间
                    if self.last_render_time is None:
                        self.last_render_time = time.time()

                    # 渲染当前帧
                    self.render_one_frame(surf)
                    # 计算下一帧的渲染时间
                    next_render_time = self.last_render_time + 1 / self.fps
                    # 计算需要休眠的时间，确保帧率控制
                    sleep_time = max(next_render_time - time.time(), 0)
                    # 休眠到下一帧渲染时间
                    await asyncio.sleep(sleep_time)
                    # 更新渲染时间戳
                    self.last_render_time = next_render_time
            else:
                # 如果有预解码数据，直接渲染增强后的帧
                for surf in segment.decode_data:
                    # 如果是首次渲染，记录开始时间
                    if self.last_render_time is None:
                        self.last_render_time = time.time()

                    # 渲染当前帧
                    self.render_one_frame(surf)
                    # 计算下一帧的渲染时间
                    next_render_time = self.last_render_time + 1 / self.fps
                    # 计算需要休眠的时间，确保帧率控制
                    sleep_time = max(next_render_time - time.time(), 0)
                    # 休眠到下一帧渲染时间
                    await asyncio.sleep(sleep_time)
                    # 更新渲染时间戳
                    self.last_render_time = next_render_time

    # 设置显示窗口的方法
    def setup_display(self, width, height):
        # 记录显示设置开始信息
        self.log.info(f"Setting up display {width}x{height}")
        # 初始化GLUT库
        glutInit(sys.argv)
        # 设置显示模式：RGBA颜色、双缓冲、深度缓冲
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH)
        # 设置窗口大小
        glutInitWindowSize(width, height)
        # 设置窗口位置
        glutInitWindowPosition(0, 0)
        # 创建窗口
        glutCreateWindow(b"iStream Player")
        # 记录显示设置完成信息
        self.log.info(f"Finished setting up display {width}x{height}")

    # 设置OpenGL环境的方法
    def setup_opengl(self):
        # 编译着色器程序
        self.program = self.compile_shaders()
        # 初始化PyCUDA
        import pycuda.autoinit
        import pycuda.gl.autoinit

        # 创建顶点数组对象
        self.vao = GLuint()
        glCreateVertexArrays(1, self.vao)

    # 创建纹理对象的方法
    def create_textures(self):
        # 创建用于GL显示的纹理
        self.texture = glGenTextures(1)
        # 激活纹理单元0
        glActiveTexture(GL_TEXTURE0)
        # 绑定纹理
        glBindTexture(GL_TEXTURE_2D, self.texture)
        # 创建2D纹理，使用RGB格式
        glTexImage2D(
            GL_TEXTURE_2D,
            0,  # 细节级别
            GL_RGB,  # 内部格式
            self.width,  # 纹理宽度
            self.height,  # 纹理高度
            0,  # 边框
            GL_RGB,  # 像素格式
            GL_UNSIGNED_BYTE,  # 数据类型
            ctypes.c_void_p(0),  # 数据指针（初始为空）
        )
        # 设置纹理包装参数
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        # 设置纹理过滤参数
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        # 解绑纹理
        glBindTexture(GL_TEXTURE_2D, 0)
        # 创建CUDA注册的图像对象，用于GPU内存共享
        self.cuda_img = pycuda.gl.RegisteredImage(
            int(self.texture), GL_TEXTURE_2D, pycuda.gl.graphics_map_flags.NONE
        )  # WRITE_DISCARD)

    # CUDA-GL握手方法 - 建立GPU内存共享
    def cuda_gl_handshake(self):
        # 创建像素缓冲区对象（PBO）
        self.pbo = glGenBuffers(1)
        # 绑定PBO
        glBindBuffer(GL_ARRAY_BUFFER, self.pbo)
        # 为PBO分配数据
        glBufferData(GL_ARRAY_BUFFER, self.data, GL_DYNAMIC_DRAW)
        # 解绑PBO
        glBindBuffer(GL_ARRAY_BUFFER, 0)
        # 初始化PyCUDA
        import pycuda.autoinit
        import pycuda.gl.autoinit

        # 创建CUDA注册的缓冲区对象
        self.cuda_pbo = pycuda.gl.RegisteredBuffer(int(self.pbo))
        # 重置顶点数组对象
        self.vao = 0
        # 创建顶点数组对象
        glGenVertexArrays(1, self.vao)
        # 绑定顶点数组对象
        glBindVertexArray(self.vao)

    # 编译着色器程序的方法
    def compile_shaders(self):
        # 顶点着色器源代码
        vertex_shader_source = """
        #version 450 core
        out vec2 uv;
        void main( void)
        {
            // 声明硬编码的位置数组
            const vec2 vertices[4] = vec2[4](vec2(-0.5,  0.5),
                                                 vec2( 0.5,  0.5),
                                                 vec2( 0.5, -0.5),
                                                 vec2(-0.5, -0.5));
            // 使用gl_VertexID索引到我们的数组
            uv=vertices[gl_VertexID]+vec2(0.5,0.5);
            uv.y = 1.0 - uv.y; // 垂直翻转纹理
            gl_Position = vec4(2*vertices[gl_VertexID],1.0,1.0);
            }
        """

        # 创建顶点着色器
        vertex_shader = glCreateShader(GL_VERTEX_SHADER)
        # 设置着色器源代码
        glShaderSource(vertex_shader, vertex_shader_source)
        # 编译着色器
        glCompileShader(vertex_shader)

        # 片段着色器源代码
        fragment_shader_source = """
        #version 450 core
        uniform sampler2D s;
        in vec2 uv;
        out vec4 color;
        void main(void)
        {
            color = vec4(texture(s, uv));
        }
        """

        # 创建片段着色器
        fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
        # 设置着色器源代码
        glShaderSource(fragment_shader, fragment_shader_source)
        # 编译着色器
        glCompileShader(fragment_shader)

        # 创建着色器程序
        program = glCreateProgram()
        # 附加顶点着色器
        glAttachShader(program, vertex_shader)
        # 附加片段着色器
        glAttachShader(program, fragment_shader)
        # 链接程序
        glLinkProgram(program)
        # 清理不再需要的着色器
        glDeleteShader(vertex_shader)
        glDeleteShader(fragment_shader)

        return program

    # 将输入转换为设备格式的方法
    def to_device(self, surf : nvc.Surface | torch.Tensor):
        # 原始帧（NVIDIA Surface）
        if isinstance(surf, nvc.Surface):
            return surf

        # 增强帧（PyTorch张量）
        if isinstance(surf, torch.Tensor):
            # 将张量移动到GPU
            surf = surf.cuda()  # move to GPU
            # 将张量转换为Surface
            surf = self.tensor_converter.tensor_to_surface(surf)
            return surf

    # 渲染单帧的方法 - 渲染器的核心渲染逻辑
    def render_one_frame(self, surf: nvc.Surface | torch.Tensor):
        # 注释掉的清屏操作
        # glClearBufferfv(GL_COLOR, 0, (0, 0, 0))
        # 注释掉的程序绑定
        # glUseProgram(self.program)

        # 将输入转换为设备格式
        surf = self.to_device(surf)

        # 通过CPU和系统内存进行纹理更新
        if self.device == "cpu":
            # 将Surface数据下载到CPU，然后用这些数据更新GL纹理
            success = self.nv_down.DownloadSingleSurface(surf, self.data)
            if not success:
                self.log.info("Could not download Cuda Surface to CPU")
                return

            # 清屏
            glClearBufferfv(GL_COLOR, 0, (0, 0, 0))
            # 绑定着色器程序
            glUseProgram(self.program)

            # 激活纹理单元0
            glActiveTexture(GL_TEXTURE0)
            # 绑定纹理
            glBindTexture(GL_TEXTURE_2D, self.texture)
            # 更新纹理子图像
            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,  # 细节级别
                0, 0,  # 偏移量
                self.width,  # 宽度
                self.height,  # 高度
                GL_RGB,  # 像素格式
                GL_UNSIGNED_BYTE,  # 数据类型
                self.data,  # 数据
            )
        else:
            # GPU渲染路径
            # 清屏
            glClearBufferfv(GL_COLOR, 0, (0, 0, 0))
            # 绑定着色器程序
            glUseProgram(self.program)

            # 从surface.Plane_Ptr()复制到PBO，然后从PBO更新纹理
            src_plane = surf.PlanePtr()
            # 映射PBO到CUDA内存
            buffer_mapping = self.cuda_pbo.map()
            # 获取设备指针和大小
            buffptr, buffsize = buffer_mapping.device_ptr_and_size()
            # 创建2D内存复制对象
            cpy = pycuda.driver.Memcpy2D()
            # 设置源设备内存
            cpy.set_src_device(src_plane.GpuMem())
            # 设置目标设备内存
            cpy.set_dst_device(buffptr)
            # 设置复制宽度（字节）
            cpy.width_in_bytes = src_plane.Width()
            # 设置源内存间距
            cpy.src_pitch = src_plane.Pitch()
            # 设置目标内存间距
            cpy.dst_pitch = self.width * 3
            # 设置复制高度
            cpy.height = src_plane.Height()
            # 执行内存复制
            cpy(aligned=True)
            # 注释掉的同步操作
            # pycuda.driver.Context.synchronize() ## not required?
            # 取消映射
            buffer_mapping.unmap()
            # 从PBO更新OpenGL纹理
            glActiveTexture(GL_TEXTURE0)
            glBindTexture(GL_TEXTURE_2D, self.texture)
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, int(self.pbo))
            glTexSubImage2D(
                GL_TEXTURE_2D,
                0,  # 细节级别
                0, 0,  # 偏移量
                self.width,  # 宽度
                self.height,  # 高度
                GL_RGB,  # 像素格式
                GL_UNSIGNED_BYTE,  # 数据类型
                ctypes.c_void_p(0),  # 数据指针（从PBO读取）
            )

        # 激活纹理单元0
        glActiveTexture(GL_TEXTURE0)
        # 绑定纹理
        glBindTexture(GL_TEXTURE_2D, self.texture)
        # 发送uniform变量到程序并绘制四边形
        glUniform1i(glGetUniformLocation(self.program, b"s"), 0)
        # 绘制四边形（全屏）
        glDrawArrays(GL_QUADS, 0, 4)
        # 交换前后缓冲区
        glutSwapBuffers()

        # 记录FPS
        self.fps_logger.log()
