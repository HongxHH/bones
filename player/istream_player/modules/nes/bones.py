from collections import OrderedDict
from typing import Dict, Optional, Tuple

import numpy as np
from istream_player.config.config import PlayerConfig
from istream_player.core.nes import NESController
# from istream_player.core.buffer import BufferManager
from istream_player.core.module import Module, ModuleOption
from istream_player.models import AdaptationSet

from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.core.enhancer import Enhancer
from istream_player.core.scheduler import Scheduler
from istream_player.core.bw_meter import BandwidthMeter
import logging

# BONES控制器实现类 - 继承自Module和NESController接口
# 实现基于神经网络的增强流媒体控制算法，优化下载和增强决策
@ModuleOption("bones", requires=[DownloadBufferImpl, EnhanceBufferImpl, Enhancer, Scheduler], default=True)
class BONESController(Module, NESController):
    # 日志记录器 - 用于记录BONES控制器的操作日志
    log = logging.getLogger("BONESController")
    
    def __init__(self):
        """
        初始化BONES控制器
        设置算法参数和状态变量
        """
        # 调用父类构造函数
        super().__init__()
        
        # BONES算法参数
        # gamma_p: 惩罚参数，用于平衡质量和延迟的权衡
        self.gamma_p = 10.
        # V_multiplier: V值的乘数，用于调整控制强度
        self.V_multiplier = 1.

        # 增强安全系数 - 用于保守估计增强时间，避免超时
        self.enhance_safety = 0.9
        # 段时长（毫秒）- 从MPD中获取的最大段持续时间
        self.seg_time = None

        # 首次下载标志 - 用于初始化阶段使用最低质量
        self.first_download = True

    async def setup(self,
                    config: PlayerConfig,
                    download_buffer: DownloadBufferImpl,
                    enhance_buffer: EnhanceBufferImpl,
                    enhancer: Enhancer,
                    scheduler: Scheduler,
                    **kwargs):
        """
        设置BONES控制器 - 初始化所有依赖组件和配置参数
        
        Args:
            config: 播放器配置对象
            download_buffer: 下载缓冲区实现
            enhance_buffer: 增强缓冲区实现
            enhancer: 视频增强器
            scheduler: 调度器
            **kwargs: 其他可选参数
        """
        # 从配置中设置缓冲区大小（转换为毫秒）
        self.buffer_size = config.buffer_duration * 1000
        # 存储所有依赖组件的引用
        self.download_buffer = download_buffer  # 下载缓冲区
        self.enhance_buffer = enhance_buffer    # 增强缓冲区
        self.enhancer = enhancer                # 视频增强器
        self.scheduler = scheduler              # 调度器

    def update_selection(self, adaptation_sets: Dict[int, AdaptationSet], index: int) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        更新表示选择 - BONES算法的主入口点
        为每个自适应集选择最优的下载和增强动作组合
        
        Args:
            adaptation_sets: 自适应集字典
            index: 当前段索引
            
        Returns:
            Tuple[Dict[int, int], Dict[int, int]]: (下载动作字典, 增强动作字典)
        """
        # 初始化 段时长（如果尚未设置）
        if self.seg_time is None:
            # 从MPD中获取最大段持续时间并转换为毫秒
            self.seg_time = self.scheduler.mpd_provider.mpd.max_segment_duration * 1000  # ms

        # 初始化动作字典
        download_actions = dict()  # 下载动作：自适应集ID -> 表示ID
        enhance_actions = dict()   # 增强动作：自适应集ID -> 增强级别

        # 为每个自适应集选择动作
        for adaptation_set in adaptation_sets.values():
            if self.first_download:
                # 首次下载：使用最低质量（表示ID=0, 增强级别=0）
                download_action, enhance_action = 0, 0
                self.first_download = False
            else:
                # 后续下载：使用BONES算法进行智能决策
                download_action, enhance_action = self.bones_decision(adaptation_set)
            # 记录选择结果
            download_actions[adaptation_set.id] = download_action
            enhance_actions[adaptation_set.id] = enhance_action

        return download_actions, enhance_actions

    def bones_decision(self, adaptation_set: AdaptationSet):
        """
        BONES决策算法 - 为单个自适应集选择最优的下载和增强动作
        基于Lyapunov优化理论，平衡质量、延迟和缓冲状态
        
        Args:
            adaptation_set: 要处理的自适应集
            
        Returns:
            Tuple[int, int]: (下载动作, 增强动作)
        """
        # 检查增强器是否就绪
        if not self.enhancer.is_ready():
            # 如果增强器未就绪，使用最低质量
            return 0, 0

        # 获取增强时间表并应用安全系数（转换为毫秒）
        self.enhance_time = self.enhancer.get_latency_table() * 1000 / self.enhance_safety

        # 提取所有表示的比特率
        bitrate = [representation.bandwidth for representation in adaptation_set.representations.values()]
        # 转换为numpy数组便于计算
        self.bitrate = np.array(bitrate)
        # 计算段大小矩阵：(比特率数量, 1)
        # 段大小 = 比特率 * 段时长 / 1000 (转换为字节)
        self.seg_size = np.array(bitrate)[:, None] * self.seg_time / 1000 # (num_bitrate, 1)

        # 获取当前缓冲状态（转换为毫秒）
        self.buff_down = self.download_buffer.buffer_level(continuous=True) * 1000  # 下载缓冲水位
        self.buff_enh = self.enhance_buffer.buffer_level(continuous=True) * 1000   # 增强缓冲水位
        # 获取增强质量表（VMAF分数）
        self.vmaf_enh_avg = self.enhancer.get_quality_table()
        
        # 记录缓冲状态用于调试
        self.log.info(f"Download buffer level: {self.buff_down}")
        self.log.info(f"Enhance buffer level: {self.buff_enh}")

        # 执行BONES控制算法
        self.action_down, self.action_enh = self._bones_control()

        return self.action_down, self.action_enh

    def _bones_control(self):
        """
        BONES控制算法核心实现
        基于Lyapunov优化理论，通过最小化目标函数来选择最优动作组合
        
        Returns:
            Tuple[int, int]: (下载动作, 增强动作)
        """
        # 计算Lyapunov参数V
        # V = (缓冲区大小 - 段时长) * 段时长 / (最大质量分数 + 惩罚参数)
        V = (self.buffer_size - self.seg_time) * self.seg_time / (np.max(self.vmaf_enh_avg) + self.gamma_p)
        # 应用V值乘数
        V = V * self.V_multiplier

        # 计算目标函数的"漂移"部分
        # 漂移 = 下载缓冲水位 * 段时长 + 增强缓冲水位 * 增强时间
        # 这代表了当前系统的延迟状态
        obj_drift = self.buff_down * self.seg_time + self.buff_enh * self.enhance_time  # (num_bitrate, num_method + 1)

        # 计算目标函数的"奖励"部分
        # 奖励 = 增强质量分数 + 惩罚参数
        # 这代表了选择该动作能获得的质量收益
        obj_reward = self.vmaf_enh_avg + self.gamma_p  # (num_bitrate, num_method + 1)

        # 计算完整的目标函数
        # 目标函数 = (漂移 - V * 奖励) / 段大小
        # 这个公式平衡了延迟惩罚和质量奖励，并考虑了段大小的影响
        obj = (obj_drift - V * obj_reward) / self.seg_size  # (num_bitrate, num_method + 1)

        # 禁用会导致超时的增强选项
        # 如果增强完成时间晚于下载时间，则禁用该选项
        mask_late = ((self.buff_enh + self.enhance_time - self.buff_down) > 0)  # (num_bitrate, num_method + 1)
        # 将超时的选项设置为无穷大，确保不会被选中
        obj[mask_late] = np.inf

        # 选择使目标函数最小化的动作组合
        decision = obj.argmin()
        # 从一维索引转换为二维索引
        action_down = decision // obj.shape[1]  # 下载动作（表示ID）
        action_enh = decision % obj.shape[1]   # 增强动作（增强级别）
        
        return int(action_down), int(action_enh)