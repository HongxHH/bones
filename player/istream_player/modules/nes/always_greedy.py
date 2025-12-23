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
import logging
from istream_player.core.bw_meter import BandwidthMeter



# 总是选择某一固定码率下载和固定的增强选择
@ModuleOption("always_greedy", requires=[DownloadBufferImpl, EnhanceBufferImpl, Enhancer, Scheduler, BandwidthMeter], default=True)
class AlwaysGreedyController(Module, NESController):
    log = logging.getLogger("AlwaysBufferGreedyController")
    def __init__(self):
        super().__init__()
        # BOLA parameters

        self.reservoir = 0.375 # 缓冲区下限，表示缓冲区下限为37.5%的缓冲区大小
        self.upper_reservoir = 0.9 # 缓冲区上限，表示缓冲区上限为90%的缓冲区大小

        self.enhance_safety = 0.9 # 增强安全系数，后面会把真实增强时间除以该数以“放大”所需时间或留安全余地。

        self.seg_time = None # 段时长，从MPD中获取的最大段持续时间
        self.first_download = True # 首次下载标志，用于初始化阶段使用最低质量


    async def setup(self,
                    config: PlayerConfig,
                    download_buffer: DownloadBufferImpl,
                    enhance_buffer: EnhanceBufferImpl,
                    enhancer: Enhancer,
                    scheduler: Scheduler,
                    bw_meter: BandwidthMeter,
                    **kwargs):
        self.buffer_size = config.buffer_duration * 1000
        self.download_buffer = download_buffer
        self.enhance_buffer = enhance_buffer
        self.enhancer = enhancer
        self.scheduler = scheduler
        self.bw_meter = bw_meter


    def update_selection(self, adaptation_sets: Dict[int, AdaptationSet], index: int) -> Tuple[Dict[int, int], Dict[int, int]]:
        # 获取段时长
        if self.seg_time is None: 
            self.seg_time = self.scheduler.mpd_provider.mpd.max_segment_duration * 1000  # ms


        download_actions = dict() # 下载动作字典，表示每个自适应集的下载动作
        enhance_actions = dict() # 增强动作字典，表示每个自适应集的增强动作

        for adaptation_set in adaptation_sets.values():
            if self.first_download: 
                self.first_download = False # 首次下载标志，用于初始化阶段使用最低质量
                download_action, enhance_action = 0, 0
            else:
                download_action, enhance_action = self.buffer_greedy_decision(adaptation_set)

            download_actions[adaptation_set.id] = download_action
            enhance_actions[adaptation_set.id] = enhance_action

        return download_actions, enhance_actions


    # 对某个 adaptation_set 的具体决策
    def buffer_greedy_decision(self, adaptation_set: AdaptationSet):
        bitrate = [representation.bandwidth for representation in adaptation_set.representations.values()] # 获取自适应集的可选码率集合
        self.bitrate = np.array(bitrate) # 转换为numpy数组便于计算
        self.util_down_avg = np.log(self.bitrate / np.min(self.bitrate)) # 计算各码率相对最低码率的对数效用。这种对数效用常用于 BOLA、或 ABR 中以体现码率递增的边际收益递减。

        self.buff_down = self.download_buffer.buffer_level(continuous=True) * 1000 # 获取下载缓冲区的缓冲时长
        self.buff_enh = self.enhance_buffer.buffer_level(continuous=True) * 1000 # 获取增强缓冲区的缓冲时长

        # 系列 self.threshold 运算：这是把码率数组映射到“缓冲阈值”上的过程
        self.threshold = self.bitrate - self.bitrate[0] # 将数组以最低码率（bitrate[0]）为基线，得到各档相对增量。
        self.threshold = (self.threshold / self.threshold[-1]) * (self.upper_reservoir - self.reservoir) # 将相对增量归一化到[0, 1]范围内，再缩放到 (upper_reservoir - reservoir) 区间。这一步把码率等级映射到缓冲区间的“水位比例”区间。
        self.threshold = (self.threshold + self.reservoir) * self.buffer_size # 将归一化后的值 得到每个码率对应的缓冲阈值（以 ms 为单位）。

        self.log.info(f"Download buffer level: {self.buff_down}")
        self.log.info(f"Enhance buffer level: {self.buff_enh}")

        action_down = self._buffer_control()
        action_enh = 2
        return action_down, action_enh


    # 下载选择 -- 基于缓冲区的映射
    def _buffer_control(self):
        """
        BOLA algorithm
        :return:
        """
        buffer_level = self.buff_down
        num_action_down = self.bitrate.shape[0]

        # piecewise linear mapping from buffer level to bitrate
        self.action_down = 0
        if buffer_level > self.threshold[-1]:
            # upper reservoir zone
            self.action_down = num_action_down - 1
        else:
            # reservoir & cushion zone
            for action_download in range(num_action_down):
                if buffer_level < self.threshold[action_download]:
                    self.action_down = action_download
                    break
        return self.action_down
