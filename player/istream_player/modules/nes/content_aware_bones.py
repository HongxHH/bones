"""
内容感知 BONES 控制器

在 BONES 的 Lyapunov 优化框架基础上，引入内容感知的质量增益和延迟预测。
核心改进：
1. 不再使用静态全局质量表，而是基于每段的内容特征预测增强质量增益
2. 不再使用静态全局延迟表，而是基于每段的内容特征预测增强延迟
3. 其余决策逻辑（Lyapunov 优化、缓冲区管理）保持不变

这是一个消融实验的关键 baseline：证明内容感知本身（不改变决策算法）就能带来提升。
"""
import logging
from typing import Dict, Optional, Tuple

import numpy as np
from istream_player.config.config import PlayerConfig
from istream_player.core.nes import NESController
from istream_player.core.module import Module, ModuleOption
from istream_player.models import AdaptationSet

from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.core.enhancer import Enhancer
from istream_player.core.scheduler import Scheduler
from istream_player.utils.content_features import ContentFeatureStore


@ModuleOption("content_aware_bones", requires=[DownloadBufferImpl, EnhanceBufferImpl, Enhancer, Scheduler], default=False)
class ContentAwareBONESController(Module, NESController):
    """
    内容感知 BONES 控制器

    与 BONES 的区别：
    - quality_table: 静态全局 -> 内容感知的逐段预测
    - latency_table: 静态全局 -> 内容感知的逐段预测
    - 决策算法: 完全相同（Lyapunov 优化）
    """
    log = logging.getLogger("ContentAwareBONESController")

    def __init__(self):
        super().__init__()

        # BONES 算法参数（与原 BONES 一致）
        self.gamma_p = 10.
        self.V_multiplier = 1.
        self.enhance_safety = 0.9
        self.seg_time = None
        self.first_download = True

        # 内容特征存储
        self.feature_store: Optional[ContentFeatureStore] = None
        self.current_segment_index = 0

        # 内容感知开关
        self.use_content_quality = True   # 是否使用内容感知质量预测
        self.use_content_latency = True   # 是否使用内容感知延迟预测

    async def setup(self,
                    config: PlayerConfig,
                    download_buffer: DownloadBufferImpl,
                    enhance_buffer: EnhanceBufferImpl,
                    enhancer: Enhancer,
                    scheduler: Scheduler,
                    **kwargs):
        self.buffer_size = config.buffer_duration * 1000
        self.download_buffer = download_buffer
        self.enhance_buffer = enhance_buffer
        self.enhancer = enhancer
        self.scheduler = scheduler

        # 加载内容特征
        feature_file = getattr(config, 'content_feature_file', None)
        if feature_file is None:
            # 默认路径
            feature_file = "data/content_features/bbb_features.json"

        self.feature_store = ContentFeatureStore(feature_file)
        self.log.info(f"Content features loaded from: {feature_file}, "
                      f"segments: {len(self.feature_store.features)}")

        # 从配置读取开关
        self.use_content_quality = getattr(config, 'use_content_quality', True)
        self.use_content_latency = getattr(config, 'use_content_latency', True)

    def update_selection(self, adaptation_sets: Dict[int, AdaptationSet], index: int) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        更新表示选择 - 内容感知 BONES 主入口

        Args:
            adaptation_sets: 自适应集字典
            index: 当前段索引

        Returns:
            (下载动作字典, 增强动作字典)
        """
        self.current_segment_index = index

        if self.seg_time is None:
            self.seg_time = self.scheduler.mpd_provider.mpd.max_segment_duration * 1000

        download_actions = dict()
        enhance_actions = dict()

        for adaptation_set in adaptation_sets.values():
            if self.first_download:
                download_action, enhance_action = 0, 0
                self.first_download = False
            else:
                download_action, enhance_action = self._content_aware_decision(adaptation_set)
            download_actions[adaptation_set.id] = download_action
            enhance_actions[adaptation_set.id] = enhance_action

        return download_actions, enhance_actions

    def _content_aware_decision(self, adaptation_set: AdaptationSet):
        """
        内容感知决策 - 核心算法

        与 BONES 的唯一区别：quality_table 和 latency_table 是内容感知的
        """
        if not self.enhancer.is_ready():
            return 0, 0

        # 获取静态表作为基础
        base_latency = self.enhancer.get_latency_table()  # (5, 5) 每帧延迟
        base_quality = self.enhancer.get_quality_table()   # (5, 5) VMAF

        # 内容感知调整
        if self.use_content_latency and self.feature_store is not None:
            # 对每个 (bitrate, level) 组合，基于内容特征调整延迟
            content_latency = np.zeros_like(base_latency)
            for br_idx in range(base_latency.shape[0]):
                for lvl_idx in range(base_latency.shape[1]):
                    if base_latency[br_idx, lvl_idx] > 0:
                        content_latency[br_idx, lvl_idx] = self.feature_store.predict_enhancement_latency(
                            segment_index=self.current_segment_index,
                            enhancement_level=lvl_idx,
                            base_latency=base_latency[br_idx, lvl_idx],
                        )
            self.enhance_time = content_latency * 1000 / self.enhance_safety
        else:
            self.enhance_time = base_latency * 1000 / self.enhance_safety

        if self.use_content_quality and self.feature_store is not None:
            # 对每个 (bitrate, level) 组合，基于内容特征调整质量
            content_quality = base_quality.copy()
            for br_idx in range(base_quality.shape[0]):
                for lvl_idx in range(1, base_quality.shape[1]):  # level 0 是无增强
                    if np.isfinite(base_quality[br_idx, lvl_idx]):
                        # 基础 VMAF（无增强）
                        base_vmaf = base_quality[br_idx, 0] if np.isfinite(base_quality[br_idx, 0]) else 60.0
                        # 预测增益
                        gain = self.feature_store.predict_quality_gain(
                            segment_index=self.current_segment_index,
                            enhancement_level=lvl_idx,
                            base_vmaf=base_vmaf,
                        )
                        content_quality[br_idx, lvl_idx] = base_vmaf + gain
            self.vmaf_enh_avg = content_quality
        else:
            self.vmaf_enh_avg = base_quality

        # 以下与原 BONES 完全相同
        bitrate = [rep.bandwidth for rep in adaptation_set.representations.values()]
        self.bitrate = np.array(bitrate)
        self.seg_size = np.array(bitrate)[:, None] * self.seg_time / 1000

        self.buff_down = self.download_buffer.buffer_level(continuous=True) * 1000
        self.buff_enh = self.enhance_buffer.buffer_level(continuous=True) * 1000

        feat = self.feature_store.get(self.current_segment_index) if self.feature_store else None
        if feat:
            self.log.debug(f"Segment {self.current_segment_index}: "
                           f"texture={feat.texture_complexity:.3f}, "
                           f"motion={feat.motion_intensity:.3f}")

        action_down, action_enh = self._bones_control()
        return action_down, action_enh

    def _bones_control(self):
        """BONES 控制算法核心（与原 BONES 完全相同）"""
        V = (self.buffer_size - self.seg_time) * self.seg_time / (np.max(self.vmaf_enh_avg) + self.gamma_p)
        V = V * self.V_multiplier

        obj_drift = self.buff_down * self.seg_time + self.buff_enh * self.enhance_time
        obj_reward = self.vmaf_enh_avg + self.gamma_p
        obj = (obj_drift - V * obj_reward) / self.seg_size

        mask_late = ((self.buff_enh + self.enhance_time - self.buff_down) > 0)
        obj[mask_late] = np.inf

        decision = obj.argmin()
        action_down = decision // obj.shape[1]
        action_enh = decision % obj.shape[1]

        return int(action_down), int(action_enh)
