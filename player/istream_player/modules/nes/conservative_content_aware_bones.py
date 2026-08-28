"""
保守内容感知 BONES 控制器 (Conservative Content-Aware BONES)

设计原则：
1. 码率决策完全复用 BONES 的 Lyapunov 优化，不做任何改动
2. 只对增强级别做内容感知微调，且仅在极端纹理情况下调整
3. 微调后必须通过时间可行性检查，否则回退到 BONES 原决策
4. 不使用双峰策略（Level 0 或 Level 3），只在相邻级别间微调

与 content_aware_bones.py 的区别：
- content_aware_bones: 完全替换 BONES 的质量表为内容感知预测，导致极端选择
- conservative_content_aware_bones: 保留 BONES 决策，只做 ±1 级别的安全微调
"""
import json
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
from istream_player.config.config import PlayerConfig
from istream_player.core.module import Module, ModuleOption
from istream_player.models import AdaptationSet

from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.core.enhancer import Enhancer
from istream_player.core.scheduler import Scheduler
from istream_player.modules.nes.bones import BONESController
from istream_player.utils.content_features import SegmentContentFeatures, ContentFeatureStore


@ModuleOption(
    "conservative_ca_bones",
    requires=[DownloadBufferImpl, EnhanceBufferImpl, Enhancer, Scheduler],
    default=False,
)
class ConservativeContentAwareBONESController(BONESController):
    """
    保守内容感知 BONES 控制器

    在 BONES 决策基础上，根据段的纹理复杂度对增强级别做 ±1 微调：
    - 高纹理段 (> texture_high_threshold)：增强级别 +1（如果时间可行）
    - 低纹理段 (< texture_low_threshold)：增强级别 -1（节省算力）
    - 中等纹理段：保持 BONES 原决策

    所有微调都必须通过时间可行性检查，否则回退。
    """

    log = logging.getLogger("ConservativeCABONESController")

    def __init__(
        self,
        texture_high_threshold: float = 0.70,
        texture_low_threshold: float = 0.40,
        feature_file: Optional[str] = None,
    ):
        """
        初始化保守内容感知 BONES 控制器

        Args:
            texture_high_threshold: 高纹理阈值，超过则尝试提升增强级别
            texture_low_threshold: 低纹理阈值，低于则降低增强级别
            feature_file: 内容特征文件路径（JSON），为 None 时从默认路径加载
        """
        super().__init__()
        self.texture_high_threshold = texture_high_threshold
        self.texture_low_threshold = texture_low_threshold
        self.feature_file = feature_file
        self.feature_store: Optional[ContentFeatureStore] = None
        self.current_segment_index = 0

        # 统计微调次数
        self.stats = {
            "total_decisions": 0,
            "bones_original": 0,
            "enhance_increased": 0,
            "enhance_decreased": 0,
            "increase_blocked_by_feasibility": 0,
        }

    async def setup(
        self,
        config: PlayerConfig,
        download_buffer: DownloadBufferImpl,
        enhance_buffer: EnhanceBufferImpl,
        enhancer: Enhancer,
        scheduler: Scheduler,
        **kwargs,
    ):
        """设置控制器，加载内容特征"""
        await super().setup(config, download_buffer, enhance_buffer, enhancer, scheduler, **kwargs)

        # 加载内容特征
        feature_path = self.feature_file
        if feature_path is None:
            # 默认路径：基于输入视频名推断
            video_name = os.path.splitext(os.path.basename(config.input))[0] if config.input else "bbb"
            feature_path = f"data/content_features/{video_name}_features.json"

        # 尝试加载特征文件
        if os.path.exists(feature_path):
            self.feature_store = ContentFeatureStore(feature_path)
            self.log.info(f"Content features loaded: {len(self.feature_store.features)} segments from {feature_path}")
        else:
            self.log.warning(f"Content feature file not found: {feature_path}, falling back to pure BONES")
            self.feature_store = None

    def update_selection(
        self, adaptation_sets: Dict[int, AdaptationSet], index: int
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        """更新表示选择，在 BONES 决策后做内容感知微调"""
        self.current_segment_index = index

        # 先调用父类的 BONES 决策
        download_actions, enhance_actions = super().update_selection(adaptation_sets, index)

        # 如果没有特征存储，直接返回 BONES 决策
        if self.feature_store is None:
            return download_actions, enhance_actions

        # 获取当前段的内容特征
        feature = self.feature_store.get(index)

        # 对每个自适应集做增强级别微调
        for as_id in adaptation_sets.keys():
            bones_enh = enhance_actions[as_id]
            bones_down = download_actions[as_id]
            adjusted_enh = self._adjust_enhance_level(bones_down, bones_enh, feature)
            enhance_actions[as_id] = adjusted_enh

        return download_actions, enhance_actions

    def _adjust_enhance_level(
        self, action_down: int, bones_enh: int, feature: Optional[SegmentContentFeatures]
    ) -> int:
        """
        根据内容特征微调增强级别

        Args:
            action_down: BONES 选择的码率级别（用于时间可行性检查）
            bones_enh: BONES 选择的增强级别
            feature: 当前段的内容特征

        Returns:
            调整后的增强级别
        """
        self.stats["total_decisions"] += 1

        # 没有特征，保持原决策
        if feature is None:
            self.stats["bones_original"] += 1
            return bones_enh

        texture = feature.texture_complexity
        max_enh_level = self.enhance_time.shape[1] - 1 if hasattr(self, "enhance_time") else 3

        # 默认保持 BONES 决策
        target_enh = bones_enh

        if texture > self.texture_high_threshold and bones_enh < max_enh_level:
            # 高纹理段：尝试提升一级
            candidate = bones_enh + 1
            if self._is_feasible(action_down, candidate):
                target_enh = candidate
                self.stats["enhance_increased"] += 1
                self.log.debug(
                    f"Seg {self.current_segment_index}: texture={texture:.3f} > {self.texture_high_threshold}, "
                    f"enhance {bones_enh} -> {target_enh} (feasible)"
                )
            else:
                self.stats["increase_blocked_by_feasibility"] += 1
                self.log.debug(
                    f"Seg {self.current_segment_index}: texture={texture:.3f} > {self.texture_high_threshold}, "
                    f"enhance {bones_enh} -> {candidate} BLOCKED (not feasible), keep {bones_enh}"
                )

        elif texture < self.texture_low_threshold and bones_enh > 0:
            # 低纹理段：降低一级（总是安全的，降低增强时间）
            target_enh = bones_enh - 1
            self.stats["enhance_decreased"] += 1
            self.log.debug(
                f"Seg {self.current_segment_index}: texture={texture:.3f} < {self.texture_low_threshold}, "
                f"enhance {bones_enh} -> {target_enh} (save compute)"
            )

        else:
            # 中等纹理：保持 BONES 决策
            self.stats["bones_original"] += 1

        return target_enh

    def _is_feasible(self, action_down: int, enh_level: int) -> bool:
        """
        检查给定码率和增强级别的组合是否时间可行

        复用 BONES 的可行性判据：buff_enh + enhance_time - buff_down <= 0

        Args:
            action_down: 码率级别
            enh_level: 增强级别

        Returns:
            是否可行
        """
        if not hasattr(self, "enhance_time") or self.enhance_time is None:
            return True
        if action_down >= self.enhance_time.shape[0] or enh_level >= self.enhance_time.shape[1]:
            return False
        enhance_time_ms = self.enhance_time[action_down, enh_level]
        return (self.buff_enh + enhance_time_ms - self.buff_down) <= 0

    def print_stats(self):
        """打印微调统计信息"""
        total = self.stats["total_decisions"]
        if total == 0:
            return
        self.log.info("=" * 60)
        self.log.info("Conservative Content-Aware BONES 微调统计:")
        self.log.info(f"  总决策数: {total}")
        self.log.info(f"  保持 BONES 原决策: {self.stats['bones_original']} ({self.stats['bones_original']/total*100:.1f}%)")
        self.log.info(f"  增强级别提升: {self.stats['enhance_increased']} ({self.stats['enhance_increased']/total*100:.1f}%)")
        self.log.info(f"  增强级别降低: {self.stats['enhance_decreased']} ({self.stats['enhance_decreased']/total*100:.1f}%)")
        self.log.info(f"  提升被可行性检查阻止: {self.stats['increase_blocked_by_feasibility']}")
        self.log.info("=" * 60)
