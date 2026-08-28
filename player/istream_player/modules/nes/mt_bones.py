"""
MT-BONES: Multi-Task BONES 控制器

在 CA-BONES 基础上扩展：
1. 多任务增强数据（5种任务组合替代4种增强级别）
2. GPU资源感知：虚拟/真实GPU利用率和显存作为状态变量
3. 内容感知任务选择：运动内容偏好去模糊，静态内容偏好轻量任务
4. 资源约束：高GPU占用时惩罚重任务组合

任务组合定义（enhance_action 0-4）：
  0: none
  1: sr (仅超分)
  2: sr_deblur (去模糊+超分)
  3: deblur (仅去模糊)
  4: sr_deblur_fi (去模糊+超分+插帧，FI暂降级为sr_deblur)
"""
import json
import logging
from typing import Dict, Tuple

import numpy as np
from istream_player.config.config import PlayerConfig
from istream_player.core.nes import NESController
from istream_player.core.module import Module, ModuleOption
from istream_player.models import AdaptationSet

from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.core.enhancer import Enhancer
from istream_player.core.scheduler import Scheduler

from .ca_bones import CABONESController


@ModuleOption("mt_bones", requires=[DownloadBufferImpl, EnhanceBufferImpl, Enhancer, Scheduler], default=False)
class MTBONESController(CABONESController):
    """
    Multi-Task BONES 控制器

    扩展 CA-BONES，加入：
    - GPU资源状态追踪（EWMA利用率 + 显存）
    - 资源感知惩罚项
    - 内容感知任务选择（运动/静态）
    """

    log = logging.getLogger("MTBONESController")

    # 任务组合资源开销（归一化）
    TASK_RESOURCES = {
        0: {'utilization': 0.0, 'memory': 0},
        1: {'utilization': 0.65, 'memory': 3},
        2: {'utilization': 0.85, 'memory': 12},
        3: {'utilization': 0.55, 'memory': 10},
        4: {'utilization': 0.95, 'memory': 25},
    }

    TASK_NAMES = {0: 'none', 1: 'sr', 2: 'sr_deblur', 3: 'deblur', 4: 'sr_deblur_fi'}

    def __init__(self):
        super().__init__()
        # 资源感知参数（经仿真器调优）
        self.resource_aware = True
        self.gpu_util_penalty = 1.0
        self.gpu_mem_limit_mb = 8000
        self.max_allowed_util = 0.60
        self.motion_threshold = 0.4
        self.content_aware_task = True

        # GPU状态（EWMA）
        self.gpu_utilization = 0.0
        self.gpu_memory_used = 0.0
        self.util_alpha = 0.3

        # 真实GPU监控（可选）
        self.gpu_monitor = None

        # 任务组合统计
        self.task_combo_counts = np.zeros(5, dtype=int)

    async def setup(self, config: PlayerConfig, download_buffer: DownloadBufferImpl,
                    enhance_buffer: EnhanceBufferImpl, enhancer: Enhancer,
                    scheduler: Scheduler, **kwargs):
        """初始化 MT-BONES 控制器"""
        # 先调用父类 setup（会加载 enhance 数据）
        await super().setup(config, download_buffer, enhance_buffer, enhancer, scheduler, **kwargs)

        # 读取多任务配置
        self.resource_aware = getattr(config, 'mt_resource_aware', True)
        self.gpu_util_penalty = getattr(config, 'mt_gpu_util_penalty', 1.0)
        self.max_allowed_util = getattr(config, 'mt_max_allowed_util', 0.60)
        self.motion_threshold = getattr(config, 'mt_motion_threshold', 0.4)
        self.content_aware_task = getattr(config, 'mt_content_aware_task', True)

        # 尝试加载真实GPU监控
        try:
            from istream_player.utils.gpu_monitor import GPUMonitor
            self.gpu_monitor = GPUMonitor(device_index=0)
            self.gpu_monitor.start()
            self.log.info("GPU monitor started for resource-aware scheduling")
        except Exception as e:
            self.log.warning(f"GPU monitor not available: {e}, using virtual GPU state")
            self.gpu_monitor = None

        self.log.info(f"MT-BONES setup: resource_aware={self.resource_aware}, "
                      f"penalty={self.gpu_util_penalty}, max_util={self.max_allowed_util}")

    def _get_current_gpu_util(self) -> float:
        """获取当前GPU利用率（真实监控或虚拟状态）"""
        if self.gpu_monitor is not None:
            try:
                state = self.gpu_monitor.get_state()
                return float(state.get('utilization', self.gpu_utilization)) / 100.0
            except Exception:
                pass
        return self.gpu_utilization

    def _compute_resource_penalty(self, task_combo: int, seg_idx: int) -> float:
        """
        计算任务组合的资源惩罚项

        考虑：
        1. 当前GPU利用率（高占用时惩罚重任务）
        2. 显存限制
        3. 内容感知（静态内容惩罚去模糊）
        """
        if not self.resource_aware or task_combo == 0:
            return 0.0

        res = self.TASK_RESOURCES[task_combo]
        penalty = 0.0

        # 1. GPU利用率惩罚
        current_util = self._get_current_gpu_util()
        projected_util = max(current_util, res['utilization'])
        if projected_util > self.max_allowed_util:
            penalty += self.gpu_util_penalty * (projected_util - self.max_allowed_util) * 10

        # 2. 显存惩罚（硬约束）
        if res['memory'] > self.gpu_mem_limit_mb:
            penalty += 100.0

        # 3. 内容感知：静态内容惩罚去模糊组合
        if self.content_aware_task and task_combo in (2, 3, 4):
            motion = self._estimate_motion_level(seg_idx)
            if motion < self.motion_threshold:
                penalty += 0.5 * (self.motion_threshold - motion)

        return penalty

    def _estimate_motion_level(self, seg_idx: int) -> float:
        """
        估算段的运动水平（基于VMAF增益作为内容复杂度代理）
        """
        if self.base_quality_seg is None or self.avg_gain is None:
            return 0.5
        if seg_idx >= self.base_quality_seg.shape[0]:
            return 0.5
        # 用SR增益（combo 1 vs combo 0）作为运动/复杂度代理
        avg_gain = np.mean(self.avg_gain[:, 1])
        return np.clip(avg_gain / 10.0, 0, 1)

    def _bones_control(self):
        """
        重写 BONES 决策核心：加入资源感知惩罚

        父类计算 obj = (obj_drift - V * obj_reward) / seg_size
        此处加入资源惩罚后再选最优
        """
        # 先调用父类计算基础 obj
        V = (self.buffer_size - self.seg_time) * self.seg_time / (np.max(self.vmaf_enh_avg) + self.gamma_p)
        V = V * self.V_multiplier

        obj_drift = self.buff_down * self.seg_time + self.buff_enh * self.enhance_time

        if self.ca_mode == 'cares' and self.lambda_values is not None:
            idx = self.current_segment_index
            lambda_i = self.lambda_values[idx] if idx < len(self.lambda_values) else 0.0
            obj_reward = self.vmaf_enh_avg + self.gamma_p - lambda_i * self.enhance_time
        else:
            obj_reward = self.vmaf_enh_avg + self.gamma_p

        if self.smooth_alpha > 0 and self.last_vmaf is not None:
            switch_penalty = self.smooth_alpha * np.abs(self.vmaf_enh_avg - self.last_vmaf)
            obj_reward = obj_reward - switch_penalty

        obj = (obj_drift - V * obj_reward) / self.seg_size

        # === MT-BONES 扩展：资源感知惩罚 ===
        if self.resource_aware:
            seg_idx = self.current_segment_index
            for br in range(obj.shape[0]):
                for combo in range(obj.shape[1]):
                    penalty = self._compute_resource_penalty(combo, seg_idx)
                    obj[br, combo] += penalty

        # 延迟约束
        mask_late = ((self.buff_enh + self.enhance_time - self.buff_down) > 0)
        obj[mask_late] = np.inf

        decision = obj.argmin()
        action_down = decision // obj.shape[1]
        action_enh = decision % obj.shape[1]

        # 更新GPU状态（EWMA）
        if self.resource_aware:
            res = self.TASK_RESOURCES.get(int(action_enh), {'utilization': 0.0, 'memory': 0})
            self.gpu_utilization = (1 - self.util_alpha) * self.gpu_utilization + self.util_alpha * res['utilization']
            self.gpu_memory_used = res['memory']

        # 统计
        self.task_combo_counts[int(action_enh)] += 1

        self.last_vmaf = float(self.vmaf_enh_avg[action_down, action_enh])

        return int(action_down), int(action_enh)

    def get_task_combo_stats(self) -> Dict:
        """返回任务组合使用统计"""
        total = self.task_combo_counts.sum()
        if total == 0:
            return {}
        stats = {}
        for i, name in self.TASK_NAMES.items():
            stats[name] = {
                'count': int(self.task_combo_counts[i]),
                'ratio': float(self.task_combo_counts[i] / total),
            }
        stats['avg_gpu_utilization'] = float(self.gpu_utilization)
        return stats

    async def cleanup(self):
        """清理资源"""
        if self.gpu_monitor is not None:
            try:
                self.gpu_monitor.stop()
            except Exception:
                pass
        await super().cleanup()
