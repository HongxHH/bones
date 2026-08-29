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

    # 任务组合资源开销（USM锐化无需模型显存，比NAFNet轻量）
    TASK_RESOURCES = {
        0: {'utilization': 0.0, 'memory': 0},
        1: {'utilization': 0.65, 'memory': 3},
        2: {'utilization': 0.70, 'memory': 3},   # sr+usm: USM增加极少显存
        3: {'utilization': 0.10, 'memory': 0},   # usm only: 无模型，纯GPU计算
        4: {'utilization': 0.75, 'memory': 3},   # sr+usm+fi (FI暂未实现)
    }

    TASK_NAMES = {0: 'none', 1: 'sr', 2: 'sr_sharpen', 3: 'sharpen', 4: 'sr_sharpen_fi'}

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

        # 内容感知（阶段1：内容特征加载与因子调制）
        self.content_features = None       # 每段的内容特征列表
        self.content_aware = True          # 是否启用内容感知
        self.motion_max = 40.0             # 运动强度归一化上限
        self.texture_max = 8000.0          # 纹理复杂度归一化上限
        self.content_factor_strength = 1.0 # 内容因子调制强度（0=禁用，1=完全启用）

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

        # 内容感知配置
        self.content_aware = getattr(config, 'mt_content_aware', True)
        self.content_factor_strength = getattr(config, 'mt_content_factor_strength', 1.0)

        # 加载内容特征文件（阶段1）
        if self.content_aware:
            content_file = getattr(config, 'mt_content_features_file', '../data/bbb/content_features_bbb.json')
            self._load_content_features(content_file)

        self.log.info(f"MT-BONES setup: resource_aware={self.resource_aware}, "
                      f"penalty={self.gpu_util_penalty}, max_util={self.max_allowed_util}, "
                      f"content_aware={self.content_aware}, "
                      f"content_features={'loaded' if self.content_features else 'none'}")

    def _get_current_gpu_util(self) -> float:
        """获取当前GPU利用率（真实监控或虚拟状态）"""
        if self.gpu_monitor is not None:
            try:
                state = self.gpu_monitor.get_state()
                return float(state.get('utilization', self.gpu_utilization)) / 100.0
            except Exception:
                pass
        return self.gpu_utilization

    def _load_content_features(self, content_file: str):
        """加载内容特征文件（阶段1）"""
        try:
            with open(content_file, 'r') as f:
                data = json.load(f)
            self.content_features = data.get('features', [])
            self.motion_max = data.get('motion_median', self.motion_max) * 2.5
            self.texture_max = data.get('texture_median', self.texture_max) * 2.5
            self.log.info(f"Content features loaded: {len(self.content_features)} segments, "
                          f"motion_max={self.motion_max:.1f}, texture_max={self.texture_max:.1f}")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.log.warning(f"Cannot load content features from {content_file}: {e}")
            self.content_features = None

    def _compute_content_factors(self, seg_idx: int) -> np.ndarray:
        """
        根据内容特征计算各任务组合的增益因子（v2：多级别SR + USM-only）

        新combo结构：
          0: none
          1: sr_light (IMDN_low, 级别1)
          2: sr_medium (IMDN_medium, 级别2)
          3: sr_heavy (IMDN_high, 级别3)
          4: usm_only (仅USM锐化)

        基于imdn_bbb_3060ti.json逐段VMAF数据的实证分析：
        - 纹理复杂度与SR增益强相关（360p r=0.83, 480p r=0.85）
        - 高纹理段SR增益是低纹理段的1.5-2.8倍
        - 场景切换段SR增益降低40-55%（360p/480p）
        - SR级别越高，纹理敏感度越大（重模型能恢复更多细节）

        Returns:
            shape=(5,) 的内容因子数组，factors[0]=1.0（无增强不受影响）
        """
        factors = np.ones(5)
        if not self.content_aware or self.content_features is None:
            return factors
        if seg_idx >= len(self.content_features):
            return factors

        feat = self.content_features[seg_idx]
        motion = min(feat.get('motion_intensity', 0) / self.motion_max, 1.0)
        texture = min(feat.get('texture_complexity', 0) / self.texture_max, 1.0)
        scene_change = feat.get('scene_change', False)

        # 纹理因子：高纹理→增益高（实证：高纹理SR增益是低纹理的1.5-2.8倍）
        # 基础纹理因子0.5~1.5，SR级别越高敏感度越大
        texture_base = 0.5 + 1.0 * texture       # 0.5 ~ 1.5
        # 级别缩放：L1=0.8x, L2=1.0x, L3=1.2x（重模型纹理敏感度更高）
        level_scaling = [0.0, 0.8, 1.0, 1.2]

        # 场景切换惩罚：实证360p/480p场景切换段增益降低40-55%
        scene_penalty_sr = 0.55 if scene_change else 1.0
        scene_penalty_usm = 0.70 if scene_change else 1.0

        s = self.content_factor_strength
        # 从1.0向目标因子插值，s=0时全为1.0（禁用内容感知）
        def lerp(target):
            return 1.0 + s * (target - 1.0)

        # combo 1-3: SR_light/medium/heavy — 纹理主导 + 场景惩罚 + 级别缩放
        for level in [1, 2, 3]:
            sr_target = texture_base * level_scaling[level] * scene_penalty_sr
            # 限制因子范围，避免过度调制
            sr_target = max(0.3, min(2.0, sr_target))
            factors[level] = lerp(sr_target)

        # combo 4: USM-only — 内容中性（实测USM增益方差大，当前不具竞争力，暂不调制）
        # 实测数据: 360p mean=+5.20(std=6.47), 480p mean=+1.68(std=6.76)
        # USM在低纹理段增益更高，但与SR质量差异大会增加振荡，暂不启用内容调制
        factors[4] = 1.0

        return factors

    def _apply_content_awareness(self, vmaf_enh_avg: np.ndarray, seg_idx: int) -> np.ndarray:
        """
        将内容因子应用到VMAF增益表（调制增益部分，不修改base）

        Args:
            vmaf_enh_avg: shape=(num_bitrate, 5) 的VMAF表
            seg_idx: 当前段索引

        Returns:
            调制后的VMAF表
        """
        if not self.content_aware or self.content_features is None:
            return vmaf_enh_avg

        factors = self._compute_content_factors(seg_idx)
        base_col = vmaf_enh_avg[:, 0:1]  # (num_br, 1)
        gains = vmaf_enh_avg - base_col   # (num_br, 5)

        # 调制增益（只调制增强列1-4，列0保持0）
        for e in range(1, 5):
            gains[:, e] *= factors[e]

        result = base_col + gains
        # 保持-inf掩码（无效增强位置）
        result[~np.isfinite(vmaf_enh_avg)] = -np.inf
        return result

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

        # 3. 内容感知：静态内容惩罚锐化组合
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
        # === MT-BONES 扩展：内容感知增益调制（阶段1）===
        seg_idx = self.current_segment_index
        self.vmaf_enh_avg = self._apply_content_awareness(self.vmaf_enh_avg, seg_idx)

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
