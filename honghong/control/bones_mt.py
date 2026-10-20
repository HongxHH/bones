"""
BONES-MT: Multi-Task BONES 调度器

将 BONES 的"增强级别"维度扩展为"任务组合"：
  0: None (基线)
  1: SR only (NanoVSR 超分)
  2: SR + Deblur (低分辨率去模糊 + 超分)
  3: Deblur only (仅去模糊，适用于高码率)
  4: SR + Deblur + FI (超分 + 去模糊 + 插帧)

核心创新：
  1. GPU 资源状态追踪（虚拟利用率 + 显存）
  2. 资源感知惩罚项（高 GPU 占用时惩罚重任务组合）
  3. 内容感知任务选择（运动内容偏好去模糊，静态内容偏好轻量任务）
  4. 延迟约束（组合延迟超过可用时间时设为不可行）
"""
import sys
sys.path.append("..")
from control.bones_ca import BONES_CA
import numpy as np
import math


class BONES_MT(BONES_CA):
    """Multi-Task BONES: 资源感知的多任务增强调度器"""

    # 任务组合定义（索引对应 enhance_level）
    TASK_COMBOS = ['none', 'sr', 'sr_deblur', 'deblur', 'sr_deblur_fi']

    # 各任务组合的 GPU 资源开销（归一化 [0,1]）
    # utilization: 推理时 GPU 利用率估计
    # memory: 模型显存占用估计（MB）
    TASK_RESOURCES = {
        0: {'utilization': 0.0, 'memory': 0},
        1: {'utilization': 0.65, 'memory': 3},    # NanoVSR
        2: {'utilization': 0.85, 'memory': 12},   # NanoVSR + NAFNet-Small
        3: {'utilization': 0.55, 'memory': 10},   # NAFNet-Small only
        4: {'utilization': 0.95, 'memory': 25},   # + RIFE-lite
    }

    def __init__(self, args):
        super(BONES_MT, self).__init__(args)

        # 资源感知参数（经参数扫描调优：penalty=1.0, max_util=0.60 最优）
        self.resource_aware = getattr(args, 'resource_aware', True)
        self.gpu_util_penalty = getattr(args, 'gpu_util_penalty', 1.0)  # 资源惩罚系数
        self.gpu_mem_limit_mb = getattr(args, 'gpu_mem_limit_mb', 8000)  # 显存上限
        self.max_allowed_util = getattr(args, 'max_allowed_util', 0.60)  # 最大允许利用率

        # 内容感知任务选择参数
        self.content_aware_task = getattr(args, 'content_aware_task', True)
        self.motion_threshold = getattr(args, 'motion_threshold', 0.4)  # 运动内容阈值

        # 虚拟 GPU 状态（仿真器中追踪）
        self.gpu_utilization = 0.0  # 当前 GPU 利用率（滑动平均）
        self.gpu_memory_used = 0.0  # 当前显存占用
        self.util_alpha = 0.3  # EWMA 系数

        # 任务组合统计
        self.task_combo_counts = np.zeros(5, dtype=int)

    def _get_content_motion_level(self, seg_idx):
        """
        估算段的运动水平（基于 VMAF 增益作为内容复杂度代理）。
        高增益 = 高复杂度 = 可能有运动/细节。
        """
        if seg_idx >= self.vmaf_enh_seg.shape[0]:
            return 0.5
        # 用 SR 增益（combo 1 vs combo 0）作为运动/复杂度代理
        avg_gain = np.mean(self.vmaf_enh_seg[seg_idx, :, 1] - self.vmaf_enh_seg[seg_idx, :, 0])
        # 归一化到 [0,1]
        motion = np.clip(avg_gain / 10.0, 0, 1)
        return motion

    def _compute_resource_penalty(self, task_combo, seg_idx):
        """
        计算任务组合的资源惩罚项。
        考虑：当前 GPU 利用率、显存限制、内容运动水平。
        """
        if not self.resource_aware or task_combo == 0:
            return 0.0

        res = self.TASK_RESOURCES[task_combo]
        penalty = 0.0

        # 1. GPU 利用率惩罚：当前利用率高时，重任务惩罚更大
        projected_util = max(self.gpu_utilization, res['utilization'])
        if projected_util > self.max_allowed_util:
            penalty += self.gpu_util_penalty * (projected_util - self.max_allowed_util) * 10

        # 2. 显存惩罚
        if res['memory'] > self.gpu_mem_limit_mb:
            penalty += 100.0  # 硬约束：超过显存上限直接不可行

        # 3. 内容感知：静态内容不偏好去模糊（惩罚去模糊组合）
        if self.content_aware_task and task_combo in [2, 3, 4]:
            motion = self._get_content_motion_level(seg_idx)
            if motion < self.motion_threshold:
                # 静态内容，去模糊收益小，增加惩罚
                penalty += 0.5 * (self.motion_threshold - motion)

        return penalty

    def _bones_control(self):
        """
        重写 BONES 决策：加入资源感知惩罚和内容感知任务选择。
        """
        if self.autotune:
            self._tune_params()

        buff_down = self.buffer_download.buffer_level
        buff_enh = self.buffer_enhance.buffer_level
        lat_scale = self.latency_factor[self.idx_down] if (self.ca_latency and self.idx_down < len(self.latency_factor)) else 1.0
        obj_drift = buff_down * self.seg_time + buff_enh * self.enhance_time * lat_scale

        # 内容感知奖励（复用父类逻辑）
        if self.content_aware and self.ca_mode == 'oracle':
            obj_reward = self.vmaf_enh_seg[self.idx_down] + self.gamma_p
        elif self.content_aware and self.ca_mode == 'predicted':
            obj_reward = self.predicted_vmaf_seg[self.idx_down] + self.gamma_p
        elif self.content_aware and self.ca_mode == 'scalar':
            c_i = self.content_factors[self.idx_down] if self.idx_down < len(self.content_factors) else 1.0
            base_vmaf = self.vmaf_enh_avg[:, 0:1]
            enh_gain = self.vmaf_enh_avg - base_vmaf
            effective_vmaf = base_vmaf + c_i * enh_gain
            obj_reward = effective_vmaf + self.gamma_p
        elif self.content_aware and self.ca_mode in ('per_level', 'base_known'):
            c_level = self.content_factors[self.idx_down]
            base_vmaf = self.vmaf_enh_avg[:, 0:1]
            enh_gain = self.vmaf_enh_avg - base_vmaf
            effective_vmaf = base_vmaf + c_level[np.newaxis, :] * enh_gain
            obj_reward = effective_vmaf + self.gamma_p
        else:
            obj_reward = self.vmaf_enh_avg + self.gamma_p

        if self.is_avg_size:
            seg_size = self.bitrate[:, None] * self.seg_time
        else:
            seg_size = self.seg_size[self.idx_down][:, None]

        obj = (obj_drift - self.V * obj_reward) / seg_size

        # === 多任务扩展：资源感知惩罚 ===
        if self.resource_aware:
            for br in range(obj.shape[0]):
                for combo in range(obj.shape[1]):
                    penalty = self._compute_resource_penalty(combo, self.idx_down)
                    obj[br, combo] += penalty

        # 延迟约束：增强时间超过可用时间时设为不可行
        mask_late = ((buff_enh + self.enhance_time * lat_scale - buff_down) > 0)
        obj[mask_late] = math.inf

        decision = obj.argmin()
        action_down = decision // obj.shape[1]
        action_enh = decision % obj.shape[1]

        # 更新 GPU 状态（EWMA）
        if self.resource_aware:
            res = self.TASK_RESOURCES[action_enh]
            self.gpu_utilization = (1 - self.util_alpha) * self.gpu_utilization + self.util_alpha * res['utilization']
            self.gpu_memory_used = res['memory']

        # 统计
        self.task_combo_counts[action_enh] += 1

        if self.quick_start and (not self.model_ready) and buff_down >= self.download_threshold:
            self.extra_download = self.model_size
            self.model_ready = True
            self.vmaf_enh_avg = self.old_vmaf

        return action_down, action_enh

    def get_task_combo_stats(self):
        """返回任务组合使用统计"""
        total = self.task_combo_counts.sum()
        if total == 0:
            return {}
        stats = {}
        for i, name in enumerate(self.TASK_COMBOS):
            stats[name] = {
                'count': int(self.task_combo_counts[i]),
                'ratio': float(self.task_combo_counts[i] / total),
            }
        stats['avg_gpu_utilization'] = float(self.gpu_utilization)
        stats['gpu_memory_used_mb'] = float(self.gpu_memory_used)
        return stats

    def control_download_finish(self):
        """重写：多任务场景下的增强缓冲推入"""
        if self.buffer_enhance.buffer_level == 0:
            self.session.set_enhance(False, 0)

        buff_down = self.buffer_download.buffer_level
        buff_enh = self.buffer_enhance.buffer_level
        action_down = self.buffer_download.history[self.idx_enh]
        action_enh = self.action_enh

        # 延迟检查
        if buff_down - buff_enh - self.seg_time < self.enhance_time[action_down][action_enh]:
            self.action_enh = 0
            action_enh = 0

        if self.action_enh != 0:
            self.buffer_enhance.push(
                segment_index=self.idx_enh,
                decision=self.action_enh,
                segment_time=self.enhance_time[action_down][action_enh]
            )
        self.idx_enh += 1

        buffer_level = self.buffer_download.buffer_level
        max_level = self.buffer_download.max_level
        time_wait = buffer_level + self.seg_time - max_level
        time_wait = max(0, time_wait)
        self.session.set_download(False, time_wait)
        return
