"""
BONES + 帧级DP + 增量传输 联合优化仿真器

结合两者优势：
1. 帧级DP：在GPU预算内为每帧选择最优增强级别（0-3），考虑内容特征
2. 增量传输：下载完成后基础层立即可播放，GPU利用缓冲区提前量渐进增强
3. 增强顺序：按VMAF增益/增强时间效率排序，高效率帧先增强

对比：
- 纯BONES：段级固定增强，下载→增强→播放（串行）
- 联合方案：BONES码率决策 + 帧级DP级别分配 + 增量传输渐进增强
"""
import sys
import os
import json
import math
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from control.bones import BONES
from simulation.gpu_grid import (STEP_MS, to_steps, budget_to_steps,
                                 assert_cost_table, solve_frame_dp,
                                 grid_residual_ms)


class BONESCombined(BONES):
    """
    BONES + 帧级DP + 增量传输 联合优化仿真器
    """

    def __init__(self, args):
        super().__init__(args)

        # 加载逐帧VMAF数据
        per_frame_path = getattr(args, 'per_frame_path',
                                  r"E:\dev\Code\Python\bones\data\bbb\per_frame_quality.json")
        with open(per_frame_path) as f:
            self.per_frame_data = json.load(f)

        # 分辨率映射
        self.bitrate_to_res = {0: 240, 1: 360, 2: 480, 3: 720, 4: 1080}
        self.res_to_idx = {240: 0, 360: 1, 480: 2}
        self.enh_to_level = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}
        self.level_to_name = {0: "base", 1: "sr_low", 2: "sr_medium", 3: "sr_high"}

        # 段级状态
        self.segment_actual_vmaf = {}  # seg_idx -> 实际播放VMAF
        self.segment_frame_decisions = {}  # seg_idx -> 每帧的增强级别
        self.segment_enhanced_frames = {}  # seg_idx -> 已增强的帧索引集合

        # 统计
        self.total_enhanced_frames = 0
        self.total_frames = 0
        self.high_priority_enhanced = 0
        self.high_priority_total = 0
        self.level_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        print(f"[BONESCombined] 逐帧数据加载完成: {len(self.per_frame_data['results'])} 个测量")

    def _get_frame_vmafs(self, seg_idx, res):
        """获取某段某分辨率下各增强级别的逐帧VMAF"""
        vmafs = {}
        for level in range(4):
            name = self.level_to_name[level]
            key = f"{res}_{name}_{seg_idx}"
            if key in self.per_frame_data["results"]:
                vmafs[level] = np.array(self.per_frame_data["results"][key])
            else:
                return None
        return vmafs

    def _frame_level_dp_with_incremental(self, seg_idx, res_idx, max_level,
                                           available_time_ms, buffer_level_ms):
        """
        帧级DP + 增量传输联合决策（修正版v3）

        正确设计：
        1. 帧级DP：预算=段增强时间（固定，和BONES原版一致），决定每帧最优级别
        2. 增量传输：可用时间=缓冲区水位+播放前半段（和BONESIncremental一致）
        3. 按增益排序，在可用时间内选择先增强哪些帧
        4. 已增强帧用DP决定的级别，未增强帧用基础质量
        5. 总增强时间和BONES原版一致（不会因时间不够牺牲质量）

        Args:
            seg_idx: 段索引
            res_idx: 分辨率索引
            max_level: 最大增强级别（BONES决策的）
            available_time_ms: 可用增强时间（缓冲区水位+播放前半段）
            buffer_level_ms: 下载完成时缓冲区水位

        Returns:
            decisions: np.array(120,) 每帧的增强级别
            enhanced_mask: np.array(120, bool) 是否已增强
            seg_vmaf: 段平均VMAF
        """
        res = self.bitrate_to_res.get(res_idx, None)
        if res is None or res not in self.res_to_idx or max_level == 0:
            decisions = np.zeros(120, dtype=int)
            enhanced_mask = np.zeros(120, dtype=bool)
            return decisions, enhanced_mask, None

        vmafs = self._get_frame_vmafs(seg_idx, res)
        if vmafs is None:
            decisions = np.full(120, max_level, dtype=int)
            enhanced_mask = np.ones(120, dtype=bool)
            return decisions, enhanced_mask, np.mean(vmafs[max_level]) if vmafs else None

        num_frames = 120
        gpu_step_ms = STEP_MS  # M1：统一 0.1ms 网格（原 10ms 向上取整使浅档超额计费）

        # 计算每帧每个级别的增强时间（0.1ms 步数，四舍五入）和VMAF
        frame_gpu_steps = np.zeros((num_frames, 4), dtype=np.int64)
        frame_vmaf = np.zeros((num_frames, 4))
        for f in range(num_frames):
            for level in range(4):
                frame_vmaf[f][level] = vmafs[level][f]
                if level == 0:
                    frame_gpu_steps[f][level] = 0
                else:
                    method_idx = level
                    if method_idx < self.enhance_time.shape[1]:
                        seg_time = self.enhance_time[res_idx][method_idx]
                        frame_gpu_steps[f][level] = to_steps(seg_time / 120.0)
                    else:
                        frame_gpu_steps[f][level] = 10 ** 9

        # M1 断言：允许级别内不允许零成本动作
        assert_cost_table(frame_gpu_steps[:, :max_level + 1])

        # 步骤1：帧级DP - 预算=段增强时间（固定），决定每帧最优级别
        total_budget_ms = self.enhance_time[res_idx][max_level] if max_level > 0 else 0
        budget_steps = budget_to_steps(total_budget_ms)  # M1：删除原 5000 步硬帽

        ideal_decisions, best_c, _ = solve_frame_dp(
            frame_gpu_steps, frame_vmaf, budget_steps, max_level=max_level)

        # 步骤2：计算每帧的理想增益和增强时间（ms，与决策同一张网格成本表）
        ideal_gains = np.array([vmafs[ideal_decisions[f]][f] - vmafs[0][f] for f in range(num_frames)])
        ideal_times_ms = np.array([frame_gpu_steps[f][ideal_decisions[f]] * gpu_step_ms
                                   for f in range(num_frames)])

        # 步骤3：增量传输 - 按增益排序，在可用时间内选择
        valid_mask = (ideal_gains > 0) & (ideal_times_ms > 0)
        valid_indices = np.where(valid_mask)[0]
        sorted_indices = valid_indices[np.argsort(-ideal_gains[valid_indices])]

        enhanced_mask = np.zeros(num_frames, dtype=bool)
        decisions = np.zeros(num_frames, dtype=int)
        used_time = 0.0

        for f in sorted_indices:
            if used_time + ideal_times_ms[f] <= available_time_ms:
                enhanced_mask[f] = True
                decisions[f] = ideal_decisions[f]
                used_time += ideal_times_ms[f]

        # 步骤4：计算段平均VMAF
        seg_vmaf = 0.0
        for f in range(num_frames):
            if enhanced_mask[f]:
                seg_vmaf += vmafs[decisions[f]][f]
            else:
                seg_vmaf += vmafs[0][f]
        seg_vmaf /= num_frames

        # 统计
        self.total_enhanced_frames += np.sum(enhanced_mask)
        self.total_frames += num_frames
        for level in range(4):
            self.level_counts[level] += np.sum(decisions == level)

        high_priority_count = int(num_frames * 0.3)
        high_priority_indices = np.argsort(-ideal_gains)[:high_priority_count]
        self.high_priority_total += high_priority_count
        self.high_priority_enhanced += np.sum(enhanced_mask[high_priority_indices])

        return decisions, enhanced_mask, seg_vmaf

    def control_download_finish(self):
        """
        下载完成收尾（联合方案）：
        - 段立即可播放（基础质量）
        - 计算可用增强时间（缓冲区提前量 + 播放前半段）
        - 帧级DP + 增量传输联合决策
        - 推入增强缓冲（用实际增强时间）
        """
        # 若增强缓冲为空，唤醒增强
        if self.buffer_enhance.buffer_level == 0:
            self.session.set_enhance(False, 0)

        buff_down = self.buffer_download.buffer_level
        buff_enh = self.buffer_enhance.buffer_level
        action_down = self.buffer_download.history[self.idx_enh]
        action_enh = self.action_enh

        # BONES原版：来不及增强则回退为不增强
        if buff_down - buff_enh - self.seg_time < self.enhance_time[action_down][action_enh]:
            action_enh = 0
            self.action_enh = 0

        seg_idx = self.idx_enh

        if action_enh != 0:
            max_level = self.enh_to_level.get(action_enh, 3)

            # 可用增强时间 = 缓冲区水位 + 播放前半段时间
            # （和BONESIncremental一致：播放前半段GPU可以继续增强）
            available_time_ms = buff_down + self.seg_time / 2

            decisions, enhanced_mask, seg_vmaf = self._frame_level_dp_with_incremental(
                seg_idx, action_down, max_level, available_time_ms, buff_down)

            self.segment_frame_decisions[seg_idx] = decisions
            self.segment_enhanced_frames[seg_idx] = enhanced_mask

            if seg_vmaf is not None:
                self.segment_actual_vmaf[seg_idx] = seg_vmaf

            # 计算实际增强时间：M2 起与决策共用同一张 0.1ms 网格成本表
            grid_steps = 0
            continuous_ms = 0.0
            res = self.bitrate_to_res.get(action_down, None)
            if res is not None and res in self.res_to_idx:
                for f in range(120):
                    if enhanced_mask[f]:
                        level = int(decisions[f])
                        if level > 0:
                            method_idx = level
                            if method_idx < self.enhance_time.shape[1]:
                                frame_ms = self.enhance_time[action_down][method_idx] / 120.0
                                grid_steps += to_steps(frame_ms)
                                continuous_ms += frame_ms
            actual_enhance_time = grid_steps * STEP_MS
            # 网格与连续毫秒成本对账（本段残差理论上界 6ms）
            residual = grid_residual_ms(continuous_ms, grid_steps)
            if abs(residual) > 6.0 + 1e-6:
                raise AssertionError(f"seg {seg_idx} 联合方案网格记账残差超界: {residual:.3f}ms")

            # 推入增强缓冲
            if actual_enhance_time > 0:
                self.buffer_enhance.push(
                    segment_index=seg_idx, decision=action_enh,
                    segment_time=actual_enhance_time)
            else:
                self.buffer_enhance.push(
                    segment_index=seg_idx, decision=0, segment_time=0)
        else:
            # 不增强
            self.segment_frame_decisions[seg_idx] = np.zeros(120, dtype=int)
            self.segment_enhanced_frames[seg_idx] = np.zeros(120, dtype=bool)
            res = self.bitrate_to_res.get(action_down, None)
            if res is not None and res in self.res_to_idx:
                vmafs = self._get_frame_vmafs(seg_idx, res)
                if vmafs is not None:
                    self.segment_actual_vmaf[seg_idx] = np.mean(vmafs[0])
                    self.total_frames += 120
                    self.level_counts[0] += 120

        self.idx_enh += 1

        # 计算等待时间，避免下载缓冲溢出
        buffer_level = self.buffer_download.buffer_level
        max_level = self.buffer_download.max_level
        time_wait = buffer_level + self.seg_time - max_level
        time_wait = max(0, time_wait)
        self.session.set_download(False, time_wait)

    def report(self, title=""):
        """汇总性能指标"""
        history_down = self.buffer_download.history
        history_enh = self.buffer_enhance.history
        num_seg = self.seg_size.shape[0]

        quality_vmaf = np.zeros((num_seg,))
        oscillation_vmaf = 0
        total_bandwidth = np.zeros((1,), dtype=np.int64)

        for i in range(num_seg):
            action_down = history_down[i]

            if i in self.segment_actual_vmaf:
                quality_vmaf[i] = self.segment_actual_vmaf[i]
            else:
                action_enh = history_enh[i]
                quality_vmaf[i] = self.vmaf_enh_seg[i][action_down][action_enh]

            if i > 0:
                oscillation_vmaf += abs(quality_vmaf[i] - quality_vmaf[i - 1])

            total_bandwidth += self.seg_size[i][action_down]

        quality_vmaf_avg = np.mean(quality_vmaf)
        oscillation_vmaf = oscillation_vmaf / (num_seg - 1)
        # idle_time 内部为毫秒，QoE 卡顿项与播放器同口径用秒（1 秒卡顿惩罚 0.1）
        avg_freeze = self.buffer_download.idle_time / 1000.0 / num_seg
        avg_bandwidth = total_bandwidth / num_seg / self.seg_time
        qoe = quality_vmaf_avg - oscillation_vmaf - avg_freeze / 10

        # 联合方案特有指标
        enh_frame_ratio = self.total_enhanced_frames / self.total_frames if self.total_frames > 0 else 0
        high_priority_ratio = (self.high_priority_enhanced / self.high_priority_total
                                if self.high_priority_total > 0 else 0)
        level_dist = {k: v / self.total_frames for k, v in self.level_counts.items()} if self.total_frames > 0 else {}

        report_simulator = {
            "playback time": self.global_time,
            "average quality vmaf": quality_vmaf_avg,
            "average oscillation vmaf": oscillation_vmaf,
            "freeze time": self.buffer_download.idle_time,
            "freeze time sec": self.buffer_download.idle_time / 1000.0,
            "freeze event": self.buffer_download.idle_event,
            "average bandwidth": avg_bandwidth,
            "qoe": qoe,
            "enhanced frame ratio": enh_frame_ratio,
            "high priority enhanced ratio": high_priority_ratio,
            "level distribution": level_dist,
            "total enhanced frames": self.total_enhanced_frames,
            "total frames": self.total_frames,
            "download history": history_down,
            "enhance history": history_enh,
        }

        if self.logger.is_verbose():
            title = "Simulator Report" if title == "" else title
            self.logger.summarize(title, report_simulator)

        return report_simulator
