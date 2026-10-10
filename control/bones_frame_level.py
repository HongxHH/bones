"""
BONES + 帧级内容感知增强调度 联合优化仿真器

BONES做段级码率决策和最大增强级别约束，帧级DP在段内分配增强资源。
对比：纯BONES（段级固定增强）vs BONES+帧级DP（段内动态分配）
"""
import sys
import os
import json
import math
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from control.bones import BONES


class BONESFrameLevel(BONES):
    """
    BONES + 帧级DP联合优化仿真器

    核心改动：
    1. BONES正常做段级联合决策（码率+最大增强级别）
    2. 下载完成后，对该段运行帧级DP，在[0, max_level]范围内分配增强资源
    3. 增强时间 = 帧级DP决策的总GPU时间（不是段级固定时间）
    4. 报告中使用帧级VMAF计算质量（不是段平均）
    """

    def __init__(self, args):
        super().__init__(args)

        # 加载逐帧VMAF数据
        per_frame_path = getattr(args, 'per_frame_path',
                                  r"E:\dev\Code\Python\bones\data\bbb\per_frame_quality.json")
        with open(per_frame_path) as f:
            self.per_frame_data = json.load(f)

        # 分辨率映射：仿真器码率索引 -> 分辨率
        # bbb_video_info.json中码率顺序: 240p, 360p, 480p, 720p, 1080p
        self.bitrate_to_res = {0: 240, 1: 360, 2: 480, 3: 720, 4: 1080}
        self.res_to_idx = {240: 0, 360: 1, 480: 2}

        # 增强级别映射：仿真器enhance索引 -> 我们的级别
        # 仿真器: 0=不增强, 1=low, 2=medium, 3=high, 4=ultra
        # 我们的: 0=base, 1=sr_low, 2=sr_medium, 3=sr_high
        self.enh_to_level = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}  # ultra映射到high
        self.level_to_name = {0: "base", 1: "sr_low", 2: "sr_medium", 3: "sr_high"}

        # 帧级决策存储：segment_index -> np.array(120,) of enhancement levels
        self.frame_decisions = {}

        # GPU时间步长（用于DP离散化）
        self.gpu_step_ms = 10

        # 段间平滑性约束
        self.prev_seg_avg_vmaf = None  # 上一段平均VMAF
        self.smooth_weight = 15.0  # 平滑性惩罚权重（越大越平滑）

        # 动态GPU预算参数
        self.gpu_budget_high_buffer = 1.3  # 缓冲区高时预算乘数
        self.gpu_budget_mid_buffer = 1.0   # 缓冲区中时预算乘数
        self.gpu_budget_low_buffer = 0.7   # 缓冲区低时预算乘数
        self.high_buffer_threshold = 15000  # 高缓冲区阈值（ms）
        self.low_buffer_threshold = 8000    # 低缓冲区阈值（ms）

        print(f"[BONESFrameLevel] 逐帧数据加载完成: {len(self.per_frame_data['results'])} 个测量")
        print(f"[BONESFrameLevel] 平滑性权重={self.smooth_weight}, 动态GPU预算已启用")

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

    def _compute_frame_gpu_time_ms(self, res_idx, level, num_frames=1):
        """计算某增强级别处理num_frames帧的GPU时间（毫秒）"""
        if level == 0:
            return 0.0
        # enhance_time shape: (num_bitrate, num_method+1)
        # method: 0=不增强, 1=low, 2=medium, 3=high
        method_idx = level  # 1=low, 2=medium, 3=high
        if method_idx >= self.enhance_time.shape[1]:
            method_idx = self.enhance_time.shape[1] - 1
        # enhance_time是整段(120帧)的时间，换算单帧
        seg_time_ms = self.enhance_time[res_idx][method_idx]
        return seg_time_ms / 120.0 * num_frames

    def _frame_level_dp(self, seg_idx, res_idx, max_level, gpu_budget_ms, prev_seg_vmaf=None):
        """
        帧级前瞻DP调度：在GPU预算约束下，最大化段内总VMAF（含段间平滑性约束）

        Args:
            seg_idx: 段索引
            res_idx: 分辨率索引（仿真器码率索引）
            max_level: 最大允许增强级别（BONES决策的）
            gpu_budget_ms: GPU时间预算（毫秒）
            prev_seg_vmaf: 上一段平均VMAF（用于平滑性约束）

        Returns:
            decisions: np.array(120,) of enhancement levels
            total_vmaf: 总VMAF（原始VMAF，不含惩罚）
            gpu_used: 实际使用的GPU时间
        """
        res = self.bitrate_to_res.get(res_idx, None)
        if res is None or res not in self.res_to_idx:
            # 不支持的分辨率（720p/1080p），退化为段级固定
            decisions = np.full(120, max_level, dtype=int)
            total_vmaf = None
            gpu_used = self._compute_frame_gpu_time_ms(res_idx, max_level, 120)
            return decisions, total_vmaf, gpu_used

        vmafs = self._get_frame_vmafs(seg_idx, res)
        if vmafs is None:
            decisions = np.full(120, max_level, dtype=int)
            gpu_used = self._compute_frame_gpu_time_ms(res_idx, max_level, 120)
            return decisions, None, gpu_used

        num_frames = 120
        budget_steps = int(gpu_budget_ms / self.gpu_step_ms) + 1

        # 预计算每帧每个级别的GPU时间（步数）和VMAF（含平滑性惩罚）
        frame_gpu_steps = np.zeros((num_frames, 4), dtype=int)
        frame_vmaf_raw = np.zeros((num_frames, 4))  # 原始VMAF
        frame_vmaf_obj = np.zeros((num_frames, 4))  # 目标函数VMAF（含惩罚）

        for f in range(num_frames):
            for level in range(4):
                if level > max_level:
                    frame_vmaf_raw[f][level] = -np.inf
                    frame_vmaf_obj[f][level] = -np.inf
                    frame_gpu_steps[f][level] = 999999
                    continue
                raw_vmaf = vmafs[level][f]
                frame_vmaf_raw[f][level] = raw_vmaf
                gpu_ms = self._compute_frame_gpu_time_ms(res_idx, level, 1)
                frame_gpu_steps[f][level] = int(gpu_ms / self.gpu_step_ms)

                # 平滑性惩罚：使段平均VMAF接近上一段
                # 惩罚 = smooth_weight * (vmaf - prev_vmaf) / num_frames
                # 这样总惩罚 = smooth_weight * |seg_avg - prev_vmaf|
                if prev_seg_vmaf is not None:
                    penalty = self.smooth_weight * (raw_vmaf - prev_seg_vmaf) / num_frames
                    frame_vmaf_obj[f][level] = raw_vmaf - penalty
                else:
                    frame_vmaf_obj[f][level] = raw_vmaf

        # DP（用含惩罚的VMAF做决策）
        dp = np.full((num_frames + 1, budget_steps), -np.inf)
        dp[0][0] = 0
        choice = np.zeros((num_frames + 1, budget_steps), dtype=int)

        for f in range(num_frames):
            for c in range(budget_steps):
                if dp[f][c] == -np.inf:
                    continue
                for level in range(4):
                    if level > max_level:
                        continue
                    new_c = c + frame_gpu_steps[f][level]
                    if new_c < budget_steps:
                        new_vmaf = dp[f][c] + frame_vmaf_obj[f][level]
                        if new_vmaf > dp[f + 1][new_c]:
                            dp[f + 1][new_c] = new_vmaf
                            choice[f + 1][new_c] = level

        # 找最优终点
        best_c = np.argmax(dp[num_frames])

        # 回溯（用原始VMAF计算总VMAF）
        decisions = np.zeros(num_frames, dtype=int)
        c = best_c
        total_vmaf = 0.0
        for f in range(num_frames - 1, -1, -1):
            level = choice[f + 1][c]
            decisions[f] = level
            total_vmaf += frame_vmaf_raw[f][level]
            c -= frame_gpu_steps[f][level]

        gpu_used = best_c * self.gpu_step_ms
        return decisions, total_vmaf, gpu_used

    def control_download_finish(self):
        """
        下载完成收尾：
        - BONES原版逻辑（唤醒增强、检查来得及、推入增强缓冲）
        - 额外：对刚下载的段运行帧级DP，替换增强时间
        """
        # 若增强缓冲为空，唤醒增强
        if self.buffer_enhance.buffer_level == 0:
            self.session.set_enhance(False, 0)

        # 检查增强是否能按时完成
        buff_down = self.buffer_download.buffer_level
        buff_enh = self.buffer_enhance.buffer_level
        action_down = self.buffer_download.history[self.idx_enh]
        action_enh = self.action_enh

        # BONES原版：来不及则回退为不增强
        if buff_down - buff_enh - self.seg_time < self.enhance_time[action_down][action_enh]:
            action_enh = 0
            self.action_enh = 0

        seg_idx = self.idx_enh

        if action_enh != 0:
            # 帧级DP：在BONES的最大增强级别约束下，分配段内增强资源
            max_level = self.enh_to_level.get(action_enh, 3)

            # 动态GPU预算：根据缓冲区状态调整
            base_budget = self.enhance_time[action_down][action_enh]
            if buff_down >= self.high_buffer_threshold:
                budget_multiplier = self.gpu_budget_high_buffer
            elif buff_down <= self.low_buffer_threshold:
                budget_multiplier = self.gpu_budget_low_buffer
            else:
                budget_multiplier = self.gpu_budget_mid_buffer
            gpu_budget_ms = base_budget * budget_multiplier

            decisions, total_vmaf, gpu_used = self._frame_level_dp(
                seg_idx, action_down, max_level, gpu_budget_ms,
                prev_seg_vmaf=self.prev_seg_avg_vmaf)

            self.frame_decisions[seg_idx] = decisions

            # 更新上一段平均VMAF（用于下一段的平滑性约束）
            if total_vmaf is not None:
                self.prev_seg_avg_vmaf = total_vmaf / 120.0
            else:
                # 不支持的分辨率，用段级平均VMAF
                self.prev_seg_avg_vmaf = self.vmaf_enh_seg[seg_idx][action_down][action_enh]

            # 用帧级DP的实际GPU时间推入增强缓冲
            if gpu_used > 0:
                self.buffer_enhance.push(
                    segment_index=seg_idx, decision=action_enh,
                    segment_time=gpu_used)
            else:
                # DP决定全部不增强
                self.buffer_enhance.push(
                    segment_index=seg_idx, decision=0,
                    segment_time=0)
        else:
            # 不增强
            self.frame_decisions[seg_idx] = np.zeros(120, dtype=int)
            # 更新上一段平均VMAF为base质量
            res = self.bitrate_to_res.get(action_down, None)
            if res is not None and res in self.res_to_idx:
                vmafs = self._get_frame_vmafs(seg_idx, res)
                if vmafs is not None:
                    self.prev_seg_avg_vmaf = np.mean(vmafs[0])
                else:
                    self.prev_seg_avg_vmaf = self.vmaf_enh_seg[seg_idx][action_down][0]
            else:
                # 不支持的分辨率（720p/1080p），用段级平均VMAF
                self.prev_seg_avg_vmaf = self.vmaf_enh_seg[seg_idx][action_down][0]

        self.idx_enh += 1

        # 计算等待时间，避免下载缓冲溢出
        buffer_level = self.buffer_download.buffer_level
        max_level = self.buffer_download.max_level
        time_wait = buffer_level + self.seg_time - max_level
        time_wait = max(0, time_wait)
        self.session.set_download(False, time_wait)

    def report(self, title=""):
        """
        汇总性能指标：使用帧级VMAF计算质量（而不是段平均）
        """
        history_down = self.buffer_download.history
        history_enh = self.buffer_enhance.history
        num_seg = self.seg_size.shape[0]

        quality_vmaf = np.zeros((num_seg,))
        oscillation_vmaf = 0
        total_bandwidth = np.zeros((1,), dtype=np.int64)

        for i in range(num_seg):
            action_down = history_down[i]
            action_enh = history_enh[i]

            if i in self.frame_decisions and action_enh != 0:
                # 使用帧级VMAF计算该段平均质量
                res = self.bitrate_to_res.get(action_down, None)
                if res is not None and res in self.res_to_idx:
                    vmafs = self._get_frame_vmafs(i, res)
                    if vmafs is not None:
                        decisions = self.frame_decisions[i]
                        seg_vmaf = np.mean([vmafs[decisions[f]][f] for f in range(120)])
                        quality_vmaf[i] = seg_vmaf
                    else:
                        quality_vmaf[i] = self.vmaf_enh_seg[i][action_down][action_enh]
                else:
                    quality_vmaf[i] = self.vmaf_enh_seg[i][action_down][action_enh]
            else:
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

        # 帧级决策统计
        enh_frame_count = 0
        total_frames = 0
        level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for i in range(num_seg):
            if i in self.frame_decisions:
                decisions = self.frame_decisions[i]
                enh_frame_count += np.sum(decisions > 0)
                total_frames += len(decisions)
                for level in range(4):
                    level_counts[level] += np.sum(decisions == level)

        enh_ratio = enh_frame_count / total_frames if total_frames > 0 else 0

        report_simulator = {
            "playback time": self.global_time,
            "average quality vmaf": quality_vmaf_avg,
            "average oscillation vmaf": oscillation_vmaf,
            "freeze time": self.buffer_download.idle_time,
            "freeze time sec": self.buffer_download.idle_time / 1000.0,
            "freeze event": self.buffer_download.idle_event,
            "average bandwidth": avg_bandwidth,
            "qoe": qoe,
            "enhanced frame ratio": enh_ratio,
            "level distribution": {k: v / total_frames for k, v in level_counts.items()} if total_frames > 0 else {},
            "download history": history_down,
            "enhance history": history_enh,
        }

        if self.logger.is_verbose():
            title = "Simulator Report" if title == "" else title
            self.logger.summarize(title, report_simulator)

        return report_simulator
