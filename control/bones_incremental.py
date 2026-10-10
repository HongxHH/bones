"""
BONES + 增量传输 + 渐进增强 仿真器

核心改动：
1. 下载完成后，段立即可播放（基础质量），不需要等增强完成
2. GPU利用缓冲区提前量，在后台增强高优先级帧
3. 播放时：已增强帧用增强质量，未增强帧用基础质量
4. 帧级DP决定增强顺序（高VMAF增益帧优先）

对比：
- 纯BONES：下载→增强→播放（串行，必须等增强完才能播放）
- BONES+增量：下载→播放（基础质量），后台渐进增强
"""
import sys
import os
import json
import math
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from control.bones import BONES


class BONESIncremental(BONES):
    """
    BONES + 增量传输 + 渐进增强仿真器

    关键状态：
    - segment_enhanced_count[seg_idx]: 该段已增强的帧数
    - segment_playable[seg_idx]: 该段是否可播放（下载完成即可播放）
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
        self.segment_enhanced_count = {}  # seg_idx -> 已增强帧数
        self.segment_max_enhance_level = {}  # seg_idx -> BONES选择的最大增强级别
        self.segment_download_time = {}  # seg_idx -> 下载完成时间（ms）
        self.segment_actual_vmaf = {}  # seg_idx -> 实际播放VMAF（含渐进增强）

        # 统计
        self.total_enhanced_frames = 0
        self.total_frames = 0
        self.high_priority_enhanced = 0
        self.high_priority_total = 0

        print(f"[BONESIncremental] 逐帧数据加载完成: {len(self.per_frame_data['results'])} 个测量")

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

    def _compute_enhanced_count(self, seg_idx, action_down, action_enh, buffer_level_ms):
        """
        计算该段在播放开始时已增强的帧数

        利用缓冲区提前量：下载完成后到播放开始前的时间 = buffer_level_ms
        GPU在这段时间内可以增强的帧数 = min(120, buffer_level_ms / frame_enhance_time)

        Args:
            seg_idx: 段索引
            action_down: 码率索引
            action_enh: 增强级别索引
            buffer_level_ms: 下载完成时的缓冲区水位（ms）

        Returns:
            enhanced_count: 已增强帧数
            max_level: 最大增强级别
        """
        if action_enh == 0:
            return 0, 0

        max_level = self.enh_to_level.get(action_enh, 3)

        # 单帧增强时间（ms）
        # enhance_time[action_down][action_enh]是整段120帧的时间
        seg_enhance_time_ms = self.enhance_time[action_down][action_enh]
        if seg_enhance_time_ms <= 0:
            return 0, max_level

        frame_enhance_time_ms = seg_enhance_time_ms / 120.0

        # 可用于增强的时间 = 缓冲区水位（下载完成到播放开始的时间）
        # 加上播放过程中继续增强的时间（保守估计：播放前半段时间）
        available_time_ms = buffer_level_ms + self.seg_time / 2  # 播放过程中还能增强一半时间

        enhanced_count = min(120, int(available_time_ms / frame_enhance_time_ms))
        return enhanced_count, max_level

    def _select_frames_to_enhance(self, seg_idx, res, max_level, enhanced_count):
        """
        帧级DP：选择VMAF增益最高的enhanced_count帧增强

        Returns:
            decisions: np.array(120,) of enhancement levels
            seg_vmaf: 该段实际平均VMAF（已增强帧用增强质量，未增强帧用基础质量）
        """
        vmafs = self._get_frame_vmafs(seg_idx, res)
        if vmafs is None or enhanced_count == 0:
            decisions = np.zeros(120, dtype=int)
            if vmafs is not None:
                seg_vmaf = np.mean(vmafs[0])
            else:
                seg_vmaf = None
            return decisions, seg_vmaf

        # 计算每帧在max_level下的VMAF增益
        gains = vmafs[max_level] - vmafs[0]

        # 选择增益最高的enhanced_count帧
        if enhanced_count >= 120:
            selected = np.ones(120, dtype=bool)
        else:
            # 取增益最高的enhanced_count个索引
            top_indices = np.argsort(gains)[-enhanced_count:]
            selected = np.zeros(120, dtype=bool)
            selected[top_indices] = True

        decisions = np.zeros(120, dtype=int)
        decisions[selected] = max_level

        # 计算段平均VMAF
        seg_vmaf = np.mean([
            vmafs[decisions[f]][f] for f in range(120)
        ])

        # 统计高优先级帧（增益前30%）的增强情况
        high_priority_count = int(120 * 0.3)
        high_priority_indices = np.argsort(gains)[-high_priority_count:]
        self.high_priority_total += high_priority_count
        self.high_priority_enhanced += np.sum(selected[high_priority_indices])

        return decisions, seg_vmaf

    def control_download_finish(self):
        """
        下载完成收尾（增量传输版本）：
        - 段立即可播放（基础质量），不需要等增强
        - 计算利用缓冲区提前量可以增强多少帧
        - 帧级DP选择高增益帧增强
        - 推入增强缓冲（用实际增强时间，可能小于段增强时间）
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

        # 记录下载完成时间
        self.segment_download_time[seg_idx] = self.global_time

        if action_enh != 0:
            # 计算利用缓冲区提前量可以增强的帧数
            enhanced_count, max_level = self._compute_enhanced_count(
                seg_idx, action_down, action_enh, buff_down)

            self.segment_enhanced_count[seg_idx] = enhanced_count
            self.segment_max_enhance_level[seg_idx] = max_level

            # 帧级DP选择高增益帧增强，计算实际VMAF
            res = self.bitrate_to_res.get(action_down, None)
            if res is not None and res in self.res_to_idx:
                decisions, seg_vmaf = self._select_frames_to_enhance(
                    seg_idx, res, max_level, enhanced_count)
                if seg_vmaf is not None:
                    self.segment_actual_vmaf[seg_idx] = seg_vmaf
                    self.total_enhanced_frames += enhanced_count
                    self.total_frames += 120

            # 用实际增强时间推入增强缓冲
            # 实际增强时间 = enhanced_count帧的增强时间（可能小于段增强时间）
            if enhanced_count > 0:
                seg_enhance_time_ms = self.enhance_time[action_down][action_enh]
                actual_enhance_time = seg_enhance_time_ms * enhanced_count / 120.0
                self.buffer_enhance.push(
                    segment_index=seg_idx, decision=action_enh,
                    segment_time=actual_enhance_time)
            else:
                self.buffer_enhance.push(
                    segment_index=seg_idx, decision=0, segment_time=0)
        else:
            # 不增强
            self.segment_enhanced_count[seg_idx] = 0
            self.segment_max_enhance_level[seg_idx] = 0
            res = self.bitrate_to_res.get(action_down, None)
            if res is not None and res in self.res_to_idx:
                vmafs = self._get_frame_vmafs(seg_idx, res)
                if vmafs is not None:
                    self.segment_actual_vmaf[seg_idx] = np.mean(vmafs[0])
                    self.total_frames += 120

        self.idx_enh += 1

        # 计算等待时间，避免下载缓冲溢出
        buffer_level = self.buffer_download.buffer_level
        max_level = self.buffer_download.max_level
        time_wait = buffer_level + self.seg_time - max_level
        time_wait = max(0, time_wait)
        self.session.set_download(False, time_wait)

    def report(self, title=""):
        """
        汇总性能指标：使用渐进增强的实际VMAF
        """
        history_down = self.buffer_download.history
        history_enh = self.buffer_enhance.history
        num_seg = self.seg_size.shape[0]

        quality_vmaf = np.zeros((num_seg,))
        oscillation_vmaf = 0
        total_bandwidth = np.zeros((1,), dtype=np.int64)

        for i in range(num_seg):
            action_down = history_down[i]

            # 使用实际VMAF（含渐进增强）
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

        # 增量传输特有指标
        enh_frame_ratio = self.total_enhanced_frames / self.total_frames if self.total_frames > 0 else 0
        high_priority_ratio = (self.high_priority_enhanced / self.high_priority_total
                                if self.high_priority_total > 0 else 0)

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
            "total enhanced frames": self.total_enhanced_frames,
            "total frames": self.total_frames,
            "download history": history_down,
            "enhance history": history_enh,
        }

        if self.logger.is_verbose():
            title = "Simulator Report" if title == "" else title
            self.logger.summarize(title, report_simulator)

        return report_simulator
