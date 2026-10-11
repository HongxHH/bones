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
from simulation.gpu_grid import STEP_MS, to_steps, grid_residual_ms
from simulation.frame_enhance_queue import FrameEnhanceQueue


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

        # M3：帧级 EDF 工作队列（播放截止墙钟 + 可调度性前瞻 + missed 显式记账）
        self.frames_per_seg = 120
        # M5：默认 EDF 因果排序；--oracle 时按真增益密度（全知上界）
        self.oracle_rank = bool(getattr(args, "oracle_rank", False))
        rank_mode = "oracle" if self.oracle_rank else "edf"
        self.frame_queue = FrameEnhanceQueue(self.frames_per_seg, rank_mode=rank_mode)
        self.total_pushed_steps = 0  # 累计推入增强缓冲的 GPU 网格步（守恒断言用）

        # 统计
        self.total_enhanced_frames = 0
        self.total_frames = 0

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

    def _schedule_segment_frames(self, seg_idx, action_down, max_level,
                                 play_start_lead_ms, backlog_ms):
        """M3：帧级 EDF 队列调度，替代旧的"整段可行性闸门 + 半段时间赠送 + 解析式帧数"。

        Returns:
            decisions: np.array(120,) 每帧承诺增强档（0=基础，错过播放截止）
            admitted_steps: 被接收作业的总 GPU 网格步
            seg_vmaf: 段帧均 VMAF
        """
        res = self.bitrate_to_res.get(action_down, None)
        vmafs = self._get_frame_vmafs(seg_idx, res) if res is not None else None
        n = self.frames_per_seg
        if vmafs is None:
            return None, None, None

        frame_cost_steps = np.zeros((n, 4), dtype=np.int64)
        frame_vmaf = np.zeros((n, 4))
        for f in range(n):
            for level in range(4):
                frame_vmaf[f][level] = vmafs[level][f]
                if level > 0:
                    frame_cost_steps[f][level] = to_steps(
                        self.enhance_time[action_down][level] / float(n))

        decisions, admitted_steps, seg_vmaf = self.frame_queue.schedule_segment(
            now_ms=self.global_time, seg_idx=seg_idx,
            play_start_lead_ms=play_start_lead_ms,
            frame_play_ms=self.seg_time / n,
            frame_cost_steps=frame_cost_steps, frame_vmaf=frame_vmaf,
            max_level=max_level, backlog_steps=to_steps(backlog_ms))
        return decisions, admitted_steps, seg_vmaf

    def control_download_finish(self):
        """
        下载完成收尾（M3 帧级 EDF 队列版）：
        - 段下载完成即可按基础质量播放，不做整段可行性闸门
        - 120 个帧作业带播放截止墙钟入队，EDF 前瞻决定哪些帧/档能在播放前完成
        - 到点未完成的帧按基础质量播放（missed，显式记账）
        - 推入增强缓冲的 GPU 时间恒等于被接收作业成本之和
        """
        # 若增强缓冲为空，唤醒增强
        if self.buffer_enhance.buffer_level == 0:
            self.session.set_enhance(False, 0)

        buff_down = self.buffer_download.buffer_level
        buff_enh = self.buffer_enhance.buffer_level
        action_down = self.buffer_download.history[self.idx_enh]
        action_enh = self.action_enh

        seg_idx = self.idx_enh

        # 记录下载完成时间
        self.segment_download_time[seg_idx] = self.global_time

        max_level = self.enh_to_level.get(action_enh, 3)

        if action_enh != 0:
            # 帧级数据可用分辨率走 EDF 队列；段第 0 帧播放提前量 = 水位 − 段长
            # （下载完成钩子中 buff_down 已含本段），不再赠送半段播放时间。
            play_start_lead_ms = buff_down - self.seg_time
            decisions, admitted_steps, seg_vmaf = self._schedule_segment_frames(
                seg_idx, action_down, max_level, play_start_lead_ms, buff_enh)

            if decisions is not None:
                self.segment_max_enhance_level[seg_idx] = max_level
                enhanced_count = int(np.sum(decisions > 0))
                self.segment_enhanced_count[seg_idx] = enhanced_count
                self.segment_actual_vmaf[seg_idx] = seg_vmaf
                self.total_enhanced_frames += enhanced_count
                self.total_frames += self.frames_per_seg

                actual_enhance_time = admitted_steps * STEP_MS
                self.total_pushed_steps += admitted_steps
                # 网格与连续毫秒成本对账（本段残差理论上界 6ms）
                continuous_ms = sum(
                    self.enhance_time[action_down][int(decisions[f])] / self.frames_per_seg
                    for f in range(self.frames_per_seg) if decisions[f] > 0)
                residual = grid_residual_ms(continuous_ms, admitted_steps)
                if abs(residual) > 6.0 + 1e-6:
                    raise AssertionError(
                        f"seg {seg_idx} 增量方案网格记账残差超界: {residual:.3f}ms")
                # 零时长条目不得入增强缓冲（会话引擎在双等待 tie 下会空转死循环）
                if actual_enhance_time > 0:
                    self.buffer_enhance.push(
                        segment_index=seg_idx, decision=action_enh,
                        segment_time=actual_enhance_time)
            else:
                # 无逐帧数据的分辨率（720p/1080p）：保留 BONES 整段可行性语义
                if buff_down - buff_enh - self.seg_time < self.enhance_time[action_down][action_enh]:
                    action_enh = 0
                    self.action_enh = 0
                    self._push_base_segment(seg_idx, action_down)
                else:
                    self.segment_enhanced_count[seg_idx] = self.frames_per_seg
                    self.segment_max_enhance_level[seg_idx] = max_level
                    full_steps = to_steps(self.enhance_time[action_down][action_enh])
                    self.buffer_enhance.push(
                        segment_index=seg_idx, decision=action_enh,
                        segment_time=full_steps * STEP_MS)
        else:
            self._push_base_segment(seg_idx, action_down)
        self.idx_enh += 1

        # 计算等待时间，避免下载缓冲溢出
        buffer_level = self.buffer_download.buffer_level
        max_level = self.buffer_download.max_level
        time_wait = buffer_level + self.seg_time - max_level
        time_wait = max(0, time_wait)
        self.session.set_download(False, time_wait)

    def _push_base_segment(self, seg_idx, action_down):
        """不增强段的记账。"""
        self.segment_enhanced_count[seg_idx] = 0
        self.segment_max_enhance_level[seg_idx] = 0
        res = self.bitrate_to_res.get(action_down, None)
        if res is not None and res in self.res_to_idx:
            vmafs = self._get_frame_vmafs(seg_idx, res)
            if vmafs is not None:
                self.segment_actual_vmaf[seg_idx] = np.mean(vmafs[0])
                self.total_frames += self.frames_per_seg
        # 基础段不入增强缓冲（history 默认 0；零时长条目会让会话引擎空转）

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

        # 增量传输特有指标：增强帧比例统一取帧队列口径（只计有逐帧数据的段，
        # 三控制器分母一致；M5 修复 FL 85.8% vs INC 99.0% 的 cosmetic 差异）
        queue_summary = self.frame_queue.summary()
        enh_frame_ratio = queue_summary["enhanced frame ratio"]

        # M3：帧队列总守恒断言（入队=增强+错过，GPU 承诺=推入）
        self.frame_queue.final_assert(self.total_pushed_steps)

        report_simulator = {
            "playback time": self.global_time,
            "average quality vmaf": quality_vmaf_avg,
            "average oscillation vmaf": oscillation_vmaf,
            "freeze time": self.buffer_download.idle_time,
            "freeze time sec": self.buffer_download.idle_time / 1000.0,
            "freeze event": self.buffer_download.idle_event,
            "average bandwidth": avg_bandwidth,
            "qoe": qoe,
            "rank mode": self.frame_queue.rank_mode,
            "enhanced frame ratio": enh_frame_ratio,
            "total enhanced frames": self.total_enhanced_frames,
            "total frames": self.total_frames,
            "frame queue": queue_summary,
            "download history": history_down,
            "enhance history": history_enh,
        }

        if self.logger.is_verbose():
            title = "Simulator Report" if title == "" else title
            self.logger.summarize(title, report_simulator)

        return report_simulator
