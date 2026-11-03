"""
BONES + 帧级渐进增强 联合仿真器（M3/M4 帧级 EDF 队列版）

M3 起与 bones_frame_level 共用 FrameEnhanceQueue：BONES 段级码率/最大增强档
决策不变，下载完成后帧作业带播放截止墙钟入队，EDF 前瞻逐帧定档，错过播放
截止的帧按基础质量播放（missed 显式记账）。原"帧级 DP + 增量选帧"双阶段
逻辑已被队列统一替代。
"""
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from control.bones import BONES
from simulation.gpu_grid import STEP_MS, to_steps, grid_residual_ms
from simulation.frame_enhance_queue import FrameEnhanceQueue


class BONESCombined(BONES):
    def __init__(self, args):
        super().__init__(args)

        from control.frame_quality_results import load_per_frame_data

        self.per_frame_data = load_per_frame_data(args)

        self.bitrate_to_res = {0: 240, 1: 360, 2: 480, 3: 720, 4: 1080}
        self.res_to_idx = {240: 0, 360: 1, 480: 2}
        self.enh_to_level = {0: 0, 1: 1, 2: 2, 3: 3, 4: 3}
        self.level_to_name = {0: "base", 1: "sr_low", 2: "sr_medium", 3: "sr_high"}

        self.frames_per_seg = 120
        self.segment_actual_vmaf = {}
        self.segment_frame_decisions = {}

        self.total_enhanced_frames = 0
        self.total_frames = 0
        self.level_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        # M3：帧级 EDF 工作队列；M5 默认 EDF 因果排序，--oracle 为全知上界
        self.oracle_rank = bool(getattr(args, "oracle_rank", False))
        rank_mode = "oracle" if self.oracle_rank else "edf"
        self.frame_queue = FrameEnhanceQueue(self.frames_per_seg, rank_mode=rank_mode)
        self.total_pushed_steps = 0

        print(f"[BONESCombined] 逐帧数据加载完成: {len(self.per_frame_data['results'])} 个测量")

    def _get_frame_vmafs(self, seg_idx, res):
        vmafs = {}
        for level in range(4):
            name = self.level_to_name[level]
            key = f"{res}_{name}_{seg_idx}"
            if key in self.per_frame_data["results"]:
                vmafs[level] = np.array(self.per_frame_data["results"][key])
            else:
                return None
        return vmafs

    def _schedule_segment_frames(self, seg_idx, res_idx, max_level,
                                 play_start_lead_ms, backlog_ms):
        res = self.bitrate_to_res.get(res_idx, None)
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
                    method_idx = min(level, self.enhance_time.shape[1] - 1)
                    frame_cost_steps[f][level] = to_steps(
                        self.enhance_time[res_idx][method_idx] / float(n))

        decisions, admitted_steps, seg_vmaf = self.frame_queue.schedule_segment(
            now_ms=self.global_time, seg_idx=seg_idx,
            play_start_lead_ms=play_start_lead_ms,
            frame_play_ms=self.seg_time / n,
            frame_cost_steps=frame_cost_steps, frame_vmaf=frame_vmaf,
            max_level=max_level, backlog_steps=to_steps(backlog_ms))
        return decisions, admitted_steps, seg_vmaf

    def control_download_finish(self):
        """下载完成收尾（M3 帧级 EDF 队列版）。"""
        if self.buffer_enhance.buffer_level == 0:
            self.session.set_enhance(False, 0)

        buff_down = self.buffer_download.buffer_level
        buff_enh = self.buffer_enhance.buffer_level
        action_down = self.buffer_download.history[self.idx_enh]
        action_enh = self.action_enh
        seg_idx = self.idx_enh

        max_level = self.enh_to_level.get(action_enh, 3)
        res = self.bitrate_to_res.get(action_down, None)
        frame_data_ok = (action_enh != 0 and res is not None and res in self.res_to_idx)

        if frame_data_ok:
            play_start_lead_ms = buff_down - self.seg_time
            decisions, admitted_steps, seg_vmaf = self._schedule_segment_frames(
                seg_idx, self.res_to_idx[res], max_level, play_start_lead_ms, buff_enh)

            if decisions is not None:
                self.segment_frame_decisions[seg_idx] = decisions
                self.segment_actual_vmaf[seg_idx] = seg_vmaf
                self.total_enhanced_frames += int(np.sum(decisions > 0))
                self.total_frames += self.frames_per_seg
                for level in range(4):
                    self.level_counts[level] += int(np.sum(decisions == level))
                self.total_pushed_steps += admitted_steps

                continuous_ms = sum(
                    self.enhance_time[self.res_to_idx[res]][min(int(lv), self.enhance_time.shape[1] - 1)] / self.frames_per_seg
                    for lv in decisions if lv > 0)
                residual = grid_residual_ms(continuous_ms, admitted_steps)
                if abs(residual) > 6.0 + 1e-6:
                    raise AssertionError(
                        f"seg {seg_idx} 联合方案网格记账残差超界: {residual:.3f}ms")
                # 零时长条目不得入增强缓冲（会话引擎双等待 tie 下空转死循环）
                if admitted_steps * STEP_MS > 0:
                    self.buffer_enhance.push(
                        segment_index=seg_idx, decision=action_enh,
                        segment_time=admitted_steps * STEP_MS)
            else:
                frame_data_ok = False

        if not frame_data_ok:
            if action_enh != 0 and (
                    buff_down - buff_enh - self.seg_time
                    < self.enhance_time[action_down][action_enh]):
                action_enh = 0
                self.action_enh = 0
            if action_enh != 0:
                cap = self.enh_to_level[action_enh]
                self.segment_frame_decisions[seg_idx] = np.full(
                    self.frames_per_seg, cap, dtype=int)
                self.buffer_enhance.push(
                    segment_index=seg_idx, decision=action_enh,
                    segment_time=self.enhance_time[action_down][action_enh])
            else:
                # 基础段不入增强缓冲（history 默认 0）
                self.segment_frame_decisions[seg_idx] = np.zeros(
                    self.frames_per_seg, dtype=int)
                res2 = self.bitrate_to_res.get(action_down, None)
                if res2 is not None and res2 in self.res_to_idx:
                    vmafs = self._get_frame_vmafs(seg_idx, res2)
                    if vmafs is not None:
                        self.segment_actual_vmaf[seg_idx] = np.mean(vmafs[0])
                        self.total_frames += self.frames_per_seg
                        self.level_counts[0] += self.frames_per_seg

        self.idx_enh += 1

        buffer_level = self.buffer_download.buffer_level
        max_buf = self.buffer_download.max_level
        time_wait = buffer_level + self.seg_time - max_buf
        time_wait = max(0, time_wait)
        self.session.set_download(False, time_wait)

    def report(self, title=""):
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
        avg_freeze = self.buffer_download.idle_time / 1000.0 / num_seg
        avg_bandwidth = total_bandwidth / num_seg / self.seg_time
        qoe = quality_vmaf_avg - oscillation_vmaf - avg_freeze / 10

        # M3：帧队列总守恒断言
        self.frame_queue.final_assert(self.total_pushed_steps)
        # M5：增强帧比例统一取队列口径（只计有逐帧数据段，三控制器分母一致）
        enh_frame_ratio = self.frame_queue.summary()["enhanced frame ratio"]

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
            "level distribution": (
                {k: v / self.total_frames for k, v in self.level_counts.items()}
                if self.total_frames > 0 else {}),
            "frame queue": self.frame_queue.summary(),
            "total enhanced frames": self.total_enhanced_frames,
            "total frames": self.total_frames,
            "download history": history_down,
            "enhance history": history_enh,
        }

        if self.logger.is_verbose():
            title = "Simulator Report" if title == "" else title
            self.logger.summarize(title, report_simulator)

        return report_simulator
