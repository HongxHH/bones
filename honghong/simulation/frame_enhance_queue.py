"""帧级增强工作队列（M3/M4）

替代三套控制器里的"整段可行性闸门 + seg_time/2 时间赠送 + 解析式可增强帧数"。

语义（与播放器 pacnet_enhancer.py 的 EDF 可调度性前瞻同族）：
- 段下载完成即入队 120 个帧作业，每帧带**绝对播放截止时刻**；
- 帧 f 的截止时刻 = 该段起播墙钟 + f × 每帧播放时长；
  段起播提前量 = 下载缓冲水位 − 段长（下载完成钩子中水位已含本段）；
- GPU 单处理机，已承诺但未完成的工作 = 增强缓冲水位 backlog；
- 调度按"单位成本增益"贪心选帧/选档，并用 EDF 可行性检验保证每个被接收的
  作业都能在截止前完成（按截止时刻排序，累计 GPU 时间 ≤ 截止余量）；
- 未被接收的档级/帧到点按基础（或更低档）质量播放，记 missed；
  部分帧只接收了低档（如 level1 未接收 level2）属合法的部分增强。

守恒：每段 入队帧数 = 增强帧(level>0) + 完全错过帧(level=0)；
累计推入增强缓冲的 GPU 时间 = Σ 接收档级成本。
"""
from simulation.gpu_grid import STEP_MS, to_steps


class FrameEnhanceQueue:
    def __init__(self, frames_per_seg=120, step_ms=STEP_MS, rank_mode="edf"):
        """rank_mode:
        - "edf"（默认，可部署因果版）：候选按播放截止时刻排序，等增益假设下
          EDF 是最优可行性保持顺序；不使用任何增强后信息。
        - "oracle"（全知上界）：候选按真实增量 VMAF/增量成本密度降序，
          只能在带 --oracle 的离线脚本中使用，不得作为可部署结果（M5/B6）。
        """
        self.frames_per_seg = frames_per_seg
        self.step_ms = step_ms
        self.rank_mode = rank_mode
        # 累计统计
        self.total_enqueued = 0
        self.total_enhanced_frames = 0   # 最终 level>0 的帧
        self.total_missed_frames = 0     # 最终 level=0 的帧
        self.total_admitted_steps = 0    # 已承诺 GPU 网格步数
        self.total_segments = 0

    @staticmethod
    def frame_deadlines(now_ms, play_start_lead_ms, frame_play_ms, num_frames):
        """每帧的绝对播放截止时刻（ms 墙钟）。"""
        return [now_ms + play_start_lead_ms + f * frame_play_ms for f in range(num_frames)]

    @staticmethod
    def _edf_feasible(admitted, backlog_steps, now_ms, step_ms):
        """EDF 单处理机可调度性检验。

        admitted: list of (deadline_ms, cost_steps)；backlog_steps: 已承诺的更早工作。
        按截止时刻升序累计工作量，任一作业超过其截止余量即不可行。
        """
        if not admitted:
            return True
        ordered = sorted(admitted, key=lambda x: x[0])
        cum = backlog_steps
        for deadline, cost in ordered:
            cum += cost
            avail_steps = (deadline - now_ms) / step_ms
            if cum > avail_steps + 1e-9:
                return False
        return True

    def schedule_segment(self, now_ms, seg_idx, play_start_lead_ms, frame_play_ms,
                         frame_cost_steps, frame_vmaf, max_level, backlog_steps):
        """对一个新下载段做帧级 EDF 调度。

        Args:
            now_ms: 当前墙钟（global_time, ms）
            seg_idx: 段索引
            play_start_lead_ms: 本段第 0 帧距现在的播放提前量（ms）
            frame_play_ms: 每帧播放时长（ms）
            frame_cost_steps: np.array(num_frames, 4) int，每帧各档 GPU 网格成本（level0=0）
            frame_vmaf: np.array(num_frames, 4)，每帧各档 VMAF
            max_level: BONES 决策的增强档上限（1..3）
            backlog_steps: 已推入增强缓冲、尚未完成的 GPU 工作（网格步）

        Returns:
            decisions: np.array(num_frames) int，每帧最终承诺的增强档（0=基础）
            admitted_steps: 本段被接收的总 GPU 网格步数
            seg_vmaf: 段帧均 VMAF（完成帧取增强档，错过帧取基础档）
        """
        import numpy as np

        num_frames = frame_vmaf.shape[0]
        deadlines = self.frame_deadlines(
            now_ms, play_start_lead_ms, frame_play_ms, num_frames)

        cur_level = [0] * num_frames
        admitted = []  # (deadline, cost_steps, frame, level)

        # 候选档级步进：(frame, level, 增量成本steps, 增量VMAF, 密度)
        candidates = []
        for f in range(num_frames):
            for l in range(1, max_level + 1):
                dc = int(frame_cost_steps[f][l] - frame_cost_steps[f][l - 1])
                dv = float(frame_vmaf[f][l] - frame_vmaf[f][l - 1])
                if dc <= 0 or dv <= 0:
                    continue
                candidates.append((f, l, dc, dv, dv / dc))

        def try_admit(ff, ll):
            """检验并接收帧 ff 的档级 ll（单步，含已承诺集合）。"""
            trial = [(deadlines[ff],
                      int(frame_cost_steps[ff][ll] - frame_cost_steps[ff][ll - 1]))]
            base = [(d, c) for (d, c, _, _) in admitted]
            if self._edf_feasible(base + trial, backlog_steps, now_ms, self.step_ms):
                admitted.append((trial[0][0], trial[0][1], ff, ll))
                cur_level[ff] = ll
                return True
            return False

        if self.rank_mode == "oracle":
            # 全知上界：按真密度降序；多趟处理档级链依赖（收 l 必须先收 l-1）
            remaining = sorted(candidates, key=lambda c: -c[4])
            for _pass in range(max_level):
                progressed = False
                next_remaining = []
                for f, l, dc, dv, density in remaining:
                    if cur_level[f] >= l:
                        continue
                    # 该档级缺失的低档链一并打包检验
                    chain = [(ff, ll) for ff, ll in
                             ((f, x) for x in range(cur_level[f] + 1, l + 1))]
                    trial = [(deadlines[ff],
                              int(frame_cost_steps[ff][ll] - frame_cost_steps[ff][ll - 1]))
                             for ff, ll in chain]
                    base = [(d, c) for (d, c, _, _) in admitted]
                    if self._edf_feasible(base + trial, backlog_steps, now_ms, self.step_ms):
                        for (ff, ll), (dd, cc) in zip(chain, trial):
                            admitted.append((dd, cc, ff, ll))
                            cur_level[ff] = ll
                        progressed = True
                    else:
                        # 当前档级不可行，更低档级留给后续趟次重试
                        if l == cur_level[f] + 1:
                            next_remaining.append((f, l, dc, dv, density))
                remaining = next_remaining
                if not progressed:
                    break
        elif self.rank_mode == "edf":
            # 可部署因果版：按播放截止时刻（帧序）扫描，每帧逐档取到最深可行档。
            # 等增益假设下 EDF 为最优可行性保持顺序；不使用增强后 VMAF。
            for f in range(num_frames):
                for l in range(1, max_level + 1):
                    if not try_admit(f, l):
                        break
        else:
            raise ValueError(f"unknown rank_mode {self.rank_mode}")

        decisions = np.array(cur_level, dtype=int)
        admitted_steps = int(sum(c for (_, c, _, _) in admitted))

        seg_vmaf = float(np.mean([frame_vmaf[f][decisions[f]] for f in range(num_frames)]))

        enhanced = int(np.sum(decisions > 0))
        missed = num_frames - enhanced
        # 守恒断言
        if enhanced + missed != num_frames:
            raise AssertionError(
                f"seg {seg_idx} 帧队列不守恒: enhanced {enhanced} + missed {missed} != {num_frames}")

        self.total_enqueued += num_frames
        self.total_enhanced_frames += enhanced
        self.total_missed_frames += missed
        self.total_admitted_steps += admitted_steps
        self.total_segments += 1

        return decisions, admitted_steps, seg_vmaf

    def final_assert(self, total_pushed_steps):
        """仿真结束时守恒总断言：入队帧全部分决，GPU 承诺量与推入量一致。"""
        if self.total_enqueued != self.total_enhanced_frames + self.total_missed_frames:
            raise AssertionError(
                f"帧队列总量不守恒: enqueued {self.total_enqueued} != "
                f"enhanced {self.total_enhanced_frames} + missed {self.total_missed_frames}")
        if self.total_admitted_steps != total_pushed_steps:
            raise AssertionError(
                f"GPU 记账不守恒: admitted {self.total_admitted_steps} != pushed {total_pushed_steps}")

    def summary(self):
        return {
            "rank mode": self.rank_mode,
            "segments scheduled": self.total_segments,
            "frames enqueued": self.total_enqueued,
            "frames enhanced": self.total_enhanced_frames,
            "frames missed (base)": self.total_missed_frames,
            "enhanced frame ratio": (
                self.total_enhanced_frames / self.total_enqueued
                if self.total_enqueued else 0.0),
            "gpu admitted ms": self.total_admitted_steps * self.step_ms,
        }
