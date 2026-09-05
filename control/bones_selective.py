import sys
sys.path.append("..")
import os
import math
import numpy as np
import argparse

from control.bones import BONES


# 选择性增强仿真器: 在BONES的Lyapunov框架内, 只替换"决策时使用的增强质量矩阵"来源
#   static : 全局静态平均(等价原始BONES, 基线)
#   oracle : 当前段真实质量(上帝视角, 选择性增强的理论上限)
#   pred   : GBDT对当前段的OOF预测质量(本方法)
# report统一用 vmaf_enh_seg 真值计质量, 三条线唯一差别是决策质量矩阵, 构成受控对比。
#
# decouple(解耦)模式: 下载码率决策永远用稳定的static全局平均(与原BONES完全一致, 预测噪声
#   无法污染下载, 杜绝码率翻转→下载超时→卡顿爆炸); 仅在码率确定后用段级质量在该码率行内
#   选择增强方法。收益纯粹来自"同码率下更聪明地选择增强"。
class SelectiveBONES(BONES):
    def __init__(self, args):
        super(SelectiveBONES, self).__init__(args)
        self.quality_mode = args.quality_mode
        self.static_quality = self.vmaf_enh_avg.copy()   # 全局静态平均(BONES原貌)
        # 收缩系数alpha: 段级质量向static收缩; 0=纯BONES, 1=完全段级
        self.alpha = getattr(args, "alpha", 1.0)
        self.decouple = getattr(args, "decouple", False)
        # 门控: 仅当段级决策相对static决策的obj优势超过margin才偏离, 否则保持BONES决策
        self.gating = getattr(args, "gating", False)
        self.margin = getattr(args, "margin", 0.0)
        self.gating_gaps = []
        # 算力预算准入: 预测/真值增益低于阈值的(码率,增强)组合屏蔽为不可选, 阈值越高越省算力
        self.budget_thresh = getattr(args, "budget_thresh", None)
        self.seg_quality_matrix = None
        if self.quality_mode != "static":
            matrix_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "..", "workspace", "gain_predictor")
            fname = {"oracle": "oracle_enh.npy",
                     "pred": "pred_enh_oof.npy"}[self.quality_mode]
            self.seg_quality_matrix = np.load(os.path.join(matrix_dir, fname))
        self.enh_level_count = np.zeros(self.vmaf_enh_avg.shape[1], dtype=int)

    def _shrink(self, seg_q, ref):
        """段级质量向静态参考收缩, 无效位(-inf)保持-inf"""
        if not (self.quality_mode == "pred" and self.alpha < 1.0):
            return seg_q
        invalid = (ref == -np.inf)
        diff = np.where(invalid, 0.0, seg_q - ref)
        mix = ref + self.alpha * diff
        mix[invalid] = -np.inf
        return mix

    def _budget_mask(self, q):
        """增益阈值准入: 增强组合相对同码率不增强的增益不足阈值则屏蔽(强制不增强该组合)"""
        if self.budget_thresh is None:
            return q
        q = q.copy()
        gain = q[:, 1:] - q[:, :1]          # (5,4), 无效位为-inf-有限=-inf, 自然被屏蔽
        q[:, 1:] = np.where(gain >= self.budget_thresh, q[:, 1:], -np.inf)
        return q

    def _decision_quality(self):
        """联合模式: 当前段决策质量矩阵 (num_bitrate, num_method+1)"""
        if self.seg_quality_matrix is None:
            return self.static_quality
        seg_q = self._shrink(self.seg_quality_matrix[self.idx_down], self.static_quality)
        return self._budget_mask(seg_q)

    def _enhance_only(self, action_down):
        """解耦阶段2: 固定下载码率, 仅用段级质量在该行选增强方法"""
        buff_down = self.buffer_download.buffer_level
        buff_enh = self.buffer_enhance.buffer_level
        ref_row = self.static_quality[action_down]
        seg_row = self._shrink(
            self.seg_quality_matrix[self.idx_down][action_down].copy(), ref_row)
        drift = buff_down * self.seg_time + buff_enh * self.enhance_time[action_down]
        reward = seg_row + self.gamma_p
        size = self.seg_size[self.idx_down][action_down]
        obj = (drift - self.V * reward) / size
        late = (buff_enh + self.enhance_time[action_down] - buff_down) > 0
        obj[late] = math.inf
        return int(obj.argmin())

    def _obj_matrix(self, quality):
        """复现_bones_control的联合目标矩阵(含来不及增强屏蔽), 返回(5,5)obj"""
        if self.autotune:
            self._tune_params()
        bd = self.buffer_download.buffer_level
        be = self.buffer_enhance.buffer_level
        drift = bd * self.seg_time + be * self.enhance_time
        obj = (drift - self.V * (quality + self.gamma_p)) / self.seg_size[self.idx_down][:, None]
        late = ((be + self.enhance_time - bd) > 0)
        obj[late] = math.inf
        return obj

    @staticmethod
    def _argmin2(mat):
        k = int(mat.argmin())
        return k // mat.shape[1], k % mat.shape[1]

    def _gated_control(self):
        """门控联合决策: 段级决策相对static决策优势不足margin则保持static决策"""
        Os = self._obj_matrix(self.static_quality)
        ds, es = self._argmin2(Os)
        seg_q = self._shrink(self.seg_quality_matrix[self.idx_down], self.static_quality)
        Op = self._obj_matrix(seg_q)
        dp, ep = self._argmin2(Op)
        if (dp, ep) == (ds, es):
            return ds, es
        gap = Op[ds, es] - Op[dp, ep]   # >0: seg推荐组合在seg评估下更优的幅度
        self.gating_gaps.append(gap)
        if gap >= self.margin:
            return dp, ep
        return ds, es

    def control_download_start(self):
        if self.gating and self.seg_quality_matrix is not None:
            self.action_down, self.action_enh = self._gated_control()
            if self.logger.is_verbose():
                self.logger.write("(control-gated) download {} enhance {}".format(
                    self.action_down, self.action_enh))
            return
        if self.decouple and self.seg_quality_matrix is not None:
            # 阶段1: 稳定static做完整联合决策, 下载码率与原BONES完全一致
            self.vmaf_enh_avg = self.static_quality
            self.action_down, _ = self._bones_control()
            # 阶段2: 固定码率, 段级质量只决定增强方法
            self.action_enh = self._enhance_only(self.action_down)
            if self.logger.is_verbose():
                self.logger.write("(control-decouple) download {} enhance {}".format(
                    self.action_down, self.action_enh))
            return
        # 联合模式: 决策前切换到当前段质量矩阵(static下恒等于全局平均)
        self.vmaf_enh_avg = self._decision_quality()
        super(SelectiveBONES, self).control_download_start()

    def control_download_finish(self):
        super(SelectiveBONES, self).control_download_finish()
        if self.idx_enh - 1 >= 0:
            self.enh_level_count[self.action_enh] += 1


def build_argparser():
    p = argparse.ArgumentParser(description='Selective-enhancement BONES simulator.')
    p.add_argument('--log_path', default="bones_selective.txt")
    p.add_argument('--manifest_path', default='../data/bbb/bbb_video_info.json')
    p.add_argument('--trace_path', default='../data/trace/dummy.json')
    p.add_argument('--enhance_path', default='../data/bbb/imdn_bbb_2080ti.json')
    p.add_argument('--video_length', default=None)
    p.add_argument('--throughput', default='ewma')
    p.add_argument('-db', '--download_buffer', default=60, type=int)
    p.add_argument('-eb', '--enhance_buffer', default=60, type=int)
    p.add_argument('-bm', '--bandwidth_multiplier', default=1, type=float)
    p.add_argument('-sm', '--speed_multiplier', default=1, type=float)
    p.add_argument('--verbose', action='store_true')
    p.add_argument('-gp', '--gamma_p', type=float, default=10)
    p.add_argument('-vm', '--V_multiplier', type=float, default=1)
    p.add_argument('-m', '--monitor', action='store_true')
    p.add_argument('-a', '--autotune', action='store_true')
    p.add_argument('-q', '--quick_start', action='store_true')
    p.add_argument('--quality_mode', choices=['static', 'oracle', 'pred'], default='pred')
    p.add_argument('--alpha', type=float, default=1.0)
    p.add_argument('--decouple', action='store_true')
    p.add_argument('--gating', action='store_true')
    p.add_argument('--margin', type=float, default=0.0)
    p.add_argument('--budget_thresh', type=float, default=None)
    return p


if __name__ == '__main__':
    args = build_argparser().parse_args()
    sim = SelectiveBONES(args)
    sim.run()
    rep = sim.report()
    print("QoE={:.3f} VMAF={:.3f} osc={:.3f} freeze={:.2f}s enh={}".format(
        rep["qoe"], rep["average quality vmaf"], rep["average oscillation vmaf"],
        rep["freeze time"], sim.enh_level_count.tolist()))
