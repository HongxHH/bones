import sys
sys.path.append("..")
from simulation import Simulator, MonitorNetworkModel
import math
import numpy as np
import argparse
import os


class BONES_CA(Simulator):
    """
    Content-Aware BONES (CA-BONES)
    将内容复杂度因子融入 Lyapunov 优化的奖励项。

    核心修改：
      原始: obj_reward = vmaf_enh_avg + gamma_p
      CA:   obj_reward = c_i * vmaf_enh_avg + gamma_p

    其中 c_i 为第 i 段的内容复杂度因子，由段级 VMAF 增强增益与平均增益的比值推导。
    """

    def __init__(self, args):
        super(BONES_CA, self).__init__(args)
        self.gamma_p = args.gamma_p
        self.V = (args.download_buffer * 1000 - self.seg_time) * self.seg_time / (self.vmaf_down_avg[-1] + self.gamma_p)
        self.V = self.V * args.V_multiplier
        self.is_avg_size = False

        # 内容感知参数
        self.content_aware = getattr(args, 'content_aware', True)
        self.c_min = getattr(args, 'c_min', 0.7)
        self.c_max = getattr(args, 'c_max', 1.8)
        self.ca_mode = getattr(args, 'ca_mode', 'base_known')  # 'scalar', 'per_level', 'base_known', 'oracle', 'predicted'

        # 多段前瞻参数
        self.lookahead_N = getattr(args, 'lookahead_N', 1)  # 1=单段贪心, N>1=多段前瞻
        self.smooth_alpha = getattr(args, 'smooth_alpha', 0.0)  # VMAF 平滑惩罚系数

        # 预计算内容因子（scalar 或 per_level）
        if self.content_aware and self.ca_mode in ('scalar', 'per_level', 'base_known'):
            self.content_factors = self._compute_content_factors(mode='per_level')
        else:
            self.content_factors = np.ones(self.seg_size.shape[0])

        # predicted 模式：加载训练好的 VMAF 增益预测模型
        if self.content_aware and self.ca_mode == 'predicted':
            self.predicted_vmaf_seg = self._load_prediction_model()

        # 内容感知延迟预测：用段级平均 VMAF 增益作为内容复杂度代理
        # 高增益段 = 高复杂度 = 高增强延迟
        self.ca_latency = getattr(args, 'ca_latency', False)
        if self.ca_latency:
            self.latency_factor = self._compute_latency_factor()
        else:
            self.latency_factor = np.ones(self.seg_size.shape[0])

        self.quick_start = args.quick_start
        self.model_size = 310 * 1000 * 8
        self.model_ready = False
        self.download_threshold = 2 * self.seg_time
        if self.quick_start and ("imdn_bbb" not in self.args.enhance_path):
            raise ValueError("Quick start currently only supports content-aware IMDN.")

        self.monitor = args.monitor
        if args.monitor:
            self.network = MonitorNetworkModel(self.logger, self.net_time, self.bandwidth, self.latency,
                                                   self.monitor_hook, self.buffer_enhance, self.buffer_download)

        self.autotune = args.autotune
        if args.autotune:
            self._load_autotune_table(args)

        if "imdn" in self.args.enhance_path:
            pass

        if self.quick_start:
            self.old_vmaf = self.vmaf_enh_avg.copy()
            self.old_vmaf[:, -2:] = -np.inf
            self.vmaf_enh_avg[:, 1:] = -np.inf

    def _compute_content_factors(self, mode='per_level'):
        """
        为每段计算内容复杂度因子。

        mode='scalar': 一个标量 c_i 调制所有增强级别
        mode='per_level': 为每个增强级别 e 计算独立的 c_i(e)

        增强增益 = vmaf_enh[r, e] - vmaf_enh[r, 0]（e > 0）
        内容因子 = 段增益 / 全局平均增益

        返回:
            scalar 模式: np.array of shape (num_segment,)
            per_level 模式: np.array of shape (num_segment, num_method+1)
        """
        num_seg = self.seg_size.shape[0]
        num_bitrate = self.vmaf_enh_seg.shape[1]
        num_method = self.vmaf_enh_seg.shape[2]

        if mode == 'scalar':
            # 标量模式：对所有 bitrate 和 e>0 取平均
            seg_gains = np.zeros(num_seg)
            for i in range(num_seg):
                gains = []
                for r in range(num_bitrate):
                    for e in range(1, num_method):
                        gain = self.vmaf_enh_seg[i, r, e] - self.vmaf_enh_seg[i, r, 0]
                        if gain > 0:
                            gains.append(gain)
                seg_gains[i] = np.mean(gains) if len(gains) > 0 else 0.0
            avg_gain = np.mean(seg_gains[seg_gains > 0]) if np.any(seg_gains > 0) else 1.0
            content_factors = seg_gains / avg_gain
            content_factors = np.clip(content_factors, self.c_min, self.c_max)
            return content_factors

        elif mode == 'per_level':
            # per-level 模式：为每个增强级别 e 计算独立的内容因子
            # shape: (num_seg, num_method)
            seg_gains_level = np.zeros((num_seg, num_method))
            for i in range(num_seg):
                for e in range(num_method):
                    if e == 0:
                        seg_gains_level[i, e] = 0.0  # e=0 无增益
                        continue
                    gains = []
                    for r in range(num_bitrate):
                        gain = self.vmaf_enh_seg[i, r, e] - self.vmaf_enh_seg[i, r, 0]
                        if gain > 0:
                            gains.append(gain)
                    seg_gains_level[i, e] = np.mean(gains) if len(gains) > 0 else 0.0

            # 全局平均增益（按级别）
            avg_gains_level = np.zeros(num_method)
            for e in range(1, num_method):
                valid = seg_gains_level[:, e] > 0
                avg_gains_level[e] = np.mean(seg_gains_level[valid, e]) if np.any(valid) else 1.0

            # per-level 内容因子
            content_factors_level = np.ones((num_seg, num_method))
            for e in range(1, num_method):
                if avg_gains_level[e] > 0:
                    content_factors_level[:, e] = seg_gains_level[:, e] / avg_gains_level[e]
            content_factors_level = np.clip(content_factors_level, self.c_min, self.c_max)
            # e=0 设为 1.0（不影响，因为增益为 0）
            content_factors_level[:, 0] = 1.0
            return content_factors_level

    def _load_prediction_model(self):
        """
        加载训练好的 VMAF 增益预测模型，为每段预测 VMAF。
        使用批量预测加速，并处理 -inf base quality。

        返回:
            predicted_vmaf_seg: np.array of shape (num_seg, num_bitrate, num_level)
        """
        import pickle
        import json

        # 加载模型
        model_path = getattr(self.args, 'predictor_path', '../workspace/vmaf_gain_predictor.pkl')
        feature_path = getattr(self.args, 'feature_path', '../player/data/content_features/bbb_features.json')

        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        model = data['model']

        # 加载内容特征
        features_data = json.load(open(feature_path))
        features = {}
        for item in features_data:
            features[item['segment_index']] = np.array([
                item['texture_complexity'],
                item['motion_intensity'],
                item['scene_change_prob'],
                item['texture_std'],
                item['motion_std'],
            ])

        num_seg, num_bitrate, num_level = self.vmaf_enh_seg.shape
        predicted_vmaf_seg = np.zeros_like(self.vmaf_enh_seg)

        # 计算有效 base quality 的平均值（用于替换 -inf）
        valid_base = self.vmaf_enh_seg[:, :, 0]
        valid_base = valid_base[np.isfinite(valid_base)]
        avg_base = np.mean(valid_base) if len(valid_base) > 0 else 60.0

        # 批量构建所有 (segment, bitrate, level>0) 的特征
        all_feats = []
        all_indices = []
        for i in range(num_seg):
            if i not in features:
                continue
            base_feat = features[i]
            for r in range(num_bitrate):
                for e in range(1, num_level):
                    all_feats.append(self._build_pred_feat(base_feat, r, e))
                    all_indices.append((i, r, e))

        # 批量预测
        if all_feats:
            all_feats_arr = np.array(all_feats)
            pred_gains = model.predict(all_feats_arr)
            for (i, r, e), pred_gain in zip(all_indices, pred_gains):
                base_q = self.vmaf_enh_seg[i, r, 0]
                if not np.isfinite(base_q):
                    base_q = avg_base
                predicted_vmaf_seg[i, r, e] = base_q + pred_gain

        # e=0: 使用实际 base quality，-inf 替换为平均值
        for i in range(num_seg):
            for r in range(num_bitrate):
                base_q = self.vmaf_enh_seg[i, r, 0]
                predicted_vmaf_seg[i, r, 0] = base_q if np.isfinite(base_q) else avg_base

        # 裁剪到合理范围 [0, 100]
        predicted_vmaf_seg = np.clip(predicted_vmaf_seg, 0, 100)

        # 保留实际 VMAF 中的 -inf 掩码（避免控制器选择无效动作）
        inf_mask = ~np.isfinite(self.vmaf_enh_seg)
        predicted_vmaf_seg[inf_mask] = -np.inf

        return predicted_vmaf_seg

    def _compute_latency_factor(self):
        """
        计算每段的延迟缩放因子。
        用段级平均 VMAF 增益作为内容复杂度代理：高增益 = 高复杂度 = 高延迟。
        归一化使全局均值为 1.0。
        """
        num_seg, num_bitrate, num_level = self.vmaf_enh_seg.shape
        seg_gains = []
        for i in range(num_seg):
            gains = []
            for r in range(num_bitrate):
                base = self.vmaf_enh_seg[i, r, 0]
                if not np.isfinite(base):
                    continue
                for e in range(1, num_level):
                    enh = self.vmaf_enh_seg[i, r, e]
                    if np.isfinite(enh):
                        gains.append(enh - base)
            seg_gains.append(np.mean(gains) if gains else 0)

        seg_gains = np.array(seg_gains)
        global_avg = np.mean(seg_gains[seg_gains > 0]) if np.any(seg_gains > 0) else 1.0
        factor = seg_gains / global_avg
        # 裁剪到合理范围 [0.5, 2.0]
        factor = np.clip(factor, 0.5, 2.0)
        return factor

    def _build_pred_feat(self, base_feat, bitrate_idx, level_idx):
        """构建预测特征向量（与训练时一致）"""
        feat = list(base_feat)
        feat.append(bitrate_idx / 4.0)
        feat.append(level_idx / 4.0)
        feat.append(base_feat[0] * base_feat[1])
        feat.append(base_feat[0] ** 2)
        feat.append(base_feat[1] ** 2)
        feat.append(base_feat[0] * level_idx / 4.0)
        feat.append(base_feat[1] * level_idx / 4.0)
        return np.array(feat)

    def control_download_start(self):
        if self.lookahead_N > 1:
            self.action_down, self.action_enh = self._lookahead_control(self.lookahead_N)
        else:
            self.action_down, self.action_enh = self._bones_control()

        if self.logger.is_verbose():
            self.logger.write("(control) download {} enhance {} content_factor {:.3f}".format(
                self.action_down, self.action_enh,
                self.content_factors[self.idx_down] if self.idx_down < len(self.content_factors) else 1.0
            ))
        return

    def _bones_control(self):
        if self.autotune:
            self._tune_params()

        buff_down = self.buffer_download.buffer_level
        buff_enh = self.buffer_enhance.buffer_level
        lat_scale = self.latency_factor[self.idx_down] if (self.ca_latency and self.idx_down < len(self.latency_factor)) else 1.0
        obj_drift = buff_down * self.seg_time + buff_enh * self.enhance_time * lat_scale

        # ===== 内容感知修改：奖励项 =====
        if self.content_aware and self.ca_mode == 'oracle':
            # Oracle 模式：直接使用段级 VMAF（上界，假设完美内容预测）
            obj_reward = self.vmaf_enh_seg[self.idx_down] + self.gamma_p
        elif self.content_aware and self.ca_mode == 'predicted':
            # Predicted 模式：用训练好的模型预测段级 VMAF 增益
            obj_reward = self.predicted_vmaf_seg[self.idx_down] + self.gamma_p
        elif self.content_aware and self.ca_mode == 'scalar':
            # Scalar 模式：一个标量内容因子调制所有增强级别
            c_i = self.content_factors[self.idx_down] if self.idx_down < len(self.content_factors) else 1.0
            base_vmaf = self.vmaf_enh_avg[:, 0:1]
            enh_gain = self.vmaf_enh_avg - base_vmaf
            effective_vmaf = base_vmaf + c_i * enh_gain
            obj_reward = effective_vmaf + self.gamma_p
        elif self.content_aware and self.ca_mode == 'per_level':
            # Per-level 模式：为每个增强级别 e 使用独立的内容因子 c_i(e)
            c_level = self.content_factors[self.idx_down]  # (num_method+1,)
            base_vmaf = self.vmaf_enh_avg[:, 0:1]  # (num_bitrate, 1)
            enh_gain = self.vmaf_enh_avg - base_vmaf  # (num_bitrate, num_method+1)
            effective_vmaf = base_vmaf + c_level[np.newaxis, :] * enh_gain
            obj_reward = effective_vmaf + self.gamma_p
        elif self.content_aware and self.ca_mode == 'base_known':
            # Base-known 模式：使用段级实际 base quality + per-level 调制增益
            # base_known_seg[i, r] = vmaf_enh_seg[i, r, 0]（下载后可测量）
            # gain 仍用 per-level 内容因子调制平均增益
            c_level = self.content_factors[self.idx_down]  # (num_method+1,)
            base_vmaf_seg = self.vmaf_enh_seg[self.idx_down, :, 0:1]  # (num_bitrate, 1)
            enh_gain_avg = self.vmaf_enh_avg - self.vmaf_enh_avg[:, 0:1]  # (num_bitrate, num_method+1)
            effective_vmaf = base_vmaf_seg + c_level[np.newaxis, :] * enh_gain_avg
            obj_reward = effective_vmaf + self.gamma_p
        else:
            # 原始 BONES：使用平均 VMAF
            obj_reward = self.vmaf_enh_avg + self.gamma_p

        if self.is_avg_size:
            seg_size = self.bitrate[:, None] * self.seg_time
        else:
            seg_size = self.seg_size[self.idx_down][:, None]

        obj = (obj_drift - self.V * obj_reward) / seg_size

        mask_late = ((buff_enh + self.enhance_time * lat_scale - buff_down) > 0)
        obj[mask_late] = math.inf

        decision = obj.argmin()
        action_down = decision // obj.shape[1]
        action_enh = decision % obj.shape[1]

        if self.quick_start and (not self.model_ready) and buff_down >= self.download_threshold:
            self.extra_download = self.model_size
            self.model_ready = True
            self.vmaf_enh_avg = self.old_vmaf
        return action_down, action_enh

    def _lookahead_control(self, N=3):
        """
        多段前瞻控制：联合优化未来 N 段的码率和增强级别选择。

        缓冲区模型（连续速率近似）：
        - 决策点到下一决策点的时间 = 当前段下载时间 dl_time
        - 在此期间：增强以 seg_time/en_time 速率从 bd 搬到 be，播放以 1:1 消费 be
        - 下载完成后 bd += seg_time
        - 每步用更新后的缓冲区计算 BONES 目标函数（漂移 + 奖励）
        """
        buff_down = self.buffer_download.buffer_level
        buff_enh = self.buffer_enhance.buffer_level
        idx = self.idx_down
        num_br = self.seg_size.shape[1]
        num_lvl = self.enhance_time.shape[1]
        seg_time = self.seg_time
        max_buffer = self.buffer_download.max_level

        # 带宽估计（作为未来 N 段的恒定预测）
        bandwidth = self.tput_history.estimate()[0] if hasattr(self, 'tput_history') else 1000.0

        best_obj = float('inf')
        best_first = (0, 0)

        from itertools import product
        for seq in product(range(num_br), range(num_lvl), repeat=N):
            actions = [(seq[2*k], seq[2*k+1]) for k in range(N)]

            bd = buff_down
            be = buff_enh
            total_obj = 0.0
            total_freeze = 0.0
            total_seg_size = 0.0
            prev_vmaf = None

            for k in range(N):
                if idx + k >= self.seg_size.shape[0]:
                    break
                d, e = actions[k]

                # 下载时间（ms）：seg_size 单位是 bytes，bandwidth 单位是 kbps
                dl_time = self.seg_size[idx + k, d] * 8.0 / bandwidth if bandwidth > 0 else 1e9
                # 增强时间（ms）
                en_time = self.enhance_time[d, e]

                # 获取该段 VMAF
                if self.content_aware and self.ca_mode == 'base_known' and idx + k < self.vmaf_enh_seg.shape[0]:
                    c_level = self.content_factors[idx + k] if idx + k < len(self.content_factors) else np.ones(num_lvl)
                    base_v = self.vmaf_enh_seg[idx + k, d, 0]
                    gain_avg = self.vmaf_enh_avg[d, e] - self.vmaf_enh_avg[d, 0]
                    vmaf = base_v + c_level[e] * gain_avg
                elif self.content_aware and self.ca_mode == 'oracle' and idx + k < self.vmaf_enh_seg.shape[0]:
                    vmaf = self.vmaf_enh_seg[idx + k, d, e]
                else:
                    vmaf = self.vmaf_enh_avg[d, e]

                # VMAF 平滑惩罚
                smooth_pen = 0.0
                if prev_vmaf is not None and hasattr(self, 'smooth_alpha') and self.smooth_alpha > 0:
                    smooth_pen = self.smooth_alpha * abs(vmaf - prev_vmaf)
                prev_vmaf = vmaf

                # 单步目标：drift - V * reward，除以 seg_size 与单步 BONES 保持一致
                drift = bd * seg_time + be * en_time
                reward = vmaf + self.gamma_p - smooth_pen
                seg_size_k = self.seg_size[idx + k, d] if idx + k < self.seg_size.shape[0] else self.seg_size[-1, d]
                step_obj = (drift - self.V * reward) / seg_size_k
                total_obj += step_obj
                total_seg_size += seg_size_k

                # === 缓冲区更新（连续速率近似）===
                T = dl_time  # 到下一决策点的时间

                # 1. 增强：从 bd 搬到 be，速率 = seg_time / en_time
                if en_time > 0 and bd > 0:
                    enh_rate = seg_time / en_time
                    enh_amount = min(bd, enh_rate * T)
                    bd -= enh_amount
                    be += enh_amount

                # 2. 播放：消费 be，速率 1:1；be 不足时产生卡顿
                if be >= T:
                    be -= T
                else:
                    freeze = T - be
                    total_freeze += freeze
                    be = 0.0

                # 3. 当前段下载完成，加入下载缓冲区
                bd += seg_time

                # 限制最大缓冲区
                bd = min(bd, max_buffer)
                be = min(be, max_buffer)

            # 显式卡顿惩罚：按 QoE 公式 avg_freeze/10 缩放，除以平均段大小与 step_obj 量纲一致
            avg_seg_size = total_seg_size / N if N > 0 and total_seg_size > 0 else 1.0
            freeze_penalty = (self.V * total_freeze / seg_time / 100.0) / avg_seg_size
            total_obj += freeze_penalty

            if total_obj < best_obj:
                best_obj = total_obj
                best_first = actions[0]

        return best_first[0], best_first[1]

    def _tune_params(self):
        num_bandwidth = 25
        min_bandwidth = 200
        step_bandwidth = 200
        num_variance = 10
        min_variance = 500
        step_variance = 500
        num_latency = 2
        min_latency = 20
        step_latency = 80

        bandwidth, latency = self.tput_history.estimate()
        variance = np.var(self.tput_history.throughput_window)

        def get_idx(val, min_val, step_val, num_val):
            if val < min_val:
                idx_val = 0
            else:
                idx_val = int((val - min_val) // step_val)
            idx_val = min(num_val - 1, idx_val)
            return idx_val
        idx_bandwidth = get_idx(bandwidth, min_bandwidth, step_bandwidth, num_bandwidth)
        idx_variance = get_idx(variance, min_variance, step_variance, num_variance)
        idx_latency = get_idx(latency, min_latency, step_latency, num_latency)

        self.gamma_p = self.gamma_p_table[idx_bandwidth, idx_variance, idx_latency]
        self.V_multiplier = self.V_multiplier_table[idx_bandwidth, idx_variance, idx_latency]
        self.V = (self.buffer_download.max_level - self.seg_time) * self.seg_time / (self.vmaf_down_avg[-1] + self.gamma_p)
        self.V = self.V * self.V_multiplier

        if self.logger.is_verbose():
            self.logger.write("(autotune) gamma_p {} V_multiplier {}".format(
                self.gamma_p, self.V_multiplier
            ))
        return

    def control_download_finish(self):
        if self.buffer_enhance.buffer_level == 0:
            self.session.set_enhance(False, 0)

        buff_down = self.buffer_download.buffer_level
        buff_enh = self.buffer_enhance.buffer_level
        action_down = self.buffer_download.history[self.idx_enh]
        action_enh = self.action_enh
        if buff_down - buff_enh - self.seg_time < self.enhance_time[action_down][action_enh]:
            self.action_enh = 0

        if self.action_enh != 0:
            self.buffer_enhance.push(segment_index=self.idx_enh, decision=self.action_enh,
                                 segment_time=self.enhance_time[self.action_down][self.action_enh])
        self.idx_enh += 1
        buffer_level = self.buffer_download.buffer_level
        max_level = self.buffer_download.max_level
        time_wait = buffer_level + self.seg_time - max_level
        time_wait = max(0, time_wait)
        self.session.set_download(False, time_wait)
        return

    def control_enhance_start(self):
        return

    def control_enhance_finish(self):
        return

    def monitor_hook(self, buffer_download, buffer_enhance, throughput, latency, remain_size, total_time):
        lat_scale = self.latency_factor[self.idx_down] if (self.ca_latency and self.idx_down < len(self.latency_factor)) else 1.0
        obj_drift = buffer_download * self.seg_time + buffer_enhance * self.enhance_time[self.action_down][self.action_enh] * lat_scale

        # Compute reward for current action
        if self.content_aware and self.ca_mode == 'oracle':
            obj_reward_now = self.vmaf_enh_seg[self.idx_down][self.action_down][self.action_enh] + self.gamma_p
        elif self.content_aware and self.ca_mode == 'predicted':
            obj_reward_now = self.predicted_vmaf_seg[self.idx_down][self.action_down][self.action_enh] + self.gamma_p
        elif self.content_aware and self.ca_mode == 'scalar':
            c_i = self.content_factors[self.idx_down] if self.idx_down < len(self.content_factors) else 1.0
            base = self.vmaf_enh_avg[self.action_down, 0]
            gain = self.vmaf_enh_avg[self.action_down][self.action_enh] - base
            obj_reward_now = base + c_i * gain + self.gamma_p
        elif self.content_aware and self.ca_mode == 'per_level':
            c_level = self.content_factors[self.idx_down]
            base = self.vmaf_enh_avg[self.action_down, 0]
            gain = self.vmaf_enh_avg[self.action_down][self.action_enh] - base
            obj_reward_now = base + c_level[self.action_enh] * gain + self.gamma_p
        elif self.content_aware and self.ca_mode == 'base_known':
            c_level = self.content_factors[self.idx_down]
            base_seg = self.vmaf_enh_seg[self.idx_down, self.action_down, 0]
            gain_avg = self.vmaf_enh_avg[self.action_down][self.action_enh] - self.vmaf_enh_avg[self.action_down, 0]
            obj_reward_now = base_seg + c_level[self.action_enh] * gain_avg + self.gamma_p
        else:
            obj_reward_now = self.vmaf_enh_avg[self.action_down][self.action_enh] + self.gamma_p
        score_now = (obj_drift - self.V * obj_reward_now) / remain_size

        # Compute rewards for all other actions
        lat_scale = self.latency_factor[self.idx_down] if (self.ca_latency and self.idx_down < len(self.latency_factor)) else 1.0
        obj_drift = buffer_download * self.seg_time + buffer_enhance * self.enhance_time * lat_scale
        if self.content_aware and self.ca_mode == 'oracle':
            obj_reward = self.vmaf_enh_seg[self.idx_down] + self.gamma_p
        elif self.content_aware and self.ca_mode == 'predicted':
            obj_reward = self.predicted_vmaf_seg[self.idx_down] + self.gamma_p
        elif self.content_aware and self.ca_mode == 'scalar':
            c_i = self.content_factors[self.idx_down] if self.idx_down < len(self.content_factors) else 1.0
            base_vmaf = self.vmaf_enh_avg[:, 0:1]
            enh_gain = self.vmaf_enh_avg - base_vmaf
            obj_reward = base_vmaf + c_i * enh_gain + self.gamma_p
        elif self.content_aware and self.ca_mode == 'per_level':
            c_level = self.content_factors[self.idx_down]
            base_vmaf = self.vmaf_enh_avg[:, 0:1]
            enh_gain = self.vmaf_enh_avg - base_vmaf
            obj_reward = base_vmaf + c_level[np.newaxis, :] * enh_gain + self.gamma_p
        elif self.content_aware and self.ca_mode == 'base_known':
            c_level = self.content_factors[self.idx_down]
            base_vmaf_seg = self.vmaf_enh_seg[self.idx_down, :, 0:1]
            enh_gain_avg = self.vmaf_enh_avg - self.vmaf_enh_avg[:, 0:1]
            obj_reward = base_vmaf_seg + c_level[np.newaxis, :] * enh_gain_avg + self.gamma_p
        else:
            obj_reward = self.vmaf_enh_avg + self.gamma_p
        seg_size = self.seg_size[self.idx_down][:, None]
        score_other = (obj_drift - self.V * obj_reward) / seg_size
        mask_late = ((buffer_enhance + self.enhance_time - buffer_download) > 0)
        score_other[mask_late] = math.inf
        decision = score_other.argmin()
        action_down = decision // score_other.shape[1]
        action_enh = decision % score_other.shape[1]

        if (action_down != self.action_down) and (action_enh != self.action_enh) and (score_other[action_down, action_enh] < score_now):
            self.action_down = action_down
            self.action_enh = action_enh
            new_size = self.seg_size[self.idx_down][self.action_down]
            if self.logger.is_verbose():
                self.logger.write("(monitor) download {} enhance {}".format(
                    self.action_down, self.action_enh
                ))
            return new_size
        return None

    def _load_autotune_table(self, args):
        path2name = {
            "../data/bbb/nas_1080ti.json": "nas",
            "../data/bbb/imdn_div2k_2080ti.json": "div2080",
            "../data/bbb/imdn_bbb_2080ti.json": "bbb2080",
            "../data/bbb/imdn_div2k_3060ti.json": "div3060",
            "../data/bbb/imdn_bbb_3060ti.json": "bbb3060",
        }
        abs_path = os.path.dirname(os.path.abspath(__file__)) + "/"
        tables = np.load(abs_path + f"autotune_bones_{path2name[args.enhance_path]}.npz")
        self.gamma_p_table, self.V_multiplier_table = tables["gamma_p_table"], tables["V_multiplier_table"]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CA-BONES: Content-Aware Neural-Enhanced Streaming.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--log_path', default="bones_ca_test.txt")
    parser.add_argument('--manifest_path', default='../data/bbb/bbb_video_info.json')
    parser.add_argument('--trace_path', default='../data/trace/dummy.json')
    parser.add_argument('--enhance_path', default='../data/bbb/nas_1080ti.json')
    parser.add_argument('--video_length', default=None)
    parser.add_argument('--throughput', default='ewma')
    parser.add_argument('-db', '--download_buffer', default=60, type=int)
    parser.add_argument('-eb', '--enhance_buffer', default=60, type=int)
    parser.add_argument('-bm', '--bandwidth_multiplier', default=1, type=float)
    parser.add_argument('-sm', '--speed_multiplier', default=1, type=float)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('-gp', '--gamma_p', type=float, default=10)
    parser.add_argument('-vm', '--V_multiplier', type=float, default=1)
    parser.add_argument('-m', '--monitor', action='store_true')
    parser.add_argument('-a', '--autotune', action='store_true')
    parser.add_argument('-q', '--quick_start', action='store_true')
    parser.add_argument('--content_aware', action='store_true', default=True)
    parser.add_argument('--ca_mode', type=str, default='base_known', choices=['scalar', 'per_level', 'base_known', 'oracle', 'predicted'])
    parser.add_argument('--c_min', type=float, default=0.7)
    parser.add_argument('--c_max', type=float, default=1.8)
    parser.add_argument('--ca_latency', action='store_true', default=False, help='Enable content-aware latency prediction')

    args = parser.parse_args()
    simulator = BONES_CA(args)
    simulator.run()
    report = simulator.report()
