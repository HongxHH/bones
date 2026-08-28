"""
CA-BONES: Content-Aware BONES 控制器

在 BONES 的 Lyapunov 优化框架基础上，引入内容感知的质量奖励。
核心改进（经仿真器验证）：
1. base_known: 使用段级实际 base quality（替代全局平均）
2. per_level 内容因子: 为每个增强级别计算独立的内容复杂度因子
3. 奖励 = base_quality_seg[i, r] + c_i(e) * avg_gain(r, e)

与 content_aware_bones.py 的区别：
- 不使用手工启发式预测（texture_factor × motion_factor）
- 使用从数据中计算的 per-level 内容因子
- 使用段级 base quality（可从下载后的帧测量，或离线预计算）
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


@ModuleOption("ca_bones", requires=[DownloadBufferImpl, EnhanceBufferImpl, Enhancer, Scheduler], default=False)
class CABONESController(Module, NESController):
    """
    Content-Aware BONES 控制器

    理论基础：内容感知 Lyapunov 优化
    - 将内容复杂度作为状态变量融入奖励函数
    - 保持 O(1/V) 近最优保证
    """
    log = logging.getLogger("CABONESController")

    def __init__(self):
        super().__init__()
        # BONES 算法参数
        self.gamma_p = 10.
        self.V_multiplier = 1.
        self.enhance_safety = 0.9
        self.seg_time = None
        self.first_download = True

        # 内容感知参数
        self.c_min = 0.7
        self.c_max = 1.8
        self.ca_mode = 'base_known'  # 'base_known', 'predicted', 'cares'

        # CARES 参数（内容感知资源惩罚）
        self.lambda_min = 0.0   # 高内容段的资源惩罚系数
        self.lambda_max = 5.0   # 低内容段的资源惩罚系数

        # 平滑约束参数（质量切换惩罚）
        self.smooth_alpha = 0.0  # VMAF切换惩罚系数，0=禁用
        self.last_vmaf = None  # 上一段的预测VMAF，用于VMAF差值切换惩罚

        # 段级数据（从 enhance JSON 加载）
        self.base_quality_seg = None   # (num_seg, num_bitrate) 段级 base VMAF
        self.content_factors = None    # (num_seg, num_method+1) per-level 内容因子
        self.avg_gain = None           # (num_bitrate, num_method+1) 平均增强增益
        self.num_segments = 0

        # 预测模式数据
        self.predicted_vmaf_seg = None  # (num_seg, num_bitrate, num_level) 模型预测的 VMAF

        # CARES 数据
        self.lambda_values = None        # (num_seg,) 每段的资源惩罚系数
        self.content_features = None     # dict: segment_index -> feature dict

    async def setup(self,
                    config: PlayerConfig,
                    download_buffer: DownloadBufferImpl,
                    enhance_buffer: EnhanceBufferImpl,
                    enhancer: Enhancer,
                    scheduler: Scheduler,
                    **kwargs):
        self.buffer_size = config.buffer_duration * 1000
        self.download_buffer = download_buffer
        self.enhance_buffer = enhance_buffer
        self.enhancer = enhancer
        self.scheduler = scheduler

        # 从配置读取参数
        self.c_min = getattr(config, 'ca_c_min', 0.7)
        self.c_max = getattr(config, 'ca_c_max', 1.8)
        self.ca_mode = getattr(config, 'ca_mode', 'base_known')
        self.lambda_min = getattr(config, 'cares_lambda_min', 0.0)
        self.lambda_max = getattr(config, 'cares_lambda_max', 5.0)
        self.smooth_alpha = getattr(config, 'smooth_alpha', 0.0)

        # 加载段级质量数据
        enhance_file = getattr(config, 'ca_enhance_file', None)
        if enhance_file is None:
            # 默认路径
            enhance_file = "../data/bbb/imdn_bbb_3060ti.json"

        self._load_segment_data(enhance_file)

        # 预测模式：加载训练好的 VMAF 增益预测模型
        if self.ca_mode == 'predicted':
            predictor_path = getattr(config, 'ca_predictor_path', '../workspace/vmaf_gain_predictor.pkl')
            feature_path = getattr(config, 'ca_feature_path', 'data/content_features/bbb_features.json')
            self._load_prediction_model(predictor_path, feature_path)

        # CARES 模式：加载内容特征，计算每段的资源惩罚系数 λ_i
        if self.ca_mode == 'cares':
            feature_path = getattr(config, 'ca_feature_path', 'data/content_features/bbb_features.json')
            self._init_cares(feature_path)

    def _load_segment_data(self, enhance_path: str):
        """
        从 enhance JSON 加载段级质量数据，计算 per-level 内容因子

        数据结构（原始 JSON）：
        - base_quality_vmaf: (num_bitrate-1, num_seg)
        - enhanced_quality_vmaf: (num_bitrate_enh, num_method, num_seg)
        """
        try:
            with open(enhance_path, 'r') as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            self.log.warning(f"Cannot load enhance data from {enhance_path}: {e}")
            self.log.warning("Falling back to static quality table (original BONES behavior)")
            return

        frame_rate = data['frame_rate']
        base_q = np.array(data['base_quality_vmaf'])      # (4, 159)
        enh_q = np.array(data['enhanced_quality_vmaf'])    # (3, 4, 159)

        num_bitrate_low = base_q.shape[0]  # 4
        num_bitrate_enh = enh_q.shape[0]   # 3
        num_method = enh_q.shape[1]         # 4
        num_seg = base_q.shape[1]           # 159
        num_bitrate = num_bitrate_low + 1   # 5 (4 lower + 1 highest)

        # 转置为 (num_seg, num_bitrate-1)
        base_q_seg = base_q.T  # (159, 4)
        # 最高码率设为满分
        base_q_seg = np.concatenate([base_q_seg, np.full((num_seg, 1), 100.0)], axis=1)  # (159, 5)

        # 转置增强质量为 (num_seg, num_bitrate_enh, num_method)
        enh_q_seg = enh_q.transpose((2, 0, 1))  # (159, 3, 4)
        enh_q_seg[enh_q_seg < 0] = -np.inf
        # 拼接缺失的码率（最高码率不增强）
        enh_q_seg = np.concatenate([
            enh_q_seg,
            np.full((num_seg, num_bitrate - num_bitrate_enh, num_method), -np.inf)
        ], axis=1)  # (159, 5, 4)
        # 拼接无增强选项（e=0 = base quality）
        vmaf_enh_seg = np.concatenate([base_q_seg[:, :, None], enh_q_seg], axis=2)  # (159, 5, 5)

        # 段级 base quality (e=0)
        self.base_quality_seg = base_q_seg  # (num_seg, num_bitrate)

        # 平均 VMAF 和平均增益
        vmaf_enh_avg = np.mean(vmaf_enh_seg, axis=0)  # (5, 5)
        base_avg = vmaf_enh_avg[:, 0:1]  # (5, 1)
        self.avg_gain = vmaf_enh_avg - base_avg  # (5, 5)

        # 计算 per-level 内容因子
        # c_i(e) = mean_r(actual_gain_i(r, e)) / mean_i(mean_r(actual_gain_i(r, e)))
        content_factors = np.ones((num_seg, num_method + 1))
        for e in range(1, num_method + 1):
            # 每段在级别 e 的平均增益（跨所有 bitrate，只考虑有效值）
            seg_gains = np.zeros(num_seg)
            for i in range(num_seg):
                gains = []
                for r in range(num_bitrate):
                    g = vmaf_enh_seg[i, r, e] - vmaf_enh_seg[i, r, 0]
                    if np.isfinite(g) and g > 0:
                        gains.append(g)
                seg_gains[i] = np.mean(gains) if len(gains) > 0 else 0.0
            # 全局平均增益（按级别）
            valid = seg_gains > 0
            avg_gain_e = np.mean(seg_gains[valid]) if np.any(valid) else 1.0
            # 内容因子
            if avg_gain_e > 0:
                content_factors[:, e] = seg_gains / avg_gain_e
        content_factors = np.clip(content_factors, self.c_min, self.c_max)
        content_factors[:, 0] = 1.0  # e=0 无增益

        self.content_factors = content_factors
        self.num_segments = num_seg

        self.log.info(f"CA-BONES data loaded: {num_seg} segments, "
                      f"{num_bitrate} bitrates, {num_method + 1} levels")
        self.log.info(f"  Content factor range: [{content_factors[:, 1:].min():.2f}, "
                      f"{content_factors[:, 1:].max():.2f}], "
                      f"mean={content_factors[:, 1:].mean():.3f}")

    def _load_prediction_model(self, predictor_path: str, feature_path: str):
        """
        加载训练好的 VMAF 增益预测模型（RandomForest, R²=0.909），
        为每段预计算预测 VMAF。

        Args:
            predictor_path: 模型文件路径 (.pkl)
            feature_path: 内容特征文件路径 (.json)
        """
        import pickle
        import os

        if not os.path.exists(predictor_path):
            self.log.warning(f"Predictor not found: {predictor_path}, falling back to base_known")
            self.ca_mode = 'base_known'
            return
        if not os.path.exists(feature_path):
            self.log.warning(f"Feature file not found: {feature_path}, falling back to base_known")
            self.ca_mode = 'base_known'
            return

        try:
            with open(predictor_path, 'rb') as f:
                data = pickle.load(f)
            model = data['model']

            with open(feature_path, 'r') as f:
                features_data = json.load(f)
            features = {}
            for item in features_data:
                features[item['segment_index']] = np.array([
                    item['texture_complexity'],
                    item['motion_intensity'],
                    item['scene_change_prob'],
                    item['texture_std'],
                    item['motion_std'],
                ])

            num_seg = self.num_segments
            num_bitrate = self.base_quality_seg.shape[1]
            num_level = self.avg_gain.shape[1]

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
            predicted_vmaf = np.full((num_seg, num_bitrate, num_level), -np.inf)
            # e=0: base quality
            for i in range(num_seg):
                for r in range(num_bitrate):
                    predicted_vmaf[i, r, 0] = self.base_quality_seg[i, r]

            if all_feats:
                all_feats_arr = np.array(all_feats)
                pred_gains = model.predict(all_feats_arr)
                for (i, r, e), pred_gain in zip(all_indices, pred_gains):
                    base_q = self.base_quality_seg[i, r]
                    if np.isfinite(base_q):
                        predicted_vmaf[i, r, e] = base_q + pred_gain

            # 裁剪到合理范围
            valid = np.isfinite(predicted_vmaf)
            predicted_vmaf[valid] = np.clip(predicted_vmaf[valid], 0, 100)

            self.predicted_vmaf_seg = predicted_vmaf
            self.log.info(f"CA-BONES predicted mode: model loaded, "
                          f"predicted VMAF shape={predicted_vmaf.shape}, "
                          f"valid={np.sum(valid)}/{valid.size}, "
                          f"range=[{np.nanmin(predicted_vmaf[valid]):.2f}, "
                          f"{np.nanmax(predicted_vmaf[valid]):.2f}]")
        except Exception as e:
            self.log.warning(f"Failed to load prediction model: {e}, falling back to base_known")
            self.ca_mode = 'base_known'

    def _build_pred_feat(self, base_feat, bitrate_idx, level_idx):
        """构建预测特征向量（与训练时一致，12维）"""
        feat = list(base_feat)
        feat.append(bitrate_idx / 4.0)
        feat.append(level_idx / 4.0)
        feat.append(base_feat[0] * base_feat[1])  # texture × motion
        feat.append(base_feat[0] ** 2)             # texture²
        feat.append(base_feat[1] ** 2)             # motion²
        feat.append(base_feat[0] * level_idx / 4.0)  # texture × level
        feat.append(base_feat[1] * level_idx / 4.0)  # motion × level
        return np.array(feat)

    def _init_cares(self, feature_path: str):
        """
        CARES 初始化：加载内容特征，计算每段的资源惩罚系数 λ_i。

        核心思想：
        - 高内容段（纹理/运动高）→ λ 小 → 资源惩罚小 → 值得增强
        - 低内容段（纹理/运动低）→ λ 大 → 资源惩罚大 → 不值得增强

        λ_i = λ_min + (λ_max - λ_min) × (1 - content_score_i)
        content_score_i = normalize(texture_i × motion_i) ∈ [0, 1]
        """
        import os
        if not os.path.exists(feature_path):
            self.log.warning(f"CARES: feature file not found: {feature_path}, falling back to base_known")
            self.ca_mode = 'base_known'
            return

        try:
            with open(feature_path, 'r') as f:
                features_data = json.load(f)
            self.content_features = {item['segment_index']: item for item in features_data}

            num_seg = self.num_segments
            lambda_values = np.ones(num_seg) * (self.lambda_min + self.lambda_max) / 2

            # 计算每段的 content_score = texture × motion
            scores = []
            for i in range(num_seg):
                feat = self.content_features.get(i, {})
                texture = feat.get('texture_complexity', 0)
                motion = feat.get('motion_intensity', 0)
                scores.append(texture * motion)
            scores = np.array(scores)

            # 归一化到 [0, 1]
            score_min, score_max = scores.min(), scores.max()
            if score_max > score_min:
                content_score = (scores - score_min) / (score_max - score_min)
            else:
                content_score = np.ones(num_seg) * 0.5

            # 计算 λ_i：高内容 → λ 小，低内容 → λ 大
            lambda_values = self.lambda_min + (self.lambda_max - self.lambda_min) * (1 - content_score)

            self.lambda_values = lambda_values

            self.log.info(f"CARES initialized: {num_seg} segments, "
                          f"λ range=[{lambda_values.min():.2f}, {lambda_values.max():.2f}], "
                          f"mean={lambda_values.mean():.2f}, "
                          f"content_score range=[{content_score.min():.3f}, {content_score.max():.3f}]")
        except Exception as e:
            self.log.warning(f"CARES init failed: {e}, falling back to base_known")
            self.ca_mode = 'base_known'

    def update_selection(self, adaptation_sets: Dict[int, AdaptationSet], index: int) -> Tuple[Dict[int, int], Dict[int, int]]:
        self.current_segment_index = index

        if self.seg_time is None:
            self.seg_time = self.scheduler.mpd_provider.mpd.max_segment_duration * 1000

        download_actions = dict()
        enhance_actions = dict()

        for adaptation_set in adaptation_sets.values():
            if self.first_download:
                download_action, enhance_action = 0, 0
                self.first_download = False
            else:
                download_action, enhance_action = self._ca_bones_decision(adaptation_set)
            download_actions[adaptation_set.id] = download_action
            enhance_actions[adaptation_set.id] = enhance_action

        return download_actions, enhance_actions

    def _ca_bones_decision(self, adaptation_set: AdaptationSet):
        """
        CA-BONES 决策核心

        奖励 = base_quality_seg[i, r] + c_i(e) * avg_gain(r, e) + gamma_p
        """
        if not self.enhancer.is_ready():
            return 0, 0

        idx = self.current_segment_index

        # 获取静态表（延迟表支持 per-segment 内容感知预测）
        self.enhance_time = self.enhancer.get_latency_table(segment_index=idx) * 1000 / self.enhance_safety
        base_quality = self.enhancer.get_quality_table()  # (5, 5) 静态平均

        bitrate = [rep.bandwidth for rep in adaptation_set.representations.values()]
        self.bitrate = np.array(bitrate)
        self.seg_size = np.array(bitrate)[:, None] * self.seg_time / 1000

        self.buff_down = self.download_buffer.buffer_level(continuous=True) * 1000
        self.buff_enh = self.enhance_buffer.buffer_level(continuous=True) * 1000

        # 计算内容感知奖励
        if self.ca_mode == 'cares' and self.lambda_values is not None:
            # CARES 模式：质量奖励用全局平均（稳定），资源惩罚内容感知（在 _bones_control 中处理）
            self.vmaf_enh_avg = base_quality
        elif self.ca_mode == 'predicted' and self.predicted_vmaf_seg is not None and idx < self.num_segments:
            # Predicted 模式：用训练好的模型预测段级 VMAF（R²=0.909）
            pred_vmaf = self.predicted_vmaf_seg[idx, :, :]  # (num_bitrate, num_level)
            # 对齐维度
            n_br = min(pred_vmaf.shape[0], base_quality.shape[0])
            n_lvl = min(pred_vmaf.shape[1], base_quality.shape[1])
            effective_vmaf = np.full_like(base_quality, -np.inf)
            effective_vmaf[:n_br, :n_lvl] = pred_vmaf[:n_br, :n_lvl]
            # 保持 -inf 掩码（静态表中无效的位置也设为 -inf）
            effective_vmaf[~np.isfinite(base_quality)] = -np.inf
            self.vmaf_enh_avg = effective_vmaf
        elif self.base_quality_seg is not None and self.content_factors is not None and idx < self.num_segments:
            # 使用段级 base quality + per-level 内容因子调制平均增益
            base_seg = self.base_quality_seg[idx, :]  # (num_bitrate,)
            c_level = self.content_factors[idx, :]     # (num_method+1,)

            # 对齐维度（静态表的 bitrate 数可能与段级数据不同）
            n_br = min(base_seg.shape[0], base_quality.shape[0])
            n_lvl = min(c_level.shape[0], base_quality.shape[1])

            effective_vmaf = np.zeros_like(base_quality)
            effective_vmaf[:n_br, :n_lvl] = (
                base_seg[:n_br, None] + c_level[None, :n_lvl] * self.avg_gain[:n_br, :n_lvl]
            )
            # 超出范围的部分用静态表
            effective_vmaf[n_br:, :] = base_quality[n_br:, :]
            effective_vmaf[:, n_lvl:] = base_quality[:, n_lvl:]
            # 保持 -inf 掩码
            effective_vmaf[~np.isfinite(base_quality)] = -np.inf

            self.vmaf_enh_avg = effective_vmaf
        else:
            # 回退到静态表
            self.vmaf_enh_avg = base_quality

        action_down, action_enh = self._bones_control()
        return int(action_down), int(action_enh)

    def _bones_control(self):
        """BONES Lyapunov 优化核心（CARES 模式加入内容感知资源惩罚）"""
        V = (self.buffer_size - self.seg_time) * self.seg_time / (np.max(self.vmaf_enh_avg) + self.gamma_p)
        V = V * self.V_multiplier

        obj_drift = self.buff_down * self.seg_time + self.buff_enh * self.enhance_time

        if self.ca_mode == 'cares' and self.lambda_values is not None:
            # CARES: 奖励 = VMAF + gamma_p - λ_i × enhance_time
            # λ_i 内容感知：高内容段 λ 小（值得增强），低内容段 λ 大（不值得增强）
            idx = self.current_segment_index
            lambda_i = self.lambda_values[idx] if idx < len(self.lambda_values) else 0.0
            obj_reward = self.vmaf_enh_avg + self.gamma_p - lambda_i * self.enhance_time
        else:
            obj_reward = self.vmaf_enh_avg + self.gamma_p

        # VMAF切换惩罚：直接惩罚预测VMAF与上一段VMAF的差值，减少 oscillation
        if self.smooth_alpha > 0 and self.last_vmaf is not None:
            switch_penalty = self.smooth_alpha * np.abs(self.vmaf_enh_avg - self.last_vmaf)
            obj_reward = obj_reward - switch_penalty

        obj = (obj_drift - V * obj_reward) / self.seg_size

        mask_late = ((self.buff_enh + self.enhance_time - self.buff_down) > 0)
        obj[mask_late] = np.inf

        decision = obj.argmin()
        action_down = decision // obj.shape[1]
        action_enh = decision % obj.shape[1]

        # 更新上一段预测VMAF，用于下一段的切换惩罚
        self.last_vmaf = float(self.vmaf_enh_avg[action_down, action_enh])

        return int(action_down), int(action_enh)
