"""
DRL 神经增强视频流控制器

使用深度强化学习（DQN）替代 BONES 的 Lyapunov 优化，做码率+增强级别的联合决策。

核心设计：
- 状态空间：网络状态 + 缓冲状态 + 内容特征 + 历史决策
- 动作空间：联合 (bitrate_level, enhancement_level)，共 5*4=20 个离散动作
- 奖励：QoE = 平均质量 - 卡顿惩罚 - 振荡惩罚
- 算法：DQN（Deep Q-Network），支持离线训练（CQL）和在线推理

与 BONES 的对比：
- BONES: 凸优化 + 静态质量/延迟表，有理论保证但无法利用高维状态
- DRL: 神经网络 + 内容感知状态，能学习复杂策略但需要训练数据
"""
import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from istream_player.config.config import PlayerConfig
from istream_player.core.nes import NESController
from istream_player.core.module import Module, ModuleOption
from istream_player.models import AdaptationSet

from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.core.enhancer import Enhancer
from istream_player.core.scheduler import Scheduler
from istream_player.core.bw_meter import BandwidthMeter
from istream_player.utils.content_features import ContentFeatureStore


class QNetwork(nn.Module):
    """
    Q 网络 - 输入状态，输出每个动作的 Q 值

    架构：3 层全连接网络，ReLU 激活
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@ModuleOption("drl_nes", requires=[DownloadBufferImpl, EnhanceBufferImpl, Enhancer, Scheduler, BandwidthMeter], default=False)
class DRLNESController(Module, NESController):
    """
    DRL 神经增强视频流控制器

    使用预训练的 Q 网络做联合 (bitrate, enhancement) 决策。
    支持内容感知（通过内容特征作为状态输入）和非内容感知模式。
    """
    log = logging.getLogger("DRLNESController")

    # 动作空间：5 个码率级别 * 4 个增强级别 = 20 个联合动作
    NUM_BITRATES = 5
    NUM_ENHANCE_LEVELS = 4
    ACTION_DIM = NUM_BITRATES * NUM_ENHANCE_LEVELS

    # 状态维度
    # 网络: last_throughput, avg_throughput, throughput_std (3)
    # 缓冲: download_buffer, enhance_buffer (2)
    # 内容: texture, motion, scene_change, texture_std, motion_std (5)
    # 历史: last_bitrate, last_enhance_level (2)
    # 段信息: segment_index_norm (1)
    STATE_DIM = 13

    def __init__(self):
        super().__init__()

        self.first_download = True
        self.current_segment_index = 0
        self.last_bitrate = 0
        self.last_enhance_level = 0

        # 网络历史
        self.throughput_history = []
        self.max_history_len = 10

        # Q 网络
        self.q_network: Optional[QNetwork] = None
        self.device = "cpu"
        self.model_path = None

        # 内容特征
        self.feature_store: Optional[ContentFeatureStore] = None
        self.use_content_features = True

        # 探索率（推理时设为 0）
        self.epsilon = 0.0

    async def setup(self,
                    config: PlayerConfig,
                    download_buffer: DownloadBufferImpl,
                    enhance_buffer: EnhanceBufferImpl,
                    enhancer: Enhancer,
                    scheduler: Scheduler,
                    bw_meter: BandwidthMeter,
                    **kwargs):
        self.buffer_size = config.buffer_duration
        self.download_buffer = download_buffer
        self.enhance_buffer = enhance_buffer
        self.enhancer = enhancer
        self.scheduler = scheduler
        self.bw_meter = bw_meter

        self.device = getattr(config, 'enhancer_device', 'cpu')
        self.model_path = getattr(config, 'drl_model_path', 'models/drl_nes_qnet.pt')

        # 加载内容特征
        feature_file = getattr(config, 'content_feature_file', 'data/content_features/bbb_features.json')
        self.use_content_features = getattr(config, 'use_content_features', True)
        if self.use_content_features:
            self.feature_store = ContentFeatureStore(feature_file)
            self.log.info(f"Content features loaded: {len(self.feature_store.features)} segments")

        # 加载 Q 网络
        self._load_model()

    def _load_model(self):
        """加载预训练的 Q 网络"""
        self.q_network = QNetwork(self.STATE_DIM, self.ACTION_DIM).to(self.device)

        if self.model_path and os.path.exists(self.model_path):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                self.q_network.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.q_network.load_state_dict(checkpoint)
            self.log.info(f"Q-network loaded from: {self.model_path}")
        else:
            self.log.warning(f"No pre-trained model found at {self.model_path}, using random weights")
            self.log.warning("Please train the model first using scripts/train_drl.py")

        self.q_network.eval()

    def update_selection(self, adaptation_sets: Dict[int, AdaptationSet], index: int) -> Tuple[Dict[int, int], Dict[int, int]]:
        """
        DRL 决策主入口

        Args:
            adaptation_sets: 自适应集字典
            index: 当前段索引

        Returns:
            (下载动作字典, 增强动作字典)
        """
        self.current_segment_index = index

        download_actions = dict()
        enhance_actions = dict()

        for adaptation_set in adaptation_sets.values():
            if self.first_download:
                download_action, enhance_action = 0, 0
                self.first_download = False
            else:
                download_action, enhance_action = self._drl_decision(adaptation_set)

            download_actions[adaptation_set.id] = download_action
            enhance_actions[adaptation_set.id] = enhance_action
            self.last_bitrate = download_action
            self.last_enhance_level = enhance_action

        return download_actions, enhance_actions

    def _drl_decision(self, adaptation_set: AdaptationSet) -> Tuple[int, int]:
        """
        使用 Q 网络做决策

        Returns:
            (bitrate_level, enhancement_level)
        """
        if not self.enhancer.is_ready():
            return 0, 0

        # 构建状态
        state = self._build_state(adaptation_set)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 前向推理
        with torch.no_grad():
            q_values = self.q_network(state_tensor).cpu().numpy()[0]

        # 过滤无效动作（最高码率不需要增强）
        quality_table = self.enhancer.get_quality_table()
        for br_idx in range(self.NUM_BITRATES):
            for lvl_idx in range(self.NUM_ENHANCE_LEVELS):
                action_idx = br_idx * self.NUM_ENHANCE_LEVELS + lvl_idx
                if br_idx < quality_table.shape[0] and lvl_idx < quality_table.shape[1]:
                    if not np.isfinite(quality_table[br_idx, lvl_idx]):
                        q_values[action_idx] = -np.inf

        # epsilon-greedy（推理时 epsilon=0）
        if np.random.random() < self.epsilon:
            valid_actions = np.where(np.isfinite(q_values))[0]
            action_idx = np.random.choice(valid_actions)
        else:
            action_idx = np.argmax(q_values)

        # 解码联合动作
        bitrate_level = action_idx // self.NUM_ENHANCE_LEVELS
        enhance_level = action_idx % self.NUM_ENHANCE_LEVELS

        return int(bitrate_level), int(enhance_level)

    def _build_state(self, adaptation_set: AdaptationSet) -> np.ndarray:
        """
        构建状态向量

        Returns:
            shape (STATE_DIM,) 的归一化状态向量
        """
        # 网络状态
        current_tput = self.bw_meter.get_last_n_throughput(1)[0] if hasattr(self.bw_meter, 'get_last_n_throughput') else 0.0
        avg_tput = np.mean(self.throughput_history[-5:]) if self.throughput_history else 0.0
        tput_std = np.std(self.throughput_history[-5:]) if len(self.throughput_history) >= 2 else 0.0

        # 归一化网络状态（假设最大 10Mbps）
        max_tput = 10_000_000
        net_state = np.array([
            min(current_tput / max_tput, 1.0),
            min(avg_tput / max_tput, 1.0),
            min(tput_std / max_tput, 1.0),
        ], dtype=np.float32)

        # 缓冲状态（归一化到 [0, 1]）
        buff_down = self.download_buffer.buffer_level(continuous=True)
        buff_enh = self.enhance_buffer.buffer_level(continuous=True)
        buffer_state = np.array([
            min(buff_down / self.buffer_size, 1.0),
            min(buff_enh / self.buffer_size, 1.0),
        ], dtype=np.float32)

        # 内容特征
        if self.use_content_features and self.feature_store is not None:
            content_state = self.feature_store.get_feature_vector(self.current_segment_index)
        else:
            content_state = np.zeros(5, dtype=np.float32)

        # 历史决策（归一化）
        history_state = np.array([
            self.last_bitrate / (self.NUM_BITRATES - 1),
            self.last_enhance_level / (self.NUM_ENHANCE_LEVELS - 1),
        ], dtype=np.float32)

        # 段索引（归一化，假设最多 200 段）
        seg_norm = min(self.current_segment_index / 200.0, 1.0)

        state = np.concatenate([
            net_state,        # 3
            buffer_state,     # 2
            content_state,    # 5
            history_state,    # 2
            [seg_norm],       # 1
        ])

        assert state.shape[0] == self.STATE_DIM, f"State dim mismatch: {state.shape[0]} vs {self.STATE_DIM}"
        return state

    def record_throughput(self, throughput: float):
        """记录吞吐量（供训练时使用）"""
        self.throughput_history.append(throughput)
        if len(self.throughput_history) > self.max_history_len:
            self.throughput_history.pop(0)
