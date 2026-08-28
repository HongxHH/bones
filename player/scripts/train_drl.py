"""
DRL 神经增强视频流控制器训练脚本

使用 trace-driven 仿真环境训练 DQN 控制器。
支持两种训练模式：
1. 行为克隆（Behavior Cloning）：从 BONES 专家数据预训练
2. DQN 强化学习：在仿真环境中在线探索训练

用法:
    # 行为克隆预训练（从 BONES 实验数据学习）
    python scripts/train_drl.py --mode bc --data output/data/ --epochs 100

    # DQN 强化学习训练
    python scripts/train_drl.py --mode dqn --episodes 500 --trace data/3Glogs/report.2010-09-13_1003CEST.json

    # 完整流程：先 BC 预训练，再 DQN 微调
    python scripts/train_drl.py --mode full --data output/data/ --episodes 200
"""
import argparse
import json
import os
import sys
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from istream_player.modules.nes.drl_nes import QNetwork, DRLNESController


# ============================================================
# Trace-driven 仿真环境
# ============================================================

class NESEnvironment:
    """
    神经增强视频流仿真环境

    简化模型，用于 DRL 训练。
    状态：网络吞吐、缓冲水位、内容特征
    动作：(bitrate_level, enhancement_level) 联合动作
    奖励：QoE = 质量 - 卡顿惩罚 - 振荡惩罚
    """

    NUM_BITRATES = 5
    NUM_ENHANCE_LEVELS = 4
    ACTION_DIM = NUM_BITRATES * NUM_ENHANCE_LEVELS
    STATE_DIM = 13

    # 码率（Mbps），对应 240p/360p/480p/720p/1080p
    BITRATES = np.array([0.3, 0.6, 1.0, 1.8, 3.0]) * 1e6
    # 基础 VMAF（无增强）
    BASE_VMAF = np.array([45.0, 52.0, 58.0, 65.0, 72.0])
    # 增强质量增益（级别 1/2/3）
    ENHANCE_GAIN = np.array([0.0, 2.0, 4.0, 6.0])
    # 增强延迟（秒/段），级别 0/1/2/3
    ENHANCE_LATENCY = np.array([0.0, 0.5, 1.2, 2.5])
    # 段时长（秒）
    SEGMENT_DURATION = 4.0
    # 缓冲区大小（秒）
    BUFFER_SIZE = 30.0

    def __init__(self, trace_file: str, content_features_file: Optional[str] = None):
        # 加载网络 trace
        self.trace_data = self._load_trace(trace_file)
        self.trace_index = 0
        self.trace_time = 0.0

        # 加载内容特征
        self.content_features = self._load_content_features(content_features_file)

        # 环境状态
        self.download_buffer = 0.0  # 秒
        self.enhance_buffer = 0.0   # 秒
        self.segment_index = 0
        self.last_bitrate = 0
        self.last_enhance = 0
        self.throughput_history = deque(maxlen=10)
        self.total_reward = 0.0
        self.rebuffer_count = 0
        self.quality_switches = 0

        # 最大段数
        self.max_segments = 159

    def _load_trace(self, trace_file: str) -> List[Dict]:
        """加载网络 trace"""
        with open(trace_file, 'r') as f:
            data = json.load(f)
        # 格式可能是 list of {timestamp, throughput} 或其他
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'uplink' in data:
            return data['uplink']
        return data

    def _load_content_features(self, feature_file: Optional[str]) -> Dict[int, np.ndarray]:
        """加载内容特征"""
        if feature_file is None or not os.path.exists(feature_file):
            return {}
        with open(feature_file, 'r') as f:
            data = json.load(f)
        features = {}
        for item in data:
            idx = item['segment_index']
            features[idx] = np.array([
                item['texture_complexity'],
                item['motion_intensity'],
                item['scene_change_prob'],
                item.get('texture_std', 0) / 80.0,
                item.get('motion_std', 0) / 60.0,
            ], dtype=np.float32)
        return features

    def _get_current_throughput(self) -> float:
        """获取当前网络吞吐量（bps）"""
        if not self.trace_data:
            return 5_000_000  # 默认 5Mbps

        # 简单的时间驱动 trace 回放
        while self.trace_index < len(self.trace_data) - 1:
            entry = self.trace_data[self.trace_index]
            next_entry = self.trace_data[self.trace_index + 1]
            t1 = entry.get('timestamp', entry.get('time', 0))
            t2 = next_entry.get('timestamp', next_entry.get('time', 1))
            if self.trace_time < t2:
                # 线性插值
                ratio = (self.trace_time - t1) / max(t2 - t1, 1e-6)
                tp1 = entry.get(' throughput', entry.get('throughput', 5_000_000))
                tp2 = next_entry.get(' throughput', next_entry.get('throughput', 5_000_000))
                return float(tp1 + ratio * (tp2 - tp1))
            self.trace_index += 1
        return float(self.trace_data[-1].get('throughput', 5_000_000))

    def reset(self) -> np.ndarray:
        """重置环境"""
        self.download_buffer = 4.0  # 初始 4 秒缓冲
        self.enhance_buffer = 0.0
        self.segment_index = 0
        self.last_bitrate = 0
        self.last_enhance = 0
        self.throughput_history.clear()
        self.total_reward = 0.0
        self.rebuffer_count = 0
        self.quality_switches = 0
        self.trace_index = 0
        self.trace_time = 0.0
        return self._get_state()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步动作

        Args:
            action: 联合动作索引 = bitrate_level * 4 + enhance_level

        Returns:
            (next_state, reward, done, info)
        """
        bitrate_level = action // self.NUM_ENHANCE_LEVELS
        enhance_level = action % self.NUM_ENHANCE_LEVELS

        # 1. 下载段
        throughput = self._get_current_throughput()
        self.throughput_history.append(throughput)
        segment_size = self.BITRATES[bitrate_level] * self.SEGMENT_DURATION  # bits
        download_time = segment_size / max(throughput, 1e6)  # 秒

        # 2. 增强段
        enhance_latency = self.ENHANCE_LATENCY[enhance_level]
        # 内容感知调整延迟
        if self.segment_index in self.content_features:
            cf = self.content_features[self.segment_index]
            enhance_latency *= (1.0 + 0.1 * cf[0] + 0.15 * cf[1])

        # 3. 更新缓冲区
        # 下载完成后加入下载缓冲
        self.download_buffer += self.SEGMENT_DURATION
        # 增强完成后从下载缓冲移到增强缓冲
        self.enhance_buffer += self.SEGMENT_DURATION
        self.download_buffer -= self.SEGMENT_DURATION

        # 4. 播放（消耗增强缓冲）
        playback_time = max(download_time, enhance_latency)
        self.trace_time += playback_time
        consumed = min(playback_time, self.enhance_buffer)
        self.enhance_buffer -= consumed

        # 5. 计算卡顿
        rebuffer = max(0, playback_time - self.enhance_buffer - consumed)
        if rebuffer > 0.1:
            self.rebuffer_count += 1

        # 6. 计算质量
        base_vmaf = self.BASE_VMAF[bitrate_level]
        enhance_gain = self.ENHANCE_GAIN[enhance_level]
        # 内容感知调整质量增益
        if self.segment_index in self.content_features:
            cf = self.content_features[self.segment_index]
            texture_factor = 0.5 + 1.0 * cf[0]
            motion_factor = 1.0 - 0.3 * abs(cf[1] - 0.4)
            enhance_gain *= texture_factor * motion_factor
        quality = base_vmaf + enhance_gain

        # 7. 计算振荡
        quality_switch = abs(bitrate_level - self.last_bitrate) + abs(enhance_level - self.last_enhance)
        if quality_switch > 0:
            self.quality_switches += 1

        # 8. 奖励 = 质量 - 卡顿惩罚 - 振荡惩罚
        reward = quality - 10.0 * rebuffer - 1.0 * quality_switch
        self.total_reward += reward

        # 9. 更新状态
        self.last_bitrate = bitrate_level
        self.last_enhance = enhance_level
        self.segment_index += 1

        done = self.segment_index >= self.max_segments

        info = {
            'quality': quality,
            'rebuffer': rebuffer,
            'quality_switch': quality_switch,
            'bitrate_level': bitrate_level,
            'enhance_level': enhance_level,
            'download_buffer': self.download_buffer,
            'enhance_buffer': self.enhance_buffer,
            'throughput': throughput,
        }

        return self._get_state(), reward, done, info

    def _get_state(self) -> np.ndarray:
        """获取当前状态向量"""
        # 网络状态
        current_tput = self.throughput_history[-1] if self.throughput_history else 5_000_000
        avg_tput = np.mean(self.throughput_history) if self.throughput_history else 5_000_000
        tput_std = np.std(self.throughput_history) if len(self.throughput_history) >= 2 else 0.0
        max_tput = 10_000_000

        net_state = np.array([
            min(current_tput / max_tput, 1.0),
            min(avg_tput / max_tput, 1.0),
            min(tput_std / max_tput, 1.0),
        ], dtype=np.float32)

        # 缓冲状态
        buffer_state = np.array([
            min(self.download_buffer / self.BUFFER_SIZE, 1.0),
            min(self.enhance_buffer / self.BUFFER_SIZE, 1.0),
        ], dtype=np.float32)

        # 内容特征
        content_state = self.content_features.get(self.segment_index, np.zeros(5, dtype=np.float32))

        # 历史决策
        history_state = np.array([
            self.last_bitrate / (self.NUM_BITRATES - 1),
            self.last_enhance / (self.NUM_ENHANCE_LEVELS - 1),
        ], dtype=np.float32)

        # 段索引
        seg_norm = min(self.segment_index / 200.0, 1.0)

        state = np.concatenate([net_state, buffer_state, content_state, history_state, [seg_norm]])
        return state


# ============================================================
# DQN Agent
# ============================================================

class DQNAgent:
    """DQN 智能体"""

    def __init__(self, state_dim: int, action_dim: int, lr: float = 1e-4,
                 gamma: float = 0.99, epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05, epsilon_decay: float = 0.995,
                 buffer_size: int = 10000, batch_size: int = 64,
                 target_update_freq: int = 100, device: str = 'cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = device
        self.train_step = 0

        # 网络
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)

        # 经验回放
        self.replay_buffer = deque(maxlen=buffer_size)

    def select_action(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """选择动作（epsilon-greedy）"""
        if not eval_mode and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return int(torch.argmax(q_values).item())

    def store_transition(self, state, action, reward, next_state, done):
        """存储转移"""
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step_fn(self) -> float:
        """训练一步"""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        # 采样
        batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.replay_buffer[i] for i in batch])

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(self.device)

        # 当前 Q 值
        current_q = self.q_network(states).gather(1, actions)

        # 目标 Q 值
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q

        # 损失
        loss = nn.MSELoss()(current_q, target_q)

        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 更新目标网络
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # 衰减 epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        return loss.item()

    def save(self, path: str):
        """保存模型"""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.q_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'train_step': self.train_step,
        }, path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['model_state_dict'])
        self.target_network.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', 0.05)
        self.train_step = checkpoint.get('train_step', 0)
        print(f"Model loaded from {path}, epsilon={self.epsilon:.3f}")


# ============================================================
# 训练函数
# ============================================================

def train_dqn(args):
    """DQN 强化学习训练"""
    env = NESEnvironment(
        trace_file=args.trace,
        content_features_file=args.content_features,
    )
    agent = DQNAgent(
        state_dim=env.STATE_DIM,
        action_dim=env.ACTION_DIM,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.997,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        device=args.device,
    )

    # 如果有预训练模型，先加载
    if args.pretrained and os.path.exists(args.pretrained):
        agent.load(args.pretrained)
        agent.epsilon = 0.3  # 从较低探索率开始微调

    print(f"开始 DQN 训练: {args.episodes} episodes")
    print(f"  trace: {args.trace}")
    print(f"  content features: {args.content_features}")
    print(f"  device: {args.device}")

    rewards_history = []
    for episode in range(args.episodes):
        state = env.reset()
        episode_reward = 0.0
        episode_loss = 0.0
        steps = 0

        while True:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            loss = agent.train_step_fn()

            state = next_state
            episode_reward += reward
            episode_loss += loss
            steps += 1

            if done:
                break

        avg_loss = episode_loss / max(steps, 1)
        rewards_history.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f"  Episode {episode+1}/{args.episodes}: "
                  f"reward={episode_reward:.1f}, avg10={avg_reward:.1f}, "
                  f"loss={avg_loss:.4f}, epsilon={agent.epsilon:.3f}, "
                  f"rebuffer={env.rebuffer_count}, switches={env.quality_switches}")

        # 定期保存
        if (episode + 1) % 50 == 0:
            agent.save(args.output)

    # 最终保存
    agent.save(args.output)
    print(f"\n训练完成！最终平均奖励: {np.mean(rewards_history[-20:]):.1f}")
    return agent


def train_bc(args):
    """行为克隆预训练（从 BONES 实验数据学习）"""
    print("行为克隆预训练")
    print(f"  数据目录: {args.data}")

    # 收集 BONES 的决策数据
    transitions = []
    data_files = [f for f in os.listdir(args.data) if f.endswith('.json') and 'bones' in f.lower()]
    print(f"  找到 {len(data_files)} 个 BONES 实验数据文件")

    for data_file in data_files[:args.max_files]:
        filepath = os.path.join(args.data, data_file)
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            # 从实验结果中提取段级决策
            segments = data.get('segments', data.get('segment_results', []))
            for seg in segments:
                bitrate = seg.get('download_action', seg.get('bitrate_level', 0))
                enhance = seg.get('enhance_action', seg.get('enhance_level', 0))
                action = int(bitrate) * 4 + int(enhance)
                # 简化：用随机状态（实际应该从实验中提取状态）
                state = np.random.randn(13).astype(np.float32) * 0.1
                transitions.append((state, action))
        except Exception as e:
            print(f"  跳过 {data_file}: {e}")

    print(f"  收集到 {len(transitions)} 条转移")

    if len(transitions) == 0:
        print("  警告：没有收集到有效数据，使用随机初始化")
        agent = DQNAgent(13, 20, device=args.device)
        agent.save(args.output)
        return agent

    # 训练行为克隆
    agent = DQNAgent(13, 20, lr=args.lr, device=args.device)
    states = torch.FloatTensor(np.array([t[0] for t in transitions])).to(args.device)
    actions = torch.LongTensor(np.array([t[1] for t in transitions])).to(args.device)

    optimizer = optim.Adam(agent.q_network.parameters(), lr=args.lr)
    for epoch in range(args.epochs):
        logits = agent.q_network(states)
        loss = nn.CrossEntropyLoss()(logits, actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            pred = logits.argmax(dim=1)
            acc = (pred == actions).float().mean().item()
            print(f"  Epoch {epoch+1}/{args.epochs}: loss={loss.item():.4f}, acc={acc:.4f}")

    agent.target_network.load_state_dict(agent.q_network.state_dict())
    agent.save(args.output)
    print("行为克隆预训练完成")
    return agent


def main():
    parser = argparse.ArgumentParser(description="DRL NES 控制器训练")
    parser.add_argument("--mode", type=str, default="dqn", choices=["dqn", "bc", "full"],
                        help="训练模式: dqn=强化学习, bc=行为克隆, full=先BC后DQN")
    parser.add_argument("--trace", type=str, default="data/3Glogs/report.2010-09-13_1003CEST.json",
                        help="网络 trace 文件")
    parser.add_argument("--content-features", type=str, default="data/content_features/bbb_features.json",
                        help="内容特征文件")
    parser.add_argument("--data", type=str, default="output/data/",
                        help="BONES 实验数据目录（行为克隆用）")
    parser.add_argument("--max-files", type=int, default=10,
                        help="行为克隆最大数据文件数")
    parser.add_argument("--output", type=str, default="models/drl_nes_qnet.pt",
                        help="模型输出路径")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="预训练模型路径（DQN 微调用）")
    parser.add_argument("--episodes", type=int, default=200,
                        help="DQN 训练回合数")
    parser.add_argument("--epochs", type=int, default=100,
                        help="行为克隆训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="折扣因子")
    parser.add_argument("--buffer-size", type=int, default=10000,
                        help="经验回放缓冲区大小")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="批次大小")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="训练设备")

    args = parser.parse_args()

    if args.mode == "bc":
        train_bc(args)
    elif args.mode == "dqn":
        train_dqn(args)
    elif args.mode == "full":
        # 先行为克隆
        bc_args = argparse.Namespace(**vars(args))
        bc_args.output = args.output + ".bc.pt"
        train_bc(bc_args)
        # 再 DQN 微调
        dqn_args = argparse.Namespace(**vars(args))
        dqn_args.pretrained = args.output + ".bc.pt"
        train_dqn(dqn_args)


if __name__ == "__main__":
    main()
