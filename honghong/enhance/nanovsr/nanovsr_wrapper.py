"""
NanoVSR 集成包装器
将 ECCV 2026 NanoVSR 模型集成到 bones 增强管线中。

NanoVSR 特点：
- 4x 超分辨率
- 双向循环传播（forward + backward）
- RepVGG 可重参数化块（推理时融合为单路 3x3 卷积）
- 超轻量（226k ~ 5.4M 参数）
- 输入：(b, t, c, h, w)，输出：(b, t, c, h*4, w*4)
"""

import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import time

# 添加 NanoVSR 源码路径
NANOVSR_DIR = os.path.dirname(os.path.abspath(__file__))
if NANOVSR_DIR not in sys.path:
    sys.path.insert(0, NANOVSR_DIR)

from models.nanovsr import NanoVSR


class NanoVSRWrapper:
    """NanoVSR 模型包装器，提供帧级增强接口"""

    # 模型配置映射（根据 README Model Zoo）
    MODEL_CONFIGS = {
        '226k': {'num_feat': 32, 'num_blocks': 8},
        '644k': {'num_feat': 48, 'num_blocks': 12},
        '1.7m': {'num_feat': 64, 'num_blocks': 20},
        '5.4m': {'num_feat': 96, 'num_blocks': 30},
    }

    def __init__(self, model_variant='644k', checkpoint_path=None, device='cuda',
                 temporal_window=15, fp16=False):
        """
        初始化 NanoVSR 包装器

        Args:
            model_variant: 模型变体 ('226k', '644k', '1.7m', '5.4m')
            checkpoint_path: 权重文件路径，None 则使用默认路径
            device: 计算设备 ('cuda' or 'cpu')
            temporal_window: 时序窗口大小（帧），用于 chunk 处理
            fp16: 是否使用半精度推理
        """
        self.model_variant = model_variant
        self.device = device
        self.temporal_window = temporal_window
        self.fp16 = fp16

        if model_variant not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model variant: {model_variant}. "
                           f"Available: {list(self.MODEL_CONFIGS.keys())}")

        config = self.MODEL_CONFIGS[model_variant]

        # 构建模型（训练拓扑，加载后融合为推理拓扑）
        self.model = NanoVSR(
            num_feat=config['num_feat'],
            num_blocks=config['num_blocks'],
            deploy=False
        )

        # 加载权重
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                NANOVSR_DIR, 'checkpoints', f'nanovsr_{model_variant}.pth'
            )

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        # 处理不同的 checkpoint 格式
        if 'params' in checkpoint:
            self.model.load_state_dict(checkpoint['params'], strict=True)
        elif 'state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        else:
            self.model.load_state_dict(checkpoint, strict=True)

        # 融合为推理模式（重参数化）
        self.model.switch_to_deploy()
        self.model = self.model.to(device)
        self.model.eval()

        if fp16 and device == 'cuda':
            self.model = self.model.half()

        # 计算参数量
        self.num_params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def enhance_frames(self, frames):
        """
        增强一组帧（4x 超分辨率）

        Args:
            frames: numpy array 或 torch tensor，形状 (t, h, w, c) 或 (t, c, h, w)
                    像素值范围 [0, 255] uint8 或 [0, 1] float

        Returns:
            enhanced_frames: numpy array (t, h*4, w*4, c)，uint8 [0, 255]
        """
        # 输入格式标准化
        if isinstance(frames, np.ndarray):
            if frames.ndim == 4:
                if frames.shape[-1] == 3:  # (t, h, w, c)
                    frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
                else:  # (t, c, h, w)
                    frames = torch.from_numpy(frames).float()
            else:
                raise ValueError(f"Expected 4D input, got shape {frames.shape}")

            # 归一化到 [0, 1]
            if frames.max() > 1.0:
                frames = frames / 255.0

        frames = frames.to(self.device)
        if self.fp16 and self.device == 'cuda':
            frames = frames.half()

        t, c, h, w = frames.shape

        # Chunk 处理：按时序窗口分块
        outputs = []
        for start in range(0, t, self.temporal_window):
            end = min(start + self.temporal_window, t)
            chunk = frames[start:end].unsqueeze(0)  # (1, chunk_t, c, h, w)

            # 模型推理
            out_chunk = self.model(chunk)  # (1, chunk_t, c, h*4, w*4)
            outputs.append(out_chunk.squeeze(0))

        output = torch.cat(outputs, dim=0)  # (t, c, h*4, w*4)

        # 后处理：clamp 到 [0, 1]，转 uint8
        output = output.clamp(0, 1)
        if self.fp16:
            output = output.float()
        output = (output * 255).round().byte()
        output = output.permute(0, 2, 3, 1).cpu().numpy()  # (t, h*4, w*4, c)

        return output

    @torch.no_grad()
    def enhance_single_frame(self, frame):
        """
        增强单帧（使用时序窗口=1，会有质量损失，仅用于测试）

        Args:
            frame: numpy array (h, w, c) uint8

        Returns:
            enhanced_frame: numpy array (h*4, w*4, c) uint8
        """
        frames = frame[np.newaxis, ...]  # (1, h, w, c)
        return self.enhance_frames(frames)[0]

    def measure_latency(self, input_size=(270, 480), num_frames=15, num_warmup=3, num_runs=10):
        """
        测量模型推理延迟

        Args:
            input_size: (h, w) 输入分辨率
            num_frames: 时序窗口帧数
            num_warmup: 预热次数
            num_runs: 测量次数

        Returns:
            dict: 延迟统计信息
        """
        h, w = input_size
        dummy_input = torch.randn(1, num_frames, 3, h, w).to(self.device)
        if self.fp16 and self.device == 'cuda':
            dummy_input = dummy_input.half()

        # 预热
        for _ in range(num_warmup):
            _ = self.model(dummy_input)
            if self.device == 'cuda':
                torch.cuda.synchronize()

        # 测量
        latencies = []
        for _ in range(num_runs):
            if self.device == 'cuda':
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = self.model(dummy_input)
            if self.device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        latencies = np.array(latencies)
        return {
            'model': f'NanoVSR-{self.model_variant}',
            'input_size': f'{h}x{w}',
            'num_frames': num_frames,
            'num_params': self.num_params,
            'fp16': self.fp16,
            'latency_mean_ms': float(np.mean(latencies)),
            'latency_std_ms': float(np.std(latencies)),
            'latency_min_ms': float(np.min(latencies)),
            'latency_max_ms': float(np.max(latencies)),
            'fps': float(num_frames / (np.mean(latencies) / 1000)),
        }

    def get_memory_usage(self):
        """获取模型显存占用（MB）"""
        if self.device != 'cuda':
            return 0
        return torch.cuda.memory_allocated() / 1024 / 1024


if __name__ == '__main__':
    # 快速测试
    print("Loading NanoVSR-644k...")
    wrapper = NanoVSRWrapper(model_variant='644k', device='cuda', temporal_window=15)
    print(f"  Parameters: {wrapper.num_params:,}")
    print(f"  Memory: {wrapper.get_memory_usage():.1f} MB")

    # 延迟测量
    print("\nMeasuring latency...")
    for res in [(180, 320), (270, 480), (360, 640)]:
        stats = wrapper.measure_latency(input_size=res, num_frames=15, num_warmup=3, num_runs=10)
        print(f"  {res[0]}x{res[1]}: {stats['latency_mean_ms']:.1f} ± {stats['latency_std_ms']:.1f} ms "
              f"({stats['fps']:.1f} FPS)")

    # 单帧增强测试
    print("\nSingle frame enhancement test...")
    dummy_frame = np.random.randint(0, 255, (270, 480, 3), dtype=np.uint8)
    enhanced = wrapper.enhance_single_frame(dummy_frame)
    print(f"  Input: {dummy_frame.shape} -> Output: {enhanced.shape}")
    print(f"  Output dtype: {enhanced.dtype}, range: [{enhanced.min()}, {enhanced.max()}]")
    print("\nAll tests passed!")
