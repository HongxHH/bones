"""
内容特征提取模块
为神经增强视频流（NES）系统提供内容感知能力。

提取三类特征：
1. 纹理复杂度（Texture Complexity）：基于 Sobel 梯度幅值，衡量画面细节丰富程度
2. 运动强度（Motion Intensity）：基于帧间差分，衡量画面运动剧烈程度
3. 场景切换概率（Scene Change Probability）：基于帧间结构相似度突变

这些特征用于：
- 预测每段的增强质量增益（替代静态全局质量表）
- 预测每段的增强延迟（替代静态全局延迟表）
- 作为 DRL 控制器的状态输入
"""
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class SegmentContentFeatures:
    """单段视频的内容特征"""
    segment_index: int
    # 纹理复杂度：0.0（平滑）~ 1.0（高细节）
    texture_complexity: float
    # 运动强度：0.0（静止）~ 1.0（剧烈运动）
    motion_intensity: float
    # 场景切换概率：0.0（无切换）~ 1.0（确定切换）
    scene_change_prob: float
    # 段内帧数
    num_frames: int
    # 纹理复杂度标准差（衡量段内内容变化）
    texture_std: float
    # 运动强度标准差
    motion_std: float

    def to_dict(self) -> dict:
        return asdict(self)


class ContentFeatureExtractor:
    """
    内容特征提取器

    从解码后的帧序列中提取内容特征。
    支持两种使用模式：
    1. 离线预计算：从视频文件提取所有段的特征，保存为 JSON
    2. 在线提取：从解码后的帧张量实时提取（当前段的特征）
    """

    def __init__(self, segment_duration: float = 4.0, fps: float = 30.0):
        """
        初始化特征提取器

        Args:
            segment_duration: 段时长（秒）
            fps: 视频帧率
        """
        self.segment_duration = segment_duration
        self.fps = fps
        self.frames_per_segment = int(segment_duration * fps)

        # 特征归一化参数（基于经验值，可通过数据集校准）
        self.texture_max = 80.0  # Sobel 梯度幅值的典型最大值
        self.motion_max = 60.0   # 帧间差分的典型最大值

    def compute_frame_texture(self, frame: np.ndarray) -> float:
        """
        计算单帧的纹理复杂度

        使用 Sobel 算子计算梯度幅值，取平均值作为纹理复杂度。

        Args:
            frame: 帧数据，shape (H, W, 3) 或 (H, W)，dtype uint8 或 float

        Returns:
            纹理复杂度（未归一化）
        """
        if frame.ndim == 3:
            # 转为灰度
            gray = np.mean(frame, axis=2).astype(np.float32)
        else:
            gray = frame.astype(np.float32)

        # Sobel 梯度
        gx = np.zeros_like(gray)
        gy = np.zeros_like(gray)
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        gradient_mag = np.sqrt(gx ** 2 + gy ** 2)

        return float(np.mean(gradient_mag))

    def compute_frame_motion(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        计算两帧之间的运动强度

        使用帧间平均绝对差（MAD）作为运动强度度量。

        Args:
            frame1: 前一帧
            frame2: 当前帧

        Returns:
            运动强度（未归一化）
        """
        if frame1.ndim == 3:
            f1 = np.mean(frame1, axis=2).astype(np.float32)
            f2 = np.mean(frame2, axis=2).astype(np.float32)
        else:
            f1 = frame1.astype(np.float32)
            f2 = frame2.astype(np.float32)

        return float(np.mean(np.abs(f1 - f2)))

    def detect_scene_change(self, frame1: np.ndarray, frame2: np.ndarray,
                            threshold: float = 30.0) -> float:
        """
        检测两帧之间是否发生场景切换

        使用帧间直方图差异作为场景切换度量。
        返回 0~1 的概率值。

        Args:
            frame1: 前一帧
            frame2: 当前帧
            threshold: 场景切换阈值

        Returns:
            场景切换概率 0.0 ~ 1.0
        """
        if frame1.ndim == 3:
            f1 = np.mean(frame1, axis=2).astype(np.float32)
            f2 = np.mean(frame2, axis=2).astype(np.float32)
        else:
            f1 = frame1.astype(np.float32)
            f2 = frame2.astype(np.float32)

        # 计算直方图差异
        hist1, _ = np.histogram(f1, bins=32, range=(0, 255))
        hist2, _ = np.histogram(f2, bins=32, range=(0, 255))
        hist1 = hist1.astype(np.float32) / (hist1.sum() + 1e-8)
        hist2 = hist2.astype(np.float32) / (hist2.sum() + 1e-8)
        hist_diff = np.sum(np.abs(hist1 - hist2)) / 2.0  # 0~1

        # 结合帧间均值差
        mean_diff = np.abs(np.mean(f1) - np.mean(f2)) / 255.0

        # 综合得分
        score = 0.6 * hist_diff + 0.4 * mean_diff

        # 转换为概率（sigmoid-like）
        prob = 1.0 / (1.0 + np.exp(-10 * (score - 0.3)))
        return float(prob)

    def extract_segment_features(self, frames: List[np.ndarray],
                                  segment_index: int) -> SegmentContentFeatures:
        """
        从一段视频的帧列表中提取内容特征

        Args:
            frames: 段内所有帧的列表
            segment_index: 段索引

        Returns:
            SegmentContentFeatures 对象
        """
        num_frames = len(frames)
        if num_frames == 0:
            return SegmentContentFeatures(
                segment_index=segment_index,
                texture_complexity=0.0,
                motion_intensity=0.0,
                scene_change_prob=0.0,
                num_frames=0,
                texture_std=0.0,
                motion_std=0.0,
            )

        # 计算每帧的纹理复杂度
        texture_values = [self.compute_frame_texture(f) for f in frames]
        texture_mean = float(np.mean(texture_values))
        texture_std = float(np.std(texture_values))

        # 计算帧间运动强度
        motion_values = []
        scene_change_probs = []
        for i in range(1, num_frames):
            motion_values.append(self.compute_frame_motion(frames[i - 1], frames[i]))
            scene_change_probs.append(
                self.detect_scene_change(frames[i - 1], frames[i])
            )

        if motion_values:
            motion_mean = float(np.mean(motion_values))
            motion_std = float(np.std(motion_values))
            scene_change_prob = float(np.max(scene_change_probs))  # 取段内最大值
        else:
            motion_mean = 0.0
            motion_std = 0.0
            scene_change_prob = 0.0

        # 归一化到 0~1
        texture_norm = min(texture_mean / self.texture_max, 1.0)
        motion_norm = min(motion_mean / self.motion_max, 1.0)

        return SegmentContentFeatures(
            segment_index=segment_index,
            texture_complexity=texture_norm,
            motion_intensity=motion_norm,
            scene_change_prob=scene_change_prob,
            num_frames=num_frames,
            texture_std=texture_std,
            motion_std=motion_std,
        )

    def extract_from_tensor(self, tensor: np.ndarray, segment_index: int) -> SegmentContentFeatures:
        """
        从 PyTorch 张量格式的解码数据中提取特征

        Args:
            tensor: 解码数据，shape (N, C, H, W) 或 (N, H, W, C)
            segment_index: 段索引

        Returns:
            SegmentContentFeatures 对象
        """
        # 转换为帧列表
        frames = []
        for i in range(tensor.shape[0]):
            frame = tensor[i]
            if frame.ndim == 3 and frame.shape[0] in [1, 3]:
                # (C, H, W) -> (H, W, C)
                frame = np.transpose(frame, (1, 2, 0))
            frames.append(frame)

        return self.extract_segment_features(frames, segment_index)


class ContentFeatureStore:
    """
    内容特征存储

    加载预计算的段级内容特征，提供按段索引查询的接口。
    同时提供基于内容特征的质量增益和延迟预测。
    """

    def __init__(self, feature_file: Optional[str] = None):
        """
        初始化特征存储

        Args:
            feature_file: 预计算特征的 JSON 文件路径
        """
        self.features: Dict[int, SegmentContentFeatures] = {}
        if feature_file and os.path.exists(feature_file):
            self.load(feature_file)

    def load(self, feature_file: str):
        """从 JSON 文件加载特征"""
        with open(feature_file, 'r') as f:
            data = json.load(f)

        self.features = {}
        for item in data:
            feat = SegmentContentFeatures(**item)
            self.features[feat.segment_index] = feat

    def save(self, feature_file: str, features: List[SegmentContentFeatures]):
        """保存特征到 JSON 文件"""
        data = [f.to_dict() for f in features]
        os.makedirs(os.path.dirname(feature_file) if os.path.dirname(feature_file) else '.', exist_ok=True)
        with open(feature_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get(self, segment_index: int) -> Optional[SegmentContentFeatures]:
        """获取指定段的特征"""
        return self.features.get(segment_index)

    def get_feature_vector(self, segment_index: int) -> np.ndarray:
        """
        获取指定段的特征向量（用于 DRL 状态输入）

        Returns:
            shape (5,) 的数组：[texture, motion, scene_change, texture_std, motion_std]
            如果段不存在，返回全零向量
        """
        feat = self.features.get(segment_index)
        if feat is None:
            return np.zeros(5, dtype=np.float32)
        return np.array([
            feat.texture_complexity,
            feat.motion_intensity,
            feat.scene_change_prob,
            feat.texture_std / 80.0,  # 归一化
            feat.motion_std / 60.0,   # 归一化
        ], dtype=np.float32)

    def predict_quality_gain(self, segment_index: int, enhancement_level: int,
                              base_vmaf: float) -> float:
        """
        基于内容特征预测增强质量增益

        核心洞察：
        - 高纹理复杂度的内容：超分增益大（细节恢复）
        - 高运动强度的内容：超分增益中等（运动模糊限制）
        - 场景切换：首帧增益不确定，保守估计

        Args:
            segment_index: 段索引
            enhancement_level: 增强级别（1=低, 2=中, 3=高）
            base_vmaf: 无增强时的基础 VMAF

        Returns:
            预测的 VMAF 增益
        """
        feat = self.features.get(segment_index)
        if feat is None:
            # 无特征时使用全局平均增益（基于经验）
            level_gains = {1: 2.0, 2: 4.0, 3: 6.0}
            return level_gains.get(enhancement_level, 0.0)

        # 级别基础增益（来自质量表的典型值）
        level_base_gain = {1: 1.5, 2: 3.5, 3: 5.5}.get(enhancement_level, 0.0)

        # 内容调制因子
        # 纹理复杂度：线性调制，高纹理 = 高增益
        texture_factor = 0.5 + 1.0 * feat.texture_complexity  # 0.5 ~ 1.5

        # 运动强度：非线性调制，中等运动增益最高，过高运动增益下降
        motion_factor = 1.0 - 0.3 * abs(feat.motion_intensity - 0.4)  # 0.88 ~ 1.0

        # 场景切换：保守降低
        scene_factor = 1.0 - 0.5 * feat.scene_change_prob  # 0.5 ~ 1.0

        gain = level_base_gain * texture_factor * motion_factor * scene_factor
        return float(gain)

    def predict_enhancement_latency(self, segment_index: int, enhancement_level: int,
                                      base_latency: float) -> float:
        """
        基于内容特征预测增强延迟

        核心洞察：
        - 高纹理复杂度：计算量略增（更多高频细节）
        - 高运动强度：帧间相关性低，无法利用缓存
        - 场景切换：首帧无参考，延迟略增

        Args:
            segment_index: 段索引
            enhancement_level: 增强级别
            base_latency: 基础延迟（来自静态延迟表）

        Returns:
            预测的增强延迟（秒）
        """
        feat = self.features.get(segment_index)
        if feat is None:
            return base_latency

        # 内容调制因子（延迟变化范围约 ±15%）
        texture_factor = 1.0 + 0.1 * feat.texture_complexity  # 1.0 ~ 1.1
        motion_factor = 1.0 + 0.15 * feat.motion_intensity    # 1.0 ~ 1.15
        scene_factor = 1.0 + 0.1 * feat.scene_change_prob     # 1.0 ~ 1.1

        latency = base_latency * texture_factor * motion_factor * scene_factor
        return float(latency)


def precompute_video_features(video_path: str, output_path: str,
                                segment_duration: float = 4.0,
                                fps: float = 30.0,
                                max_segments: Optional[int] = None,
                                downscale_width: int = 480,
                                frame_sample_interval: int = 4):
    """
    从视频文件预计算所有段的内容特征（优化版）

    Args:
        video_path: 视频文件路径
        output_path: 输出 JSON 文件路径
        segment_duration: 段时长（秒）
        fps: 目标帧率
        max_segments: 最大段数（None 表示全部）
        downscale_width: 特征计算时的降采样宽度（0=不降采样）
        frame_sample_interval: 运动/场景特征的帧采样间隔（每N帧取1帧，加速计算）
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("需要安装 opencv-python: pip install opencv-python")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_segment = int(segment_duration * video_fps)

    print(f"视频信息: {video_path}")
    print(f"  帧率: {video_fps:.1f} fps, 总帧数: {total_frames}")
    print(f"  段时长: {segment_duration}s, 每段帧数: {frames_per_segment}")
    print(f"  降采样宽度: {downscale_width}px, 帧采样间隔: {frame_sample_interval}")

    all_features = []
    segment_index = 0
    frame_count = 0

    # 预分配缓冲区
    gray_frames = []  # 存储降采样后的灰度帧

    def flush_segment():
        """处理当前段的所有帧并生成特征"""
        nonlocal gray_frames
        if not gray_frames:
            return None

        n = len(gray_frames)
        # 纹理复杂度：对所有帧计算 Sobel 梯度幅值均值
        texture_values = []
        for gray in gray_frames:
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
            mag = cv2.magnitude(gx, gy)
            texture_values.append(float(mag.mean()))

        texture_mean = float(np.mean(texture_values))
        texture_std = float(np.std(texture_values))

        # 运动强度和场景切换：采样帧对
        motion_values = []
        scene_probs = []
        for i in range(frame_sample_interval, n, frame_sample_interval):
            prev = gray_frames[i - frame_sample_interval].astype(np.float32)
            curr = gray_frames[i].astype(np.float32)
            # 运动强度：帧间 MAD
            motion = float(np.mean(np.abs(curr - prev)))
            motion_values.append(motion)
            # 场景切换：直方图差异
            hist1 = cv2.calcHist([gray_frames[i - frame_sample_interval]], [0], None, [32], [0, 256])
            hist2 = cv2.calcHist([gray_frames[i]], [0], None, [32], [0, 256])
            cv2.normalize(hist1, hist1, 1, 0, cv2.NORM_L1)
            cv2.normalize(hist2, hist2, 1, 0, cv2.NORM_L1)
            hist_diff = float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA))
            mean_diff = abs(float(prev.mean()) - float(curr.mean())) / 255.0
            score = 0.6 * hist_diff + 0.4 * mean_diff
            scene_prob = 1.0 / (1.0 + np.exp(-10 * (score - 0.3)))
            scene_probs.append(scene_prob)

        motion_mean = float(np.mean(motion_values)) if motion_values else 0.0
        motion_std = float(np.std(motion_values)) if len(motion_values) > 1 else 0.0
        scene_change_prob = float(max(scene_probs)) if scene_probs else 0.0

        # 归一化
        texture_norm = min(texture_mean / 80.0, 1.0)
        motion_norm = min(motion_mean / 60.0, 1.0)

        feat = SegmentContentFeatures(
            segment_index=segment_index,
            texture_complexity=texture_norm,
            motion_intensity=motion_norm,
            scene_change_prob=scene_change_prob,
            num_frames=n,
            texture_std=texture_std,
            motion_std=motion_std,
        )
        gray_frames = []
        return feat

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 降采样
        if downscale_width > 0 and frame.shape[1] > downscale_width:
            h, w = frame.shape[:2]
            new_h = int(h * downscale_width / w)
            frame = cv2.resize(frame, (downscale_width, new_h), interpolation=cv2.INTER_AREA)

        # 转灰度
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frames.append(gray)
        frame_count += 1

        if len(gray_frames) >= frames_per_segment:
            feat = flush_segment()
            if feat is not None:
                all_features.append(feat)
                print(f"  段 {segment_index}: texture={feat.texture_complexity:.3f}, "
                      f"motion={feat.motion_intensity:.3f}, scene={feat.scene_change_prob:.3f} "
                      f"({frame_count}/{total_frames})")
            segment_index += 1

            if max_segments and segment_index >= max_segments:
                break

    # 处理剩余帧
    if gray_frames:
        feat = flush_segment()
        if feat is not None:
            all_features.append(feat)

    cap.release()

    # 保存
    store = ContentFeatureStore()
    store.save(output_path, all_features)
    print(f"\n已保存 {len(all_features)} 段特征到: {output_path}")

    if all_features:
        textures = [f.texture_complexity for f in all_features]
        motions = [f.motion_intensity for f in all_features]
        print(f"统计:")
        print(f"  纹理复杂度: mean={np.mean(textures):.3f}, std={np.std(textures):.3f}, "
              f"min={np.min(textures):.3f}, max={np.max(textures):.3f}")
        print(f"  运动强度: mean={np.mean(motions):.3f}, std={np.std(motions):.3f}, "
              f"min={np.min(motions):.3f}, max={np.max(motions):.3f}")

    return all_features
