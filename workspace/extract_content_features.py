"""
内容特征提取脚本（阶段1）

对1080p参考视频按4秒分段，提取每段的内容特征：
  - motion_intensity: 运动强度（帧间SAD归一化）
  - texture_complexity: 纹理复杂度（Laplacian方差）
  - scene_change: 场景切换标记（帧间突变）
  - scene_change_score: 场景切换置信度

特征提取在降采样(320x180)上进行，保证速度。
输出：content_features.json，供NES内容感知决策使用。
"""
import json
import time
import numpy as np
import cv2
from pathlib import Path


def compute_laplacian_variance(gray: np.ndarray) -> float:
    """计算Laplacian方差，衡量纹理复杂度/清晰度"""
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def compute_frame_sad(prev: np.ndarray, curr: np.ndarray) -> float:
    """计算两帧间的平均绝对差（SAD），衡量运动强度"""
    diff = np.abs(curr.astype(np.float32) - prev.astype(np.float32))
    return float(diff.mean())


def extract_segment_features(frames: list, segment_idx: int) -> dict:
    """
    从一段的采样帧中提取内容特征

    Args:
        frames: 降采样后的灰度帧列表（均匀采样10帧）
        segment_idx: 段索引

    Returns:
        包含内容特征的字典
    """
    if len(frames) < 2:
        return {
            "segment_index": segment_idx,
            "motion_intensity": 0.0,
            "texture_complexity": 0.0,
            "scene_change": False,
            "scene_change_score": 0.0,
            "num_sampled_frames": len(frames),
        }

    # 1. 纹理复杂度：所有采样帧的Laplacian方差均值
    texture_scores = [compute_laplacian_variance(f) for f in frames]
    texture_complexity = float(np.mean(texture_scores))

    # 2. 运动强度：相邻采样帧间SAD的均值
    sad_scores = []
    for i in range(1, len(frames)):
        sad = compute_frame_sad(frames[i - 1], frames[i])
        sad_scores.append(sad)
    motion_intensity = float(np.mean(sad_scores))

    # 3. 场景切换：最大SAD超过阈值
    max_sad = float(np.max(sad_scores)) if sad_scores else 0.0
    # 场景切换阈值：运动强度的3倍，且绝对SAD > 25（经验阈值）
    scene_threshold = max(25.0, motion_intensity * 3.0)
    scene_change = max_sad > scene_threshold
    scene_change_score = max_sad / scene_threshold if scene_threshold > 0 else 0.0

    return {
        "segment_index": segment_idx,
        "motion_intensity": round(motion_intensity, 4),
        "texture_complexity": round(texture_complexity, 2),
        "scene_change": scene_change,
        "scene_change_score": round(scene_change_score, 4),
        "num_sampled_frames": len(frames),
    }


def main():
    video_path = r"E:\dev\Code\Python\bones\data\video_mp4box\output_1080p.mp4"
    output_path = r"E:\dev\Code\Python\bones\data\bbb\content_features_bbb.json"

    # 参数
    seg_duration = 4.0  # 每段4秒
    target_fps = 30.0
    frames_per_seg = int(seg_duration * target_fps)  # 120
    samples_per_seg = 10  # 每段均匀采样10帧
    downscale_w = 320
    downscale_h = 180

    print(f"视频: {video_path}")
    print(f"分段: {seg_duration}s/段, {frames_per_seg}帧/段, 采样{samples_per_seg}帧/段")
    print(f"降采样: {downscale_w}x{downscale_h}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    num_segments = int(np.ceil(duration / seg_duration))
    print(f"视频: {total_frames}帧, {duration:.1f}s, {num_segments}段")

    # 计算每段需要采样的帧索引（在段内均匀分布）
    sample_offsets = np.linspace(0, frames_per_seg - 1, samples_per_seg, dtype=int)

    features = []
    t0 = time.time()

    for seg_idx in range(num_segments):
        seg_start_frame = seg_idx * frames_per_seg
        sampled_frames = []

        for offset in sample_offsets:
            frame_idx = seg_start_frame + offset
            if frame_idx >= total_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break
            # 降采样 + 灰度
            frame_small = cv2.resize(frame, (downscale_w, downscale_h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
            sampled_frames.append(gray)

        feat = extract_segment_features(sampled_frames, seg_idx)
        features.append(feat)

        if (seg_idx + 1) % 20 == 0:
            elapsed = time.time() - t0
            avg_per_seg = elapsed / (seg_idx + 1)
            print(f"  已处理 {seg_idx + 1}/{num_segments} 段, "
                  f"平均 {avg_per_seg*1000:.1f}ms/段, "
                  f"预计剩余 {avg_per_seg * (num_segments - seg_idx - 1):.1f}s")

    cap.release()
    total_time = time.time() - t0
    print(f"\n完成: {len(features)}段, 总耗时 {total_time:.1f}s, "
          f"平均 {total_time/len(features)*1000:.1f}ms/段")

    # 统计特征分布
    motions = [f["motion_intensity"] for f in features]
    textures = [f["texture_complexity"] for f in features]
    scene_changes = sum(1 for f in features if f["scene_change"])

    print(f"\n特征分布统计:")
    print(f"  运动强度: mean={np.mean(motions):.2f}, std={np.std(motions):.2f}, "
          f"min={np.min(motions):.2f}, max={np.max(motions):.2f}")
    print(f"  纹理复杂度: mean={np.mean(textures):.1f}, std={np.std(textures):.1f}, "
          f"min={np.min(textures):.1f}, max={np.max(textures):.1f}")
    print(f"  场景切换段: {scene_changes}/{len(features)} ({scene_changes/len(features)*100:.1f}%)")

    # 内容分类（用于阶段2的增益表测量）
    motion_arr = np.array(motions)
    texture_arr = np.array(textures)
    motion_median = np.median(motion_arr)
    texture_median = np.median(texture_arr)

    for f in features:
        f["content_class"] = (
            ("high_motion" if f["motion_intensity"] > motion_median else "low_motion") + "_" +
            ("high_texture" if f["texture_complexity"] > texture_median else "low_texture")
        )

    # 保存
    output = {
        "video": str(video_path),
        "segment_duration_s": seg_duration,
        "fps": target_fps,
        "frames_per_segment": frames_per_seg,
        "samples_per_segment": samples_per_seg,
        "downscale_resolution": [downscale_w, downscale_h],
        "num_segments": len(features),
        "motion_median": round(float(motion_median), 4),
        "texture_median": round(float(texture_median), 2),
        "extraction_time_s": round(total_time, 2),
        "avg_extraction_ms_per_seg": round(total_time / len(features) * 1000, 1),
        "features": features,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n已保存到: {output_path}")

    # 打印前5段和后5段的特征，用于人工验证
    print(f"\n前5段特征:")
    for f in features[:5]:
        print(f"  seg{f['segment_index']:3d}: motion={f['motion_intensity']:6.2f}, "
              f"texture={f['texture_complexity']:7.1f}, scene={f['scene_change']}, "
              f"class={f['content_class']}")
    print(f"...")
    print(f"后5段特征:")
    for f in features[-5:]:
        print(f"  seg{f['segment_index']:3d}: motion={f['motion_intensity']:6.2f}, "
              f"texture={f['texture_complexity']:7.1f}, scene={f['scene_change']}, "
              f"class={f['content_class']}")


if __name__ == "__main__":
    main()
