"""
VBR编码脚本：根据内容特征动态调整每段码率

核心思想：
- 高运动/高纹理段分配更多码率（提升质量）
- 低运动/低纹理段分配更少码率（节省带宽）
- 保持平均码率与CBR相同（归一化内容因子）

用法：
python vbr_encode.py --resolution 1080p --bitrate_idx 4 --num_segments 10
"""
import os
import sys
import json
import argparse
import subprocess
import numpy as np
from pathlib import Path


def load_content_features(features_path):
    """加载内容特征数据"""
    with open(features_path) as f:
        data = json.load(f)
    features = data["features"]
    motion = np.array([f["motion_intensity"] for f in features])
    texture = np.array([f["texture_complexity"] for f in features])
    scene_change = np.array([f.get("scene_change", False) for f in features])
    return motion, texture, scene_change


def compute_vbr_bitrates(motion, texture, base_bitrate_kbps,
                          motion_weight=0.6, texture_weight=0.4,
                          min_factor=0.5, max_factor=2.0):
    """
    计算每段的VBR码率

    Args:
        motion: 每段运动强度数组
        texture: 每段纹理复杂度数组
        base_bitrate_kbps: 基础码率（CBR码率）
        motion_weight: 运动权重
        texture_weight: 纹理权重
        min_factor: 最小码率因子
        max_factor: 最大码率因子

    Returns:
        vbr_bitrates: 每段的VBR码率（kbps）
        content_factors: 每段的内容因子
    """
    # 归一化到[0,1]
    motion_norm = (motion - motion.min()) / (motion.max() - motion.min() + 1e-8)
    texture_norm = (texture - texture.min()) / (texture.max() - texture.min() + 1e-8)

    # 计算内容因子
    content_factors = motion_weight * motion_norm + texture_weight * texture_norm

    # 归一化使平均值=1（保持平均码率不变）
    content_factors = content_factors / content_factors.mean()

    # 限制范围
    content_factors = np.clip(content_factors, min_factor, max_factor)

    # 再次归一化（clip后平均值可能变化）
    content_factors = content_factors / content_factors.mean()

    # 计算VBR码率
    vbr_bitrates = base_bitrate_kbps * content_factors

    return vbr_bitrates, content_factors


def encode_segment_vbr(input_path, output_path, start_time_s, duration_s,
                       target_bitrate_kbps, resolution, fps=30):
    """
    用ffmpeg对单段做VBR编码

    Args:
        input_path: 输入视频路径
        output_path: 输出视频路径
        start_time_s: 开始时间（秒）
        duration_s: 持续时间（秒）
        target_bitrate_kbps: 目标码率（kbps）
        resolution: 分辨率（如"1080p"）
        fps: 帧率

    Returns:
        success: 是否成功
        file_size: 输出文件大小（bytes）
    """
    # 分辨率映射
    res_map = {
        "240p": "426:240",
        "360p": "640:360",
        "480p": "854:480",
        "720p": "1280:720",
        "1080p": "1920:1080",
    }
    scale = res_map.get(resolution, "1920:1080")

    # ffmpeg命令：VBR编码（对齐原始BONES编码参数）
    # -preset veryfast, keyint=120固定GOP, maxrate=2x, bufsize=4x, bicubic缩放
    maxrate = int(target_bitrate_kbps * 2)
    bufsize = int(target_bitrate_kbps * 4)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time_s),
        "-i", input_path,
        "-t", str(duration_s),
        "-vf", f"scale={scale}:flags=bicubic,fps={fps}",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-b:v", f"{int(target_bitrate_kbps)}k",
        "-maxrate", f"{maxrate}k",
        "-bufsize", f"{bufsize}k",
        "-x264opts", "keyint=120:min-keyint=120:scenecut=0:no-scenecut",
        "-an",  # 无音频
        "-movflags", "+faststart",
        output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  ffmpeg error: {result.stderr[-200:]}")
            return False, 0
        file_size = os.path.getsize(output_path)
        return True, file_size
    except subprocess.TimeoutExpired:
        print(f"  ffmpeg timeout")
        return False, 0
    except Exception as e:
        print(f"  ffmpeg exception: {e}")
        return False, 0


def measure_vmaf(reference_path, distorted_path, resolution):
    """
    用ffmpeg libvmaf测量VMAF

    Args:
        reference_path: 参考视频路径（原始高质量）
        distorted_path: 失真视频路径
        resolution: 分辨率

    Returns:
        vmaf_score: VMAF分数（0-100），失败返回None
    """
    # 参考视频需要缩放到相同分辨率
    res_map = {
        "240p": "426:240",
        "360p": "640:360",
        "480p": "854:480",
        "720p": "1280:720",
        "1080p": "1920:1080",
    }
    scale = res_map.get(resolution, "1920:1080")

    cmd = [
        "ffmpeg", "-y",
        "-i", distorted_path,
        "-i", reference_path,
        "-filter_complex",
        f"[0:v]scale={scale}:flags=bicubic[dist];[1:v]scale={scale}:flags=bicubic,fps=30[ref];[dist][ref]libvmaf=log_path=/dev/null:log_fmt=json",
        "-f", "null", "-"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        for line in result.stderr.split('\n'):
            if "VMAF score:" in line:
                return float(line.split("VMAF score:")[1].strip())
        for line in result.stdout.split('\n'):
            if "VMAF score:" in line:
                return float(line.split("VMAF score:")[1].strip())
        return None
    except subprocess.TimeoutExpired:
        print(f"  VMAF measurement timeout (300s)")
        return None
    except Exception as e:
        print(f"  VMAF measurement error: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="VBR编码脚本")
    parser.add_argument("--input", default=r"E:\dev\Code\Python\bones\data\bbb\bbb_2160p_60fps.mp4",
                        help="输入视频路径")
    parser.add_argument("--features", default=r"E:\dev\Code\Python\bones\data\bbb\content_features_bbb.json",
                        help="内容特征路径")
    parser.add_argument("--output_dir", default=r"E:\dev\Code\Python\bones\data\bbb_vbr",
                        help="输出目录")
    parser.add_argument("--resolution", default="1080p",
                        choices=["240p", "360p", "480p", "720p", "1080p"],
                        help="目标分辨率")
    parser.add_argument("--bitrate_idx", type=int, default=4,
                        help="码率档位索引（0-4，对应398/802/1203/2406/4738 kbps）")
    parser.add_argument("--num_segments", type=int, default=10,
                        help="编码段数（0=全部159段）")
    parser.add_argument("--motion_weight", type=float, default=0.6,
                        help="运动权重")
    parser.add_argument("--texture_weight", type=float, default=0.4,
                        help="纹理权重")
    parser.add_argument("--measure_vmaf", action="store_true",
                        help="是否测量VMAF（需要参考视频）")
    args = parser.parse_args()

    # 码率档位（原始BONES编码的目标码率）
    bitrates_kbps = [400, 800, 1200, 2400, 4800]
    base_bitrate = bitrates_kbps[args.bitrate_idx]

    # 加载内容特征
    motion, texture, scene_change = load_content_features(args.features)
    num_total = len(motion)
    num_segments = args.num_segments if args.num_segments > 0 else num_total
    num_segments = min(num_segments, num_total)

    print(f"VBR编码配置:")
    print(f"  分辨率: {args.resolution}")
    print(f"  基础码率: {base_bitrate} kbps")
    print(f"  编码段数: {num_segments}/{num_total}")
    print(f"  运动权重: {args.motion_weight}, 纹理权重: {args.texture_weight}")
    print()

    # 计算VBR码率
    vbr_bitrates, content_factors = compute_vbr_bitrates(
        motion[:num_segments], texture[:num_segments], base_bitrate,
        args.motion_weight, args.texture_weight
    )

    print(f"内容因子统计: min={content_factors.min():.3f}, max={content_factors.max():.3f}, mean={content_factors.mean():.3f}")
    print(f"VBR码率统计: min={vbr_bitrates.min():.0f}, max={vbr_bitrates.max():.0f}, mean={vbr_bitrates.mean():.0f} kbps")
    print()

    # 创建输出目录
    output_dir = Path(args.output_dir) / args.resolution / f"br{args.bitrate_idx}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 编码每段
    segment_duration = 4.0  # 每段4秒
    results = []

    for seg_idx in range(num_segments):
        start_time = seg_idx * segment_duration
        output_path = str(output_dir / f"segment_{seg_idx:04d}.mp4")
        target_br = vbr_bitrates[seg_idx]

        print(f"[{seg_idx+1}/{num_segments}] 段{seg_idx}: "
              f"motion={motion[seg_idx]:.1f}, texture={texture[seg_idx]:.1f}, "
              f"factor={content_factors[seg_idx]:.3f}, target_br={target_br:.0f}kbps")

        success, file_size = encode_segment_vbr(
            args.input, output_path, start_time, segment_duration,
            target_br, args.resolution
        )

        if success:
            actual_br = file_size * 8 / segment_duration / 1000  # kbps
            print(f"  成功: size={file_size/1024:.1f}KB, actual_br={actual_br:.0f}kbps")

            vmaf = None
            if args.measure_vmaf:
                vmaf = measure_vmaf(args.input, output_path, args.resolution)
                if vmaf is not None:
                    print(f"  VMAF: {vmaf:.2f}")

            results.append({
                "segment_index": seg_idx,
                "motion_intensity": float(motion[seg_idx]),
                "texture_complexity": float(texture[seg_idx]),
                "content_factor": float(content_factors[seg_idx]),
                "target_bitrate_kbps": float(target_br),
                "actual_file_size_bytes": file_size,
                "actual_bitrate_kbps": float(actual_br),
                "vmaf": vmaf,
            })
        else:
            print(f"  失败")
            results.append({
                "segment_index": seg_idx,
                "motion_intensity": float(motion[seg_idx]),
                "texture_complexity": float(texture[seg_idx]),
                "content_factor": float(content_factors[seg_idx]),
                "target_bitrate_kbps": float(target_br),
                "actual_file_size_bytes": 0,
                "actual_bitrate_kbps": 0,
                "vmaf": None,
                "error": True,
            })

    # 保存结果
    result_path = str(output_dir / "vbr_encoding_results.json")
    with open(result_path, "w") as f:
        json.dump({
            "config": {
                "resolution": args.resolution,
                "base_bitrate_kbps": base_bitrate,
                "bitrate_idx": args.bitrate_idx,
                "num_segments": num_segments,
                "motion_weight": args.motion_weight,
                "texture_weight": args.texture_weight,
            },
            "content_factor_stats": {
                "min": float(content_factors.min()),
                "max": float(content_factors.max()),
                "mean": float(content_factors.mean()),
            },
            "segments": results,
        }, f, indent=2)

    print(f"\n结果已保存到: {result_path}")

    # 汇总统计
    successful = [r for r in results if not r.get("error")]
    if successful:
        avg_actual_br = np.mean([r["actual_bitrate_kbps"] for r in successful])
        print(f"\n汇总:")
        print(f"  成功段数: {len(successful)}/{num_segments}")
        print(f"  平均实际码率: {avg_actual_br:.0f} kbps (目标: {base_bitrate} kbps)")
        vmafs = [r["vmaf"] for r in successful if r["vmaf"] is not None]
        if vmafs:
            print(f"  平均VMAF: {np.mean(vmafs):.2f}")


if __name__ == "__main__":
    main()
