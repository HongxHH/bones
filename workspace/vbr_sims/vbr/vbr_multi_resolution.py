"""
多分辨率VBR实验：测试不同分辨率下内容感知VBR编码的效果

每个分辨率使用60%最优比例（高运动段1500kbps对应比例，其他段基础码率）
全量159段
"""
import os
import sys
import json
import subprocess
import numpy as np
from pathlib import Path


def encode_segment(input_path, output_path, start_time_s, duration_s,
                   target_bitrate_kbps, resolution="480p", fps=30, preset="veryfast"):
    """用ffmpeg编码单段视频"""
    res_map = {
        "240p": "426:240", "360p": "640:360", "480p": "854:480",
        "720p": "1280:720", "1080p": "1920:1080",
    }
    scale = res_map.get(resolution, "854:480")
    maxrate = int(target_bitrate_kbps * 2)
    bufsize = int(target_bitrate_kbps * 4)

    cmd = [
        "ffmpeg", "-y", "-ss", str(start_time_s), "-i", input_path,
        "-t", str(duration_s), "-vf", f"scale={scale}:flags=bicubic,fps={fps}",
        "-c:v", "libx264", "-preset", preset,
        "-b:v", f"{int(target_bitrate_kbps)}k",
        "-maxrate", f"{maxrate}k", "-bufsize", f"{bufsize}k",
        "-x264opts", "keyint=120:min-keyint=120:scenecut=0:no-scenecut",
        "-an", "-movflags", "+faststart", output_path
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            return False, 0
        return True, os.path.getsize(output_path)
    except Exception:
        return False, 0


def merge_videos(segment_paths, output_path, concat_file_path):
    """合并多段视频"""
    with open(concat_file_path, "w") as f:
        for p in segment_paths:
            f.write(f"file '{p}'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file_path,
                    "-c", "copy", output_path], capture_output=True, timeout=120)


def measure_vmaf(reference_path, distorted_path, resolution="480p"):
    """用ffmpeg libvmaf测量VMAF"""
    res_map = {
        "240p": "426:240", "360p": "640:360", "480p": "854:480",
        "720p": "1280:720", "1080p": "1920:1080",
    }
    scale = res_map.get(resolution, "854:480")
    cmd = [
        "ffmpeg", "-y", "-i", distorted_path, "-i", reference_path,
        "-filter_complex",
        f"[0:v]scale={scale}:flags=bicubic[dist];[1:v]scale={scale}:flags=bicubic[ref];[dist][ref]libvmaf=log_path=/dev/null:log_fmt=json",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        for line in result.stderr.split('\n') + result.stdout.split('\n'):
            if "VMAF score:" in line:
                return float(line.split("VMAF score:")[1].strip())
        return None
    except Exception:
        return None


def run_single_resolution(input_video, features_path, output_base_dir, resolution,
                          num_segments, segment_duration, cbr_bitrate, vbr_high_bitrate,
                          high_motion_ratio, motion):
    """对单个分辨率运行VBR实验"""
    output_dir = output_base_dir / resolution
    output_dir.mkdir(parents=True, exist_ok=True)

    sorted_idx = np.argsort(-motion)
    num_high = int(num_segments * high_motion_ratio)
    high_motion_segments = set(sorted_idx[:num_high].tolist())

    print(f"\n{'='*60}")
    print(f"分辨率: {resolution}")
    print(f"{'='*60}")
    print(f"CBR码率: {cbr_bitrate}kbps, VBR高运动码率: {vbr_high_bitrate}kbps")
    print(f"高运动段比例: {high_motion_ratio:.0%} ({num_high}段)")

    # 步骤1：生成参考视频
    print(f"  [1/4] 生成参考视频...")
    ref_dir = output_dir / "reference"
    ref_dir.mkdir(exist_ok=True)
    ref_segments = []
    for seg_idx in range(num_segments):
        ref_path = str(ref_dir / f"ref_{seg_idx:04d}.mp4")
        success, size = encode_segment(input_video, ref_path, seg_idx * segment_duration,
                                        segment_duration, 10000, resolution, preset="medium")
        if success:
            ref_segments.append(ref_path)
    ref_merged = str(output_dir / "reference_merged.mp4")
    merge_videos(ref_segments, ref_merged, str(output_dir / "ref_concat.txt"))
    print(f"    参考视频完成: {len(ref_segments)}段")

    # 步骤2：生成CBR视频
    print(f"  [2/4] 生成CBR视频...")
    cbr_dir = output_dir / "cbr"
    cbr_dir.mkdir(exist_ok=True)
    cbr_segments = []
    cbr_sizes = []
    for seg_idx in range(num_segments):
        cbr_path = str(cbr_dir / f"cbr_{seg_idx:04d}.mp4")
        success, size = encode_segment(input_video, cbr_path, seg_idx * segment_duration,
                                        segment_duration, cbr_bitrate, resolution)
        if success:
            cbr_segments.append(cbr_path)
            cbr_sizes.append(size)
    cbr_merged = str(output_dir / "cbr_merged.mp4")
    merge_videos(cbr_segments, cbr_merged, str(output_dir / "cbr_concat.txt"))
    cbr_avg_br = np.mean(cbr_sizes) * 8 / segment_duration / 1000
    print(f"    CBR视频完成: {len(cbr_segments)}段, 平均码率={cbr_avg_br:.0f}kbps")

    # 步骤3：生成VBR视频
    print(f"  [3/4] 生成VBR视频...")
    vbr_dir = output_dir / "vbr"
    vbr_dir.mkdir(exist_ok=True)
    vbr_segments = []
    vbr_sizes = []
    for seg_idx in range(num_segments):
        target_br = vbr_high_bitrate if seg_idx in high_motion_segments else cbr_bitrate
        vbr_path = str(vbr_dir / f"vbr_{seg_idx:04d}.mp4")
        success, size = encode_segment(input_video, vbr_path, seg_idx * segment_duration,
                                        segment_duration, target_br, resolution)
        if success:
            vbr_segments.append(vbr_path)
            vbr_sizes.append(size)
    vbr_merged = str(output_dir / "vbr_merged.mp4")
    merge_videos(vbr_segments, vbr_merged, str(output_dir / "vbr_concat.txt"))
    vbr_avg_br = np.mean(vbr_sizes) * 8 / segment_duration / 1000
    print(f"    VBR视频完成: {len(vbr_segments)}段, 平均码率={vbr_avg_br:.0f}kbps")

    # 步骤4：测量VMAF
    print(f"  [4/4] 测量VMAF...")
    cbr_vmaf = measure_vmaf(ref_merged, cbr_merged, resolution)
    vbr_vmaf = measure_vmaf(ref_merged, vbr_merged, resolution)

    result = {
        "resolution": resolution,
        "cbr_bitrate": cbr_bitrate,
        "vbr_high_bitrate": vbr_high_bitrate,
        "high_motion_ratio": high_motion_ratio,
        "num_high_motion": num_high,
        "cbr_avg_bitrate": float(cbr_avg_br),
        "vbr_avg_bitrate": float(vbr_avg_br),
        "bitrate_increase_pct": float((vbr_avg_br - cbr_avg_br) / cbr_avg_br * 100),
        "cbr_vmaf": cbr_vmaf,
        "vbr_vmaf": vbr_vmaf,
        "vmaf_diff": float(vbr_vmaf - cbr_vmaf) if (cbr_vmaf and vbr_vmaf) else None,
    }

    cbr_str = f"{cbr_vmaf:.2f}" if cbr_vmaf else "N/A"
    vbr_str = f"{vbr_vmaf:.2f}" if vbr_vmaf else "N/A"
    diff_str = f"{result['vmaf_diff']:+.2f}" if result['vmaf_diff'] is not None else "N/A"
    print(f"    CBR VMAF: {cbr_str}, VBR VMAF: {vbr_str}, diff: {diff_str}")
    print(f"    码率提升: +{result['bitrate_increase_pct']:.1f}%")

    return result


def main():
    input_video = r"E:\dev\Code\Python\bones\data\bbb\bbb_2160p_60fps.mp4"
    features_path = r"E:\dev\Code\Python\bones\data\bbb\content_features_bbb.json"
    output_base_dir = Path(r"E:\dev\Code\Python\bones\data\vbr_multi_resolution")
    output_base_dir.mkdir(parents=True, exist_ok=True)

    num_segments = 159
    segment_duration = 4.0
    high_motion_ratio = 0.60  # 最优比例

    # 各分辨率的基础码率（对应BONES的码率档位）
    resolution_configs = [
        {"resolution": "240p", "cbr_bitrate": 400, "vbr_high_bitrate": 500},
        {"resolution": "360p", "cbr_bitrate": 800, "vbr_high_bitrate": 1000},
        {"resolution": "480p", "cbr_bitrate": 1200, "vbr_high_bitrate": 1500},
        {"resolution": "720p", "cbr_bitrate": 2400, "vbr_high_bitrate": 3000},
    ]

    # 加载内容特征
    with open(features_path) as f:
        features = json.load(f)
    motion = np.array([f["motion_intensity"] for f in features["features"][:num_segments]])

    print("=" * 70)
    print("多分辨率VBR实验")
    print("=" * 70)
    print(f"段数: {num_segments}, 高运动段比例: {high_motion_ratio:.0%}")
    print(f"分辨率: {[c['resolution'] for c in resolution_configs]}")
    print()

    all_results = []
    for config in resolution_configs:
        result = run_single_resolution(
            input_video, features_path, output_base_dir,
            config["resolution"], num_segments, segment_duration,
            config["cbr_bitrate"], config["vbr_high_bitrate"],
            high_motion_ratio, motion
        )
        all_results.append(result)

    # 汇总
    print(f"\n{'='*70}")
    print("多分辨率实验结果汇总")
    print(f"{'='*70}")
    print(f"{'分辨率':<10} {'CBR码率':<10} {'CBR VMAF':<10} {'VBR VMAF':<10} "
          f"{'VMAF提升':<10} {'码率提升':<10} {'效率':<10}")
    print("-" * 75)
    for r in all_results:
        efficiency = r['vmaf_diff'] / r['bitrate_increase_pct'] if r['vmaf_diff'] is not None and r['bitrate_increase_pct'] > 0 else 0
        cbr_str = f"{r['cbr_vmaf']:.2f}" if r['cbr_vmaf'] else "N/A"
        vbr_str = f"{r['vbr_vmaf']:.2f}" if r['vbr_vmaf'] else "N/A"
        diff_str = f"{r['vmaf_diff']:+.2f}" if r['vmaf_diff'] is not None else "N/A"
        print(f"{r['resolution']:<10} {r['cbr_bitrate']:<10} {cbr_str:<10} {vbr_str:<10} "
              f"{diff_str:<10} +{r['bitrate_increase_pct']:<9.1f}% {efficiency:<10.4f}")

    # 保存结果
    result_path = str(output_base_dir / "multi_resolution_results.json")
    with open(result_path, "w") as f:
        json.dump({
            "config": {
                "num_segments": num_segments,
                "high_motion_ratio": high_motion_ratio,
                "resolution_configs": resolution_configs,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\n结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
