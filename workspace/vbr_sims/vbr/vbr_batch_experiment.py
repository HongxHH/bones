"""
批量VBR实验：测试不同高运动段比例的VMAF提升

优化：参考视频和CBR视频只编码一次，然后对每个比例重新编码VBR并测量VMAF。
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
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
                    "-c", "copy", output_path], capture_output=True, timeout=60)


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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        for line in result.stderr.split('\n') + result.stdout.split('\n'):
            if "VMAF score:" in line:
                return float(line.split("VMAF score:")[1].strip())
        return None
    except Exception:
        return None


def main():
    # 配置
    input_video = r"E:\dev\Code\Python\bones\data\bbb\bbb_2160p_60fps.mp4"
    features_path = r"E:\dev\Code\Python\bones\data\bbb\content_features_bbb.json"
    output_dir = Path(r"E:\dev\Code\Python\bones\data\vbr_full_experiment")
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution = "480p"
    num_segments = 159  # 全部段
    segment_duration = 4.0
    cbr_bitrate = 1200
    vbr_high_bitrate = 1500
    ref_bitrate = 10000

    # 测试的高运动段比例（全量实验测试更高比例，寻找边际递减点）
    ratios = [0.50, 0.60, 0.70]

    # 加载内容特征
    with open(features_path) as f:
        features = json.load(f)
    motion = np.array([f["motion_intensity"] for f in features["features"][:num_segments]])
    sorted_idx = np.argsort(-motion)

    print("=" * 70)
    print("批量VBR实验：测试不同高运动段比例")
    print("=" * 70)
    print(f"分辨率: {resolution}, 段数: {num_segments}")
    print(f"CBR码率: {cbr_bitrate}kbps, VBR高运动码率: {vbr_high_bitrate}kbps")
    print(f"测试比例: {ratios}")
    print()

    # 步骤1：生成参考视频（只做一次）
    print("[步骤1] 生成参考视频（只做一次）...")
    ref_dir = output_dir / "reference"
    ref_dir.mkdir(exist_ok=True)
    ref_segments = []
    for seg_idx in range(num_segments):
        ref_path = str(ref_dir / f"ref_{seg_idx:04d}.mp4")
        success, size = encode_segment(input_video, ref_path, seg_idx * segment_duration,
                                        segment_duration, ref_bitrate, resolution, preset="medium")
        if success:
            ref_segments.append(ref_path)
    ref_merged = str(output_dir / "reference_merged.mp4")
    merge_videos(ref_segments, ref_merged, str(output_dir / "ref_concat.txt"))
    print(f"  参考视频完成: {len(ref_segments)}段")
    print()

    # 步骤2：生成CBR视频（只做一次）
    print("[步骤2] 生成CBR视频（只做一次）...")
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
    print(f"  CBR视频完成: {len(cbr_segments)}段, 平均码率={cbr_avg_br:.0f}kbps")
    print()

    # 步骤3：测量CBR的VMAF（只做一次）
    print("[步骤3] 测量CBR的VMAF...")
    cbr_vmaf = measure_vmaf(ref_merged, cbr_merged, resolution)
    print(f"  CBR VMAF: {cbr_vmaf:.2f}" if cbr_vmaf else "  CBR VMAF: 测量失败")
    print()

    # 步骤4：对每个比例生成VBR视频并测量VMAF
    results = []
    for ratio in ratios:
        print(f"[比例 {ratio:.0%}] 生成VBR视频并测量VMAF...")
        num_high = int(num_segments * ratio)
        high_motion_segments = set(sorted_idx[:num_high].tolist())

        vbr_dir = output_dir / f"vbr_ratio_{int(ratio*100)}"
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

        vbr_merged = str(output_dir / f"vbr_merged_{int(ratio*100)}.mp4")
        merge_videos(vbr_segments, vbr_merged, str(output_dir / f"vbr_concat_{int(ratio*100)}.txt"))
        vbr_avg_br = np.mean(vbr_sizes) * 8 / segment_duration / 1000

        vbr_vmaf = measure_vmaf(ref_merged, vbr_merged, resolution)

        result = {
            "ratio": ratio,
            "num_high_motion": num_high,
            "vbr_avg_bitrate": float(vbr_avg_br),
            "cbr_avg_bitrate": float(cbr_avg_br),
            "bitrate_increase_pct": float((vbr_avg_br - cbr_avg_br) / cbr_avg_br * 100),
            "vbr_vmaf": vbr_vmaf,
            "cbr_vmaf": cbr_vmaf,
            "vmaf_diff": float(vbr_vmaf - cbr_vmaf) if (vbr_vmaf and cbr_vmaf) else None,
        }
        results.append(result)

        vmaf_str = f"{vbr_vmaf:.2f}" if vbr_vmaf else "N/A"
        diff_str = f"{result['vmaf_diff']:+.2f}" if result['vmaf_diff'] is not None else "N/A"
        print(f"  高运动段: {num_high}段 ({ratio:.0%})")
        print(f"  平均码率: {vbr_avg_br:.0f}kbps (CBR={cbr_avg_br:.0f}kbps, +{result['bitrate_increase_pct']:.1f}%)")
        print(f"  VMAF: {vmaf_str} (CBR={cbr_vmaf:.2f}, diff={diff_str})")
        print()

    # 汇总
    print("=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    print(f"{'比例':<8} {'高运动段':<10} {'码率提升':<10} {'VMAF提升':<10} {'VMAF/码率效率':<15}")
    print("-" * 55)
    for r in results:
        efficiency = r['vmaf_diff'] / r['bitrate_increase_pct'] if r['vmaf_diff'] is not None and r['bitrate_increase_pct'] > 0 else 0
        print(f"{r['ratio']:<8.0%} {r['num_high_motion']:<10} "
              f"+{r['bitrate_increase_pct']:<9.1f}% "
              f"{r['vmaf_diff']:<+10.2f} "
              f"{efficiency:<15.4f}")

    # 保存结果
    result_path = str(output_dir / "batch_experiment_results.json")
    with open(result_path, "w") as f:
        json.dump({
            "config": {
                "resolution": resolution, "num_segments": num_segments,
                "cbr_bitrate": cbr_bitrate, "vbr_high_bitrate": vbr_high_bitrate,
                "ratios": ratios,
            },
            "cbr": {"avg_bitrate": float(cbr_avg_br), "vmaf": cbr_vmaf},
            "results": results,
        }, f, indent=2)
    print(f"\n结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
