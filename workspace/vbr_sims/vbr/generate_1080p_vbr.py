"""
补充1080p VBR数据：生成1080p CBR/VBR视频，测量逐段VMAF，保存中间数据

1080p基础码率: 4738kbps
高运动段码率: 4738 * 1.25 = 5922kbps
高运动段比例: 60%
"""
import os
import sys
import json
import subprocess
import numpy as np
from pathlib import Path


def encode_segment(input_path, output_path, start_time_s, duration_s,
                   target_bitrate_kbps, resolution="1080p", fps=30, preset="veryfast"):
    """用ffmpeg编码单段视频（如果文件已存在且非空则跳过）"""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        return True, os.path.getsize(output_path)

    res_map = {
        "240p": "426:240", "360p": "640:360", "480p": "854:480",
        "720p": "1280:720", "1080p": "1920:1080",
    }
    scale = res_map.get(resolution, "1920:1080")
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            return False, 0
        return True, os.path.getsize(output_path)
    except Exception:
        return False, 0


def measure_vmaf_single(reference_path, distorted_path, resolution="1080p"):
    """测量单段视频的VMAF"""
    res_map = {
        "240p": "426:240", "360p": "640:360", "480p": "854:480",
        "720p": "1280:720", "1080p": "1920:1080",
    }
    scale = res_map.get(resolution, "1920:1080")
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
    intermediate_dir = Path(r"E:\dev\Code\Python\bones\data\bbb_vbr_integrated\intermediate")
    intermediate_dir.mkdir(parents=True, exist_ok=True)

    output_dir = Path(r"E:\dev\Code\Python\bones\data\bbb_vbr_integrated\1080p")
    output_dir.mkdir(parents=True, exist_ok=True)

    num_segments = 159
    segment_duration = 4.0
    high_motion_ratio = 0.60
    bitrate_increase = 1.25

    resolution = "1080p"
    bitrate_idx = 4  # 1080p是第5个码率档（索引4）
    base_bitrate = 4738
    high_bitrate = int(base_bitrate * bitrate_increase)

    # 检查中间文件是否已存在
    intermediate_file = intermediate_dir / f"{resolution}_data.json"
    if intermediate_file.exists():
        print(f"[跳过] {resolution} 中间数据已存在: {intermediate_file}")
        with open(intermediate_file) as f:
            data = json.load(f)
        print(f"  CBR平均VMAF: {np.mean(data['cbr_vmafs']):.2f}")
        print(f"  VBR平均VMAF: {np.mean(data['vbr_vmafs']):.2f}")
        print(f"  提升量: {np.mean(data['vbr_vmafs']) - np.mean(data['cbr_vmafs']):+.2f}")
        return

    print("=" * 70)
    print(f"补充1080p VBR数据")
    print("=" * 70)
    print(f"基础码率: {base_bitrate}kbps, 高运动码率: {high_bitrate}kbps")
    print(f"高运动段比例: {high_motion_ratio:.0%}")
    print()

    # 加载内容特征
    with open(features_path) as f:
        features = json.load(f)
    motion = np.array([f["motion_intensity"] for f in features["features"][:num_segments]])
    sorted_idx = np.argsort(-motion)
    num_high = int(num_segments * high_motion_ratio)
    high_motion_segments = set(sorted_idx[:num_high].tolist())

    ref_dir = output_dir / "reference"
    ref_dir.mkdir(exist_ok=True)
    cbr_dir = output_dir / "cbr"
    cbr_dir.mkdir(exist_ok=True)
    vbr_dir = output_dir / "vbr"
    vbr_dir.mkdir(exist_ok=True)

    # 步骤1：生成参考视频
    print(f"[1/4] 生成参考视频（10Mbps）...")
    for seg_idx in range(num_segments):
        ref_path = str(ref_dir / f"ref_{seg_idx:04d}.mp4")
        encode_segment(input_video, ref_path, seg_idx * segment_duration,
                       segment_duration, 10000, resolution, preset="medium")
        if (seg_idx + 1) % 20 == 0:
            print(f"  参考视频: {seg_idx+1}/{num_segments}")
    print(f"  参考视频完成")

    # 步骤2：生成CBR视频
    print(f"[2/4] 生成CBR视频（{base_bitrate}kbps）...")
    cbr_sizes = []
    for seg_idx in range(num_segments):
        cbr_path = str(cbr_dir / f"cbr_{seg_idx:04d}.mp4")
        success, size = encode_segment(input_video, cbr_path, seg_idx * segment_duration,
                                        segment_duration, base_bitrate, resolution)
        cbr_sizes.append(size * 8 if success else 0)
        if (seg_idx + 1) % 20 == 0:
            print(f"  CBR视频: {seg_idx+1}/{num_segments}")
    print(f"  CBR视频完成, 平均码率={np.mean(cbr_sizes)/segment_duration/1000:.0f}kbps")

    # 步骤3：生成VBR视频
    print(f"[3/4] 生成VBR视频（高运动{high_bitrate}kbps, 其他{base_bitrate}kbps）...")
    vbr_sizes = []
    for seg_idx in range(num_segments):
        target_br = high_bitrate if seg_idx in high_motion_segments else base_bitrate
        vbr_path = str(vbr_dir / f"vbr_{seg_idx:04d}.mp4")
        success, size = encode_segment(input_video, vbr_path, seg_idx * segment_duration,
                                        segment_duration, target_br, resolution)
        vbr_sizes.append(size * 8 if success else 0)
        if (seg_idx + 1) % 20 == 0:
            print(f"  VBR视频: {seg_idx+1}/{num_segments}")
    print(f"  VBR视频完成, 平均码率={np.mean(vbr_sizes)/segment_duration/1000:.0f}kbps")

    # 步骤4：测量逐段VMAF
    print(f"[4/4] 测量逐段VMAF...")
    cbr_vmafs = []
    vbr_vmafs = []
    for seg_idx in range(num_segments):
        ref_path = str(ref_dir / f"ref_{seg_idx:04d}.mp4")
        cbr_path = str(cbr_dir / f"cbr_{seg_idx:04d}.mp4")
        vbr_path = str(vbr_dir / f"vbr_{seg_idx:04d}.mp4")

        cbr_vmaf = measure_vmaf_single(ref_path, cbr_path, resolution)
        vbr_vmaf = measure_vmaf_single(ref_path, vbr_path, resolution)

        cbr_vmafs.append(cbr_vmaf if cbr_vmaf is not None else 0)
        vbr_vmafs.append(vbr_vmaf if vbr_vmaf is not None else 0)

        if (seg_idx + 1) % 10 == 0:
            recent_cbr = [x for x in cbr_vmafs[-10:] if x > 0]
            recent_vbr = [x for x in vbr_vmafs[-10:] if x > 0]
            avg_cbr = np.mean(recent_cbr) if recent_cbr else 0
            avg_vbr = np.mean(recent_vbr) if recent_vbr else 0
            print(f"  VMAF测量: {seg_idx+1}/{num_segments}, "
                  f"近10段CBR={avg_cbr:.2f}, VBR={avg_vbr:.2f}, diff={avg_vbr-avg_cbr:+.2f}")

    # 计算有效段的平均
    valid_cbr = [x for x in cbr_vmafs if x > 0]
    valid_vbr = [x for x in vbr_vmafs if x > 0]
    avg_cbr = np.mean(valid_cbr) if valid_cbr else 0
    avg_vbr = np.mean(valid_vbr) if valid_vbr else 0

    print(f"\n  CBR平均VMAF: {avg_cbr:.2f} (有效段{len(valid_cbr)}/{num_segments})")
    print(f"  VBR平均VMAF: {avg_vbr:.2f} (有效段{len(valid_vbr)}/{num_segments})")
    print(f"  平均提升量: {avg_vbr - avg_cbr:+.2f}")

    # 保存中间数据
    with open(intermediate_file, "w") as f:
        json.dump({
            "resolution": resolution,
            "bitrate_idx": bitrate_idx,
            "base_bitrate": base_bitrate,
            "high_bitrate": high_bitrate,
            "high_motion_ratio": high_motion_ratio,
            "cbr_sizes": cbr_sizes,
            "vbr_sizes": vbr_sizes,
            "cbr_vmafs": cbr_vmafs,
            "vbr_vmafs": vbr_vmafs,
        }, f)
    print(f"\n中间数据已保存: {intermediate_file}")
    print("完成！")


if __name__ == "__main__":
    main()
