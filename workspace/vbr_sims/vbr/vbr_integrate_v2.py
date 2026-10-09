"""
VBR集成到BONES仿真器（修正版）

关键修正：
1. 不直接替换base_quality，而是计算VBR相对于CBR的提升量，加到原始base_quality上
2. 保存逐段VMAF和文件大小到中间文件，避免重复计算
3. segment_sizes用VBR实际文件大小替换
"""
import os
import sys
import json
import subprocess
import numpy as np
from pathlib import Path


def encode_segment(input_path, output_path, start_time_s, duration_s,
                   target_bitrate_kbps, resolution="480p", fps=30, preset="veryfast"):
    """用ffmpeg编码单段视频（如果文件已存在且非空则跳过）"""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        return True, os.path.getsize(output_path)

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


def measure_vmaf_single(reference_path, distorted_path, resolution="480p"):
    """测量单段视频的VMAF"""
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
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
    original_manifest = r"E:\dev\Code\Python\bones\data\bbb\bbb_video_info.json"
    original_enhance = r"E:\dev\Code\Python\bones\data\bbb\imdn_bbb_2080ti.json"

    output_dir = Path(r"E:\dev\Code\Python\bones\data\bbb_vbr_integrated")
    output_dir.mkdir(parents=True, exist_ok=True)
    intermediate_dir = output_dir / "intermediate"
    intermediate_dir.mkdir(exist_ok=True)

    num_segments = 159
    segment_duration = 4.0
    high_motion_ratio = 0.60
    bitrate_increase = 1.25

    # 码率档配置
    resolution_configs = [
        {"resolution": "240p", "base_bitrate": 398, "bitrate_idx": 0},
        {"resolution": "360p", "base_bitrate": 802, "bitrate_idx": 1},
        {"resolution": "480p", "base_bitrate": 1203, "bitrate_idx": 2},
        {"resolution": "720p", "base_bitrate": 2406, "bitrate_idx": 3},
    ]

    # 加载内容特征
    with open(features_path) as f:
        features = json.load(f)
    motion = np.array([f["motion_intensity"] for f in features["features"][:num_segments]])
    sorted_idx = np.argsort(-motion)
    num_high = int(num_segments * high_motion_ratio)
    high_motion_segments = set(sorted_idx[:num_high].tolist())

    # 加载原始数据
    with open(original_manifest) as f:
        manifest = json.load(f)
    with open(original_enhance) as f:
        enhance = json.load(f)

    original_segment_sizes = np.array(manifest['segment_sizes_bits'])  # (159, 5)
    original_base_vmaf = np.array(enhance['base_quality_vmaf'])  # (4, 159)

    print("=" * 70)
    print("VBR集成到BONES仿真器（修正版）")
    print("=" * 70)
    print(f"段数: {num_segments}, 高运动段比例: {high_motion_ratio:.0%} ({num_high}段)")
    print(f"高运动段码率提升: {bitrate_increase:.0%}")
    print(f"处理分辨率: {[c['resolution'] for c in resolution_configs]}")
    print()

    # 存储每个分辨率的逐段数据
    all_vbr_sizes = {}  # {bitrate_idx: [size_bits, ...]}
    all_vbr_vmafs = {}  # {bitrate_idx: [vmaf, ...]}
    all_cbr_vmafs = {}  # {bitrate_idx: [vmaf, ...]}

    for config in resolution_configs:
        resolution = config["resolution"]
        base_bitrate = config["base_bitrate"]
        bitrate_idx = config["bitrate_idx"]
        high_bitrate = int(base_bitrate * bitrate_increase)

        # 检查中间文件是否已存在
        intermediate_file = intermediate_dir / f"{resolution}_data.json"
        if intermediate_file.exists():
            print(f"\n[跳过] {resolution} 中间数据已存在，加载中...")
            with open(intermediate_file) as f:
                data = json.load(f)
            all_vbr_sizes[bitrate_idx] = data['vbr_sizes']
            all_vbr_vmafs[bitrate_idx] = data['vbr_vmafs']
            all_cbr_vmafs[bitrate_idx] = data['cbr_vmafs']
            print(f"  CBR平均VMAF: {np.mean(data['cbr_vmafs']):.2f}")
            print(f"  VBR平均VMAF: {np.mean(data['vbr_vmafs']):.2f}")
            print(f"  提升量: {np.mean(data['vbr_vmafs']) - np.mean(data['cbr_vmafs']):+.2f}")
            continue

        print(f"\n{'='*60}")
        print(f"处理分辨率: {resolution} (码率档索引: {bitrate_idx})")
        print(f"基础码率: {base_bitrate}kbps, 高运动码率: {high_bitrate}kbps")
        print(f"{'='*60}")

        res_dir = output_dir / resolution
        res_dir.mkdir(exist_ok=True)
        ref_dir = res_dir / "reference"
        ref_dir.mkdir(exist_ok=True)
        cbr_dir = res_dir / "cbr"
        cbr_dir.mkdir(exist_ok=True)
        vbr_dir = res_dir / "vbr"
        vbr_dir.mkdir(exist_ok=True)

        # 步骤1：生成参考视频
        print(f"  [1/4] 生成参考视频...")
        for seg_idx in range(num_segments):
            ref_path = str(ref_dir / f"ref_{seg_idx:04d}.mp4")
            encode_segment(input_video, ref_path, seg_idx * segment_duration,
                           segment_duration, 10000, resolution, preset="medium")
            if (seg_idx + 1) % 40 == 0:
                print(f"    参考视频: {seg_idx+1}/{num_segments}")
        print(f"    参考视频完成")

        # 步骤2：生成CBR视频（用于测量CBR VMAF基准）
        print(f"  [2/4] 生成CBR视频...")
        for seg_idx in range(num_segments):
            cbr_path = str(cbr_dir / f"cbr_{seg_idx:04d}.mp4")
            encode_segment(input_video, cbr_path, seg_idx * segment_duration,
                           segment_duration, base_bitrate, resolution)
            if (seg_idx + 1) % 40 == 0:
                print(f"    CBR视频: {seg_idx+1}/{num_segments}")
        print(f"    CBR视频完成")

        # 步骤3：生成VBR视频并测量文件大小
        print(f"  [3/4] 生成VBR视频...")
        vbr_sizes = []
        for seg_idx in range(num_segments):
            target_br = high_bitrate if seg_idx in high_motion_segments else base_bitrate
            vbr_path = str(vbr_dir / f"vbr_{seg_idx:04d}.mp4")
            success, size = encode_segment(input_video, vbr_path, seg_idx * segment_duration,
                                            segment_duration, target_br, resolution)
            vbr_sizes.append(size * 8 if success else original_segment_sizes[seg_idx, bitrate_idx])
            if (seg_idx + 1) % 40 == 0:
                print(f"    VBR视频: {seg_idx+1}/{num_segments}")
        print(f"    VBR视频完成")

        # 步骤4：测量逐段VMAF（CBR和VBR）
        print(f"  [4/4] 测量逐段VMAF...")
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

            if (seg_idx + 1) % 20 == 0:
                print(f"    VMAF测量: {seg_idx+1}/{num_segments}, "
                      f"CBR={np.mean(cbr_vmafs[-20:]):.2f}, VBR={np.mean(vbr_vmafs[-20:]):.2f}")

        print(f"    CBR平均VMAF: {np.mean(cbr_vmafs):.2f}")
        print(f"    VBR平均VMAF: {np.mean(vbr_vmafs):.2f}")
        print(f"    提升量: {np.mean(vbr_vmafs) - np.mean(cbr_vmafs):+.2f}")

        # 保存中间数据
        all_vbr_sizes[bitrate_idx] = vbr_sizes
        all_vbr_vmafs[bitrate_idx] = vbr_vmafs
        all_cbr_vmafs[bitrate_idx] = cbr_vmafs

        with open(intermediate_file, "w") as f:
            json.dump({
                "resolution": resolution,
                "bitrate_idx": bitrate_idx,
                "vbr_sizes": vbr_sizes,
                "cbr_vmafs": cbr_vmafs,
                "vbr_vmafs": vbr_vmafs,
            }, f)
        print(f"    中间数据已保存: {intermediate_file}")

    # 生成新的manifest和enhance文件
    print(f"\n{'='*60}")
    print("生成新的manifest和enhance文件...")

    # 新的segment_sizes：用VBR实际文件大小
    new_segment_sizes = original_segment_sizes.copy()
    for bitrate_idx, sizes in all_vbr_sizes.items():
        new_segment_sizes[:, bitrate_idx] = sizes

    # 新的base_quality_vmaf：原始值 + VBR提升量（逐段）
    new_base_vmaf = original_base_vmaf.copy()
    for bitrate_idx in all_vbr_vmafs.keys():
        # 逐段计算提升量，然后加到原始base_quality上
        for seg_idx in range(num_segments):
            cbr_vmaf = all_cbr_vmafs[bitrate_idx][seg_idx]
            vbr_vmaf = all_vbr_vmafs[bitrate_idx][seg_idx]
            if cbr_vmaf > 0 and vbr_vmaf > 0:
                improvement = vbr_vmaf - cbr_vmaf
                new_base_vmaf[bitrate_idx, seg_idx] = original_base_vmaf[bitrate_idx, seg_idx] + improvement
            # 如果测量失败，保持原始值

    # 创建新的manifest
    new_manifest = {
        "segment_duration_ms": manifest["segment_duration_ms"],
        "bitrates_kbps": manifest["bitrates_kbps"],
        "segment_sizes_bits": new_segment_sizes.tolist(),
    }
    new_manifest_path = str(output_dir / "bbb_video_info_vbr.json")
    with open(new_manifest_path, "w") as f:
        json.dump(new_manifest, f)
    print(f"  新manifest: {new_manifest_path}")

    # 创建新的enhance（enhanced_quality保持不变，因为增强在下载后进行，与编码方式无关）
    new_enhance = {
        "frame_rate": enhance["frame_rate"],
        "enhancement_fps": enhance["enhancement_fps"],
        "base_quality_vmaf": new_base_vmaf.tolist(),
        "base_quality_ssim": enhance["base_quality_ssim"],
        "base_quality_psnr": enhance["base_quality_psnr"],
        "enhanced_quality_vmaf": enhance["enhanced_quality_vmaf"],
        "enhanced_quality_ssim": enhance["enhanced_quality_ssim"],
        "enhanced_quality_psnr": enhance["enhanced_quality_psnr"],
    }
    new_enhance_path = str(output_dir / "imdn_bbb_2080ti_vbr.json")
    with open(new_enhance_path, "w") as f:
        json.dump(new_enhance, f)
    print(f"  新enhance: {new_enhance_path}")

    # 汇总对比
    print(f"\n{'='*60}")
    print("汇总对比")
    print(f"{'='*60}")
    print(f"{'码率档':<10} {'原始VMAF':<12} {'新VMAF':<12} {'VMAF提升':<12} {'原始码率':<12} {'新码率':<12} {'码率变化':<12}")
    print("-" * 80)
    for config in resolution_configs:
        bitrate_idx = config["bitrate_idx"]
        resolution = config["resolution"]
        orig_vmaf = np.mean(original_base_vmaf[bitrate_idx])
        new_vmaf = np.mean(new_base_vmaf[bitrate_idx])
        orig_br = np.mean(original_segment_sizes[:, bitrate_idx]) / segment_duration / 1000
        new_br = np.mean(new_segment_sizes[:, bitrate_idx]) / segment_duration / 1000
        print(f"{resolution:<10} {orig_vmaf:<12.2f} {new_vmaf:<12.2f} {new_vmaf-orig_vmaf:<+12.2f} "
              f"{orig_br:<12.0f} {new_br:<12.0f} {(new_br-orig_br)/orig_br*100:<+11.1f}%")

    print(f"\n完成！新的manifest和enhance文件已保存到: {output_dir}")
    print(f"下一步: 使用新的manifest和enhance文件运行BONES仿真，对比CBR和VBR的QoE")


if __name__ == "__main__":
    main()
