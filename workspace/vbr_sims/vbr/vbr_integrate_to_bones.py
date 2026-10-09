"""
VBR集成到BONES仿真器：为每个码率档生成VBR逐段数据，创建新的manifest和enhance文件

VBR策略：60%高运动段提升25%码率，其他段保持基础码率
"""
import os
import sys
import json
import subprocess
import numpy as np
from pathlib import Path
import shutil


def encode_segment(input_path, output_path, start_time_s, duration_s,
                   target_bitrate_kbps, resolution="480p", fps=30, preset="veryfast"):
    """用ffmpeg编码单段视频（如果文件已存在且非空则跳过）"""
    # 如果文件已存在且非空，直接返回
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

    num_segments = 159
    segment_duration = 4.0
    high_motion_ratio = 0.60  # 最优比例
    bitrate_increase = 1.25  # 高运动段码率提升25%

    # 码率档配置（对应BONES的5个码率档）
    # 索引0-4: 240p/360p/480p/720p/1080p
    # 原始目标码率: [398, 802, 1203, 2406, 4738]
    resolution_configs = [
        {"resolution": "240p", "base_bitrate": 398, "bitrate_idx": 0},
        {"resolution": "360p", "base_bitrate": 802, "bitrate_idx": 1},
        {"resolution": "480p", "base_bitrate": 1203, "bitrate_idx": 2},
        {"resolution": "720p", "base_bitrate": 2406, "bitrate_idx": 3},
        # 1080p保持CBR不变（高分辨率VBR收益小，且编码耗时）
    ]

    # 加载内容特征
    with open(features_path) as f:
        features = json.load(f)
    motion = np.array([f["motion_intensity"] for f in features["features"][:num_segments]])
    sorted_idx = np.argsort(-motion)
    num_high = int(num_segments * high_motion_ratio)
    high_motion_segments = set(sorted_idx[:num_high].tolist())

    print("=" * 70)
    print("VBR集成到BONES仿真器")
    print("=" * 70)
    print(f"段数: {num_segments}, 高运动段比例: {high_motion_ratio:.0%} ({num_high}段)")
    print(f"高运动段码率提升: {bitrate_increase:.0%}")
    print(f"处理分辨率: {[c['resolution'] for c in resolution_configs]}")
    print()

    # 加载原始数据
    with open(original_manifest) as f:
        manifest = json.load(f)
    with open(original_enhance) as f:
        enhance = json.load(f)

    # 复制原始数据作为基础
    original_segment_sizes = np.array(manifest['segment_sizes_bits'])  # (159, 5)
    new_segment_sizes = original_segment_sizes.copy()  # (159, 5)
    new_base_quality_vmaf = np.array(enhance['base_quality_vmaf']).copy()  # (4, 159)
    new_base_quality_ssim = np.array(enhance['base_quality_ssim']).copy()
    new_base_quality_psnr = np.array(enhance['base_quality_psnr']).copy()

    # 为每个分辨率生成VBR数据
    for config in resolution_configs:
        resolution = config["resolution"]
        base_bitrate = config["base_bitrate"]
        bitrate_idx = config["bitrate_idx"]
        high_bitrate = int(base_bitrate * bitrate_increase)

        print(f"\n{'='*60}")
        print(f"处理分辨率: {resolution} (码率档索引: {bitrate_idx})")
        print(f"基础码率: {base_bitrate}kbps, 高运动码率: {high_bitrate}kbps")
        print(f"{'='*60}")

        res_dir = output_dir / resolution
        res_dir.mkdir(exist_ok=True)
        ref_dir = res_dir / "reference"
        ref_dir.mkdir(exist_ok=True)
        vbr_dir = res_dir / "vbr"
        vbr_dir.mkdir(exist_ok=True)

        # 步骤1：生成参考视频（每段）
        print(f"  [1/3] 生成参考视频...")
        ref_segments = []
        for seg_idx in range(num_segments):
            ref_path = str(ref_dir / f"ref_{seg_idx:04d}.mp4")
            success, size = encode_segment(input_video, ref_path, seg_idx * segment_duration,
                                            segment_duration, 10000, resolution, preset="medium")
            if success:
                ref_segments.append(ref_path)
            if (seg_idx + 1) % 20 == 0:
                print(f"    参考视频: {seg_idx+1}/{num_segments}")
        print(f"    参考视频完成: {len(ref_segments)}段")

        # 步骤2：生成VBR视频并测量文件大小
        print(f"  [2/3] 生成VBR视频并测量文件大小...")
        vbr_sizes = []
        vbr_segments = []
        for seg_idx in range(num_segments):
            target_br = high_bitrate if seg_idx in high_motion_segments else base_bitrate
            vbr_path = str(vbr_dir / f"vbr_{seg_idx:04d}.mp4")
            success, size = encode_segment(input_video, vbr_path, seg_idx * segment_duration,
                                            segment_duration, target_br, resolution)
            if success:
                vbr_segments.append(vbr_path)
                vbr_sizes.append(size * 8)  # 转换为比特
            else:
                vbr_sizes.append(new_segment_sizes[seg_idx, bitrate_idx])  # 失败则用原始值
            if (seg_idx + 1) % 20 == 0:
                print(f"    VBR视频: {seg_idx+1}/{num_segments}")
        print(f"    VBR视频完成: {len(vbr_segments)}段")

        # 更新segment_sizes
        new_segment_sizes[:, bitrate_idx] = vbr_sizes

        # 步骤3：测量逐段VMAF
        print(f"  [3/3] 测量逐段VMAF...")
        vbr_vmafs = []
        for seg_idx in range(num_segments):
            ref_path = str(ref_dir / f"ref_{seg_idx:04d}.mp4")
            vbr_path = str(vbr_dir / f"vbr_{seg_idx:04d}.mp4")
            if os.path.exists(ref_path) and os.path.exists(vbr_path):
                vmaf = measure_vmaf_single(ref_path, vbr_path, resolution)
                if vmaf is not None:
                    vbr_vmafs.append(vmaf)
                else:
                    vbr_vmafs.append(new_base_quality_vmaf[bitrate_idx, seg_idx])  # 失败则用原始值
            else:
                vbr_vmafs.append(new_base_quality_vmaf[bitrate_idx, seg_idx])
            if (seg_idx + 1) % 20 == 0:
                print(f"    VMAF测量: {seg_idx+1}/{num_segments}, 当前平均VMAF={np.mean(vbr_vmafs):.2f}")
        print(f"    VMAF测量完成: 平均VMAF={np.mean(vbr_vmafs):.2f}")

        # 更新base_quality_vmaf
        new_base_quality_vmaf[bitrate_idx, :] = vbr_vmafs

        # 打印对比
        original_avg_vmaf = np.mean(enhance['base_quality_vmaf'][bitrate_idx])
        new_avg_vmaf = np.mean(vbr_vmafs)
        original_avg_size = np.mean(original_segment_sizes[:, bitrate_idx]) * 8 / segment_duration / 1000
        new_avg_size = np.mean(vbr_sizes) / segment_duration / 1000
        print(f"\n  对比:")
        print(f"    原始平均VMAF: {original_avg_vmaf:.2f}, VBR平均VMAF: {new_avg_vmaf:.2f}, diff: {new_avg_vmaf-original_avg_vmaf:+.2f}")
        print(f"    原始平均码率: {original_avg_size:.0f}kbps, VBR平均码率: {new_avg_size:.0f}kbps, diff: {new_avg_size-original_avg_size:+.0f}kbps ({(new_avg_size-original_avg_size)/original_avg_size*100:+.1f}%)")

    # 创建新的manifest文件
    print(f"\n{'='*60}")
    print("创建新的manifest和enhance文件...")
    new_manifest = {
        "segment_duration_ms": manifest["segment_duration_ms"],
        "bitrates_kbps": manifest["bitrates_kbps"],
        "segment_sizes_bits": new_segment_sizes.tolist(),
    }
    new_manifest_path = str(output_dir / "bbb_video_info_vbr.json")
    with open(new_manifest_path, "w") as f:
        json.dump(new_manifest, f)
    print(f"  新manifest: {new_manifest_path}")

    # 创建新的enhance文件
    new_enhance = {
        "frame_rate": enhance["frame_rate"],
        "enhancement_fps": enhance["enhancement_fps"],
        "base_quality_vmaf": new_base_quality_vmaf.tolist(),
        "base_quality_ssim": new_base_quality_ssim.tolist(),
        "base_quality_psnr": new_base_quality_psnr.tolist(),
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
    print("汇总对比（各码率档平均VMAF）")
    print(f"{'='*60}")
    print(f"{'码率档':<10} {'原始VMAF':<12} {'VBR VMAF':<12} {'VMAF提升':<12} {'码率提升':<12}")
    print("-" * 60)
    for config in resolution_configs:
        bitrate_idx = config["bitrate_idx"]
        resolution = config["resolution"]
        orig_vmaf = np.mean(enhance['base_quality_vmaf'][bitrate_idx])
        new_vmaf = np.mean(new_base_quality_vmaf[bitrate_idx])
        orig_br = np.mean(original_segment_sizes[:, bitrate_idx]) * 8 / segment_duration / 1000
        new_br = np.mean(new_segment_sizes[:, bitrate_idx]) / segment_duration / 1000
        print(f"{resolution:<10} {orig_vmaf:<12.2f} {new_vmaf:<12.2f} {new_vmaf-orig_vmaf:<+12.2f} {(new_br-orig_br)/orig_br*100:<+11.1f}%")

    print(f"\n完成！新的manifest和enhance文件已保存到: {output_dir}")
    print(f"下一步: 使用新的manifest和enhance文件运行BONES仿真，对比CBR和VBR的QoE")


if __name__ == "__main__":
    main()
