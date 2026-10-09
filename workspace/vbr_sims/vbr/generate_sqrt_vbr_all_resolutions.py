"""
阶段二扩展：平方根分配的5分辨率VBR数据生成

将最优的平方根分配策略扩展到240p/360p/480p/720p/1080p，
生成逐段VMAF数据，用于后续全轨迹纯净验证。
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


def get_sqrt_bitrates(motion, base_bitrate, max_increase=0.25):
    """平方根分配：码率 = 基础 × (1 + max_increase × sqrt(百分位))"""
    num_segments = len(motion)
    sorted_indices = np.argsort(motion)
    percentile = np.zeros(num_segments)
    for rank, idx in enumerate(sorted_indices):
        percentile[idx] = rank / (num_segments - 1)
    bitrates = base_bitrate * (1 + max_increase * np.sqrt(percentile))
    return bitrates, percentile


def main():
    # 配置
    input_video = r"E:\dev\Code\Python\bones\data\bbb\bbb_2160p_60fps.mp4"
    features_path = r"E:\dev\Code\Python\bones\data\bbb\content_features_bbb.json"
    output_base = Path(r"E:\dev\Code\Python\bones\data\bbb_vbr_sqrt")
    output_base.mkdir(parents=True, exist_ok=True)

    num_segments = 159
    segment_duration = 4.0
    max_increase = 0.25

    # 各分辨率配置（基础码率）
    resolutions = {
        "240p": 250,
        "360p": 500,
        "480p": 1203,
        "720p": 2500,
        "1080p": 5000,
    }

    # 加载内容特征
    with open(features_path) as f:
        features = json.load(f)
    motion = np.array([f["motion_intensity"] for f in features["features"][:num_segments]])

    print("=" * 70)
    print("阶段二扩展：平方根分配的5分辨率VBR数据生成")
    print("=" * 70)
    print(f"策略: 平方根分配, 最大提升: {max_increase*100:.0f}%")
    print(f"分辨率: {list(resolutions.keys())}")
    print()

    all_results = {}

    for res_name, base_bitrate in resolutions.items():
        print(f"\n[分辨率: {res_name}, 基础码率: {base_bitrate}kbps]")

        # 计算平方根分配的逐段码率
        bitrates, percentile = get_sqrt_bitrates(motion, base_bitrate, max_increase)
        avg_bitrate = np.mean(bitrates)
        print(f"  平均码率: {avg_bitrate:.0f}kbps (+{(avg_bitrate/base_bitrate-1)*100:.1f}%)")
        print(f"  码率范围: {np.min(bitrates):.0f} ~ {np.max(bitrates):.0f}kbps")

        # 创建目录
        res_dir = output_base / res_name
        ref_dir = res_dir / "reference"
        cbr_dir = res_dir / "cbr"
        vbr_dir = res_dir / "vbr"
        for d in [ref_dir, cbr_dir, vbr_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 生成参考视频
        print("  生成参考视频...")
        for seg_idx in range(num_segments):
            ref_path = str(ref_dir / f"ref_{seg_idx:04d}.mp4")
            encode_segment(input_video, ref_path, seg_idx * segment_duration,
                           segment_duration, 10000, res_name, preset="medium")
            if (seg_idx + 1) % 40 == 0:
                print(f"    参考视频: {seg_idx+1}/{num_segments}")

        # 生成CBR视频
        print("  生成CBR视频...")
        for seg_idx in range(num_segments):
            cbr_path = str(cbr_dir / f"cbr_{seg_idx:04d}.mp4")
            encode_segment(input_video, cbr_path, seg_idx * segment_duration,
                           segment_duration, base_bitrate, res_name)
            if (seg_idx + 1) % 40 == 0:
                print(f"    CBR视频: {seg_idx+1}/{num_segments}")

        # 生成VBR视频（平方根分配）
        print("  生成VBR视频（平方根分配）...")
        for seg_idx in range(num_segments):
            vbr_path = str(vbr_dir / f"vbr_{seg_idx:04d}.mp4")
            encode_segment(input_video, vbr_path, seg_idx * segment_duration,
                           segment_duration, bitrates[seg_idx], res_name)
            if (seg_idx + 1) % 40 == 0:
                print(f"    VBR视频: {seg_idx+1}/{num_segments}")

        # 测量逐段VMAF
        print("  测量逐段VMAF...")
        cbr_vmafs = []
        vbr_vmafs = []
        cbr_sizes = []
        vbr_sizes = []
        for seg_idx in range(num_segments):
            ref_path = str(ref_dir / f"ref_{seg_idx:04d}.mp4")
            cbr_path = str(cbr_dir / f"cbr_{seg_idx:04d}.mp4")
            vbr_path = str(vbr_dir / f"vbr_{seg_idx:04d}.mp4")

            cbr_vmaf = measure_vmaf_single(ref_path, cbr_path, res_name)
            vbr_vmaf = measure_vmaf_single(ref_path, vbr_path, res_name)

            cbr_vmafs.append(cbr_vmaf if cbr_vmaf is not None else 0)
            vbr_vmafs.append(vbr_vmaf if vbr_vmaf is not None else 0)
            cbr_sizes.append(os.path.getsize(cbr_path) if os.path.exists(cbr_path) else 0)
            vbr_sizes.append(os.path.getsize(vbr_path) if os.path.exists(vbr_path) else 0)

            if (seg_idx + 1) % 20 == 0:
                recent_cbr = [x for x in cbr_vmafs[-20:] if x > 0]
                recent_vbr = [x for x in vbr_vmafs[-20:] if x > 0]
                avg_cbr = np.mean(recent_cbr) if recent_cbr else 0
                avg_vbr = np.mean(recent_vbr) if recent_vbr else 0
                print(f"    VMAF测量: {seg_idx+1}/{num_segments}, "
                      f"CBR={avg_cbr:.2f}, VBR={avg_vbr:.2f}, diff={avg_vbr-avg_cbr:+.2f}")

        # 计算结果
        valid_cbr = [x for x in cbr_vmafs if x > 0]
        valid_vbr = [x for x in vbr_vmafs if x > 0]
        avg_cbr = np.mean(valid_cbr) if valid_cbr else 0
        avg_vbr = np.mean(valid_vbr) if valid_vbr else 0
        improvements = np.array(vbr_vmafs) - np.array(cbr_vmafs)
        valid_imp = improvements[(np.array(cbr_vmafs) > 0) & (np.array(vbr_vmafs) > 0)]
        avg_imp = np.mean(valid_imp) if len(valid_imp) > 0 else 0

        all_results[res_name] = {
            "resolution": res_name,
            "base_bitrate": base_bitrate,
            "avg_bitrate": float(avg_bitrate),
            "bitrate_increase_pct": float((avg_bitrate / base_bitrate - 1) * 100),
            "cbr_vmaf": float(avg_cbr),
            "vbr_vmaf": float(avg_vbr),
            "vmaf_improvement": float(avg_imp),
            "cbr_vmafs": cbr_vmafs,
            "vbr_vmafs": vbr_vmafs,
            "cbr_sizes": cbr_sizes,
            "vbr_sizes": vbr_sizes,
            "bitrates": bitrates.tolist(),
        }

        print(f"  CBR平均VMAF: {avg_cbr:.2f}")
        print(f"  VBR平均VMAF: {avg_vbr:.2f}")
        print(f"  VMAF提升: {avg_imp:+.2f}")
        print(f"  码率提升: +{(avg_bitrate/base_bitrate-1)*100:.1f}%")

    # 汇总
    print("\n" + "=" * 70)
    print("5分辨率平方根分配汇总")
    print("=" * 70)
    print(f"{'分辨率':<10} {'基础码率':<10} {'平均码率':<10} {'码率提升':<10} {'CBR VMAF':<12} {'VBR VMAF':<12} {'VMAF提升':<10}")
    print("-" * 75)
    for res_name in resolutions:
        r = all_results[res_name]
        print(f"{res_name:<10} {r['base_bitrate']:<10} {r['avg_bitrate']:<10.0f} "
              f"+{r['bitrate_increase_pct']:<9.1f}% {r['cbr_vmaf']:<12.2f} "
              f"{r['vbr_vmaf']:<12.2f} {r['vmaf_improvement']:<+10.2f}")

    # 保存中间数据
    intermediate_dir = output_base / "intermediate"
    intermediate_dir.mkdir(exist_ok=True)
    for res_name in resolutions:
        r = all_results[res_name]
        intermediate_path = str(intermediate_dir / f"{res_name}_data.json")
        with open(intermediate_path, "w") as f:
            json.dump({
                "cbr_vmafs": r["cbr_vmafs"],
                "vbr_vmafs": r["vbr_vmafs"],
                "cbr_sizes": r["cbr_sizes"],
                "vbr_sizes": r["vbr_sizes"],
                "bitrates": r["bitrates"],
            }, f)
        print(f"  中间数据已保存: {intermediate_path}")

    # 保存汇总结果
    summary_path = str(output_base / "sqrt_vbr_summary.json")
    save_summary = {}
    for res_name in resolutions:
        r = all_results[res_name]
        save_summary[res_name] = {k: v for k, v in r.items()
                                   if k not in ["cbr_vmafs", "vbr_vmafs", "cbr_sizes", "vbr_sizes", "bitrates"]}
    with open(summary_path, "w") as f:
        json.dump(save_summary, f, indent=2)
    print(f"\n汇总结果已保存: {summary_path}")


if __name__ == "__main__":
    main()
