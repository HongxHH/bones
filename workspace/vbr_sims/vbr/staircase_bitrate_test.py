"""
阶梯式连续分配测试

在480p上测试3种阶梯式分配（3档/4档/5档），
保持平均码率提升约+15~16.5%，对比二值分配和平方根分配的效果。

阶梯式分配：将运动强度分成几档，每档有不同的码率提升。
档位越多越接近连续分配，档位越少越接近二值分配（振荡更小）。
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
    if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
        return True, os.path.getsize(output_path)

    res_map = {"240p": "426:240", "360p": "640:360", "480p": "854:480",
               "720p": "1280:720", "1080p": "1920:1080"}
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
    res_map = {"240p": "426:240", "360p": "640:360", "480p": "854:480",
               "720p": "1280:720", "1080p": "1920:1080"}
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


def get_staircase_bitrates(motion, base_bitrate, increase_levels):
    """
    阶梯式分配：将运动强度按百分位分成N档，每档有不同的码率提升

    increase_levels: list of increase percentages, e.g. [0.05, 0.15, 0.25] for 3 levels
    """
    num_segments = len(motion)
    num_levels = len(increase_levels)
    sorted_indices = np.argsort(motion)
    percentile = np.zeros(num_segments)
    for rank, idx in enumerate(sorted_indices):
        percentile[idx] = rank / (num_segments - 1)

    # 按百分位分档
    bitrates = np.zeros(num_segments)
    for i in range(num_segments):
        level_idx = min(int(percentile[i] * num_levels), num_levels - 1)
        bitrates[i] = base_bitrate * (1 + increase_levels[level_idx])

    return bitrates, percentile


def main():
    # 配置
    input_video = r"E:\dev\Code\Python\bones\data\bbb\bbb_2160p_60fps.mp4"
    features_path = r"E:\dev\Code\Python\bones\data\bbb\content_features_bbb.json"
    output_base = Path(r"E:\dev\Code\Python\bones\data\staircase_bitrate_test")
    output_base.mkdir(parents=True, exist_ok=True)

    num_segments = 159
    segment_duration = 4.0
    resolution = "480p"
    base_bitrate = 1203

    # 3种阶梯式分配策略
    # 每档的码率提升百分比，平均提升约+15~16.5%
    strategies = [
        ("stair_3level", "3档分配 [5%,15%,25%]", [0.05, 0.15, 0.25]),
        ("stair_4level", "4档分配 [5%,10%,20%,30%]", [0.05, 0.10, 0.20, 0.30]),
        ("stair_5level", "5档分配 [5%,10%,15%,20%,25%]", [0.05, 0.10, 0.15, 0.20, 0.25]),
    ]

    # 加载内容特征
    with open(features_path) as f:
        features = json.load(f)
    motion = np.array([f["motion_intensity"] for f in features["features"][:num_segments]])

    print("=" * 70)
    print("阶梯式连续分配测试（480p）")
    print("=" * 70)
    print(f"基础码率: {base_bitrate}kbps")
    print(f"测试策略: {[s[1] for s in strategies]}")
    print()

    # 复用之前的参考视频和CBR视频
    prev_ref_dir = Path(r"E:\dev\Code\Python\bones\data\continuous_bitrate_test\480p\reference")
    prev_cbr_dir = Path(r"E:\dev\Code\Python\bones\data\continuous_bitrate_test\480p\cbr")

    ref_dir = output_base / "reference"
    cbr_dir = output_base / "cbr"
    ref_dir.mkdir(parents=True, exist_ok=True)
    cbr_dir.mkdir(parents=True, exist_ok=True)

    # 复制或生成参考视频
    print("[步骤1] 准备参考视频...")
    for seg_idx in range(num_segments):
        ref_path = str(ref_dir / f"ref_{seg_idx:04d}.mp4")
        if not os.path.exists(ref_path) or os.path.getsize(ref_path) < 1000:
            if prev_ref_dir.exists():
                prev_path = str(prev_ref_dir / f"ref_{seg_idx:04d}.mp4")
                if os.path.exists(prev_path):
                    shutil.copy2(prev_path, ref_path)
                    continue
            encode_segment(input_video, ref_path, seg_idx * segment_duration,
                           segment_duration, 10000, resolution, preset="medium")
        if (seg_idx + 1) % 40 == 0:
            print(f"  参考视频: {seg_idx+1}/{num_segments}")
    print("  参考视频完成")

    # 复制或生成CBR视频
    print("\n[步骤2] 准备CBR视频...")
    for seg_idx in range(num_segments):
        cbr_path = str(cbr_dir / f"cbr_{seg_idx:04d}.mp4")
        if not os.path.exists(cbr_path) or os.path.getsize(cbr_path) < 1000:
            if prev_cbr_dir.exists():
                prev_path = str(prev_cbr_dir / f"cbr_{seg_idx:04d}.mp4")
                if os.path.exists(prev_path):
                    shutil.copy2(prev_path, cbr_path)
                    continue
            encode_segment(input_video, cbr_path, seg_idx * segment_duration,
                           segment_duration, base_bitrate, resolution)
        if (seg_idx + 1) % 40 == 0:
            print(f"  CBR视频: {seg_idx+1}/{num_segments}")
    print("  CBR视频完成")

    # 对每种策略生成VBR视频并测量VMAF
    all_results = {}

    for strategy_id, strategy_name, increase_levels in strategies:
        print(f"\n[策略: {strategy_name}]")

        # 计算每段目标码率
        bitrates, percentile = get_staircase_bitrates(motion, base_bitrate, increase_levels)
        avg_bitrate = np.mean(bitrates)
        print(f"  平均码率: {avg_bitrate:.0f}kbps (+{(avg_bitrate/base_bitrate-1)*100:.1f}%)")
        print(f"  码率范围: {np.min(bitrates):.0f} ~ {np.max(bitrates):.0f}kbps")
        print(f"  各档码率: {[f'{base_bitrate*(1+inc):.0f}' for inc in increase_levels]}")

        # 生成VBR视频
        vbr_dir = output_base / strategy_id
        vbr_dir.mkdir(exist_ok=True)
        for seg_idx in range(num_segments):
            vbr_path = str(vbr_dir / f"vbr_{seg_idx:04d}.mp4")
            encode_segment(input_video, vbr_path, seg_idx * segment_duration,
                           segment_duration, bitrates[seg_idx], resolution)
            if (seg_idx + 1) % 40 == 0:
                print(f"  VBR视频: {seg_idx+1}/{num_segments}")
        print("  VBR视频完成")

        # 测量逐段VMAF
        print("  测量逐段VMAF...")
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

        # 计算相邻段码率差的平均值（衡量平滑度，越小越平滑）
        bitrate_diffs = np.abs(np.diff(bitrates))
        avg_bitrate_diff = np.mean(bitrate_diffs)

        all_results[strategy_id] = {
            "strategy_id": strategy_id,
            "strategy_name": strategy_name,
            "increase_levels": increase_levels,
            "avg_bitrate": float(avg_bitrate),
            "bitrate_increase_pct": float((avg_bitrate / base_bitrate - 1) * 100),
            "cbr_vmaf": float(avg_cbr),
            "vbr_vmaf": float(avg_vbr),
            "vmaf_improvement": float(avg_imp),
            "avg_bitrate_diff": float(avg_bitrate_diff),
            "cbr_vmafs": cbr_vmafs,
            "vbr_vmafs": vbr_vmafs,
            "bitrates": bitrates.tolist(),
        }

        print(f"  CBR平均VMAF: {avg_cbr:.2f}")
        print(f"  VBR平均VMAF: {avg_vbr:.2f}")
        print(f"  VMAF提升: {avg_imp:+.2f}")
        print(f"  相邻段平均码率差: {avg_bitrate_diff:.0f}kbps（越小越平滑）")

    # 汇总对比
    print("\n" + "=" * 70)
    print("汇总对比（480p）")
    print("=" * 70)
    print(f"{'策略':<25} {'平均码率':<10} {'码率提升':<10} {'VMAF提升':<10} {'相邻码率差':<10} {'平滑度'}")
    print("-" * 80)

    # 添加基线对比
    baselines = [
        ("二值分配（基线）", 1383, 14.9, 1.18, None),
        ("平方根分配", 1403, 16.6, 1.29, None),
    ]
    for name, avg_br, br_inc, vmaf_imp, _ in baselines:
        print(f"{name:<25} {avg_br:<10.0f} +{br_inc:<9.1f}% {vmaf_imp:<+10.2f} {'N/A':<10} {'N/A'}")

    for strategy_id, strategy_name, increase_levels in strategies:
        r = all_results[strategy_id]
        smoothness = "平滑" if r["avg_bitrate_diff"] < 50 else ("中等" if r["avg_bitrate_diff"] < 100 else "不平滑")
        print(f"{r['strategy_name']:<25} {r['avg_bitrate']:<10.0f} "
              f"+{r['bitrate_increase_pct']:<9.1f}% {r['vmaf_improvement']:<+10.2f} "
              f"{r['avg_bitrate_diff']:<10.0f} {smoothness}")

    # 找到最优策略
    print(f"\n基线参考：二值分配 VMAF提升 +1.18，平方根分配 VMAF提升 +1.29")
    print("目标：找到VMAF提升接近平方根分配（+1.29），但码率分配更平滑（相邻码率差小）的策略")

    # 保存结果
    result_path = str(output_base / "staircase_bitrate_comparison.json")
    save_results = {}
    for strategy_id in all_results:
        r = all_results[strategy_id]
        save_results[strategy_id] = {k: v for k, v in r.items()
                                      if k not in ["cbr_vmafs", "vbr_vmafs", "bitrates"]}
    with open(result_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
