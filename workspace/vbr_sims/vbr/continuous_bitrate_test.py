"""
阶段二：连续码率分配曲线对比测试

在480p上测试4种码率分配策略：
1. 二值分配（当前基线）：前60%高运动段+25%，其他不变
2. 线性分配：码率 = 基础 × (1 + 0.25 × 运动百分位)
3. S型分配：码率 = 基础 × (1 + 0.25 × sigmoid(6×(百分位-0.5)))
4. 平方根分配：码率 = 基础 × (1 + 0.25 × sqrt(百分位))

测量每种策略的逐段VMAF，对比效果。
"""
import os
import sys
import json
import subprocess
import numpy as np
from pathlib import Path


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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


def get_bitrate_allocation(strategy, motion, base_bitrate, max_increase=0.25):
    """
    根据策略计算每段的目标码率

    strategy: "binary" | "linear" | "sigmoid" | "sqrt"
    motion: 运动强度数组
    base_bitrate: 基础码率
    max_increase: 最大提升幅度
    """
    num_segments = len(motion)
    # 计算运动强度百分位（0=最低，1=最高）
    sorted_indices = np.argsort(motion)
    percentile = np.zeros(num_segments)
    for rank, idx in enumerate(sorted_indices):
        percentile[idx] = rank / (num_segments - 1)

    if strategy == "binary":
        # 二值分配：前60%高运动段+25%，其他不变
        threshold = np.percentile(motion, 40)  # 前60%高运动
        bitrates = np.where(motion >= threshold, base_bitrate * (1 + max_increase), base_bitrate)
    elif strategy == "linear":
        # 线性分配：码率 = 基础 × (1 + max_increase × 百分位)
        bitrates = base_bitrate * (1 + max_increase * percentile)
    elif strategy == "sigmoid":
        # S型分配：码率 = 基础 × (1 + max_increase × sigmoid(6×(百分位-0.5)))
        bitrates = base_bitrate * (1 + max_increase * sigmoid(6 * (percentile - 0.5)))
    elif strategy == "sqrt":
        # 平方根分配：码率 = 基础 × (1 + max_increase × sqrt(百分位))
        bitrates = base_bitrate * (1 + max_increase * np.sqrt(percentile))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return bitrates, percentile


def main():
    # 配置
    input_video = r"E:\dev\Code\Python\bones\data\bbb\bbb_2160p_60fps.mp4"
    features_path = r"E:\dev\Code\Python\bones\data\bbb\content_features_bbb.json"
    output_base = Path(r"E:\dev\Code\Python\bones\data\continuous_bitrate_test")
    output_base.mkdir(parents=True, exist_ok=True)

    num_segments = 159
    segment_duration = 4.0
    resolution = "480p"
    base_bitrate = 1203
    max_increase = 0.25

    strategies = ["binary", "linear", "sigmoid", "sqrt"]
    strategy_names = {
        "binary": "二值分配（基线）",
        "linear": "线性分配",
        "sigmoid": "S型分配",
        "sqrt": "平方根分配",
    }

    # 加载内容特征
    with open(features_path) as f:
        features = json.load(f)
    motion = np.array([f["motion_intensity"] for f in features["features"][:num_segments]])

    print("=" * 70)
    print("阶段二：连续码率分配曲线对比测试（480p）")
    print("=" * 70)
    print(f"基础码率: {base_bitrate}kbps, 最大提升: {max_increase*100:.0f}%")
    print(f"测试策略: {[strategy_names[s] for s in strategies]}")
    print()

    # 生成参考视频（所有策略共用）
    print("[步骤1] 生成参考视频（10Mbps）...")
    ref_dir = output_base / "reference"
    ref_dir.mkdir(exist_ok=True)
    for seg_idx in range(num_segments):
        ref_path = str(ref_dir / f"ref_{seg_idx:04d}.mp4")
        encode_segment(input_video, ref_path, seg_idx * segment_duration,
                       segment_duration, 10000, resolution, preset="medium")
        if (seg_idx + 1) % 40 == 0:
            print(f"  参考视频: {seg_idx+1}/{num_segments}")
    print("  参考视频完成")

    # 生成CBR视频（所有策略共用，作为基准）
    print("\n[步骤2] 生成CBR视频（基准）...")
    cbr_dir = output_base / "cbr"
    cbr_dir.mkdir(exist_ok=True)
    for seg_idx in range(num_segments):
        cbr_path = str(cbr_dir / f"cbr_{seg_idx:04d}.mp4")
        encode_segment(input_video, cbr_path, seg_idx * segment_duration,
                       segment_duration, base_bitrate, resolution)
        if (seg_idx + 1) % 40 == 0:
            print(f"  CBR视频: {seg_idx+1}/{num_segments}")
    print("  CBR视频完成")

    # 对每种策略生成VBR视频并测量VMAF
    all_results = {}

    for strategy in strategies:
        print(f"\n[策略: {strategy_names[strategy]}]")

        # 计算每段目标码率
        bitrates, percentile = get_bitrate_allocation(strategy, motion, base_bitrate, max_increase)
        avg_bitrate = np.mean(bitrates)
        print(f"  平均码率: {avg_bitrate:.0f}kbps (基础{base_bitrate}kbps, +{(avg_bitrate/base_bitrate-1)*100:.1f}%)")
        print(f"  码率范围: {np.min(bitrates):.0f} ~ {np.max(bitrates):.0f}kbps")

        # 生成VBR视频
        vbr_dir = output_base / strategy
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
                      f"近20段CBR={avg_cbr:.2f}, VBR={avg_vbr:.2f}, diff={avg_vbr-avg_cbr:+.2f}")

        # 计算结果
        valid_cbr = [x for x in cbr_vmafs if x > 0]
        valid_vbr = [x for x in vbr_vmafs if x > 0]
        avg_cbr = np.mean(valid_cbr) if valid_cbr else 0
        avg_vbr = np.mean(valid_vbr) if valid_vbr else 0
        avg_improve = avg_vbr - avg_cbr

        # 逐段提升量
        improvements = np.array(vbr_vmafs) - np.array(cbr_vmafs)
        valid_imp = improvements[(np.array(cbr_vmafs) > 0) & (np.array(vbr_vmafs) > 0)]
        avg_imp_valid = np.mean(valid_imp) if len(valid_imp) > 0 else 0

        all_results[strategy] = {
            "strategy": strategy,
            "strategy_name": strategy_names[strategy],
            "avg_bitrate": float(avg_bitrate),
            "bitrate_increase_pct": float((avg_bitrate / base_bitrate - 1) * 100),
            "cbr_vmaf": float(avg_cbr),
            "vbr_vmaf": float(avg_vbr),
            "vmaf_improvement": float(avg_imp_valid),
            "cbr_vmafs": cbr_vmafs,
            "vbr_vmafs": vbr_vmafs,
            "bitrates": bitrates.tolist(),
            "percentile": percentile.tolist(),
        }

        print(f"  CBR平均VMAF: {avg_cbr:.2f}")
        print(f"  VBR平均VMAF: {avg_vbr:.2f}")
        print(f"  VMAF提升: {avg_imp_valid:+.2f}")
        print(f"  码率提升: +{(avg_bitrate/base_bitrate-1)*100:.1f}%")

    # 汇总对比
    print("\n" + "=" * 70)
    print("汇总对比（480p）")
    print("=" * 70)
    print(f"{'策略':<15} {'平均码率':<12} {'码率提升':<10} {'CBR VMAF':<12} {'VBR VMAF':<12} {'VMAF提升':<10}")
    print("-" * 75)
    for strategy in strategies:
        r = all_results[strategy]
        print(f"{r['strategy_name']:<15} {r['avg_bitrate']:<12.0f} +{r['bitrate_increase_pct']:<9.1f}% "
              f"{r['cbr_vmaf']:<12.2f} {r['vbr_vmaf']:<12.2f} {r['vmaf_improvement']:<+10.2f}")

    # 找到最优策略
    best_strategy = max(strategies, key=lambda s: all_results[s]['vmaf_improvement'])
    print(f"\n最优策略: {all_results[best_strategy]['strategy_name']} "
          f"(VMAF提升 {all_results[best_strategy]['vmaf_improvement']:+.2f})")

    # 保存结果
    result_path = str(output_base / "continuous_bitrate_comparison.json")
    save_results = {}
    for strategy in strategies:
        r = all_results[strategy]
        save_results[strategy] = {k: v for k, v in r.items() if k not in ["cbr_vmafs", "vbr_vmafs", "bitrates", "percentile"]}
    with open(result_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
