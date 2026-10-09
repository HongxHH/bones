"""
小规模VBR编码实验：验证同一分辨率下内容感知码率分配是否能提升VMAF

实验设计：
- 分辨率：480p (854x480)
- 段数：前20段（每段4秒，共80秒）
- CBR方案：所有段用1200kbps
- VBR方案：高运动段(10段)用1500kbps，低运动段(10段)用900kbps，平均1200kbps
- 参考视频：480p 10Mbps高质量编码（作为VMAF测量的参考）
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
        "240p": "426:240",
        "360p": "640:360",
        "480p": "854:480",
        "720p": "1280:720",
        "1080p": "1920:1080",
    }
    scale = res_map.get(resolution, "854:480")
    maxrate = int(target_bitrate_kbps * 2)
    bufsize = int(target_bitrate_kbps * 4)

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start_time_s),
        "-i", input_path,
        "-t", str(duration_s),
        "-vf", f"scale={scale}:flags=bicubic,fps={fps}",
        "-c:v", "libx264",
        "-preset", preset,
        "-b:v", f"{int(target_bitrate_kbps)}k",
        "-maxrate", f"{maxrate}k",
        "-bufsize", f"{bufsize}k",
        "-x264opts", "keyint=120:min-keyint=120:scenecut=0:no-scenecut",
        "-an",
        "-movflags", "+faststart",
        output_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"    ffmpeg error: {result.stderr[-200:]}")
            return False, 0
        file_size = os.path.getsize(output_path)
        return True, file_size
    except Exception as e:
        print(f"    ffmpeg exception: {e}")
        return False, 0


def measure_vmaf(reference_path, distorted_path, resolution="480p"):
    """用ffmpeg libvmaf测量VMAF（参考和失真都是同分辨率，速度快）"""
    res_map = {
        "240p": "426:240",
        "360p": "640:360",
        "480p": "854:480",
        "720p": "1280:720",
        "1080p": "1920:1080",
    }
    scale = res_map.get(resolution, "854:480")

    cmd = [
        "ffmpeg", "-y",
        "-i", distorted_path,
        "-i", reference_path,
        "-filter_complex",
        f"[0:v]scale={scale}:flags=bicubic[dist];[1:v]scale={scale}:flags=bicubic[ref];[dist][ref]libvmaf=log_path=/dev/null:log_fmt=json",
        "-f", "null", "-"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        for line in result.stderr.split('\n'):
            if "VMAF score:" in line:
                return float(line.split("VMAF score:")[1].strip())
        for line in result.stdout.split('\n'):
            if "VMAF score:" in line:
                return float(line.split("VMAF score:")[1].strip())
        return None
    except subprocess.TimeoutExpired:
        print(f"    VMAF timeout (180s)")
        return None
    except Exception as e:
        print(f"    VMAF error: {e}")
        return None


def main():
    # 配置
    input_video = r"E:\dev\Code\Python\bones\data\bbb\bbb_2160p_60fps.mp4"
    features_path = r"E:\dev\Code\Python\bones\data\bbb\content_features_bbb.json"
    output_dir = Path(r"E:\dev\Code\Python\bones\data\vbr_experiment_v3")
    output_dir.mkdir(parents=True, exist_ok=True)

    resolution = "480p"
    num_segments = 20
    segment_duration = 4.0
    cbr_bitrate = 1200  # kbps
    vbr_high_bitrate = 1500  # kbps (+25%, 仅高运动段)
    vbr_low_bitrate = 1200  # kbps (不降低，保持CBR码率)
    ref_bitrate = 10000  # kbps (高质量参考)
    high_motion_ratio = 0.2  # 前20%为高运动段

    # 加载内容特征
    with open(features_path) as f:
        features = json.load(f)
    motion = np.array([f["motion_intensity"] for f in features["features"][:num_segments]])

    # 划分高/低运动段
    sorted_idx = np.argsort(-motion)
    num_high = int(num_segments * high_motion_ratio)
    high_motion_segments = set(sorted_idx[:num_high].tolist())
    low_motion_segments = set(sorted_idx[num_high:].tolist())

    print("=" * 70)
    print("小规模VBR编码实验")
    print("=" * 70)
    print(f"分辨率: {resolution}")
    print(f"段数: {num_segments}")
    print(f"CBR码率: {cbr_bitrate}kbps")
    print(f"VBR: 高运动段{vbr_high_bitrate}kbps ({len(high_motion_segments)}段), "
          f"低运动段{vbr_low_bitrate}kbps ({len(low_motion_segments)}段)")
    print(f"VBR平均码率: {(vbr_high_bitrate * len(high_motion_segments) + vbr_low_bitrate * len(low_motion_segments)) / num_segments:.0f}kbps")
    print()

    # 步骤1：生成高质量参考视频（前20段合并）
    print("[步骤1] 生成高质量参考视频...")
    ref_dir = output_dir / "reference"
    ref_dir.mkdir(exist_ok=True)
    ref_segments = []
    for seg_idx in range(num_segments):
        start_time = seg_idx * segment_duration
        ref_path = str(ref_dir / f"ref_{seg_idx:04d}.mp4")
        success, size = encode_segment(input_video, ref_path, start_time, segment_duration,
                                        ref_bitrate, resolution, preset="medium")
        if success:
            ref_segments.append(ref_path)
            print(f"  段{seg_idx}: {size/1024:.1f}KB")
        else:
            print(f"  段{seg_idx}: 失败")

    # 合并参考视频
    ref_merged = str(output_dir / "reference_merged.mp4")
    concat_file = str(output_dir / "ref_concat.txt")
    with open(concat_file, "w") as f:
        for seg in ref_segments:
            f.write(f"file '{seg}'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
                    "-c", "copy", ref_merged], capture_output=True, timeout=60)
    print(f"  参考视频合并完成: {ref_merged}")
    print()

    # 步骤2：CBR编码
    print("[步骤2] CBR编码 (1200kbps)...")
    cbr_dir = output_dir / "cbr"
    cbr_dir.mkdir(exist_ok=True)
    cbr_results = []
    for seg_idx in range(num_segments):
        start_time = seg_idx * segment_duration
        cbr_path = str(cbr_dir / f"cbr_{seg_idx:04d}.mp4")
        success, size = encode_segment(input_video, cbr_path, start_time, segment_duration,
                                        cbr_bitrate, resolution)
        actual_br = size * 8 / segment_duration / 1000 if success else 0
        cbr_results.append({"path": cbr_path, "size": size, "actual_br": actual_br, "success": success})
        print(f"  段{seg_idx}: {size/1024:.1f}KB, 实际码率={actual_br:.0f}kbps")

    # 合并CBR视频
    cbr_merged = str(output_dir / "cbr_merged.mp4")
    concat_file = str(output_dir / "cbr_concat.txt")
    with open(concat_file, "w") as f:
        for r in cbr_results:
            if r["success"]:
                f.write(f"file '{r['path']}'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
                    "-c", "copy", cbr_merged], capture_output=True, timeout=60)
    print()

    # 步骤3：VBR编码
    print("[步骤3] VBR编码 (高运动1500kbps, 低运动900kbps)...")
    vbr_dir = output_dir / "vbr"
    vbr_dir.mkdir(exist_ok=True)
    vbr_results = []
    for seg_idx in range(num_segments):
        start_time = seg_idx * segment_duration
        if seg_idx in high_motion_segments:
            target_br = vbr_high_bitrate
            label = "HIGH"
        else:
            target_br = vbr_low_bitrate
            label = "LOW "
        vbr_path = str(vbr_dir / f"vbr_{seg_idx:04d}.mp4")
        success, size = encode_segment(input_video, vbr_path, start_time, segment_duration,
                                        target_br, resolution)
        actual_br = size * 8 / segment_duration / 1000 if success else 0
        vbr_results.append({"path": vbr_path, "size": size, "actual_br": actual_br,
                            "target_br": target_br, "motion": motion[seg_idx],
                            "motion_type": label, "success": success})
        print(f"  段{seg_idx:2d} [{label}] motion={motion[seg_idx]:5.2f}, "
              f"目标={target_br}kbps, 实际={actual_br:.0f}kbps, {size/1024:.1f}KB")

    # 合并VBR视频
    vbr_merged = str(output_dir / "vbr_merged.mp4")
    concat_file = str(output_dir / "vbr_concat.txt")
    with open(concat_file, "w") as f:
        for r in vbr_results:
            if r["success"]:
                f.write(f"file '{r['path']}'\n")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", concat_file,
                    "-c", "copy", vbr_merged], capture_output=True, timeout=60)
    print()

    # 步骤4：VMAF测量
    print("[步骤4] VMAF测量...")

    # 测量CBR的VMAF
    print("  测量CBR VMAF...")
    cbr_vmaf = measure_vmaf(ref_merged, cbr_merged, resolution)
    print(f"    CBR VMAF: {cbr_vmaf:.2f}" if cbr_vmaf else "    CBR VMAF: 测量失败")

    # 测量VBR的VMAF
    print("  测量VBR VMAF...")
    vbr_vmaf = measure_vmaf(ref_merged, vbr_merged, resolution)
    print(f"    VBR VMAF: {vbr_vmaf:.2f}" if vbr_vmaf else "    VBR VMAF: 测量失败")

    # 步骤5：逐段VMAF测量（更详细的分析）
    print()
    print("[步骤5] 逐段VMAF测量（前10段）...")
    cbr_segment_vmafs = []
    vbr_segment_vmafs = []
    for seg_idx in range(min(10, num_segments)):
        # 生成单段参考视频
        seg_ref = str(ref_dir / f"ref_{seg_idx:04d}.mp4")

        # CBR单段VMAF
        cbr_seg_path = cbr_results[seg_idx]["path"]
        cbr_seg_vmaf = measure_vmaf(seg_ref, cbr_seg_path, resolution)
        cbr_segment_vmafs.append(cbr_seg_vmaf)

        # VBR单段VMAF
        vbr_seg_path = vbr_results[seg_idx]["path"]
        vbr_seg_vmaf = measure_vmaf(seg_ref, vbr_seg_path, resolution)
        vbr_segment_vmafs.append(vbr_seg_vmaf)

        motion_label = "HIGH" if seg_idx in high_motion_segments else "LOW "
        cbr_str = f"{cbr_seg_vmaf:.2f}" if cbr_seg_vmaf else "N/A"
        vbr_str = f"{vbr_seg_vmaf:.2f}" if vbr_seg_vmaf else "N/A"
        diff = f"{vbr_seg_vmaf - cbr_seg_vmaf:+.2f}" if (cbr_seg_vmaf and vbr_seg_vmaf) else "N/A"
        print(f"  段{seg_idx:2d} [{motion_label}] CBR={cbr_str}, VBR={vbr_str}, diff={diff}")

    # 汇总
    print()
    print("=" * 70)
    print("实验结果汇总")
    print("=" * 70)

    cbr_avg_br = np.mean([r["actual_br"] for r in cbr_results if r["success"]])
    vbr_avg_br = np.mean([r["actual_br"] for r in vbr_results if r["success"]])

    print(f"平均码率: CBR={cbr_avg_br:.0f}kbps, VBR={vbr_avg_br:.0f}kbps")
    if cbr_vmaf and vbr_vmaf:
        print(f"整体VMAF: CBR={cbr_vmaf:.2f}, VBR={vbr_vmaf:.2f}, diff={vbr_vmaf - cbr_vmaf:+.2f}")

    valid_cbr = [v for v in cbr_segment_vmafs if v is not None]
    valid_vbr = [v for v in vbr_segment_vmafs if v is not None]
    if valid_cbr and valid_vbr:
        print(f"前10段平均VMAF: CBR={np.mean(valid_cbr):.2f}, VBR={np.mean(valid_vbr):.2f}, "
              f"diff={np.mean(valid_vbr) - np.mean(valid_cbr):+.2f}")

    # 高运动段 vs 低运动段
    high_cbr = [cbr_segment_vmafs[i] for i in range(10) if i in high_motion_segments and cbr_segment_vmafs[i] is not None]
    high_vbr = [vbr_segment_vmafs[i] for i in range(10) if i in high_motion_segments and vbr_segment_vmafs[i] is not None]
    low_cbr = [cbr_segment_vmafs[i] for i in range(10) if i in low_motion_segments and cbr_segment_vmafs[i] is not None]
    low_vbr = [vbr_segment_vmafs[i] for i in range(10) if i in low_motion_segments and vbr_segment_vmafs[i] is not None]

    if high_cbr and high_vbr:
        print(f"\n高运动段（前10段中）: CBR={np.mean(high_cbr):.2f}, VBR={np.mean(high_vbr):.2f}, "
              f"diff={np.mean(high_vbr) - np.mean(high_cbr):+.2f}")
    if low_cbr and low_vbr:
        print(f"低运动段（前10段中）: CBR={np.mean(low_cbr):.2f}, VBR={np.mean(low_vbr):.2f}, "
              f"diff={np.mean(low_vbr) - np.mean(low_cbr):+.2f}")

    # 保存结果
    result_path = str(output_dir / "experiment_results.json")
    with open(result_path, "w") as f:
        json.dump({
            "config": {
                "resolution": resolution,
                "num_segments": num_segments,
                "cbr_bitrate": cbr_bitrate,
                "vbr_high_bitrate": vbr_high_bitrate,
                "vbr_low_bitrate": vbr_low_bitrate,
                "high_motion_segments": sorted(list(high_motion_segments)),
                "low_motion_segments": sorted(list(low_motion_segments)),
            },
            "overall": {
                "cbr_avg_bitrate": float(cbr_avg_br),
                "vbr_avg_bitrate": float(vbr_avg_br),
                "cbr_vmaf": cbr_vmaf,
                "vbr_vmaf": vbr_vmaf,
            },
            "segment_vmafs": {
                "cbr": cbr_segment_vmafs,
                "vbr": vbr_segment_vmafs,
            },
        }, f, indent=2)
    print(f"\n结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
