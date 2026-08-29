"""
第二步：精确测量USM-only逐段VMAF增益

方法：
1. 从内容特征中抽样不同类别的段（高/低纹理 × 高/低运动）
2. 对每段，从压缩视频（360p）和参考视频（1080p）提取帧
3. 应用USM（与enhancer相同的PyTorch实现）+ bicubic upscale
4. 用ffmpeg计算VMAF
5. 记录逐段USM增益，分析与内容特征的关系

由于完整测量159段×4码率太耗时，先抽样20段×2码率（360p/480p）。
"""
import os
import sys
import json
import subprocess
import tempfile
import numpy as np
import torch
import cv2

# 添加player路径
sys.path.insert(0, r"E:\dev\Code\Python\bones\player")

# USM参数（与enhancer_multitask.py一致）
USM_AMOUNT = 1.5
USM_RADIUS = 3
DEVICE = "cuda:0"

# 视频路径
VIDEO_DIR = r"E:\dev\Code\Python\bones\data\video_mp4box"
REF_VIDEO = os.path.join(VIDEO_DIR, "output_1080p.mp4")  # 参考视频
COMPRESSED_VIDEOS = {
    "360p": os.path.join(VIDEO_DIR, "output_360p.mp4"),
    "480p": os.path.join(VIDEO_DIR, "output_480p.mp4"),
}

# 内容特征
with open(r"E:\dev\Code\Python\bones\data\bbb\content_features_bbb.json") as f:
    content_data = json.load(f)
features = content_data["features"]
motion_median = content_data["motion_median"]
texture_median = content_data["texture_median"]

FPS = 30
SEG_DURATION = 4  # 秒
FRAMES_PER_SEG = FPS * SEG_DURATION  # 120帧


def gaussian_kernel_2d(kernel_size, sigma, channels=3):
    """与enhancer相同的高斯核实现"""
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)
    return kernel_2d.to(DEVICE)


def unsharp_mask(tensor, amount=USM_AMOUNT, radius=USM_RADIUS):
    """与enhancer相同的USM实现"""
    kernel_size = radius * 2 + 1
    sigma = radius / 2.0
    channels = tensor.shape[1]
    padding = kernel_size // 2
    kernel = gaussian_kernel_2d(kernel_size, sigma, channels)
    blurred = torch.nn.functional.conv2d(tensor, kernel, padding=padding, groups=channels)
    sharpened = tensor + amount * (tensor - blurred)
    return torch.clamp(sharpened, 0.0, 1.0)


def extract_segment_frames(video_path, seg_idx, num_frames=5):
    """从视频中提取指定段的采样帧"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return []

    # 跳转到段起始帧
    start_frame = seg_idx * FRAMES_PER_SEG
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    # 均匀采样num_frames帧
    sample_indices = np.linspace(0, FRAMES_PER_SEG - 1, num_frames, dtype=int)
    current = 0
    for idx in sample_indices:
        # 跳转到目标帧
        while current < idx:
            ret, _ = cap.read()
            if not ret:
                break
            current += 1
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
            current += 1
        else:
            break

    cap.release()
    return frames


def apply_usm_and_upscale(frames, target_size=(1920, 1080)):
    """对帧应用USM + bicubic upscale，返回处理后的帧"""
    result = []
    for frame in frames:
        # BGR -> RGB, HWC -> CHW, 归一化到[0,1]
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(DEVICE)  # [1, 3, H, W]

        # 应用USM
        with torch.no_grad():
            tensor = unsharp_mask(tensor)
            # bicubic upscale到1080p
            tensor = torch.nn.functional.interpolate(
                tensor, size=(target_size[1], target_size[0]),
                mode='bicubic', align_corners=False
            )

        # 转回numpy HWC, [0,255], RGB -> BGR
        out = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = (out * 255).clip(0, 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        result.append(out)

    return result


def compute_vmaf(enhanced_frames, ref_frames, tmp_dir):
    """用ffmpeg计算两组帧之间的VMAF"""
    # 保存帧为临时视频
    enh_path = os.path.join(tmp_dir, "enhanced.mp4")
    ref_path = os.path.join(tmp_dir, "reference.mp4")

    h, w = enhanced_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_enh = cv2.VideoWriter(enh_path, fourcc, FPS, (w, h))
    out_ref = cv2.VideoWriter(ref_path, fourcc, FPS, (w, h))
    for f in enhanced_frames:
        out_enh.write(f)
    for f in ref_frames:
        out_ref.write(f)
    out_enh.release()
    out_ref.release()

    # 用ffmpeg计算VMAF
    cmd = [
        "ffmpeg", "-i", enh_path, "-i", ref_path,
        "-lavfi", "libvmaf",
        "-f", "null", "-"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        # 解析VMAF分数
        for line in result.stderr.split('\n'):
            if "VMAF score" in line:
                score = float(line.split("VMAF score:")[1].strip())
                return score
    except Exception as e:
        print(f"  VMAF computation error: {e}")

    return None


def main():
    print("=" * 70)
    print("第二步：精确测量USM-only逐段VMAF增益")
    print("=" * 70)

    # 抽样：从四个内容类别各抽5段
    categories = {
        "high_texture_high_motion": [],
        "high_texture_low_motion": [],
        "low_texture_high_motion": [],
        "low_texture_low_motion": [],
    }

    for i, feat in enumerate(features):
        ht = feat["texture_complexity"] > texture_median
        hm = feat["motion_intensity"] > motion_median
        if ht and hm:
            categories["high_texture_high_motion"].append(i)
        elif ht and not hm:
            categories["high_texture_low_motion"].append(i)
        elif not ht and hm:
            categories["low_texture_high_motion"].append(i)
        else:
            categories["low_texture_low_motion"].append(i)

    # 每类抽5段（均匀分布）
    sample_segments = {}
    for cat, seg_list in categories.items():
        if len(seg_list) >= 5:
            indices = np.linspace(0, len(seg_list) - 1, 5, dtype=int)
            sample_segments[cat] = [seg_list[i] for i in indices]
        else:
            sample_segments[cat] = seg_list

    total_samples = sum(len(v) for v in sample_segments.values())
    print(f"抽样: {total_samples}段, 每类5段")
    for cat, segs in sample_segments.items():
        print(f"  {cat}: {segs}")

    # 创建临时目录
    tmp_dir = tempfile.mkdtemp(prefix="usm_vmaf_")
    print(f"临时目录: {tmp_dir}")

    # 测量每个抽样段的USM增益
    results = []
    for cat, seg_indices in sample_segments.items():
        print(f"\n--- {cat} ---")
        for seg_idx in seg_indices:
            feat = features[seg_idx]
            print(f"  段{seg_idx}: motion={feat['motion_intensity']:.1f}, "
                  f"texture={feat['texture_complexity']:.0f}, scene_change={feat['scene_change']}")

            seg_result = {
                "seg_idx": seg_idx,
                "category": cat,
                "motion": feat["motion_intensity"],
                "texture": feat["texture_complexity"],
                "scene_change": feat["scene_change"],
                "usm_gains": {},
            }

            for br_name, br_path in COMPRESSED_VIDEOS.items():
                # 提取压缩帧和参考帧
                comp_frames = extract_segment_frames(br_path, seg_idx, num_frames=5)
                ref_frames = extract_segment_frames(REF_VIDEO, seg_idx, num_frames=5)

                if len(comp_frames) < 3 or len(ref_frames) < 3:
                    print(f"    {br_name}: 帧不足 (comp={len(comp_frames)}, ref={len(ref_frames)})")
                    continue

                # 确保帧数一致
                min_frames = min(len(comp_frames), len(ref_frames))
                comp_frames = comp_frames[:min_frames]
                ref_frames = ref_frames[:min_frames]

                # 计算base VMAF（bicubic upscale，无USM）
                base_frames = []
                for f in comp_frames:
                    img = cv2.resize(f, (1920, 1080), interpolation=cv2.INTER_CUBIC)
                    base_frames.append(img)
                base_vmaf = compute_vmaf(base_frames, ref_frames, tmp_dir)

                # 计算USM VMAF
                usm_frames = apply_usm_and_upscale(comp_frames)
                usm_vmaf = compute_vmaf(usm_frames, ref_frames, tmp_dir)

                if base_vmaf is not None and usm_vmaf is not None:
                    gain = usm_vmaf - base_vmaf
                    seg_result["usm_gains"][br_name] = {
                        "base_vmaf": base_vmaf,
                        "usm_vmaf": usm_vmaf,
                        "gain": gain,
                    }
                    print(f"    {br_name}: base={base_vmaf:.2f}, usm={usm_vmaf:.2f}, gain={gain:+.2f}")
                else:
                    print(f"    {br_name}: VMAF计算失败 (base={base_vmaf}, usm={usm_vmaf})")

            results.append(seg_result)

    # 保存结果
    output_path = r"E:\dev\Code\Python\bones\workspace\usm_vmaf_measurements.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n结果已保存到: {output_path}")

    # 汇总分析
    print("\n" + "=" * 70)
    print("汇总分析")
    print("=" * 70)
    for br in ["360p", "480p"]:
        gains_by_cat = {}
        for r in results:
            if br in r["usm_gains"]:
                cat = r["category"]
                if cat not in gains_by_cat:
                    gains_by_cat[cat] = []
                gains_by_cat[cat].append(r["usm_gains"][br]["gain"])

        print(f"\n[{br}] USM-only增益按内容类别:")
        all_gains = []
        for cat, gains in sorted(gains_by_cat.items()):
            if gains:
                print(f"  {cat} ({len(gains)}段): mean={np.mean(gains):+.2f}, "
                      f"min={np.min(gains):+.2f}, max={np.max(gains):+.2f}")
                all_gains.extend(gains)
        if all_gains:
            print(f"  总体: mean={np.mean(all_gains):+.2f}, std={np.std(all_gains):.2f}, "
                  f"正增益占比={sum(1 for g in all_gains if g > 0)/len(all_gains)*100:.0f}%")

    # 清理临时目录
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"\n临时目录已清理")


if __name__ == "__main__":
    main()
