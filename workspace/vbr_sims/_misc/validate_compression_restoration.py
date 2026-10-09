"""
压缩伪影恢复方法 VMAF增益验证
对比多种传统压缩伪影恢复方法对bbb视频的效果
方法: 1) Unsharp Mask 2) Bilateral Filter 3) Bilateral+Unsharp组合 4) NL Means去噪+锐化
"""
import subprocess
import os
import numpy as np
import cv2
import json
import time

BASE_DIR = r'E:\dev\Code\Python\bones\data\video_mp4box'
WORK_DIR = r'E:\dev\Code\Python\bones\workspace\deblur_vmaf_test'
os.makedirs(WORK_DIR, exist_ok=True)

SEGMENTS = [
    {'idx': 10, 'start': 40},
    {'idx': 40, 'start': 160},
    {'idx': 70, 'start': 280},
    {'idx': 100, 'start': 400},
    {'idx': 130, 'start': 520},
]
DURATION = 4

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

def extract_clip(input_path, output_path, start, duration):
    cmd = f'ffmpeg -y -ss {start} -i "{input_path}" -t {duration} -c:v libx264 -preset fast -crf 0 -an "{output_path}"'
    run_cmd(cmd)
    return os.path.exists(output_path)

# ===== 各种压缩伪影恢复方法 =====

def method_unsharp(frame, amount=1.5, radius=3):
    """Unsharp Mask：锐化，恢复压缩丢失的高频细节"""
    blurred = cv2.GaussianBlur(frame, (0, 0), radius)
    return cv2.addWeighted(frame, 1 + amount, blurred, -amount, 0)

def method_bilateral(frame, d=9, sigma_color=75, sigma_space=75):
    """Bilateral Filter：边缘保持去噪，减少压缩噪声/块效应"""
    return cv2.bilateralFilter(frame, d, sigma_color, sigma_space)

def method_bilateral_unsharp(frame):
    """Bilateral去噪 + Unsharp锐化：先去噪再恢复细节"""
    denoised = cv2.bilateralFilter(frame, 9, 50, 50)
    return method_unsharp(denoised, amount=1.2, radius=2)

def method_nlmeans_unsharp(frame):
    """Non-Local Means去噪 + Unsharp锐化"""
    denoised = cv2.fastNlMeansDenoisingColored(frame, None, 3, 3, 7, 21)
    return method_unsharp(denoised, amount=1.2, radius=2)

def method_deblock_unsharp(frame):
    """简单去块效应 + Unsharp锐化：对块边界做平滑"""
    # 用小核高斯模糊近似去块效应（减少块边界不连续）
    deblocked = cv2.GaussianBlur(frame, (3, 3), 0.5)
    return method_unsharp(deblocked, amount=1.5, radius=3)

METHODS = {
    'unsharp': method_unsharp,
    'bilateral': method_bilateral,
    'bilateral_unsharp': method_bilateral_unsharp,
    'nlmeans_unsharp': method_nlmeans_unsharp,
    'deblock_unsharp': method_deblock_unsharp,
}

def process_video(input_path, output_path, method_name):
    """处理视频的每一帧"""
    method_func = METHODS[method_name]
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        processed = method_func(frame)
        out.write(processed)
        frame_count += 1

    cap.release()
    out.release()
    elapsed = time.time() - t0
    print(f"  {method_name}: {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")
    return frame_count > 0

def measure_vmaf(reference_path, distorted_path, rescale=True):
    if rescale:
        cmd = (f'ffmpeg -i "{reference_path}" -i "{distorted_path}" '
               f'-lavfi "[1:v]scale=1920:1080[scaled];'
               f'[0:v]setpts=PTS-STARTPTS[reference];'
               f'[scaled]setpts=PTS-STARTPTS[distorted];'
               f'[distorted][reference]libvmaf=n_threads=4" -f null -')
    else:
        cmd = (f'ffmpeg -i "{reference_path}" -i "{distorted_path}" '
               f'-lavfi "[0:v]setpts=PTS-STARTPTS[reference];'
               f'[1:v]setpts=PTS-STARTPTS[distorted];'
               f'[distorted][reference]libvmaf=n_threads=4" -f null -')
    output = run_cmd(cmd)
    import re
    match = re.search(r"VMAF score: ([\d.]+)", output)
    if match:
        return float(match.group(1))
    match = re.search(r'"vmaf": ([\d.]+)', output)
    if match:
        return float(match.group(1))
    return None

def main():
    print("=" * 70)
    print("压缩伪影恢复方法 VMAF增益验证（多方法对比）")
    print("=" * 70)

    # 1. 提取测试片段
    print("\n[1/3] Extracting test clips...")
    resolutions = ['480p']  # 先只测480p（最有价值的场景）
    clips = {}
    for res in resolutions:
        input_file = os.path.join(BASE_DIR, f'output_{res}.mp4')
        clips[res] = []
        for seg in SEGMENTS:
            out_file = os.path.join(WORK_DIR, f'{res}_seg{seg["idx"]}.mp4')
            if not os.path.exists(out_file):
                extract_clip(input_file, out_file, seg['start'], DURATION)
            clips[res].append(out_file)

    # 2. 运行各种方法
    print("\n[2/3] Running restoration methods...")
    processed_clips = {m: {} for m in METHODS}
    for method_name in METHODS:
        for res in resolutions:
            processed_clips[method_name][res] = []
            for i, clip_path in enumerate(clips[res]):
                seg_idx = SEGMENTS[i]['idx']
                out_path = os.path.join(WORK_DIR, f'{res}_seg{seg_idx}_{method_name}.mp4')
                if not os.path.exists(out_path):
                    print(f"  {method_name} {res} segment {seg_idx}...")
                    process_video(clip_path, out_path, method_name)
                processed_clips[method_name][res].append(out_path)

    # 3. 测量VMAF
    print("\n[3/3] Measuring VMAF scores...")
    reference_1080p = os.path.join(BASE_DIR, 'output_1080p.mp4')
    results = []

    for res in resolutions:
        print(f"\n  --- {res} ---")
        for i, seg in enumerate(SEGMENTS):
            seg_idx = seg['idx']
            orig_clip = clips[res][i]
            ref_clip = os.path.join(WORK_DIR, f'1080p_ref_seg{seg_idx}.mp4')
            if not os.path.exists(ref_clip):
                extract_clip(reference_1080p, ref_clip, seg['start'], DURATION)

            rescale = (res != '1080p')
            vmaf_orig = measure_vmaf(ref_clip, orig_clip, rescale=rescale)

            method_results = {}
            for method_name in METHODS:
                deblur_clip = processed_clips[method_name][res][i]
                vmaf_method = measure_vmaf(ref_clip, deblur_clip, rescale=rescale)
                gain = vmaf_method - vmaf_orig if (vmaf_orig and vmaf_method) else None
                method_results[method_name] = {'vmaf': vmaf_method, 'gain': gain}

            # 打印该段所有方法的结果
            print(f"  Seg {seg_idx:3d} (orig={vmaf_orig:.2f}):")
            for method_name in METHODS:
                r = method_results[method_name]
                gain_str = f"{r['gain']:+.2f}" if r['gain'] is not None else "N/A"
                print(f"    {method_name:20s}: vmaf={r['vmaf']:.2f}, gain={gain_str}")

            for method_name in METHODS:
                r = method_results[method_name]
                results.append({
                    'method': method_name,
                    'resolution': res,
                    'segment_idx': seg_idx,
                    'vmaf_original': vmaf_orig,
                    'vmaf_restored': r['vmaf'],
                    'vmaf_gain': r['gain'],
                })

    # 汇总统计
    print("\n" + "=" * 70)
    print("汇总统计（480p，5段平均）")
    print("=" * 70)
    print(f"{'方法':25s} {'平均VMAF':>10s} {'平均增益':>10s} {'最小增益':>10s} {'最大增益':>10s} {'正增益段':>8s}")
    print("-" * 70)
    for method_name in METHODS:
        method_results = [r for r in results if r['method'] == method_name and r['vmaf_gain'] is not None]
        if method_results:
            gains = [r['vmaf_gain'] for r in method_results]
            vmafs = [r['vmaf_restored'] for r in method_results]
            print(f"{method_name:25s} {np.mean(vmafs):10.2f} {np.mean(gains):+10.2f} {np.min(gains):+10.2f} {np.max(gains):+10.2f} {sum(1 for g in gains if g > 0):>5d}/5")

    result_file = os.path.join(WORK_DIR, 'vmaf_gain_compression_restoration_results.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果保存到: {result_file}")

if __name__ == '__main__':
    main()
