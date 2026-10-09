"""
去模糊VMAF增益小样本验证（使用传统方法，无需预训练权重）
方法1: Unsharp Mask（锐化/去模糊）
方法2: Wiener滤波（去模糊）
选5段(20秒)480p + 5段1080p，测量VMAF增益
"""
import subprocess
import os
import sys
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

def unsharp_mask(frame, amount=1.5, radius=3):
    """Unsharp Mask：经典锐化/去模糊方法"""
    blurred = cv2.GaussianBlur(frame, (0, 0), radius)
    sharpened = cv2.addWeighted(frame, 1 + amount, blurred, -amount, 0)
    return sharpened

def wiener_deblur(frame, kernel_size=3, noise_var=0.01):
    """Wiener滤波去模糊"""
    # 转换为浮点
    img_float = frame.astype(np.float64) / 255.0
    result = np.zeros_like(img_float)
    for c in range(3):
        channel = img_float[:, :, c]
        # 简单的Wiener滤波近似
        psf = np.ones((kernel_size, kernel_size)) / (kernel_size ** 2)
        # FFT
        img_fft = np.fft.fft2(channel)
        psf_fft = np.fft.fft2(psf, s=channel.shape)
        # Wiener公式
        wiener = np.conj(psf_fft) / (np.abs(psf_fft) ** 2 + noise_var)
        result[:, :, c] = np.abs(np.fft.ifft2(img_fft * wiener))
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result

def process_video(input_path, output_path, method='unsharp', **kwargs):
    """处理视频的每一帧"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if method == 'unsharp':
            processed = unsharp_mask(frame, **kwargs)
        elif method == 'wiener':
            processed = wiener_deblur(frame, **kwargs)
        else:
            processed = frame
        out.write(processed)
        frame_count += 1

    cap.release()
    out.release()
    elapsed = time.time() - t0
    print(f"  {method}: {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")
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
    print("=" * 60)
    print("去模糊VMAF增益小样本验证（传统方法）")
    print("=" * 60)

    # 1. 提取测试片段
    print("\n[1/3] Extracting test clips...")
    resolutions = ['480p', '1080p']
    clips = {}
    for res in resolutions:
        input_file = os.path.join(BASE_DIR, f'output_{res}.mp4')
        clips[res] = []
        for seg in SEGMENTS:
            out_file = os.path.join(WORK_DIR, f'{res}_seg{seg["idx"]}.mp4')
            if not os.path.exists(out_file):
                print(f"  Extracting {res} segment {seg['idx']}...")
                extract_clip(input_file, out_file, seg['start'], DURATION)
            clips[res].append(out_file)

    # 2. 运行去模糊处理
    print("\n[2/3] Running deblur on clips...")
    methods = ['unsharp', 'wiener']
    processed_clips = {m: {} for m in methods}
    for method in methods:
        for res in resolutions:
            processed_clips[method][res] = []
            for i, clip_path in enumerate(clips[res]):
                seg_idx = SEGMENTS[i]['idx']
                out_path = os.path.join(WORK_DIR, f'{res}_seg{seg_idx}_{method}.mp4')
                if not os.path.exists(out_path):
                    print(f"  {method} {res} segment {seg_idx}...")
                    if method == 'unsharp':
                        process_video(clip_path, out_path, method='unsharp', amount=1.0, radius=3)
                    elif method == 'wiener':
                        process_video(clip_path, out_path, method='wiener', kernel_size=3, noise_var=0.01)
                processed_clips[method][res].append(out_path)

    # 3. 测量VMAF
    print("\n[3/3] Measuring VMAF scores...")
    reference_1080p = os.path.join(BASE_DIR, 'output_1080p.mp4')
    results = []

    for method in methods:
        print(f"\n  --- {method} ---")
        for res in resolutions:
            print(f"\n  {res}:")
            for i, seg in enumerate(SEGMENTS):
                seg_idx = seg['idx']
                orig_clip = clips[res][i]
                deblur_clip = processed_clips[method][res][i]
                ref_clip = os.path.join(WORK_DIR, f'1080p_ref_seg{seg_idx}.mp4')
                if not os.path.exists(ref_clip):
                    extract_clip(reference_1080p, ref_clip, seg['start'], DURATION)

                rescale = (res != '1080p')
                vmaf_orig = measure_vmaf(ref_clip, orig_clip, rescale=rescale)
                vmaf_deblur = measure_vmaf(ref_clip, deblur_clip, rescale=rescale)

                gain = vmaf_deblur - vmaf_orig if (vmaf_orig and vmaf_deblur) else None
                gain_str = f"{gain:+.2f}" if gain is not None else "N/A"
                print(f"  Seg {seg_idx:3d}: orig={vmaf_orig:.2f}, deblur={vmaf_deblur:.2f}, gain={gain_str}")

                results.append({
                    'method': method,
                    'resolution': res,
                    'segment_idx': seg_idx,
                    'vmaf_original': vmaf_orig,
                    'vmaf_deblurred': vmaf_deblur,
                    'vmaf_gain': gain,
                })

    # 汇总统计
    print("\n" + "=" * 60)
    print("汇总统计")
    print("=" * 60)
    for method in methods:
        print(f"\n  === {method} ===")
        for res in resolutions:
            res_results = [r for r in results if r['method'] == method and r['resolution'] == res and r['vmaf_gain'] is not None]
            if res_results:
                gains = [r['vmaf_gain'] for r in res_results]
                origs = [r['vmaf_original'] for r in res_results]
                deblurs = [r['vmaf_deblurred'] for r in res_results]
                print(f"\n  {res}:")
                print(f"    原始VMAF:  mean={np.mean(origs):.2f}")
                print(f"    去模糊VMAF: mean={np.mean(deblurs):.2f}")
                print(f"    增益:       mean={np.mean(gains):+.2f}, min={np.min(gains):+.2f}, max={np.max(gains):+.2f}")
                print(f"    正增益段数: {sum(1 for g in gains if g > 0)}/{len(gains)}")

    result_file = os.path.join(WORK_DIR, 'vmaf_gain_traditional_results.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果保存到: {result_file}")

if __name__ == '__main__':
    main()
