"""
全分辨率 Unsharp Mask VMAF增益 + 延迟测量
分辨率: 360p, 480p, 720p, 1080p
输出: 各分辨率的VMAF增益、延迟/FPS
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

# Unsharp Mask参数（经过验证的最佳配置）
USM_AMOUNT = 1.5
USM_RADIUS = 3

def run_cmd(cmd):
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout + result.stderr

def extract_clip(input_path, output_path, start, duration):
    cmd = f'ffmpeg -y -ss {start} -i "{input_path}" -t {duration} -c:v libx264 -preset fast -crf 0 -an "{output_path}"'
    run_cmd(cmd)
    return os.path.exists(output_path)

def unsharp_mask(frame, amount=USM_AMOUNT, radius=USM_RADIUS):
    blurred = cv2.GaussianBlur(frame, (0, 0), radius)
    return cv2.addWeighted(frame, 1 + amount, blurred, -amount, 0)

def process_video(input_path, output_path):
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
        out.write(unsharp_mask(frame))
        frame_count += 1
    cap.release()
    out.release()
    elapsed = time.time() - t0
    return frame_count, elapsed, width, height

def measure_latency(input_path, num_warmup=10, num_iter=100):
    """精确测量单帧延迟"""
    cap = cv2.VideoCapture(input_path)
    frames = []
    for _ in range(num_warmup + num_iter):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    if len(frames) < num_warmup + 1:
        return None, None, None

    # warmup
    for f in frames[:num_warmup]:
        unsharp_mask(f)

    # measure
    latencies = []
    for f in frames[num_warmup:num_warmup+num_iter]:
        t0 = time.perf_counter()
        unsharp_mask(f)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    h, w = frames[0].shape[:2]
    return np.mean(latencies), np.percentile(latencies, 99), (w, h)

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
    print("全分辨率 Unsharp Mask VMAF增益 + 延迟测量")
    print(f"参数: amount={USM_AMOUNT}, radius={USM_RADIUS}")
    print("=" * 70)

    resolutions = ['360p', '480p', '720p', '1080p']
    reference_1080p = os.path.join(BASE_DIR, 'output_1080p.mp4')

    # 提取所有测试片段
    print("\n[1/4] Extracting test clips...")
    clips = {}
    for res in resolutions:
        input_file = os.path.join(BASE_DIR, f'output_{res}.mp4')
        clips[res] = []
        for seg in SEGMENTS:
            out_file = os.path.join(WORK_DIR, f'{res}_seg{seg["idx"]}.mp4')
            if not os.path.exists(out_file):
                extract_clip(input_file, out_file, seg['start'], DURATION)
            clips[res].append(out_file)

    # 提取参考片段
    ref_clips = []
    for seg in SEGMENTS:
        ref_file = os.path.join(WORK_DIR, f'1080p_ref_seg{seg["idx"]}.mp4')
        if not os.path.exists(ref_file):
            extract_clip(reference_1080p, ref_file, seg['start'], DURATION)
        ref_clips.append(ref_file)

    # 处理所有片段
    print("\n[2/4] Processing clips with Unsharp Mask...")
    processed_clips = {}
    for res in resolutions:
        processed_clips[res] = []
        for i, clip_path in enumerate(clips[res]):
            seg_idx = SEGMENTS[i]['idx']
            out_path = os.path.join(WORK_DIR, f'{res}_seg{seg_idx}_usm.mp4')
            if not os.path.exists(out_path):
                fc, elapsed, w, h = process_video(clip_path, out_path)
                print(f"  {res} seg{seg_idx}: {fc} frames, {elapsed:.2f}s, {fc/elapsed:.1f} FPS, {w}x{h}")
            processed_clips[res].append(out_path)

    # 测量延迟
    print("\n[3/4] Measuring per-frame latency...")
    latency_results = {}
    for res in resolutions:
        # 用第一段做延迟测量
        mean_ms, p99_ms, dims = measure_latency(clips[res][0])
        if mean_ms:
            fps = 1000.0 / mean_ms
            latency_results[res] = {
                'mean_ms': round(mean_ms, 3),
                'p99_ms': round(p99_ms, 3),
                'fps': round(fps, 1),
                'resolution': f"{dims[0]}x{dims[1]}"
            }
            print(f"  {res} ({dims[0]}x{dims[1]}): mean={mean_ms:.3f}ms, p99={p99_ms:.3f}ms, {fps:.1f} FPS")

    # 测量VMAF
    print("\n[4/4] Measuring VMAF scores...")
    vmaf_results = []
    for res in resolutions:
        print(f"\n  --- {res} ---")
        rescale = (res != '1080p')
        for i, seg in enumerate(SEGMENTS):
            seg_idx = seg['idx']
            vmaf_orig = measure_vmaf(ref_clips[i], clips[res][i], rescale=rescale)
            vmaf_usm = measure_vmaf(ref_clips[i], processed_clips[res][i], rescale=rescale)
            gain = vmaf_usm - vmaf_orig if (vmaf_orig and vmaf_usm) else None
            gain_str = f"{gain:+.2f}" if gain is not None else "N/A"
            print(f"  Seg {seg_idx:3d}: orig={vmaf_orig:.2f}, usm={vmaf_usm:.2f}, gain={gain_str}")
            vmaf_results.append({
                'resolution': res,
                'segment_idx': seg_idx,
                'vmaf_original': vmaf_orig,
                'vmaf_usm': vmaf_usm,
                'vmaf_gain': gain,
            })

    # 汇总
    print("\n" + "=" * 70)
    print("汇总统计")
    print("=" * 70)
    summary = {}
    for res in resolutions:
        res_results = [r for r in vmaf_results if r['resolution'] == res and r['vmaf_gain'] is not None]
        if res_results:
            gains = [r['vmaf_gain'] for r in res_results]
            origs = [r['vmaf_original'] for r in res_results]
            usms = [r['vmaf_usm'] for r in res_results]
            lat = latency_results.get(res, {})
            summary[res] = {
                'vmaf_original_mean': round(np.mean(origs), 2),
                'vmaf_usm_mean': round(np.mean(usms), 2),
                'vmaf_gain_mean': round(np.mean(gains), 2),
                'vmaf_gain_min': round(np.min(gains), 2),
                'vmaf_gain_max': round(np.max(gains), 2),
                'positive_segments': sum(1 for g in gains if g > 0),
                'total_segments': len(gains),
                'latency_mean_ms': lat.get('mean_ms'),
                'latency_p99_ms': lat.get('p99_ms'),
                'fps': lat.get('fps'),
                'resolution': lat.get('resolution'),
            }
            print(f"\n  {res} ({lat.get('resolution', '?')}):")
            print(f"    VMAF: orig={np.mean(origs):.2f} -> usm={np.mean(usms):.2f} (gain={np.mean(gains):+.2f})")
            print(f"    增益范围: [{np.min(gains):+.2f}, {np.max(gains):+.2f}], 正增益: {sum(1 for g in gains if g > 0)}/{len(gains)}")
            print(f"    延迟: mean={lat.get('mean_ms', '?')}ms, p99={lat.get('p99_ms', '?')}ms, {lat.get('fps', '?')} FPS")

    # 保存
    result_file = os.path.join(WORK_DIR, 'usm_full_resolution_results.json')
    with open(result_file, 'w') as f:
        json.dump({'summary': summary, 'details': vmaf_results, 'params': {'amount': USM_AMOUNT, 'radius': USM_RADIUS}}, f, indent=2)
    print(f"\n结果保存到: {result_file}")

if __name__ == '__main__':
    main()
