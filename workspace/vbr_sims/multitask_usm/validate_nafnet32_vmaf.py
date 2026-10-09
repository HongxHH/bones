"""
NAFNet-GoPro-width32 预训练模型 VMAF增益验证
使用正确的架构(width=32, enc_blk_nums=[1,1,1,28])和预训练权重
"""
import subprocess
import os
import sys
import numpy as np
import torch
import cv2
import json
import time

# 添加NAFNet路径
NAFNET_PATH = r'E:\dev\Code\Python\bones\enhance\NAFNet'
if NAFNET_PATH not in sys.path:
    sys.path.insert(0, NAFNET_PATH)

BASE_DIR = r'E:\dev\Code\Python\bones\data\video_mp4box'
WORK_DIR = r'E:\dev\Code\Python\bones\workspace\deblur_vmaf_test'
WEIGHT_PATH = r'E:\dev\Code\Python\bones\enhance\NAFNet\experiments\pretrained_models\NAFNet-GoPro-width32.pth'
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

def load_nafnet():
    """加载NAFNet-GoPro-width32预训练模型"""
    from basicsr.models.archs.NAFNet_arch import NAFNet
    model = NAFNet(
        img_channel=3, width=32, middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1]
    )
    # 加载权重
    ckpt = torch.load(WEIGHT_PATH, map_location='cpu')
    if 'params' in ckpt:
        state_dict = ckpt['params']
    else:
        state_dict = ckpt
    model.load_state_dict(state_dict, strict=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device).half().eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"NAFNet-GoPro-width32 loaded on {device}, params: {num_params:,}")
    return model, device

def process_video_nafnet(input_path, output_path, model, device):
    """用NAFNet处理视频的每一帧"""
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    t0 = time.time()
    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # BGR → RGB → tensor [1,3,H,W]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            tensor = tensor.to(device).half()
            # NAFNet推理
            output = model(tensor).float()
            output = torch.clamp(output, 0, 1)
            # tensor → numpy → BGR
            out_frame = (output.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            out_bgr = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
            out.write(out_bgr)
            frame_count += 1

    cap.release()
    out.release()
    elapsed = time.time() - t0
    print(f"  Processed {frame_count} frames in {elapsed:.1f}s ({frame_count/elapsed:.1f} FPS)")
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
    print("NAFNet-GoPro-width32 预训练模型 VMAF增益验证")
    print("=" * 60)

    # 1. 加载NAFNet
    print("\n[1/4] Loading NAFNet-GoPro-width32 pretrained model...")
    model, device = load_nafnet()

    # 2. 提取测试片段
    print("\n[2/4] Extracting test clips...")
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

    # 3. 运行NAFNet去模糊
    print("\n[3/4] Running NAFNet deblur on clips...")
    deblurred_clips = {}
    for res in resolutions:
        deblurred_clips[res] = []
        for i, clip_path in enumerate(clips[res]):
            seg_idx = SEGMENTS[i]['idx']
            out_path = os.path.join(WORK_DIR, f'{res}_seg{seg_idx}_nafnet32.mp4')
            if not os.path.exists(out_path):
                print(f"  NAFNet {res} segment {seg_idx}...")
                process_video_nafnet(clip_path, out_path, model, device)
            deblurred_clips[res].append(out_path)

    # 4. 测量VMAF
    print("\n[4/4] Measuring VMAF scores...")
    reference_1080p = os.path.join(BASE_DIR, 'output_1080p.mp4')
    results = []

    for res in resolutions:
        print(f"\n  --- {res} ---")
        for i, seg in enumerate(SEGMENTS):
            seg_idx = seg['idx']
            orig_clip = clips[res][i]
            deblur_clip = deblurred_clips[res][i]
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
                'method': 'NAFNet-width32',
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
    for res in resolutions:
        res_results = [r for r in results if r['resolution'] == res and r['vmaf_gain'] is not None]
        if res_results:
            gains = [r['vmaf_gain'] for r in res_results]
            origs = [r['vmaf_original'] for r in res_results]
            deblurs = [r['vmaf_deblurred'] for r in res_results]
            print(f"\n  {res}:")
            print(f"    原始VMAF:  mean={np.mean(origs):.2f}, min={np.min(origs):.2f}, max={np.max(origs):.2f}")
            print(f"    去模糊VMAF: mean={np.mean(deblurs):.2f}, min={np.min(deblurs):.2f}, max={np.max(deblurs):.2f}")
            print(f"    增益:       mean={np.mean(gains):+.2f}, min={np.min(gains):+.2f}, max={np.max(gains):+.2f}")
            print(f"    正增益段数: {sum(1 for g in gains if g > 0)}/{len(gains)}")

    result_file = os.path.join(WORK_DIR, 'vmaf_gain_nafnet32_results.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n结果保存到: {result_file}")

if __name__ == '__main__':
    main()
