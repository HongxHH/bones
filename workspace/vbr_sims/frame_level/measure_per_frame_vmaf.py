"""
细粒度逐帧VMAF测量脚本

测量每个分辨率×每个增强方法×每段×每帧的VMAF，同时提取逐帧内容特征。

测量方法：
1. 从压缩视频和参考视频中读取每段的120帧
2. 对压缩帧应用增强（SR各级别/USM），或不增强（base）
3. 用ffmpeg libvmaf逐帧测量VMAF（输出JSON日志）
4. 同时提取逐帧内容特征（运动强度、纹理复杂度、场景切换）

输出：per_frame_quality.json（结构化数据）
"""
import os
import sys
import json
import time
import subprocess
import tempfile
import numpy as np
import cv2
import torch

# 配置
VIDEO_DIR = r"E:\dev\Code\Python\bones\data\video_mp4box"
MODEL_DIR = r"E:\dev\Code\Python\bones\player\istream_player\modules\enhancer\imdn_weight"
OUTPUT_FILE = r"E:\dev\Code\Python\bones\data\bbb\per_frame_quality.json"
WORK_DIR = r"E:\dev\Code\Python\bones\workspace\perframe_tmp"

RESOLUTIONS = [240, 360, 480]  # 测量的输入分辨率
METHODS = ["base", "sr_low", "sr_medium", "sr_high", "usm_only"]
NUM_SEGMENTS = 159
FRAMES_PER_SEG = 120  # 4秒 × 30fps
FPS = 30
DEVICE = "cuda:0"

# USM参数（与enhancer_multitask.py一致）
USM_AMOUNT = 1.5
USM_RADIUS = 3


def load_imdn_models():
    """加载IMDN模型（low/medium/high，scale 2/3/4）"""
    sys.path.insert(0, r"E:\dev\Code\Python\bones\player")
    from istream_player.modules.enhancer.imdn_model import IMDN, IMDN_RTC

    models = {}
    for scale in [2, 3, 4]:
        # low: IMDN_RTC with num_modules=3, nf=6
        m_low = IMDN_RTC(upscale=scale, num_modules=3, nf=6)
        m_low.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"IMDN_low_bbb_x{scale}.pth")))
        m_low = m_low.to(DEVICE).eval()
        models[(scale, "low")] = m_low

        # medium: IMDN_RTC default
        m_med = IMDN_RTC(upscale=scale)
        m_med.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"IMDN_medium_bbb_x{scale}.pth")))
        m_med = m_med.to(DEVICE).eval()
        models[(scale, "medium")] = m_med

        # high: IMDN with nf=32
        m_high = IMDN(upscale=scale, nf=32)
        m_high.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"IMDN_high_bbb_x{scale}.pth")))
        m_high = m_high.to(DEVICE).eval()
        models[(scale, "high")] = m_high

    print(f"Loaded {len(models)} IMDN models")
    return models


def gaussian_kernel_2d(kernel_size, sigma, channels=3):
    """与enhancer相同的高斯核"""
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g.unsqueeze(1) * g.unsqueeze(0)
    kernel_2d = kernel_2d.unsqueeze(0).unsqueeze(0)
    kernel_2d = kernel_2d.repeat(channels, 1, 1, 1)
    return kernel_2d.to(DEVICE)


def apply_usm(tensor, amount=USM_AMOUNT, radius=USM_RADIUS):
    """与enhancer相同的USM实现"""
    kernel_size = radius * 2 + 1
    sigma = radius / 2.0
    channels = tensor.shape[1]
    padding = kernel_size // 2
    kernel = gaussian_kernel_2d(kernel_size, sigma, channels)
    blurred = torch.nn.functional.conv2d(tensor, kernel, padding=padding, groups=channels)
    sharpened = tensor + amount * (tensor - blurred)
    return torch.clamp(sharpened, 0.0, 1.0)


def enhance_frames(frames, method, scale, models):
    """
    对帧列表应用增强

    Args:
        frames: list of BGR numpy frames (H, W, 3)
        method: "base", "sr_low", "sr_medium", "sr_high", "usm_only"
        scale: 2, 3, or 4
        models: loaded IMDN models dict

    Returns:
        list of enhanced BGR numpy frames (1920x1080)
    """
    if method == "base":
        # 双三次上采样，不增强
        return [cv2.resize(f, (1920, 1080), interpolation=cv2.INTER_CUBIC) for f in frames]

    result = []
    for frame in frames:
        # BGR -> RGB, HWC -> CHW, 归一化
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            if method == "usm_only":
                # USM + bicubic upscale
                tensor = apply_usm(tensor)
                tensor = torch.nn.functional.interpolate(
                    tensor, size=(1080, 1920), mode="bicubic", align_corners=False
                )
            else:
                # SR: sr_low/sr_medium/sr_high
                level = method.replace("sr_", "")
                model = models[(scale, level)]
                tensor = model(tensor)
                tensor = torch.nn.functional.interpolate(
                    tensor, size=(1080, 1920), mode="bicubic", align_corners=False
                )

        # 转回numpy BGR
        out = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        out = (out * 255).clip(0, 255).astype(np.uint8)
        out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        result.append(out)

    return result


def extract_content_features(frames):
    """
    提取逐帧内容特征

    Returns:
        motion: list of float (运动强度，帧差)
        texture: list of float (纹理复杂度，Laplacian方差)
        scene_change: list of bool (场景切换标记)
    """
    motion = []
    texture = []
    scene_change = []

    prev_gray = None
    for i, frame in enumerate(frames):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 纹理复杂度：Laplacian方差
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        tex = laplacian.var()
        texture.append(float(tex))

        # 运动强度：帧差MSE
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            mot = float(np.mean(diff))
            motion.append(mot)
            # 场景切换：帧差超过阈值
            scene_change.append(mot > 30.0)
        else:
            motion.append(0.0)
            scene_change.append(False)

        prev_gray = gray

    return motion, texture, scene_change


def measure_per_frame_vmaf(enhanced_frames, reference_frames, log_name):
    """
    用ffmpeg libvmaf测量逐帧VMAF

    注意：ffmpeg libvmaf的log_path只支持相对路径，必须通过cwd指定工作目录。

    Args:
        enhanced_frames: list of enhanced BGR frames
        reference_frames: list of reference BGR frames
        log_name: 相对log文件名（如"vmaf_0.json"）

    Returns:
        list of per-frame VMAF scores
    """
    # 保存为临时视频
    tmp_dir = tempfile.mkdtemp(dir=WORK_DIR)
    enh_path = os.path.join(tmp_dir, "enhanced.mp4")
    ref_path = os.path.join(tmp_dir, "reference.mp4")

    h, w = enhanced_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_enh = cv2.VideoWriter(enh_path, fourcc, FPS, (w, h))
    out_ref = cv2.VideoWriter(ref_path, fourcc, FPS, (w, h))
    for f in enhanced_frames:
        out_enh.write(f)
    for f in reference_frames:
        out_ref.write(f)
    out_enh.release()
    out_ref.release()

    # ffmpeg libvmaf - 使用相对路径log_name，cwd=WORK_DIR
    log_full = os.path.join(WORK_DIR, log_name)
    if os.path.exists(log_full):
        os.remove(log_full)

    filter_str = (
        f"[0:v]setpts=PTS-STARTPTS[distorted];"
        f"[1:v]setpts=PTS-STARTPTS[reference];"
        f"[distorted][reference]libvmaf=log_path={log_name}:log_fmt=json:n_threads=4"
    )
    cmd = ["ffmpeg", "-y", "-i", enh_path, "-i", ref_path,
           "-lavfi", filter_str, "-f", "null", "-"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=WORK_DIR)
    except subprocess.TimeoutExpired:
        print("  ffmpeg timeout!")
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None

    # 解析逐帧VMAF
    if os.path.exists(log_full):
        with open(log_full) as f:
            log = json.load(f)
        frames_log = log.get("frames", [])
        vmafs = [fr["metrics"]["vmaf"] for fr in frames_log]
        # 清理临时文件
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.remove(log_full)
        return vmafs
    else:
        print("  Log file not created!")
        for line in result.stderr.split("\n")[-5:]:
            if line.strip():
                print("   ", line.strip())
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return None


def main():
    print("=" * 70)
    print("细粒度逐帧VMAF测量")
    print("=" * 70)
    print(f"Resolutions: {RESOLUTIONS}")
    print(f"Methods: {METHODS}")
    print(f"Segments: {NUM_SEGMENTS}")
    print(f"Frames per segment: {FRAMES_PER_SEG}")

    # 创建工作目录
    os.makedirs(WORK_DIR, exist_ok=True)

    # 加载已有进度（支持断点续跑）
    progress = {}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            progress = json.load(f)
        print(f"Loaded existing progress: {len(progress.get('results', {}))} entries")

    results = progress.get("results", {})
    content_features = progress.get("content_features", {})

    # 加载模型
    print("\nLoading IMDN models...")
    models = load_imdn_models()

    # 打开视频
    print("\nOpening video files...")
    ref_cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, "output_1080p.mp4"))
    comp_caps = {}
    for res in RESOLUTIONS:
        comp_caps[res] = cv2.VideoCapture(os.path.join(VIDEO_DIR, f"output_{res}p.mp4"))
        print(f"  {res}p: {comp_caps[res].get(cv2.CAP_PROP_FRAME_COUNT)} frames")

    # 主测量循环
    total = len(RESOLUTIONS) * len(METHODS) * NUM_SEGMENTS
    done = 0
    start_time = time.time()

    for seg_idx in range(NUM_SEGMENTS):
        seg_start = seg_idx * FRAMES_PER_SEG

        # 读取参考帧
        ref_cap.set(cv2.CAP_PROP_POS_FRAMES, seg_start)
        ref_frames = []
        for _ in range(FRAMES_PER_SEG):
            ret, f = ref_cap.read()
            if ret:
                ref_frames.append(f)
            else:
                break

        if len(ref_frames) < FRAMES_PER_SEG:
            print(f"Segment {seg_idx}: not enough reference frames ({len(ref_frames)}), skipping")
            continue

        # 提取内容特征（只需要一次，基于参考帧）
        if str(seg_idx) not in content_features:
            motion, texture, scene_change = extract_content_features(ref_frames)
            content_features[str(seg_idx)] = {
                "motion": motion,
                "texture": texture,
                "scene_change": scene_change,
            }

        for res in RESOLUTIONS:
            scale = min(1920 // (res * 16 // 9), 1080 // res)
            # 读取压缩帧
            comp_caps[res].set(cv2.CAP_PROP_POS_FRAMES, seg_start)
            comp_frames = []
            for _ in range(FRAMES_PER_SEG):
                ret, f = comp_caps[res].read()
                if ret:
                    comp_frames.append(f)
                else:
                    break

            if len(comp_frames) < FRAMES_PER_SEG:
                print(f"Segment {seg_idx}, {res}p: not enough compressed frames ({len(comp_frames)}), skipping")
                continue

            for method in METHODS:
                key = f"{res}_{method}_{seg_idx}"
                if key in results:
                    done += 1
                    continue

                # 应用增强
                enhanced = enhance_frames(comp_frames, method, scale, models)

                # 测量逐帧VMAF（log_name为相对路径，ffmpeg只支持相对路径）
                log_name = f"vmaf_{key}.json"
                vmafs = measure_per_frame_vmaf(enhanced, ref_frames, log_name)

                if vmafs is not None and len(vmafs) == FRAMES_PER_SEG:
                    results[key] = vmafs
                    done += 1
                    elapsed = time.time() - start_time
                    rate = done / elapsed if elapsed > 0 else 0
                    remaining = (total - done) / rate if rate > 0 else 0
                    if done % 10 == 0:
                        print(f"  Progress: {done}/{total} ({done/total*100:.1f}%), "
                              f"elapsed={elapsed/60:.1f}min, ETA={remaining/60:.1f}min, "
                              f"last: {key} mean_vmaf={np.mean(vmafs):.2f}")
                else:
                    print(f"  FAILED: {key} (got {len(vmafs) if vmafs else 0} frames)")

                # 每10个结果保存一次
                if done % 10 == 0 and done > 0:
                    save_progress(results, content_features)

    # 最终保存
    save_progress(results, content_features)
    print(f"\nDone! Total: {done}/{total}")
    print(f"Output: {OUTPUT_FILE}")


def save_progress(results, content_features):
    """保存进度"""
    output = {
        "metadata": {
            "resolutions": RESOLUTIONS,
            "methods": METHODS,
            "num_segments": NUM_SEGMENTS,
            "frames_per_segment": FRAMES_PER_SEG,
            "fps": FPS,
        },
        "content_features": content_features,
        "results": results,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f)
    print(f"  Saved progress: {len(results)} entries")


if __name__ == "__main__":
    main()
