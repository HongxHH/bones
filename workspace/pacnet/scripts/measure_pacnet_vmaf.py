"""
PAC-Net 三出口 VMAF 精确测量

测量 240/360/480p 三档下，PAC-Net e1/e2/e3 以及 bicubic baseline 的 VMAF，
生成播放器用的 5x5 quality_table (pacnet_quality.json)。

采样：16段 (range(0,159,10))，每段1帧，与阶段0评测一致。
方法：cv2 抽帧 -> PyTorch 增强 -> 存临时视频 -> ffmpeg libvmaf 计算。
"""
import os
import sys
import json
import subprocess
import tempfile
import numpy as np
import torch
import cv2

sys.path.insert(0, r"E:\dev\Code\Python\bones\player")
sys.path.insert(0, r"E:\dev\Code\Python\bones\workspace\pacnet")

from pacnet_model import PACNet

DEVICE = "cuda:0"
VIDEO_DIR = r"E:\dev\Code\Python\bones\data\video_mp4box"
REF_VIDEO = os.path.join(VIDEO_DIR, "output_1080p.mp4")
COMPRESSED_VIDEOS = {
    "240p": os.path.join(VIDEO_DIR, "output_240p.mp4"),
    "360p": os.path.join(VIDEO_DIR, "output_360p.mp4"),
    "480p": os.path.join(VIDEO_DIR, "output_480p.mp4"),
}
WEIGHT_PATH = r"E:\dev\Code\Python\bones\workspace\pacnet\weights\stage2_mix_nocond_x4_best.pth"
OUTPUT_JSON = r"E:\dev\Code\Python\bones\workspace\pacnet\pacnet_vmaf_measurements.json"
QUALITY_TABLE_PATH = r"E:\dev\Code\Python\bones\player\pacnet_quality.json"

FPS = 30
SEG_DURATION = 4
FRAMES_PER_SEG = FPS * SEG_DURATION
SAMPLE_SEGMENTS = list(range(0, 159, 10))  # 16段
TARGET_SIZE = (1920, 1080)  # (W, H)


def load_pacnet():
    model = PACNet(nf=32, upscale=4, exits=(2, 4, 6), cond=False)
    state = torch.load(WEIGHT_PATH, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(DEVICE).eval()
    print(f"PAC-Net loaded from {WEIGHT_PATH}")
    return model


def extract_frame_at(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame


def pacnet_enhance(model, frame_bgr, exit_idx):
    """BGR frame -> PAC-Net enhance -> BGR frame at 1080p"""
    img = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = model(tensor, exit_idx=exit_idx)
        out = torch.nn.functional.interpolate(
            out, size=(TARGET_SIZE[1], TARGET_SIZE[0]),
            mode="bicubic", align_corners=False
        )
    out = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    out = (out * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


def bicubic_upscale(frame_bgr):
    return cv2.resize(frame_bgr, TARGET_SIZE, interpolation=cv2.INTER_CUBIC)


def compute_vmaf(frames_list, tmp_dir, tag):
    """用 ffmpeg libvmaf 计算一组帧与参考帧的 VMAF"""
    enh_path = os.path.join(tmp_dir, f"enh_{tag}.mp4")
    ref_path = os.path.join(tmp_dir, f"ref_{tag}.mp4")
    h, w = frames_list[0][0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_enh = cv2.VideoWriter(enh_path, fourcc, FPS, (w, h))
    out_ref = cv2.VideoWriter(ref_path, fourcc, FPS, (w, h))
    for enh_f, ref_f in frames_list:
        out_enh.write(enh_f)
        out_ref.write(ref_f)
    out_enh.release()
    out_ref.release()

    cmd = ["ffmpeg", "-i", enh_path, "-i", ref_path,
           "-lavfi", "libvmaf", "-f", "null", "-"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        for line in result.stderr.split('\n'):
            if "VMAF score" in line:
                score = float(line.split("VMAF score:")[1].strip())
                return score
    except Exception as e:
        print(f"  VMAF computation failed: {e}")
    return None


def main():
    model = load_pacnet()
    results = {}  # results[res][method] = list of per-seg VMAF

    for res_name, video_path in COMPRESSED_VIDEOS.items():
        print(f"\n=== Measuring {res_name} ===")
        results[res_name] = {"bicubic": [], "e1": [], "e2": [], "e3": []}

        with tempfile.TemporaryDirectory() as tmp_dir:
            for method in ["bicubic", "e1", "e2", "e3"]:
                print(f"  Method: {method}")
                pairs = []
                for seg_idx in SAMPLE_SEGMENTS:
                    frame_idx = seg_idx * FRAMES_PER_SEG + FRAMES_PER_SEG // 2  # 段中间帧
                    lr_frame = extract_frame_at(video_path, frame_idx)
                    ref_frame = extract_frame_at(REF_VIDEO, frame_idx)
                    if lr_frame is None or ref_frame is None:
                        continue
                    if method == "bicubic":
                        enh_frame = bicubic_upscale(lr_frame)
                    else:
                        exit_idx = {"e1": 0, "e2": 1, "e3": 2}[method]
                        enh_frame = pacnet_enhance(model, lr_frame, exit_idx)
                    pairs.append((enh_frame, ref_frame))

                if pairs:
                    vmaf = compute_vmaf(pairs, tmp_dir, f"{res_name}_{method}")
                    if vmaf is not None:
                        results[res_name][method] = vmaf
                        print(f"    {res_name} {method}: VMAF={vmaf:.2f} ({len(pairs)} frames)")
                    else:
                        print(f"    {res_name} {method}: VMAF computation failed")
                else:
                    print(f"    {res_name} {method}: no frames")

    # 保存原始测量结果
    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw measurements saved to {OUTPUT_JSON}")

    # 生成 5x5 quality_table
    # 行: 0=240p, 1=360p, 2=480p, 3=720p, 4=1080p
    # 列: 0=不增强, 1=e1, 2=e2, 3=e3, 4=无效(-1)
    quality_table = [[-1.0] * 5 for _ in range(5)]

    # 240p (row 0)
    if "240p" in results:
        r = results["240p"]
        quality_table[0][0] = r.get("bicubic", -1.0)
        quality_table[0][1] = r.get("e1", -1.0)
        quality_table[0][2] = r.get("e2", -1.0)
        quality_table[0][3] = r.get("e3", -1.0)

    # 360p (row 1)
    if "360p" in results:
        r = results["360p"]
        quality_table[1][0] = r.get("bicubic", -1.0)
        quality_table[1][1] = r.get("e1", -1.0)
        quality_table[1][2] = r.get("e2", -1.0)
        quality_table[1][3] = r.get("e3", -1.0)

    # 480p (row 2)
    if "480p" in results:
        r = results["480p"]
        quality_table[2][0] = r.get("bicubic", -1.0)
        quality_table[2][1] = r.get("e1", -1.0)
        quality_table[2][2] = r.get("e2", -1.0)
        quality_table[2][3] = r.get("e3", -1.0)

    # 720p (row 3): 不增强 VMAF 从原表借用，增强列 -1（720p 不需要 x4 超分）
    quality_table[3][0] = 90.10  # 从 imdn_bbb_quality.json 借用
    # 1080p (row 4): GT
    quality_table[4][0] = 100.0

    with open(QUALITY_TABLE_PATH, "w") as f:
        json.dump(quality_table, f, indent=2)
    print(f"Quality table saved to {QUALITY_TABLE_PATH}")

    # 打印汇总
    print("\n=== PAC-Net VMAF Quality Table ===")
    print(f"{'':>8} {'no_enh':>8} {'e1':>8} {'e2':>8} {'e3':>8}")
    for i, res in enumerate(["240p", "360p", "480p", "720p", "1080p"]):
        row = quality_table[i]
        print(f"{res:>8} {row[0]:>8.2f} {row[1]:>8.2f} {row[2]:>8.2f} {row[3]:>8.2f}")


if __name__ == "__main__":
    main()
