"""
PAC-Net 跨视频验证：Tears of Steel VMAF 测量

在新视频（真人电影，1920x800, 24fps）上测量 PAC-Net 三出口 VMAF 增益，
验证跨视频泛化能力。采样16段均匀分布。
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
VIDEO_DIR = r"E:\dev\Code\Python\bones\data\tears_of_steel"
REF_VIDEO = r"E:\dev\Data\tears_of_steel_1080p.mp4"  # 原始高清作为GT
LOWRES_VIDEO = os.path.join(VIDEO_DIR, "output_240p.mp4")
WEIGHT_PATH = r"E:\dev\Code\Python\bones\workspace\pacnet\weights\stage2_mix_nocond_x4_best.pth"
OUTPUT_JSON = r"E:\dev\Code\Python\bones\workspace\pacnet\tos_vmaf_measurements.json"

FPS = 24
TARGET_SIZE = (1920, 800)  # (W, H)

# 先获取总帧数
cap = cv2.VideoCapture(LOWRES_VIDEO)
TOTAL_FRAMES = int(cap.get(7))
cap.release()

# 均匀采样16段
N_SAMPLES = 16
FRAME_INDICES = np.linspace(0, TOTAL_FRAMES - 10, N_SAMPLES, dtype=int)
print(f"Tears of Steel: {TOTAL_FRAMES} frames @ {FPS}fps, sampling {N_SAMPLES} frames")
print(f"Frame indices: {FRAME_INDICES[:5]}...")


def load_pacnet():
    model = PACNet(nf=32, upscale=4, exits=(2, 4, 6), cond=False)
    state = torch.load(WEIGHT_PATH, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(DEVICE).eval()
    print(f"PAC-Net loaded")
    return model


def extract_frame_at(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def pacnet_enhance(model, frame_bgr, exit_idx):
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
                return float(line.split("VMAF score:")[1].strip())
    except Exception as e:
        print(f"  VMAF failed: {e}")
    return None


def main():
    model = load_pacnet()
    results = {"bicubic": [], "e1": [], "e2": [], "e3": []}
    per_frame = {}

    with tempfile.TemporaryDirectory() as tmp_dir:
        for method in ["bicubic", "e1", "e2", "e3"]:
            print(f"\nMethod: {method}")
            pairs = []
            for i, frame_idx in enumerate(FRAME_INDICES):
                lr_frame = extract_frame_at(LOWRES_VIDEO, int(frame_idx))
                ref_frame = extract_frame_at(REF_VIDEO, int(frame_idx))
                if lr_frame is None or ref_frame is None:
                    print(f"  frame {frame_idx}: skip (None)")
                    continue
                if method == "bicubic":
                    enh_frame = bicubic_upscale(lr_frame)
                else:
                    exit_idx = {"e1": 0, "e2": 1, "e3": 2}[method]
                    enh_frame = pacnet_enhance(model, lr_frame, exit_idx)
                pairs.append((enh_frame, ref_frame))

            if pairs:
                vmaf = compute_vmaf(pairs, tmp_dir, f"tos_{method}")
                results[method] = vmaf
                print(f"  {method}: VMAF={vmaf:.2f} ({len(pairs)} frames)")

    # 对比BBB数据
    bbb_data = {
        "bicubic": 39.30,  # 240p no_enh
        "e1": 45.94,
        "e2": 48.91,
        "e3": 54.42,
    }

    print("\n=== Tears of Steel vs BBB 对比（240p VMAF）===")
    print(f"{'Method':>10} {'TOS':>8} {'BBB':>8} {'差值':>8}")
    for method in ["bicubic", "e1", "e2", "e3"]:
        tos = results[method]
        bbb = bbb_data[method]
        print(f"{method:>10} {tos:>8.2f} {bbb:>8.2f} {tos-bbb:>+8.2f}")

    # 计算增益
    tos_bicubic = results["bicubic"]
    print(f"\n=== PAC-Net 增益（vs bicubic baseline）===")
    for method in ["e1", "e2", "e3"]:
        gain = results[method] - tos_bicubic
        bbb_gain = bbb_data[method] - bbb_data["bicubic"]
        print(f"  {method}: TOS增益={gain:+.2f}, BBB增益={bbb_gain:+.2f}, 差异={gain-bbb_gain:+.2f}")

    out = {
        "video": "Tears of Steel (真人电影, 1920x800, 24fps)",
        "total_frames": TOTAL_FRAMES,
        "n_samples": N_SAMPLES,
        "vmaf": results,
        "bbb_comparison": bbb_data,
    }
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved to {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
