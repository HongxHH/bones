"""
PAC-Net 跨视频验证：Tears of Steel 完整 quality_table 测量

测量 240/360/480p 三档下 PAC-Net e1/e2/e3 + bicubic 的 VMAF，
以及 720p/1080p 的不增强 VMAF，生成 5x5 quality_table。
24fps，1920x800 目标分辨率。
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
REF_VIDEO = r"E:\dev\Data\tears_of_steel_1080p.mp4"
COMPRESSED_VIDEOS = {
    "240p": os.path.join(VIDEO_DIR, "output_240p.mp4"),
    "360p": os.path.join(VIDEO_DIR, "output_360p.mp4"),
    "480p": os.path.join(VIDEO_DIR, "output_480p.mp4"),
    "720p": os.path.join(VIDEO_DIR, "output_720p.mp4"),
}
WEIGHT_PATH = r"E:\dev\Code\Python\bones\workspace\pacnet\weights\stage2_mix_nocond_x4_best.pth"
OUTPUT_JSON = r"E:\dev\Code\Python\bones\workspace\pacnet\tos_vmaf_full.json"
QUALITY_TABLE_PATH = r"E:\dev\Code\Python\bones\data\tears_of_steel\tos_quality.json"

FPS = 24
TARGET_SIZE = (1920, 800)
N_SAMPLES = 16


def load_pacnet():
    model = PACNet(nf=32, upscale=4, exits=(2, 4, 6), cond=False)
    state = torch.load(WEIGHT_PATH, map_location="cpu")
    model.load_state_dict(state)
    model = model.to(DEVICE).eval()
    print("PAC-Net loaded")
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
    results = {}

    for res_name in ["240p", "360p", "480p", "720p"]:
        video_path = COMPRESSED_VIDEOS[res_name]
        if not os.path.exists(video_path):
            print(f"\nSKIP {res_name} (file not found)")
            continue

        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(7))
        cap.release()
        frame_indices = np.linspace(0, total - 10, N_SAMPLES, dtype=int)

        methods = ["bicubic", "e1", "e2", "e3"] if res_name in ["240p", "360p", "480p"] else ["bicubic"]
        results[res_name] = {}

        print(f"\n=== {res_name} ({total} frames) ===")

        with tempfile.TemporaryDirectory() as tmp_dir:
            for method in methods:
                print(f"  {method}...", end=" ", flush=True)
                pairs = []
                for fi in frame_indices:
                    lr_frame = extract_frame_at(video_path, int(fi))
                    ref_frame = extract_frame_at(REF_VIDEO, int(fi))
                    if lr_frame is None or ref_frame is None:
                        continue
                    if method == "bicubic":
                        enh_frame = bicubic_upscale(lr_frame)
                    else:
                        exit_idx = {"e1": 0, "e2": 1, "e3": 2}[method]
                        enh_frame = pacnet_enhance(model, lr_frame, exit_idx)
                    pairs.append((enh_frame, ref_frame))

                if pairs:
                    vmaf = compute_vmaf(pairs, tmp_dir, f"tos_{res_name}_{method}")
                    results[res_name][method] = vmaf
                    print(f"VMAF={vmaf:.2f} ({len(pairs)} frames)")
                else:
                    print("no frames")

    # 生成 5x5 quality_table
    # 行: 0=240p, 1=360p, 2=480p, 3=720p, 4=1080p
    # 列: 0=不增强, 1=e1, 2=e2, 3=e3, 4=无效
    qt = [[-1.0] * 5 for _ in range(5)]
    for row, res in enumerate(["240p", "360p", "480p"]):
        if res in results:
            r = results[res]
            qt[row][0] = r.get("bicubic", -1.0)
            qt[row][1] = r.get("e1", -1.0)
            qt[row][2] = r.get("e2", -1.0)
            qt[row][3] = r.get("e3", -1.0)
    if "720p" in results:
        qt[3][0] = results["720p"].get("bicubic", -1.0)
    qt[4][0] = 100.0  # 1080p GT

    with open(OUTPUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    with open(QUALITY_TABLE_PATH, "w") as f:
        json.dump(qt, f, indent=2)

    print("\n=== TOS Quality Table ===")
    print(f"{'':>8} {'no_enh':>8} {'e1':>8} {'e2':>8} {'e3':>8}")
    for i, res in enumerate(["240p", "360p", "480p", "720p", "1080p"]):
        print(f"{res:>8} {qt[i][0]:>8.2f} {qt[i][1]:>8.2f} {qt[i][2]:>8.2f} {qt[i][3]:>8.2f}")
    print(f"\nSaved: {QUALITY_TABLE_PATH}")


if __name__ == "__main__":
    main()
