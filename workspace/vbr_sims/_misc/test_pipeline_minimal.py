"""
最小化测试：验证逐帧VMAF测量管线
只测1段(120帧) × 1种方法(base)，逐步验证每个环节
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

VIDEO_DIR = r"E:\dev\Code\Python\bones\data\video_mp4box"
MODEL_DIR = r"E:\dev\Code\Python\bones\player\istream_player\modules\enhancer\imdn_weight"
WORK_DIR = r"E:\dev\Code\Python\bones\workspace\perframe_tmp"
DEVICE = "cuda:0"
FRAMES_PER_SEG = 120
FPS = 30

os.makedirs(WORK_DIR, exist_ok=True)

print("=" * 60)
print("Step 1: 读取视频帧")
print("=" * 60)
t0 = time.time()
cap360 = cv2.VideoCapture(os.path.join(VIDEO_DIR, "output_360p.mp4"))
cap1080 = cv2.VideoCapture(os.path.join(VIDEO_DIR, "output_1080p.mp4"))

frames_360 = []
frames_1080 = []
for i in range(FRAMES_PER_SEG):
    r1, f1 = cap360.read()
    r2, f2 = cap1080.read()
    if r1 and r2:
        frames_360.append(f1)
        frames_1080.append(f2)
cap360.release()
cap1080.release()
print(f"  读取 {len(frames_360)} 帧, 耗时 {time.time()-t0:.2f}s")
print(f"  360p shape: {frames_360[0].shape}")
print(f"  1080p shape: {frames_1080[0].shape}")

print("\n" + "=" * 60)
print("Step 2: base方法 - 双三次上采样")
print("=" * 60)
t0 = time.time()
enhanced = [cv2.resize(f, (1920, 1080), interpolation=cv2.INTER_CUBIC) for f in frames_360]
print(f"  上采样 {len(enhanced)} 帧, 耗时 {time.time()-t0:.2f}s")
print(f"  enhanced shape: {enhanced[0].shape}")

print("\n" + "=" * 60)
print("Step 3: 写入临时视频文件")
print("=" * 60)
t0 = time.time()
tmp_dir = tempfile.mkdtemp(dir=WORK_DIR)
enh_path = os.path.join(tmp_dir, "enhanced.mp4")
ref_path = os.path.join(tmp_dir, "reference.mp4")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out_enh = cv2.VideoWriter(enh_path, fourcc, FPS, (1920, 1080))
out_ref = cv2.VideoWriter(ref_path, fourcc, FPS, (1920, 1080))
for f in enhanced:
    out_enh.write(f)
for f in frames_1080:
    out_ref.write(f)
out_enh.release()
out_ref.release()
print(f"  写入视频, 耗时 {time.time()-t0:.2f}s")
print(f"  enhanced file: {os.path.getsize(enh_path)/1024/1024:.2f} MB")
print(f"  reference file: {os.path.getsize(ref_path)/1024/1024:.2f} MB")

# 验证视频可读取
cap_check = cv2.VideoCapture(enh_path)
check_count = int(cap_check.get(cv2.CAP_PROP_FRAME_COUNT))
cap_check.release()
print(f"  验证enhanced视频帧数: {check_count}")

print("\n" + "=" * 60)
print("Step 4: ffmpeg逐帧VMAF测量（带超时保护）")
print("=" * 60)
log_path = os.path.join(WORK_DIR, "test_vmaf_log.json")
if os.path.exists(log_path):
    os.remove(log_path)

# 使用正斜杠路径
log_path_ff = log_path.replace("\\", "/")
enh_path_ff = enh_path.replace("\\", "/")
ref_path_ff = ref_path.replace("\\", "/")

filter_str = (
    f"[0:v]setpts=PTS-STARTPTS[distorted];"
    f"[1:v]setpts=PTS-STARTPTS[reference];"
    f"[distorted][reference]libvmaf=log_path={log_path_ff}:log_fmt=json:n_threads=4"
)
cmd = ["ffmpeg", "-y", "-i", enh_path_ff, "-i", ref_path_ff,
       "-lavfi", filter_str, "-f", "null", "-"]
print(f"  命令: {' '.join(cmd[:5])}...")

t0 = time.time()
try:
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    elapsed = time.time() - t0
    print(f"  ffmpeg完成, 耗时 {elapsed:.2f}s, returncode={result.returncode}")

    # 打印VMAF分数
    for line in result.stderr.split("\n"):
        if "VMAF score" in line:
            print(f"  {line.strip()}")

    # 检查log文件
    if os.path.exists(log_path):
        with open(log_path) as f:
            log = json.load(f)
        frames_log = log.get("frames", [])
        vmafs = [fr["metrics"]["vmaf"] for fr in frames_log]
        print(f"  逐帧VMAF: {len(vmafs)} 帧")
        print(f"  mean={np.mean(vmafs):.2f}, std={np.std(vmafs):.2f}, "
              f"min={np.min(vmafs):.2f}, max={np.max(vmafs):.2f}")
        print(f"  前10帧: {[round(v,1) for v in vmafs[:10]]}")
    else:
        print("  ERROR: log文件未创建!")
        print("  stderr最后500字符:")
        print(result.stderr[-500:])

except subprocess.TimeoutExpired:
    print(f"  ffmpeg超时! ({time.time()-t0:.2f}s)")
    print("  这就是之前挂起的原因 - ffmpeg在Windows上timeout可能失效")

# 清理
import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)
if os.path.exists(log_path):
    os.remove(log_path)

print("\n" + "=" * 60)
print("Step 5: 测试SR模型推理速度")
print("=" * 60)
sys.path.insert(0, r"E:\dev\Code\Python\bones\player")
from istream_player.modules.enhancer.imdn_model import IMDN, IMDN_RTC

scale = 3  # 360p -> 1080p
model = IMDN_RTC(upscale=scale)
model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"IMDN_medium_bbb_x{scale}.pth")))
model = model.to(DEVICE).eval()

# 测试单帧推理
img = cv2.cvtColor(frames_360[0], cv2.COLOR_BGR2RGB)
tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
tensor = tensor.unsqueeze(0).to(DEVICE)

# warmup
with torch.no_grad():
    _ = model(tensor)
torch.cuda.synchronize()

t0 = time.time()
N = 10
with torch.no_grad():
    for _ in range(N):
        out = model(tensor)
torch.cuda.synchronize()
elapsed = time.time() - t0
print(f"  IMDN_medium x{scale}: {N}帧耗时 {elapsed:.2f}s, 单帧 {elapsed/N*1000:.1f}ms")
print(f"  120帧预计耗时: {elapsed/N*120:.1f}s")
print(f"  输出shape: {out.shape}")

print("\n" + "=" * 60)
print("测试完成")
print("=" * 60)
