"""测试ffmpeg log_path在Python subprocess中的正确用法"""
import os
import json
import subprocess
import tempfile
import numpy as np
import cv2

WORK_DIR = r"E:\dev\Code\Python\bones\workspace\perframe_tmp"
os.makedirs(WORK_DIR, exist_ok=True)

# 准备测试视频
cap360 = cv2.VideoCapture(r"E:\dev\Code\Python\bones\data\video_mp4box\output_360p.mp4")
cap1080 = cv2.VideoCapture(r"E:\dev\Code\Python\bones\data\video_mp4box\output_1080p.mp4")
frames_360, frames_1080 = [], []
for i in range(120):
    r1, f1 = cap360.read()
    r2, f2 = cap1080.read()
    if r1 and r2:
        frames_360.append(cv2.resize(f1, (1920, 1080), interpolation=cv2.INTER_CUBIC))
        frames_1080.append(f2)
cap360.release()
cap1080.release()

tmp_dir = tempfile.mkdtemp(dir=WORK_DIR)
enh_path = os.path.join(tmp_dir, "enhanced.mp4")
ref_path = os.path.join(tmp_dir, "reference.mp4")
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out1 = cv2.VideoWriter(enh_path, fourcc, 30, (1920, 1080))
out2 = cv2.VideoWriter(ref_path, fourcc, 30, (1920, 1080))
for f in frames_360: out1.write(f)
for f in frames_1080: out2.write(f)
out1.release(); out2.release()

# 测试1: 相对路径log_path（在cwd中创建）
print("Test 1: 相对路径 log_path")
log_rel = "test_log_rel.json"
if os.path.exists(log_rel): os.remove(log_rel)
filter1 = f"[0:v]setpts=PTS-STARTPTS[d];[1:v]setpts=PTS-STARTPTS[r];[d][r]libvmaf=log_path={log_rel}:log_fmt=json:n_threads=4"
cmd1 = ["ffmpeg", "-y", "-i", enh_path, "-i", ref_path, "-lavfi", filter1, "-f", "null", "-"]
r1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=60, cwd=WORK_DIR)
print(f"  returncode={r1.returncode}")
print(f"  log exists in cwd={os.path.exists(os.path.join(WORK_DIR, log_rel))}")
if os.path.exists(os.path.join(WORK_DIR, log_rel)):
    d = json.load(open(os.path.join(WORK_DIR, log_rel)))
    print(f"  frames={len(d.get('frames',[]))}")
    os.remove(os.path.join(WORK_DIR, log_rel))

# 测试2: 绝对路径但用正斜杠
print("\nTest 2: 绝对路径(正斜杠) log_path")
log_abs = os.path.join(WORK_DIR, "test_log_abs.json").replace("\\", "/")
if os.path.exists(log_abs): os.remove(log_abs)
filter2 = f"[0:v]setpts=PTS-STARTPTS[d];[1:v]setpts=PTS-STARTPTS[r];[d][r]libvmaf=log_path={log_abs}:log_fmt=json:n_threads=4"
cmd2 = ["ffmpeg", "-y", "-i", enh_path, "-i", ref_path, "-lavfi", filter2, "-f", "null", "-"]
r2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=60)
print(f"  returncode={r2.returncode}")
print(f"  log exists={os.path.exists(log_abs)}")
if os.path.exists(log_abs):
    d = json.load(open(log_abs))
    print(f"  frames={len(d.get('frames',[]))}")
    os.remove(log_abs)

# 测试3: log_path用引号包裹
print("\nTest 3: log_path带引号")
log_quoted = f"'{log_abs}'"
filter3 = f"[0:v]setpts=PTS-STARTPTS[d];[1:v]setpts=PTS-STARTPTS[r];[d][r]libvmaf=log_path={log_quoted}:log_fmt=json:n_threads=4"
cmd3 = ["ffmpeg", "-y", "-i", enh_path, "-i", ref_path, "-lavfi", filter3, "-f", "null", "-"]
r3 = subprocess.run(cmd3, capture_output=True, text=True, timeout=60)
print(f"  returncode={r3.returncode}")
print(f"  log exists={os.path.exists(log_abs)}")
if os.path.exists(log_abs):
    d = json.load(open(log_abs))
    print(f"  frames={len(d.get('frames',[]))}")
    os.remove(log_abs)

# 清理
import shutil
shutil.rmtree(tmp_dir, ignore_errors=True)
print("\nDone")
