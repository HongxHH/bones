"""验证ffmpeg libvmaf逐帧VMAF输出功能"""
import cv2, subprocess, os, json, tempfile, numpy as np

# 从360p和1080p视频各取前120帧
cap360 = cv2.VideoCapture(r"E:\dev\Code\Python\bones\data\video_mp4box\output_360p.mp4")
cap1080 = cv2.VideoCapture(r"E:\dev\Code\Python\bones\data\video_mp4box\output_1080p.mp4")

tmp = tempfile.mkdtemp()
w360, h360 = 640, 360
w1080, h1080 = 1920, 1080
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out360 = cv2.VideoWriter(os.path.join(tmp, "360p.mp4"), fourcc, 30, (w360, h360))
out1080 = cv2.VideoWriter(os.path.join(tmp, "1080p.mp4"), fourcc, 30, (w1080, h1080))

for i in range(120):
    ret1, f1 = cap360.read()
    ret2, f2 = cap1080.read()
    if ret1 and ret2:
        out360.write(f1)
        out1080.write(f2)
cap360.release()
cap1080.release()
out360.release()
out1080.release()
print("Videos written, frames=120")

# 用ffmpeg libvmaf逐帧测量，输出JSON日志
log_path = "E:/dev/Code/Python/bones/workspace/vmaf_log_test.json"
if os.path.exists(log_path):
    os.remove(log_path)
filter_str = (
    f"[0:v]scale=1920:1080[scaled];"
    f"[1:v]setpts=PTS-STARTPTS[reference];"
    f"[scaled]setpts=PTS-STARTPTS[distorted];"
    f"[distorted][reference]libvmaf=log_path={log_path}:log_fmt=json:n_threads=4"
)
cmd = ["ffmpeg", "-i", os.path.join(tmp, "360p.mp4"), "-i", os.path.join(tmp, "1080p.mp4"),
       "-lavfi", filter_str, "-f", "null", "-"]
result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
for line in result.stderr.split("\n"):
    if "VMAF score" in line:
        print("Average VMAF:", line.strip())

# 解析逐帧VMAF
if os.path.exists(log_path):
    with open(log_path) as f:
        log = json.load(f)
    frames = log.get("frames", [])
    print(f"Per-frame VMAF: {len(frames)} frames")
    if frames:
        vmafs = [fr["metrics"]["vmaf"] for fr in frames]
        print(f"  mean={np.mean(vmafs):.2f}, std={np.std(vmafs):.2f}, "
              f"min={np.min(vmafs):.2f}, max={np.max(vmafs):.2f}")
        print(f"  first 10: {[round(v, 1) for v in vmafs[:10]]}")
else:
    print("Log file not found!")
    print("stderr tail:", result.stderr[-500:])
