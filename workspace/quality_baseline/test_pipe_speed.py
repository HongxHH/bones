# -*- coding: utf-8 -*-
"""120 帧堆叠 raw 管道双 libvmaf 速度测试（复刻 #13 拓扑，新 ffmpeg）。"""
import os, sys, time, subprocess
import numpy as np, cv2

FF = sys.argv[1]
REPO = r"E:\dev\Code\Python\bones"
LR = os.path.join(REPO, r"data\video_mp4box\output_240p.mp4")
REF = os.path.join(REPO, r"data\video_mp4box\output_1080p.mp4")
W, H, FPS = 1920, 1080, 30
TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
os.makedirs(TMP, exist_ok=True)
logs = {"bic": "t_bic.json", "sr": "t_sr.json"}
for p in logs.values():
    fp = os.path.join(TMP, p)
    if os.path.exists(fp):
        os.remove(fp)
fc = ("[0:v]crop=1920:1080:0:0[d1];[0:v]crop=1920:1080:0:1080[d2];"
      "[1:v]split[r1][r2];"
      f"[d1][r1]libvmaf=log_path={logs['bic']}:log_fmt=json[a];"
      f"[d2][r2]libvmaf=log_path={logs['sr']}:log_fmt=json[b]")
cmd = [FF, "-y", "-v", "error",
       "-f", "rawvideo", "-pixel_format", "bgr24",
       "-video_size", f"{W}x{2*H}", "-framerate", str(FPS), "-i", "-",
       "-i", REF, "-filter_complex", fc,
       "-map", "[a]", "-map", "[b]", "-f", "null", "-"]
t0 = time.time()
proc = subprocess.Popen(cmd, cwd=TMP, stdin=subprocess.PIPE,
                        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
cap = cv2.VideoCapture(LR)
n = 0
while n < 120:
    ret, f = cap.read()
    if not ret:
        break
    bic = cv2.resize(f, (W, H), interpolation=cv2.INTER_CUBIC)
    sr = cv2.resize(f, (W, H), interpolation=cv2.INTER_CUBIC)  # 计时用，同 bic
    proc.stdin.write(np.vstack([bic, sr]).tobytes())
    n += 1
cap.release()
proc.stdin.close()
err = proc.stderr.read().decode("utf-8", "ignore")
rc = proc.wait()
dt = time.time() - t0
print(f"pipe stacked 2-arm 120 frames: {dt:.2f}s rc={rc}")
if rc != 0:
    print(err[-2000:])
