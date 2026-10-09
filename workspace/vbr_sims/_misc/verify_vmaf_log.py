import json
import numpy as np

d = json.load(open(r"E:\dev\Code\Python\bones\workspace\vmaf_log_test.json"))
frames = d.get("frames", [])
print("Frames:", len(frames))
print("Keys:", list(frames[0].keys()))
print("Metrics:", list(frames[0]["metrics"].keys()))
vmafs = [f["metrics"]["vmaf"] for f in frames]
print("VMAF: mean=%.2f, std=%.2f, min=%.2f, max=%.2f" % (np.mean(vmafs), np.std(vmafs), np.min(vmafs), np.max(vmafs)))
print("First 10:", [round(v, 1) for v in vmafs[:10]])
print("Frame 0:", frames[0])
