import json
from collections import Counter

with open(r"E:\dev\Code\Python\bones\data\bbb\content_features_bbb.json") as f:
    d = json.load(f)

print("num_segments:", d["num_segments"])
print("extraction_time_s:", d["extraction_time_s"])
print("avg_extraction_ms_per_seg:", d["avg_extraction_ms_per_seg"])
print("motion_median:", d["motion_median"])
print("texture_median:", d["texture_median"])
print()

print("前5段:")
for f in d["features"][:5]:
    print("  seg%3d: motion=%6.2f, texture=%7.1f, scene=%s, class=%s" % (
        f["segment_index"], f["motion_intensity"], f["texture_complexity"],
        f["scene_change"], f["content_class"]))

print()
print("后5段:")
for f in d["features"][-5:]:
    print("  seg%3d: motion=%6.2f, texture=%7.1f, scene=%s, class=%s" % (
        f["segment_index"], f["motion_intensity"], f["texture_complexity"],
        f["scene_change"], f["content_class"]))

print()
classes = Counter(f["content_class"] for f in d["features"])
print("内容类别分布:")
for k, v in sorted(classes.items()):
    print("  %s: %d (%.1f%%)" % (k, v, v / len(d["features"]) * 100))

scene_count = sum(1 for f in d["features"] if f["scene_change"])
print("场景切换段: %d/%d" % (scene_count, len(d["features"])))

# 运动强度和纹理复杂度的分布
import numpy as np
motions = np.array([f["motion_intensity"] for f in d["features"]])
textures = np.array([f["texture_complexity"] for f in d["features"]])
print()
print("运动强度分布: mean=%.2f, std=%.2f, min=%.2f, max=%.2f" % (
    motions.mean(), motions.std(), motions.min(), motions.max()))
print("纹理复杂度分布: mean=%.1f, std=%.1f, min=%.1f, max=%.1f" % (
    textures.mean(), textures.std(), textures.min(), textures.max()))

# 高运动段和低运动段的纹理差异
high_motion = [f for f in d["features"] if "high_motion" in f["content_class"]]
low_motion = [f for f in d["features"] if "low_motion" in f["content_class"]]
print()
print("高运动段数: %d, 平均纹理: %.1f" % (len(high_motion), np.mean([f["texture_complexity"] for f in high_motion])))
print("低运动段数: %d, 平均纹理: %.1f" % (len(low_motion), np.mean([f["texture_complexity"] for f in low_motion])))
