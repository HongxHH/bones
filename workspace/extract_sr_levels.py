"""从imdn_bbb_3060ti.json提取各级别SR的平均VMAF，用于新质量表"""
import json
import numpy as np

with open(r"E:\dev\Code\Python\bones\data\bbb\imdn_bbb_3060ti.json") as f:
    d = json.load(f)

base_q = np.array(d["base_quality_vmaf"])      # (4, 159)
enh_q = np.array(d["enhanced_quality_vmaf"])    # (3, 4, 159)

bitrate_names = ["240p", "360p", "480p", "720p"]
print("=== Base VMAF (avg across 159 segments) ===")
for i in range(4):
    valid = base_q[i][base_q[i] > 0]
    print(f"  {bitrate_names[i]}: {np.mean(valid):.2f} (n={len(valid)})")

print("\n=== Enhanced VMAF by level (avg across 159 segments) ===")
for br in range(3):  # 240p, 360p, 480p
    print(f"\n  [{bitrate_names[br]}]:")
    for level in range(4):
        valid = enh_q[br, level][enh_q[br, level] > 0]
        if len(valid) > 0:
            gain = np.mean(valid) - np.mean(base_q[br][base_q[br] > 0])
            print(f"    level {level+1}: VMAF={np.mean(valid):.2f}, gain={gain:+.2f} (n={len(valid)})")

print("\n=== Quality table for new combo design ===")
print("Columns: 0=base, 1=SR_light(L1), 2=SR_medium(L2), 3=SR_heavy(L3), 4=USM-only")
for br in range(4):
    base_val = np.mean(base_q[br][base_q[br] > 0]) if br < 4 else 100.0
    if br < 3:
        l1 = np.mean(enh_q[br, 0][enh_q[br, 0] > 0])
        l2 = np.mean(enh_q[br, 1][enh_q[br, 1] > 0])
        l3 = np.mean(enh_q[br, 2][enh_q[br, 2] > 0])
        print(f"  {bitrate_names[br]}: base={base_val:.2f}, L1={l1:.2f}, L2={l2:.2f}, L3={l3:.2f}")
    else:
        print(f"  {bitrate_names[br]}: base={base_val:.2f}, L1=inf, L2=inf, L3=inf (scale=1)")
