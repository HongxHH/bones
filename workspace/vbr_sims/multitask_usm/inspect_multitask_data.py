"""
检查multitask数据文件结构，并更新Unsharp Mask的测量数据
"""
import json
import numpy as np

DATA_FILE = r'E:\dev\Code\Python\bones\data\bbb\multitask_bbb_rtx5080.json'

with open(DATA_FILE, 'r') as f:
    data = json.load(f)

print("=== Top-level keys ===")
for k in data.keys():
    v = data[k]
    if isinstance(v, list):
        print(f"  {k}: list, len={len(v)}")
        if len(v) > 0 and isinstance(v[0], list):
            print(f"    [0] len={len(v[0])}")
            if len(v[0]) > 0 and isinstance(v[0][0], list):
                print(f"    [0][0] len={len(v[0][0])}")
    elif isinstance(v, dict):
        print(f"  {k}: dict, keys={list(v.keys())}")
    else:
        print(f"  {k}: {type(v).__name__} = {v}")

print("\n=== enhancement_fps ===")
fps = data['enhancement_fps']
print(f"  Shape: {len(fps)} x {len(fps[0])}")
for i, row in enumerate(fps):
    print(f"  res[{i}]: {row}")

print("\n=== _metadata ===")
print(json.dumps(data['_metadata'], indent=2))

# 检查enhanced_quality_vmaf的结构
print("\n=== enhanced_quality_vmaf ===")
eq = data['enhanced_quality_vmaf']
print(f"  Shape: {len(eq)} resolutions")
for i, res_data in enumerate(eq):
    print(f"  res[{i}]: {len(res_data)} combos")
    for j, combo_data in enumerate(res_data):
        print(f"    combo[{j}]: {len(combo_data)} segments, first={combo_data[0]:.2f}, last={combo_data[-1]:.2f}")

# 检查base_quality_vmaf
print("\n=== base_quality_vmaf ===")
bq = data['base_quality_vmaf']
print(f"  Shape: {len(bq)} resolutions x {len(bq[0])} segments")
