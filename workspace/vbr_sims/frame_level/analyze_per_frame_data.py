"""分析逐帧VMAF测量结果，验证内容感知假设"""
import json
import numpy as np
from collections import defaultdict

DATA_FILE = r"E:\dev\Code\Python\bones\data\bbb\per_frame_quality.json"

with open(DATA_FILE) as f:
    data = json.load(f)

metadata = data["metadata"]
content_features = data["content_features"]
results = data["results"]

print("=" * 70)
print("数据概览")
print("=" * 70)
print(f"分辨率: {metadata['resolutions']}")
print(f"方法: {metadata['methods']}")
print(f"段数: {metadata['num_segments']}")
print(f"每段帧数: {metadata['frames_per_segment']}")
print(f"测量结果数: {len(results)}")
print(f"内容特征段数: {len(content_features)}")

# 检查完整性
expected = len(metadata["resolutions"]) * len(metadata["methods"]) * metadata["num_segments"]
print(f"预期测量数: {expected}")
print(f"缺失: {expected - len(results)} (最后一段帧数不足)")

print("\n" + "=" * 70)
print("1. 逐帧VMAF波动分析（段内std）")
print("=" * 70)

# 按分辨率×方法统计段内VMAF波动
seg_std = defaultdict(list)
seg_mean = defaultdict(list)
for key, vmafs in results.items():
    parts = key.split("_")
    res = int(parts[0])
    method = "_".join(parts[1:-1])
    seg_idx = int(parts[-1])
    vmafs = np.array(vmafs)
    seg_std[(res, method)].append(np.std(vmafs))
    seg_mean[(res, method)].append(np.mean(vmafs))

print(f"{'分辨率':<8} {'方法':<12} {'段数':<6} {'平均VMAF':<10} {'平均段内std':<12} {'最大段内std':<12}")
print("-" * 70)
for res in metadata["resolutions"]:
    for method in metadata["methods"]:
        stds = seg_std.get((res, method), [])
        means = seg_mean.get((res, method), [])
        if stds:
            print(f"{res}p{'':<5} {method:<12} {len(stds):<6} {np.mean(means):<10.2f} "
                  f"{np.mean(stds):<12.2f} {np.max(stds):<12.2f}")

print("\n" + "=" * 70)
print("2. 内容特征与VMAF增益的相关性")
print("=" * 70)

# 对每个分辨率，计算base vs sr_medium的逐帧增益，与运动/纹理的相关性
for res in [360]:
    print(f"\n--- {res}p: base vs sr_medium ---")
    gains_all = []
    motions_all = []
    textures_all = []
    scene_changes = []

    for seg_idx in range(158):  # 跳过最后一段
        base_key = f"{res}_base_{seg_idx}"
        sr_key = f"{res}_sr_medium_{seg_idx}"
        if base_key in results and sr_key in results and str(seg_idx) in content_features:
            base_vmafs = np.array(results[base_key])
            sr_vmafs = np.array(results[sr_key])
            gains = sr_vmafs - base_vmafs
            cf = content_features[str(seg_idx)]
            motion = np.array(cf["motion"])
            texture = np.array(cf["texture"])
            sc = np.array(cf["scene_change"])

            gains_all.extend(gains)
            motions_all.extend(motion)
            textures_all.extend(texture)
            scene_changes.extend(sc)

    gains_all = np.array(gains_all)
    motions_all = np.array(motions_all)
    textures_all = np.array(textures_all)
    scene_changes = np.array(scene_changes)

    print(f"  总帧数: {len(gains_all)}")
    print(f"  平均VMAF增益: {np.mean(gains_all):.2f}")
    print(f"  增益std: {np.std(gains_all):.2f}")
    print(f"  增益范围: [{np.min(gains_all):.2f}, {np.max(gains_all):.2f}]")
    print(f"  负增益帧比例: {np.mean(gains_all < 0)*100:.1f}%")

    # 相关性
    if len(gains_all) > 0:
        corr_motion = np.corrcoef(gains_all, motions_all)[0, 1]
        corr_texture = np.corrcoef(gains_all, textures_all)[0, 1]
        print(f"  增益与运动强度相关系数: {corr_motion:.3f}")
        print(f"  增益与纹理复杂度相关系数: {corr_texture:.3f}")

    # 按运动强度分桶
    print(f"\n  按运动强度分桶（前30%高运动 vs 后30%低运动）:")
    motion_threshold_high = np.percentile(motions_all, 70)
    motion_threshold_low = np.percentile(motions_all, 30)
    high_motion_gains = gains_all[motions_all >= motion_threshold_high]
    low_motion_gains = gains_all[motions_all <= motion_threshold_low]
    print(f"    高运动帧({len(high_motion_gains)}): 平均增益={np.mean(high_motion_gains):.2f}")
    print(f"    低运动帧({len(low_motion_gains)}): 平均增益={np.mean(low_motion_gains):.2f}")
    print(f"    差异: {np.mean(high_motion_gains) - np.mean(low_motion_gains):.2f}")

    # 场景切换帧
    sc_gains = gains_all[scene_changes]
    non_sc_gains = gains_all[~scene_changes]
    print(f"\n  场景切换帧({len(sc_gains)}): 平均增益={np.mean(sc_gains):.2f}")
    print(f"  非场景切换帧({len(non_sc_gains)}): 平均增益={np.mean(non_sc_gains):.2f}")

print("\n" + "=" * 70)
print("3. 不同增强级别的边际收益")
print("=" * 70)

for res in [360]:
    print(f"\n--- {res}p ---")
    for method in ["sr_low", "sr_medium", "sr_high", "usm_only"]:
        gains = []
        for seg_idx in range(158):
            base_key = f"{res}_base_{seg_idx}"
            sr_key = f"{res}_{method}_{seg_idx}"
            if base_key in results and sr_key in results:
                g = np.array(results[sr_key]) - np.array(results[base_key])
                gains.extend(g)
        gains = np.array(gains)
        print(f"  {method:<12}: 平均增益={np.mean(gains):.2f}, std={np.std(gains):.2f}, "
              f"负增益比例={np.mean(gains<0)*100:.1f}%")

print("\n" + "=" * 70)
print("4. 段平均 vs 帧级：QoE估算差异")
print("=" * 70)

# 模拟：如果用段平均VMAF做决策，vs 用帧级VMAF做决策
# 简单对比：段平均的std vs 帧级的std
for res in [360]:
    seg_means = []
    frame_vmafs = []
    for seg_idx in range(158):
        key = f"{res}_sr_medium_{seg_idx}"
        if key in results:
            vmafs = np.array(results[key])
            seg_means.append(np.mean(vmafs))
            frame_vmafs.extend(vmafs)
    seg_means = np.array(seg_means)
    frame_vmafs = np.array(frame_vmafs)
    print(f"\n{res}p sr_medium:")
    print(f"  段平均VMAF: mean={np.mean(seg_means):.2f}, std={np.std(seg_means):.2f}")
    print(f"  帧级VMAF:   mean={np.mean(frame_vmafs):.2f}, std={np.std(frame_vmafs):.2f}")
    print(f"  帧级波动是段级的 {np.std(frame_vmafs)/np.std(seg_means):.1f} 倍")

print("\n分析完成")
