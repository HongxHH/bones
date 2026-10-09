"""
阶段2：内容感知增益表分析

利用已有的逐段VMAF数据（imdn_bbb_3060ti.json）与内容特征（content_features_bbb.json）交叉分析，
验证不同内容类别下SR增益是否有显著差异。

如果差异显著（>2分），则内容感知有价值，后续测量USM/SR+USM的逐段增益。
"""
import json
import numpy as np
from collections import defaultdict


def main():
    # 加载逐段VMAF数据
    with open(r"E:\dev\Code\Python\bones\data\bbb\imdn_bbb_3060ti.json") as f:
        vmaf_data = json.load(f)

    base_q = np.array(vmaf_data["base_quality_vmaf"])      # (4, 159)
    enh_q = np.array(vmaf_data["enhanced_quality_vmaf"])    # (3, 4, 159)

    # 加载内容特征
    with open(r"E:\dev\Code\Python\bones\data\bbb\content_features_bbb.json") as f:
        content_data = json.load(f)
    features = content_data["features"]
    motion_median = content_data["motion_median"]
    texture_median = content_data["texture_median"]

    print("=" * 70)
    print("阶段2：内容感知增益分析")
    print("=" * 70)
    print(f"VMAF数据: base={base_q.shape}, enhanced={enh_q.shape}")
    print(f"内容特征: {len(features)}段, motion_median={motion_median:.2f}, texture_median={texture_median:.1f}")
    print()

    # 码率索引：0=240p, 1=360p, 2=480p, 3=720p
    # enhanced只覆盖前3个码率（240p, 360p, 480p）
    bitrate_names = ["240p", "360p", "480p", "720p"]

    # 对每个码率，分析SR增益（enhancement level 1 = SR only）与内容的关系
    print("-" * 70)
    print("分析1：SR增益（level 1）按内容类别分布")
    print("-" * 70)

    for br_idx in range(3):  # 240p, 360p, 480p
        br_name = bitrate_names[br_idx]
        gains = []
        for seg_idx in range(159):
            base = base_q[br_idx, seg_idx]
            sr_vmaf = enh_q[br_idx, 0, seg_idx]  # level 1 = index 0
            if base > 0 and sr_vmaf > 0:
                gains.append({
                    "seg_idx": seg_idx,
                    "base": base,
                    "sr_vmaf": sr_vmaf,
                    "gain": sr_vmaf - base,
                    "motion": features[seg_idx]["motion_intensity"],
                    "texture": features[seg_idx]["texture_complexity"],
                    "scene_change": features[seg_idx]["scene_change"],
                })

        if not gains:
            continue

        # 按运动强度分组
        gains_sorted = sorted(gains, key=lambda x: x["motion"])
        low_motion = gains_sorted[:len(gains_sorted)//2]
        high_motion = gains_sorted[len(gains_sorted)//2:]

        # 按纹理复杂度分组
        gains_sorted_t = sorted(gains, key=lambda x: x["texture"])
        low_texture = gains_sorted_t[:len(gains_sorted_t)//2]
        high_texture = gains_sorted_t[len(gains_sorted_t)//2:]

        # 四象限分类
        def classify(g):
            m = "high_motion" if g["motion"] > motion_median else "low_motion"
            t = "high_texture" if g["texture"] > texture_median else "low_texture"
            return f"{m}_{t}"

        quadrants = defaultdict(list)
        for g in gains:
            quadrants[classify(g)].append(g["gain"])

        print(f"\n[{br_name}] 有效段数: {len(gains)}")
        print(f"  SR增益总体: mean={np.mean([g['gain'] for g in gains]):.2f}, "
              f"std={np.std([g['gain'] for g in gains]):.2f}, "
              f"min={np.min([g['gain'] for g in gains]):.2f}, "
              f"max={np.max([g['gain'] for g in gains]):.2f}")

        print(f"  按运动分组:")
        print(f"    低运动({len(low_motion)}段): mean gain={np.mean([g['gain'] for g in low_motion]):.2f}")
        print(f"    高运动({len(high_motion)}段): mean gain={np.mean([g['gain'] for g in high_motion]):.2f}")
        print(f"    差异: {np.mean([g['gain'] for g in low_motion]) - np.mean([g['gain'] for g in high_motion]):.2f}")

        print(f"  按纹理分组:")
        print(f"    低纹理({len(low_texture)}段): mean gain={np.mean([g['gain'] for g in low_texture]):.2f}")
        print(f"    高纹理({len(high_texture)}段): mean gain={np.mean([g['gain'] for g in high_texture]):.2f}")
        print(f"    差异: {np.mean([g['gain'] for g in high_texture]) - np.mean([g['gain'] for g in low_texture]):.2f}")

        print(f"  四象限分类:")
        for q in sorted(quadrants.keys()):
            qgains = quadrants[q]
            print(f"    {q} ({len(qgains)}段): mean gain={np.mean(qgains):.2f}, std={np.std(qgains):.2f}")

    # 分析2：场景切换对增益的影响
    print()
    print("-" * 70)
    print("分析2：场景切换对SR增益的影响")
    print("-" * 70)

    for br_idx in range(3):
        br_name = bitrate_names[br_idx]
        scene_gains = []
        non_scene_gains = []
        for seg_idx in range(159):
            base = base_q[br_idx, seg_idx]
            sr_vmaf = enh_q[br_idx, 0, seg_idx]
            if base > 0 and sr_vmaf > 0:
                gain = sr_vmaf - base
                if features[seg_idx]["scene_change"]:
                    scene_gains.append(gain)
                else:
                    non_scene_gains.append(gain)

        if scene_gains and non_scene_gains:
            print(f"  [{br_name}] 场景切换段({len(scene_gains)}): mean gain={np.mean(scene_gains):.2f}")
            print(f"  [{br_name}] 非场景切换段({len(non_scene_gains)}): mean gain={np.mean(non_scene_gains):.2f}")
            print(f"  [{br_name}] 差异: {np.mean(non_scene_gains) - np.mean(scene_gains):.2f}")

    # 分析3：运动强度与SR增益的相关性
    print()
    print("-" * 70)
    print("分析3：运动强度/纹理复杂度与SR增益的皮尔逊相关系数")
    print("-" * 70)

    for br_idx in range(3):
        br_name = bitrate_names[br_idx]
        motions = []
        textures = []
        gains = []
        for seg_idx in range(159):
            base = base_q[br_idx, seg_idx]
            sr_vmaf = enh_q[br_idx, 0, seg_idx]
            if base > 0 and sr_vmaf > 0:
                motions.append(features[seg_idx]["motion_intensity"])
                textures.append(features[seg_idx]["texture_complexity"])
                gains.append(sr_vmaf - base)

        if len(gains) > 5:
            corr_motion = np.corrcoef(motions, gains)[0, 1]
            corr_texture = np.corrcoef(textures, gains)[0, 1]
            print(f"  [{br_name}] motion-gain correlation: {corr_motion:.3f}")
            print(f"  [{br_name}] texture-gain correlation: {corr_texture:.3f}")

    # 总结
    print()
    print("=" * 70)
    print("总结")
    print("=" * 70)
    print("如果不同内容类别的SR增益差异>2分，且相关系数|r|>0.3，则内容感知有显著价值。")
    print("如果差异<1分，则内容感知价值有限，需要重新考虑方向。")


if __name__ == "__main__":
    main()
