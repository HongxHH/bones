"""
Step 0 + Step 1: 数据对齐、样本集构建与特征工程

输入: data/bbb/per_frame_quality.json
  - results["{res}_{method}_{seg}"]: 逐帧VMAF list (120帧)
  - content_features["{seg}"]: 逐帧 motion/texture/scene_change
输出:
  - workspace/gain_predictor/dataset_seg.csv   段级数据集(主)
  - workspace/gain_predictor/dataset_frame.csv 帧级数据集(用于段内方差分析)

硬约束: 所有特征只能来自"播放端LR侧可获得"的信息(motion/texture/scene_change/base质量/分辨率档),
       绝不使用任何来自HR参考或SR结果的信息作为特征(SR结果只用于构造标签y)。
"""
import json
import os
import numpy as np
import pandas as pd

ROOT = r"E:\dev\Code\Python\bones"
SRC = os.path.join(ROOT, "data", "bbb", "per_frame_quality.json")
OUT_DIR = os.path.join(ROOT, "workspace", "gain_predictor")

# 有效VMAF范围(过滤黑帧/异常帧: VMAF<=0 或 >100)
VMAF_MIN, VMAF_MAX = 1.0, 100.0


def safe_stats(arr, prefix):
    """对一维数组生成统计特征字典"""
    arr = np.asarray(arr, dtype=np.float64)
    if len(arr) == 0:
        return {f"{prefix}_mean": np.nan, f"{prefix}_std": np.nan,
                f"{prefix}_max": np.nan, f"{prefix}_median": np.nan,
                f"{prefix}_q90": np.nan, f"{prefix}_q10": np.nan}
    return {
        f"{prefix}_mean": float(np.mean(arr)),
        f"{prefix}_std": float(np.std(arr)),
        f"{prefix}_max": float(np.max(arr)),
        f"{prefix}_median": float(np.median(arr)),
        f"{prefix}_q90": float(np.quantile(arr, 0.9)),
        f"{prefix}_q10": float(np.quantile(arr, 0.1)),
    }


def main():
    with open(SRC) as f:
        data = json.load(f)

    meta = data["metadata"]
    results = data["results"]
    content = data["content_features"]
    resolutions = meta["resolutions"]              # [240,360,480]
    levels = ["sr_low", "sr_medium", "sr_high"]
    num_seg = meta["num_segments"]
    fps_seg = meta["frames_per_segment"]

    res_to_idx = {240: 0, 360: 1, 480: 2, 720: 3, 1080: 4}

    seg_rows = []
    frame_rows = []

    for res in resolutions:
        for seg in range(num_seg):
            seg_key = str(seg)
            if seg_key not in content:
                continue

            base_key = f"{res}_base_{seg}"
            if base_key not in results:
                continue

            base = np.asarray(results[base_key], dtype=np.float64)
            sr_arrs = {}
            ok = True
            for lv in levels:
                k = f"{res}_{lv}_{seg}"
                if k not in results:
                    ok = False
                    break
                sr_arrs[lv] = np.asarray(results[k], dtype=np.float64)
            if not ok:
                continue

            # LR侧内容特征(逐帧)
            motion = np.asarray(content[seg_key]["motion"], dtype=np.float64)
            texture = np.asarray(content[seg_key]["texture"], dtype=np.float64)
            scene = np.asarray(content[seg_key]["scene_change"], dtype=bool)

            n = min(len(base), len(motion), fps_seg)
            base = base[:n]; motion = motion[:n]; texture = texture[:n]; scene = scene[:n]
            for lv in levels:
                sr_arrs[lv] = sr_arrs[lv][:n]

            # 有效帧mask: base和所有sr都在合理范围
            valid = np.ones(n, dtype=bool)
            valid &= (base > VMAF_MIN) & (base <= VMAF_MAX)
            for lv in levels:
                valid &= (sr_arrs[lv] > VMAF_MIN) & (sr_arrs[lv] <= VMAF_MAX)
            if valid.sum() < n * 0.5:  # 过半帧无效则丢弃该段
                continue

            # ---------- 帧级数据集 ----------
            for fi in np.where(valid)[0]:
                row = {"res": res, "res_idx": res_to_idx[int(res)],
                       "seg": seg, "frame": fi,
                       "motion": float(motion[fi]),
                       "texture": float(texture[fi]),
                       "scene_change": int(bool(scene[fi])),
                       "base_vmaf": float(base[fi])}
                for lv in levels:
                    row[f"gain_{lv}"] = float(sr_arrs[lv][fi] - base[fi])
                frame_rows.append(row)

            # ---------- 段级特征(只在有效帧上统计) ----------
            m_v, t_v = motion[valid], texture[valid]
            base_v = base[valid]
            feat = {"res": res, "res_idx": res_to_idx[int(res)], "seg": seg,
                    "n_valid_frames": int(valid.sum()),
                    "scene_change_count": int(scene[valid].sum()),
                    "high_motion_ratio": float(np.mean(m_v > np.quantile(m_v, 0.75))) if len(m_v) else np.nan}
            feat.update(safe_stats(m_v, "motion"))
            feat.update(safe_stats(t_v, "texture"))
            feat.update(safe_stats(base_v, "base"))
            # 段级标签: 有效帧上的平均增益
            for lv in levels:
                g = sr_arrs[lv][valid] - base_v
                feat[f"gain_{lv}_mean"] = float(np.mean(g))
                feat[f"gain_{lv}_std"] = float(np.std(g))
            seg_rows.append(feat)

    df_seg = pd.DataFrame(seg_rows)
    df_frame = pd.DataFrame(frame_rows)

    seg_path = os.path.join(OUT_DIR, "dataset_seg.csv")
    frame_path = os.path.join(OUT_DIR, "dataset_frame.csv")
    df_seg.to_csv(seg_path, index=False, encoding="utf-8-sig")
    df_frame.to_csv(frame_path, index=False, encoding="utf-8-sig")

    print("=== 数据集构建完成 ===")
    print(f"段级样本: {len(df_seg)}  (期望≈{len(resolutions)}分辨率×158段)")
    print(f"帧级样本: {len(df_frame)}")
    print(f"\n各分辨率段数:\n{df_seg.groupby('res').size()}")
    print(f"\n段级特征列({len(df_seg.columns)}列):")
    print(list(df_seg.columns))
    print(f"\n标签分布(段级平均增益):")
    for lv in levels:
        c = f"gain_{lv}_mean"
        print(f"  {lv}: 全局均值={df_seg[c].mean():.2f}, 段间std={df_seg[c].std():.2f}, "
              f"范围=[{df_seg[c].min():.2f},{df_seg[c].max():.2f}]")
    print(f"\n已保存:\n  {seg_path}\n  {frame_path}")


if __name__ == "__main__":
    main()
