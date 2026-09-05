"""
Step 2: 统计基线分析
1. 各分辨率×增强级别的增益分布
2. 段内 vs 段间 方差分解(论证段平均是否足够、为何需要更细粒度)
3. LR侧特征 与 SR增益 的相关性矩阵
输出: 控制台结论 + figures/ 下图表
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"E:\dev\Code\Python\bones"
OUT = os.path.join(ROOT, "workspace", "gain_predictor")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

LEVELS = ["sr_low", "sr_medium", "sr_high"]
FEATURES = ["motion_mean", "motion_std", "motion_max", "motion_q90",
            "texture_mean", "texture_std", "texture_max",
            "base_mean", "base_std", "scene_change_count", "high_motion_ratio"]


def main():
    df_seg = pd.read_csv(os.path.join(OUT, "dataset_seg.csv"))
    df_frame = pd.read_csv(os.path.join(OUT, "dataset_frame.csv"))

    print("=" * 70)
    print("1. 各分辨率×增强级别 的段级增益分布")
    print("=" * 70)
    dist = df_seg.groupby("res")[[f"gain_{lv}_mean" for lv in LEVELS]].agg(["mean", "std", "min", "max"])
    print(dist.round(2).to_string())

    # 图1: 三级别增益分布直方图(按分辨率分面)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    for ax, res in zip(axes, sorted(df_seg["res"].unique())):
        sub = df_seg[df_seg["res"] == res]
        for lv, c in zip(LEVELS, ["#4C72B0", "#55A868", "#C44E52"]):
            ax.hist(sub[f"gain_{lv}_mean"], bins=30, alpha=0.55, label=lv, color=c)
        ax.set_title(f"{res}p")
        ax.set_xlabel("Segment-mean SR gain (VMAF)")
        ax.legend(fontsize=8)
    axes[0].set_ylabel("# segments")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "s1_gain_distribution.png"), dpi=130)
    plt.close()

    print("\n" + "=" * 70)
    print("2. 段内 vs 段间 方差分解(帧级, sr_high增益)")
    print("=" * 70)
    # 对每个分辨率: 总方差 = 段间方差(段均值的方差) + 段内方差(各段内方差的平均)
    for res in sorted(df_frame["res"].unique()):
        sub = df_frame[df_frame["res"] == res].copy()
        sub["gain"] = sub["gain_sr_high"]
        grand_mean = sub["gain"].mean()
        between = sub.groupby("seg")["gain"].mean().var()           # 段间
        within = sub.groupby("seg")["gain"].var().mean()            # 段内平均
        total = sub["gain"].var()
        ratio_between = between / total * 100
        print(f"{res}p: 总方差={total:.2f} | 段间={between:.2f}({ratio_between:.1f}%) "
              f"| 段内={within:.2f}({100-ratio_between:.1f}%)")

    print("\n" + "=" * 70)
    print("3. LR侧特征 与 段级增益 的 Pearson / Spearman 相关性")
    print("=" * 70)
    corr_tables = {}
    for lv in LEVELS:
        y = df_seg[f"gain_{lv}_mean"]
        pr, sp = {}, {}
        for f in FEATURES:
            x = df_seg[f]
            pr[f] = np.corrcoef(x, y)[0, 1]
            sp[f] = pd.Series(x).corr(pd.Series(y), method="spearman")
        corr_tables[lv] = pd.DataFrame({"pearson": pr, "spearman": sp})
        print(f"\n--- {lv} (按|pearson|排序) ---")
        t = corr_tables[lv].reindex(corr_tables[lv]["pearson"].abs().sort_values(ascending=False).index)
        print(t.round(3).to_string())

    # 图2: 关键特征 vs 增益 散点(sr_high)
    key_feats = ["motion_mean", "base_mean", "texture_mean"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    colors = {240: "#4C72B0", 360: "#55A868", 480: "#C44E52"}
    for ax, f in zip(axes, key_feats):
        for res in sorted(df_seg["res"].unique()):
            sub = df_seg[df_seg["res"] == res]
            ax.scatter(sub[f], sub["gain_sr_high_mean"], s=10, alpha=0.5,
                       color=colors[res], label=f"{res}p")
        ax.set_xlabel(f)
        ax.set_ylabel("sr_high gain" if f == key_feats[0] else "")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "s2_feature_vs_gain.png"), dpi=130)
    plt.close()

    # 图3: 相关性热力图(sr_high)
    mat = df_seg[FEATURES + ["gain_sr_high_mean"]].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(mat.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(mat.columns))); ax.set_xticklabels(mat.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(mat.index))); ax.set_yticklabels(mat.index, fontsize=8)
    for i in range(len(mat.index)):
        for j in range(len(mat.columns)):
            ax.text(j, i, f"{mat.values[i,j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "s3_corr_heatmap.png"), dpi=130)
    plt.close()

    print(f"\n图表已保存到 {FIG}")


if __name__ == "__main__":
    main()
