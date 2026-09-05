"""step20 帧级价值检验的证据图：尺度对比 / 不同口径帧级额外价值 / 最差帧段聚集度。"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

J = json.load(open("workspace/gain_predictor/step20_frame_value.json", encoding="utf-8"))
RES = [240, 360, 480]
M = ["sr_low", "sr_medium", "sr_high"]
labels = [f"{r}\n{m.replace('sr_','')}" for r in RES for m in M]

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
plt.subplots_adjust(wspace=0.28)

# (1) 段间 vs 段内增益 std
between = [J["scale"][f"{r}_{m}"]["between_std"] for r in RES for m in M]
within = [J["scale"][f"{r}_{m}"]["within_std"] for r in RES for m in M]
x = np.arange(len(labels))
ax[0].bar(x - 0.2, between, 0.4, label="between-seg std", color="#c0392b")
ax[0].bar(x + 0.2, within, 0.4, label="within-seg std", color="#7f8c8d")
ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8)
ax[0].set_ylabel("gain VMAF std"); ax[0].set_title("(a) Gain variance is across segments, not within")
ax[0].legend(fontsize=8)

# (2) 预算50%、sr_high，不同质量口径下 帧级相对段级的额外价值（两种选帧准则）
# 数值取自 step20 运行结果（sr_high, budget=0.5）
agg = ["mean", "p10", "p5", "min"]
by_gain = {240: [0.30, 1.35, 0.31, -7.94], 360: [0.23, 0.68, 0.42, 0.0], 480: [0.19, 0.65, 0.74, 0.0]}
by_base = {240: [0.29, 2.48, 6.68, 3.36], 360: [-0.05, 3.68, 4.16, 6.89], 480: [0.01, 2.13, 3.07, 4.32]}
xx = np.arange(len(agg)); w = 0.25
for i, r in enumerate(RES):
    ax[1].bar(xx + (i - 1) * w - w / 2, by_gain[r], w, color="#95a5a6", label=f"{r}p by-gain" if i == 0 else "")
for i, r in enumerate(RES):
    ax[1].bar(xx + (i - 1) * w + w / 2, by_base[r], w,
              color=["#2980b9", "#27ae60", "#e67e22"][i], label=f"{r}p save-worst")
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_xticks(xx); ax[1].set_xticklabels(agg)
ax[1].set_ylabel("frame-over-seg extra VMAF @50% budget")
ax[1].set_title("(b) Frame edge only under worst-tail metric", fontsize=10)
ax[1].legend(fontsize=7, ncol=2)

# (3) 最差5%帧的段聚集
share_seg = [J["worst_frame_cluster"][str(r)]["seg_share"] * 100 for r in RES]
intra = []
for r in RES:
    n = J["worst_frame_cluster"][str(r)]["avg_p5frames_per_carrying_seg"] / 120 * 100
    intra.append(n)
ax2b = ax[2]
ax[2].bar(np.arange(3) - 0.2, share_seg, 0.4, color="#8e44ad", label="% segments carrying worst-5% frames")
ax2b.bar(np.arange(3) + 0.2, intra, 0.4, color="#d35400", label="% frames in those segments that are worst")
ax[2].set_xticks(np.arange(3)); ax[2].set_xticklabels([f"{r}p" for r in RES])
ax[2].set_ylabel("%"); ax[2].set_title("(c) Bad quality clusters by whole segment", fontsize=10)
ax[2].legend(fontsize=7)

plt.tight_layout()
out = "workspace/gain_predictor/figures/s8_frame_value_test.png"
plt.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
