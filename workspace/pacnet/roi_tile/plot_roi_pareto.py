# -*- coding: utf-8 -*-
"""画 ROI tile 方案 vs 均匀提分辨率前沿的等码率帕累托图（读 roi_oracle.json）。"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
j = json.load(open(os.path.join(HERE, "roi_oracle.json"), encoding="utf-8"))
uf = sorted(j["uniform_frontier"], key=lambda u: u["bits_per_frame"])
ux = [u["bits_per_frame"] / 1000 for u in uf]
uv = [u["vmaf"] for u in uf]

fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
tiers = list(j["tiers"].keys())
for ax, tier in zip(axes.flat, tiers):
    d = j["tiers"][tier]
    ax.plot(ux, uv, "k-o", lw=2, ms=4, label="uniform + l2 (frontier)")
    ax.scatter([d["B_low"] / 1000], [d["baseline_l2"]["vmaf"]], c="black", marker="s", s=60)
    ax.annotate("base l2", (d["B_low"] / 1000, d["baseline_l2"]["vmaf"]),
                textcoords="offset points", xytext=(5, -10), fontsize=7)
    styles = {"perframe": ("^", "--"), "fixed": ("x", "-"), "center": ("+", ":")}
    crfcols = {26: "tab:red", 30: "tab:orange", 34: "tab:purple"}
    for arm, rows in d["arms"].items():
        crf = int(arm.split("_")[0][3:]); strat = arm.split("_")[1]
        mk, ls = styles[strat]
        xs = [r["bits_per_frame"] / 1000 for r in rows if r["frac"] > 0]
        ys = [r["vmaf"] for r in rows if r["frac"] > 0]
        ax.plot(xs, ys, ls, marker=mk, ms=5, color=crfcols[crf],
                alpha=0.8, label=f"ROI crf{crf} {strat}" if strat == "perframe" else None)
    ax.set_title(tier, fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_xlabel("bits/frame (k)")
    ax.set_ylabel("VMAF")
handles, labels = axes.flat[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8)
fig.suptitle("ROI native tiles vs uniform-resolution frontier (equal bitrate, TOS 32 frames + l2 SR)")
fig.tight_layout(rect=[0, 0.05, 1, 0.97])
out = os.path.join(HERE, "roi_pareto.png")
fig.savefig(out, dpi=130)
print("saved", out)
