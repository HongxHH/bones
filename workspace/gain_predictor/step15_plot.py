"""Step 15: 汇总各方案相对BONES(static)的端到端增益, 出对比图"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"E:\dev\Code\Python\bones\workspace\gain_predictor\figures"
os.makedirs(OUT, exist_ok=True)

# (标签, ΔQoE, 胜率%, 类别)  数值来自step10/11/13/14全量130轨迹
rows = [
    ("Oracle joint\n(upper bound)", 0.635, 79.2, "upper"),
    ("Pred joint\n(alpha=1)", -0.253, 42.3, "joint"),
    ("Pred shrink\nalpha=0.75", 0.131, 52.3, "joint"),
    ("Pred gated\nmargin=1.5", 0.041, 43.8, "joint"),
    ("Oracle decoupled", -5.118, 42.3, "decouple"),
    ("Pred decoupled", -5.964, 30.8, "decouple"),
]
labels = [r[0] for r in rows]
dqoe = [r[1] for r in rows]
win = [r[2] for r in rows]
colors = {"upper": "#2a9d8f", "joint": "#4c72b0", "decouple": "#c44e52"}
bar_colors = [colors[r[3]] for r in rows]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
b1 = ax1.bar(labels, dqoe, color=bar_colors, edgecolor="black", linewidth=0.6)
ax1.axhline(0, color="black", linewidth=1)
ax1.set_ylabel("Mean delta-QoE vs BONES(static)")
ax1.set_title("End-to-end QoE gain over 130 traces")
for b, v in zip(b1, dqoe):
    ax1.text(b.get_x()+b.get_width()/2, v + (0.12 if v >= 0 else -0.28),
             f"{v:+.2f}", ha="center", fontsize=9)
ax1.set_ylim(-7, 1.4)
ax1.tick_params(axis='x', labelsize=8)

b2 = ax2.bar(labels, win, color=bar_colors, edgecolor="black", linewidth=0.6)
ax2.axhline(50, color="gray", linestyle="--", linewidth=1)
ax2.set_ylabel("Win rate vs BONES (%)")
ax2.set_title("Per-trace win rate (dashed=50%)")
for b, v in zip(b2, win):
    ax2.text(b.get_x()+b.get_width()/2, v+1.2, f"{v:.1f}%", ha="center", fontsize=9)
ax2.set_ylim(0, 100)
ax2.tick_params(axis='x', labelsize=8)

plt.tight_layout()
p = os.path.join(OUT, "s6_sim_endtoend_gain.png")
plt.savefig(p, dpi=140, bbox_inches="tight")
print("saved", p)
