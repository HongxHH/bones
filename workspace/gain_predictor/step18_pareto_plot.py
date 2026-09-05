"""Step 18: 算力节省-QoE 帕累托对比图 (oracle上限 vs pred), 数据读step16结果"""
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"E:\dev\Code\Python\bones\workspace\gain_predictor\figures"
d = json.load(open(r"E:\dev\Code\Python\bones\workspace\gain_predictor\step16_budget_pareto.json"))

fig, ax = plt.subplots(figsize=(8.2, 5.2))
styles = {"oracle": ("#2a9d8f", "o", "Oracle (true per-seg gain, upper bound)"),
          "pred": ("#c44e52", "s", "Pred (GBDT predicted gain, ours)")}
for mode in ["oracle", "pred"]:
    xs, ys = [], []
    for r in d["curves"][mode]:
        xs.append(r["gpu_save_pct"]); ys.append(r["dqoe"])
    c, mk, lab = styles[mode]
    ax.plot(xs, ys, marker=mk, color=c, linewidth=2, markersize=7, label=lab)
ax.axhline(0, color="black", linewidth=1, linestyle="--")
ax.fill_between([-3, 80], -0.3, 0.3, color="gray", alpha=0.12, label="QoE neutral band (+-0.3)")
ax.annotate("Oracle: ~23% GPU saved\nat QoE parity", xy=(23.5, -0.05), xytext=(30, 1.2),
            arrowprops=dict(arrowstyle="->", color="#2a9d8f"), fontsize=9, color="#1f6f66")
ax.annotate("Pred: QoE collapses\nbefore saving GPU", xy=(3, -5.9), xytext=(18, -4.6),
            arrowprops=dict(arrowstyle="->", color="#c44e52"), fontsize=9, color="#96383b")
ax.set_xlabel("GPU enhancement time saved vs BONES (%)")
ax.set_ylabel("Mean delta-QoE vs BONES(static)")
ax.set_title("Compute-budget Pareto: quality-constrained enhancement (130 traces)")
ax.set_xlim(-4, 80); ax.set_ylim(-7, 2)
ax.legend(loc="lower left", fontsize=9); ax.grid(alpha=0.3)
plt.tight_layout()
p = os.path.join(OUT, "s7_budget_pareto.png")
plt.savefig(p, dpi=140, bbox_inches="tight"); print("saved", p)
