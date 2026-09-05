"""Step 23: 服务端引导证伪实验证据图（英文标签防乱码）。"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = r"E:\dev\Code\Python\bones"
d = json.load(open(os.path.join(ROOT, "workspace/gain_predictor/step22_server_guided.json"), encoding="utf-8"))
SMS = [1.0, 0.6, 0.4, 0.25]
xl = ["1.0\n(fast)", "0.6", "0.4", "0.25\n(slow)"]

def pair(sm, name, k="qoe"):
    return d[f"sm_{sm}"]["summary"]["pairs"][name][k]["mean"]

fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

# (a) 解耦执行各策略相对 BONES 的 ΔQoE 随 GPU 稀缺度
ax[0].axhline(0, color="k", lw=1, ls="--")
for name, c, mk in [("edf_max_minus_bones_orig", "#2b6cb0", "o"),
                    ("edf_smooth_minus_bones_orig", "#38a169", "s"),
                    ("edf_oracle_minus_bones_orig", "#dd6b20", "^")]:
    y = [pair(sm, name) for sm in SMS]
    ax[0].plot(range(4), y, marker=mk, color=c, lw=2, ms=7,
               label=name.replace("_minus_bones_orig", ""))
ax[0].set_xticks(range(4)); ax[0].set_xticklabels(xl)
ax[0].set_xlabel("GPU speed multiplier (left=fast, right=constrained)")
ax[0].set_ylabel(r"$\Delta$QoE vs BONES")
ax[0].set_title("(a) Decoupled non-blocking execution vs BONES")
ax[0].legend(); ax[0].grid(alpha=.3)

# (b) 服务端真值选方法的净增量（oracle - max），应≈0
vals = [pair(sm, "edf_oracle_minus_edf_max") for sm in SMS]
b = ax[1].bar(range(4), vals, color="#dd6b20", width=.55)
ax[1].axhline(0, color="k", lw=1)
ax[1].set_xticks(range(4)); ax[1].set_xticklabels(xl)
ax[1].set_xlabel("GPU speed multiplier")
ax[1].set_ylabel(r"$\Delta$QoE (oracle-method $-$ strongest-feasible)")
ax[1].set_title("(b) Server-side true-gain method choice adds ~0")
ax[1].set_ylim(-0.25, 0.25); ax[1].grid(alpha=.3, axis="y")
for r, v in zip(b, vals):
    ax[1].text(r.get_x()+r.get_width()/2, v+0.01, f"{v:+.3f}", ha="center", fontsize=9)

# (c) 稀缺时振荡代价吃掉 VMAF 增益（edf_max - bones 的 Δvmaf 与 Δosc）
dv = [pair(sm, "edf_max_minus_bones_orig", "vmaf") for sm in SMS]
do = [pair(sm, "edf_max_minus_bones_orig", "osc") for sm in SMS]
w = .35
ax[2].bar(np.arange(4)-w/2, dv, w, label=r"$\Delta$VMAF (gain)", color="#2b6cb0")
ax[2].bar(np.arange(4)+w/2, do, w, label=r"$\Delta$oscillation (cost)", color="#c53030")
ax[2].axhline(0, color="k", lw=1)
ax[2].set_xticks(range(4)); ax[2].set_xticklabels(xl)
ax[2].set_xlabel("GPU speed multiplier")
ax[2].set_title("(c) Under scarcity, oscillation cost erodes VMAF gain")
ax[2].legend(); ax[2].grid(alpha=.3, axis="y")

plt.tight_layout()
out = os.path.join(ROOT, "workspace/gain_predictor/figures/s9_server_guided_falsify.png")
plt.savefig(out, dpi=140, bbox_inches="tight")
print("saved", out)
