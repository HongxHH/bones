"""M5 步骤1b：补测"预测量每帧 base VMAF"对帧增益的排序能力。

base VMAF 的合法性：BONES 本身依赖随视频离线预测量的静态质量表（每段一个值）；
每帧 base VMAF 是同一张表细化到帧，测量时不跑增强模型，属下载前可得信息，
不是运行时 oracle。此处显式与"纯压缩域特征"分开报告。

另测：base 与 dv 的直接 Spearman；加入 base 后的 GBR_q15。
"""
import json
import os
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
with open(os.path.join(REPO, "data", "bbb", "per_frame_quality.json"), encoding="utf-8") as f:
    data = json.load(f)
cf, res = data["content_features"], data["results"]
RESOLUTIONS = [240, 360, 480]
LEVELS = [1, 2, 3]
LN = {1: "sr_low", 2: "sr_medium", 3: "sr_high"}
N_FOLD = 5

rows = []
for seg_s in cf:
    seg = int(seg_s)
    m, t, sc = cf[seg_s]["motion"], cf[seg_s]["texture"], cf[seg_s]["scene_change"]
    for ri, r in enumerate(RESOLUTIONS):
        base = np.array(res[f"{r}_base_{seg}"])
        prev = base
        dvs = {}
        for l in LEVELS:
            cur = np.array(res[f"{r}_{LN[l]}_{seg}"])
            dvs[l] = cur - prev
            prev = cur
        for f in range(len(m)):
            rows.append((seg, ri, f, float(m[f]), float(t[f]),
                         1.0 if sc[f] else 0.0, float(base[f]),
                         float(dvs[1][f]), float(dvs[2][f]), float(dvs[3][f])))

rows = np.array(rows)
segs = np.unique(rows[:, 0])
blocks = np.array_split(segs, N_FOLD)
feat_idx = [3, 4, 5, 1, 2]          # motion texture scene res f
feat_idx_base = [3, 4, 5, 1, 2, 6]  # + base vmaf

sp_direct = {l: [] for l in LEVELS}
sp_gbr = {l: [] for l in LEVELS}
over = {l: [] for l in LEVELS}
regret = {l: {fr: [] for fr in (0.25, 0.5, 0.75)} for l in LEVELS}

for fold in range(N_FOLD):
    test_segs = set(blocks[fold])
    te = np.array([r[0] in test_segs for r in rows])
    tr = ~te
    groups = {}
    for i in np.where(te)[0]:
        groups.setdefault((int(rows[i, 1]), int(rows[i, 0])), []).append(i)
    for l in LEVELS:
        y = rows[:, 7 + l - 1]
        model = HistGradientBoostingRegressor(
            max_iter=200, max_depth=4, learning_rate=0.06,
            min_samples_leaf=40, loss="quantile", quantile=0.15, random_state=0)
        model.fit(rows[tr][:, feat_idx_base], y[tr])
        for key, idxs in groups.items():
            idxs = np.array(idxs)
            yt = y[idxs]
            base_v = rows[idxs, 6]
            yp = np.clip(model.predict(rows[idxs][:, feat_idx_base]), 0, None)
            if np.std(yt) > 1e-9:
                sp_direct[l].append(spearmanr(-base_v, yt).statistic)  # base 越低增益预期越高
                sp_gbr[l].append(spearmanr(yp, yt).statistic)
            over[l].append(float(np.mean(yp > yt + 1e-9)))
            for fr in (0.25, 0.5, 0.75):
                k = max(1, int(round(fr * len(idxs))))
                vo = np.sort(yt)[::-1][:k]
                vp = yt[np.argsort(-yp, kind="stable")[:k]]
                regret[l][fr].append((vo.sum() - vp.sum()) / k)

out = {"note": "base VMAF = offline pre-measured per-frame base table (BONES static-table paradigm)",
       "levels": {}}
for l in LEVELS:
    out["levels"][f"level{l}"] = {
        "spearman_minus_base_mean": round(float(np.mean(sp_direct[l])), 4),
        "spearman_gbr_with_base_mean": round(float(np.mean(sp_gbr[l])), 4),
        "overestimate_rate": round(float(np.mean(over[l])), 4),
        "regret_top25": round(float(np.mean(regret[l][0.25])), 4),
        "regret_top50": round(float(np.mean(regret[l][0.5])), 4),
        "regret_top75": round(float(np.mean(regret[l][0.75])), 4),
    }
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "causal_rank_base_feature.json")
json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
for l in LEVELS:
    d = out["levels"][f"level{l}"]
    print(f"L{l}: sp(-base) {d['spearman_minus_base_mean']:.3f}  "
          f"sp(GBR+base) {d['spearman_gbr_with_base_mean']:.3f}  "
          f"over {d['overestimate_rate']:.3f}  "
          f"regret {d['regret_top25']:.3f}/{d['regret_top50']:.3f}/{d['regret_top75']:.3f}")
print("saved ->", p)
