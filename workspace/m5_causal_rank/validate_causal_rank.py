"""M5 步骤1：帧增益因果排序器离线验证（决策前信息，不偷看增强后 VMAF）。

数据：data/bbb/per_frame_quality.json
- 因果特征（下载完成即可得，无需跑增强模型）：每帧 motion / texture / scene_change
- 目标：每帧各档增量 VMAF dv_l = vmaf(level_l) - vmaf(level_{l-1}), l=1..3

评价口径（不用 R²；#4 教训：不准的预测比没预测更有害）：
1. 排序相关：按 (res, level, seg) 组内，pred dv 与 true dv 的 Spearman
   ——队列按密度 dv/dc 排序，dc 在组内对所有帧相同，组内排序即 dv 排序。
2. 后悔值：预算只够增强 top 25%/50%/75% 帧时，pred 排序相对 oracle 排序的 VMAF 损失。
3. 保守性：pred dv 高估率（pred>true）必须有上界；用分位数回归给下界预测。

交叉验证：158 段分 5 个连续块，4 块训练 1 块测试（禁止 shuffle；#4 shuffle 乐观口径教训）。

对比候选排序器：
- H0：texture 单特征（参数无关，最健壮，作底线）
- GBR-m：HistGradientBoosting 中位数回归
- GBR-q：HistGradientBoosting 分位数回归（alpha=0.15，保守下界）
"""
import json
import os
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PER_FRAME = os.path.join(REPO, "data", "bbb", "per_frame_quality.json")
RESOLUTIONS = [240, 360, 480]
LEVELS = [1, 2, 3]
LEVEL_NAMES = {1: "sr_low", 2: "sr_medium", 3: "sr_high"}
N_FOLD = 5
TOP_FRACS = (0.25, 0.50, 0.75)


def load_rows():
    with open(PER_FRAME, encoding="utf-8") as f:
        data = json.load(f)
    cf = data["content_features"]
    res = data["results"]
    rows = []  # dict(seg,res,f,motion,texture,scene, dv[1..3])
    for seg_s in cf:
        seg = int(seg_s)
        m = cf[seg_s]["motion"]
        t = cf[seg_s]["texture"]
        sc = cf[seg_s]["scene_change"]
        for r_idx, r in enumerate(RESOLUTIONS):
            base = np.array(res[f"{r}_base_{seg}"])
            dv = {}
            prev = base
            for l in LEVELS:
                cur = np.array(res[f"{r}_{LEVEL_NAMES[l]}_{seg}"])
                dv[l] = cur - prev
                prev = cur
            for f in range(len(m)):
                rows.append({
                    "seg": seg, "res": r, "res_idx": r_idx, "f": f,
                    "motion": float(m[f]), "texture": float(t[f]),
                    "scene": 1.0 if sc[f] else 0.0,
                    "dv1": float(dv[1][f]), "dv2": float(dv[2][f]), "dv3": float(dv[3][f]),
                })
    return rows


def feature_matrix(rows):
    arr = np.array([[r["motion"], r["texture"], r["scene"],
                     r["res_idx"], r["f"]] for r in rows], dtype=float)
    return arr


def fit_gbr(train_x, train_y, quantile=None):
    kwargs = dict(max_iter=200, max_depth=4, learning_rate=0.06,
                  min_samples_leaf=40, random_state=0)
    if quantile is not None:
        kwargs["loss"] = "quantile"
        kwargs["quantile"] = quantile
    m = HistGradientBoostingRegressor(**kwargs)
    m.fit(train_x, train_y)
    return m


def evaluate():
    rows = load_rows()
    segs = sorted({r["seg"] for r in rows})
    blocks = np.array_split(segs, N_FOLD)
    x_all = feature_matrix(rows)
    seg_arr = np.array([r["seg"] for r in rows])

    # 结果累积：(model, level) -> spearman list; regret; overrate
    models = ["H0_texture", "GBR_median", "GBR_q15"]
    sp = {(m, l): [] for m in models for l in LEVELS}
    regret = {(m, l): {fr: [] for fr in TOP_FRACS} for m in models for l in LEVELS}
    overrate = {(m, l): [] for m in models for l in LEVELS}

    for fold in range(N_FOLD):
        test_segs = set(blocks[fold])
        tr_mask = np.array([s not in test_segs for s in seg_arr])
        te_mask = ~tr_mask
        gbr_m, gbr_q = {}, {}
        for l in LEVELS:
            y = np.array([r[f"dv{l}"] for r in rows])
            gbr_m[l] = fit_gbr(x_all[tr_mask], y[tr_mask])
            gbr_q[l] = fit_gbr(x_all[tr_mask], y[tr_mask], quantile=0.15)

        # 组内评价：按 (res, seg) 分组测试行
        idx_by_group = {}
        for i in np.where(te_mask)[0]:
            r = rows[i]
            idx_by_group.setdefault((r["res"], r["seg"]), []).append(i)

        for (r_, s_), idxs in idx_by_group.items():
            idxs = np.array(idxs)
            tex = x_all[idxs, 1]
            for l in LEVELS:
                y_true = np.array([rows[i][f"dv{l}"] for i in idxs])
                preds = {
                    # texture 排序：预测值直接用 texture（单调，保守性在外部统一 clip>=0）
                    "H0_texture": tex.copy(),
                    "GBR_median": gbr_m[l].predict(x_all[idxs]),
                    "GBR_q15": np.clip(gbr_q[l].predict(x_all[idxs], ), 0.0, None),
                }
                for name, yp in preds.items():
                    if np.std(y_true) > 1e-9 and np.std(yp) > 1e-9:
                        rho = spearmanr(yp, y_true).statistic
                        sp[(name, l)].append(rho)
                    overrate[(name, l)].append(float(np.mean(yp > y_true + 1e-9)))
                    k_total = len(idxs)
                    order_true = np.argsort(-y_true, kind="stable")
                    for fr in TOP_FRACS:
                        k = max(1, int(round(fr * k_total)))
                        v_oracle = np.sum(y_true[order_true[:k]])
                        order_pred = np.argsort(-yp, kind="stable")[:k]
                        v_pred = np.sum(y_true[order_pred])
                        # 每帧平均后悔（VMAF 分）
                        regret[(name, l)][fr].append((v_oracle - v_pred) / k)

    def agg(vals):
        a = np.array(vals)
        return float(np.mean(a)), float(np.median(a))

    out = {"cv": f"{N_FOLD} contiguous blocks, train4/test1, no shuffle",
           "n_rows": len(rows), "models": {}}
    for name in models:
        mout = {}
        for l in LEVELS:
            rho_mean, rho_med = agg(sp[(name, l)])
            or_mean, _ = agg(overrate[(name, l)])
            regout = {}
            for fr in TOP_FRACS:
                rm, rmed = agg(regret[(name, l)][fr])
                regout[f"top{int(fr*100)}_regret_vmaf_mean"] = round(rm, 4)
            mout[f"level{l}"] = {
                "spearman_mean": round(rho_mean, 4),
                "spearman_median": round(rho_med, 4),
                "overestimate_rate_mean": round(or_mean, 4),
                **regout,
            }
        out["models"][name] = mout

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "causal_rank_offline.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 控制台简表
    print(f"rows={len(rows)} (158 seg x 3 res x 120 frames)")
    for name in models:
        print(f"\n=== {name} ===")
        for l in LEVELS:
            d = out["models"][name][f"level{l}"]
            print(f" L{l}: spearman {d['spearman_mean']:.3f}/{d['spearman_median']:.3f}"
                  f"  overrate {d['overestimate_rate_mean']:.3f}"
                  f"  regret top25/50/75 = "
                  f"{d['top25_regret_vmaf_mean']:.3f}/{d['top50_regret_vmaf_mean']:.3f}"
                  f"/{d['top75_regret_vmaf_mean']:.3f}")
    print(f"\nsaved -> {out_path}")


if __name__ == "__main__":
    evaluate()
