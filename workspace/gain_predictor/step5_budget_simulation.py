"""
Step 5: 选择性增强收益模拟(离线量化价值)

场景: 算力/能耗预算只允许增强一部分段时, 应该优先增强谁?
  - BONES静态表: 同分辨率所有段给同一个平均增益, 无段级区分度 -> 预算受限下等价随机选段
  - Oracle: 用真实增益选Top-k(理论上限)
  - Ours: 用OOF(交叉验证无偏)预测增益选Top-k

输出: 预算-增益曲线 + Ours相对随机的提升 + Ours达到Oracle的效率
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

ROOT = r"E:\dev\Code\Python\bones"
OUT = os.path.join(ROOT, "workspace", "gain_predictor")
FIG = os.path.join(OUT, "figures")
FEATURES = ["res_idx", "motion_mean", "motion_std", "motion_max", "motion_q90",
            "texture_mean", "texture_std", "texture_max",
            "base_mean", "base_std", "scene_change_count", "high_motion_ratio"]
RNG = 42
BUDGETS = np.arange(0.1, 1.01, 0.1)


def oof_predict(df, target):
    """5折OOF: 每段预测值来自没见过它的模型, 保证无偏"""
    oof_gbdt = np.zeros(len(df)); oof_ridge = np.zeros(len(df))
    df = df.reset_index(drop=True)
    folds = []
    # 每档分辨率内独立划分再合并
    idx_map = np.zeros(len(df), dtype=int)
    parts_tr, parts_te = [], []
    kf_global = {}
    for res, g in df.groupby("res"):
        kf_global[res] = list(KFold(5, shuffle=True, random_state=RNG).split(g))
    for fold in range(5):
        tri, tei = [], []
        for res, g in df.groupby("res"):
            g = g.reset_index()
            a, b = kf_global[res][fold]
            tri += g.iloc[a]["index"].tolist()
            tei += g.iloc[b]["index"].tolist()
        tr = df.loc[tri]; te = df.loc[tei]
        gbdt = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=500, l2_regularization=1.0,
            min_samples_leaf=8, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=RNG)
        ridge = make_pipeline(StandardScaler(), Ridge(1.0))
        gbdt.fit(tr[FEATURES].values, tr[target].values)
        ridge.fit(tr[FEATURES].values, tr[target].values)
        oof_gbdt[tei] = gbdt.predict(te[FEATURES].values)
        oof_ridge[tei] = ridge.predict(te[FEATURES].values)
    return oof_gbdt, oof_ridge


def budget_curve(y_true, y_pred, budgets, n_random=200):
    """返回各预算下 random/oracle/pred 的选中段平均增益"""
    n = len(y_true)
    rand_curve, ora_curve, pred_curve = [], [], []
    order_true = np.argsort(-y_true)
    order_pred = np.argsort(-y_pred)
    rng = np.random.RandomState(RNG)
    for b in budgets:
        k = max(1, int(round(n*b)))
        # random: 蒙特卡洛
        rs = [y_true[rng.choice(n, k, replace=False)].mean() for _ in range(n_random)]
        rand_curve.append(np.mean(rs))
        ora_curve.append(y_true[order_true[:k]].mean())
        pred_curve.append(y_true[order_pred[:k]].mean())  # 用预测排序, 但拿真实增益算收益
    return np.array(rand_curve), np.array(ora_curve), np.array(pred_curve)


def main():
    df = pd.read_csv(os.path.join(OUT, "dataset_seg.csv"))
    summary = {}
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ax, lv in zip(axes, ["sr_low", "sr_medium", "sr_high"]):
        target = f"gain_{lv}_mean"
        oof_g, oof_r = oof_predict(df, target)
        df = df.copy()
        df["oof_gbdt"], df["oof_ridge"] = oof_g, oof_r

        # 三档分辨率合并(各自排序后等权平均曲线, 消除档位量级差异)
        agg = {s: np.zeros(len(BUDGETS)) for s in ["rand", "ora", "gbdt", "ridge"]}
        per_res = {}
        for res in sorted(df["res"].unique()):
            sub = df[df["res"] == res]
            yt = sub[target].values
            r, o, pg = budget_curve(yt, sub["oof_gbdt"].values, BUDGETS)
            _, _, pr = budget_curve(yt, sub["oof_ridge"].values, BUDGETS)
            agg["rand"] += r/3; agg["ora"] += o/3; agg["gbdt"] += pg/3; agg["ridge"] += pr/3
            per_res[int(res)] = {"random": r.tolist(), "oracle": o.tolist(), "gbdt": pg.tolist()}

        # 效率: Ours达到Oracle的比例(相对random基线归一)
        def efficiency(pred):
            return (pred - agg["rand"]) / (agg["ora"] - agg["rand"] + 1e-9)
        eff_gbdt = efficiency(agg["gbdt"])
        summary[lv] = {
            "gbdt_vs_random_at50": float(agg["gbdt"][4]-agg["rand"][4]),
            "gbdt_oracle_eff_at50": float(eff_gbdt[4]),
            "gbdt_oracle_eff_mean": float(np.mean(eff_gbdt[1:8])),
            "curve": {k: v.tolist() for k, v in agg.items()},
            "eff_gbdt": eff_gbdt.tolist(),
        }

        ax.plot(BUDGETS*100, agg["ora"], "-o", ms=3, color="#222222", label="Oracle (upper bound)")
        ax.plot(BUDGETS*100, agg["gbdt"], "-s", ms=3, color="#C44E52", label="Ours-GBDT")
        ax.plot(BUDGETS*100, agg["ridge"], "--^", ms=3, color="#55A868", label="Ours-Ridge")
        ax.plot(BUDGETS*100, agg["rand"], ":", color="#4C72B0", label="BONES/random")
        ax.set_title(lv); ax.set_xlabel("enhancement budget (% segments)")
        ax.set_ylabel("avg gain of selected (VMAF)"); ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG, "s5_budget_gain_curve.png"), dpi=130)
    plt.close()

    print("="*72)
    print("选择性增强收益(50%预算下, 选中段平均VMAF增益)")
    print("="*72)
    for lv, s in summary.items():
        print(f"{lv:<10} GBDT比BONES/随机多选增益={s['gbdt_vs_random_at50']:+.2f} VMAF | "
              f"达到Oracle效率@50%={s['gbdt_oracle_eff_at50']*100:.1f}% | "
              f"平均效率(10-80%)={s['gbdt_oracle_eff_mean']*100:.1f}%")

    with open(os.path.join(OUT, "step5_budget_simulation.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n已保存 step5_budget_simulation.json 与 figures/s5_budget_gain_curve.png")


if __name__ == "__main__":
    main()
