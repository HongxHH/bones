"""
Step 4 补充: 更严谨的泛化评估
A. 5折分层交叉验证(按段, 每折覆盖三档分辨率), 报告均值±std —— 主结果, 模拟"训练集覆盖各类场景"
B. 留一分辨率(LORO): 用两档训练预测第三档 —— 检验跨分辨率外推(关系能否用于720/1080)
C. temporal 保留为"单视频外推压力测试", 仅作对照
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import make_pipeline

ROOT = r"E:\dev\Code\Python\bones"
OUT = os.path.join(ROOT, "workspace", "gain_predictor")
FEATURES = ["res_idx", "motion_mean", "motion_std", "motion_max", "motion_q90",
            "texture_mean", "texture_std", "texture_max",
            "base_mean", "base_std", "scene_change_count", "high_motion_ratio"]
LEVELS = ["sr_low", "sr_medium", "sr_high"]
RNG = 42


def metrics(yt, yp):
    mae = np.mean(np.abs(yt - yp))
    r2 = 1 - np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2)
    sp = spearmanr(yp, yt).statistic if np.std(yp) > 1e-9 else np.nan
    n = len(yt); k = max(1, n//2)
    top50 = len(set(np.argsort(yt)[-k:]) & set(np.argsort(yp)[-k:])) / k
    return mae, r2, sp, top50


def make_models():
    return {
        "ridge": make_pipeline(StandardScaler(), Ridge(1.0)),
        "gbdt": HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=500, l2_regularization=1.0,
            min_samples_leaf=8, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=RNG),
    }


def cv_5fold(df, target):
    """每档分辨率内独立做5折, 合并折结果, 保证每折test覆盖三档"""
    folds = {res: list(KFold(5, shuffle=True, random_state=RNG).split(sub))
             for res, sub in df.groupby("res")}
    res_groups = {res: sub.reset_index() for res, sub in df.groupby("res")}
    agg = {m: {"MAE": [], "R2": [], "SP": [], "TOP50": []} for m in ["ridge", "gbdt"]}
    for fold in range(5):
        tr_parts, te_parts = [], []
        for res, g in res_groups.items():
            tr_i, te_i = folds[res][fold]
            tr_parts.append(g.iloc[tr_i]); te_parts.append(g.iloc[te_i])
        tr, te = pd.concat(tr_parts), pd.concat(te_parts)
        for name, mdl in make_models().items():
            mdl.fit(tr[FEATURES].values, tr[target].values)
            mae, r2, sp, t50 = metrics(te[target].values, mdl.predict(te[FEATURES].values))
            agg[name]["MAE"].append(mae); agg[name]["R2"].append(r2)
            agg[name]["SP"].append(sp); agg[name]["TOP50"].append(t50)
    out = {}
    for name, d in agg.items():
        out[name] = {k: [float(np.mean(v)), float(np.std(v))] for k, v in d.items()}
    return out


def loro(df, target):
    """留一分辨率: 用另两档训练, 预测留出档(去掉res_idx, 否则模型记住档位编号)"""
    feats = [f for f in FEATURES if f != "res_idx"]
    out = {}
    for hold in sorted(df["res"].unique()):
        tr = df[df["res"] != hold]; te = df[df["res"] == hold]
        mdl = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=500, l2_regularization=1.0,
            min_samples_leaf=8, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=RNG)
        mdl.fit(tr[feats].values, tr[target].values)
        mae, r2, sp, t50 = metrics(te[target].values, mdl.predict(te[feats].values))
        out[f"hold_{hold}"] = {"MAE": float(mae), "R2": float(r2), "SP": float(sp), "TOP50": float(t50)}
    return out


def main():
    df = pd.read_csv(os.path.join(OUT, "dataset_seg.csv"))
    final = {"cv5": {}, "loro": {}}
    print("="*80)
    print("A. 5折分层交叉验证 (均值±std) —— 主结果")
    print("="*80)
    for lv in LEVELS:
        t = f"gain_{lv}_mean"
        r = cv_5fold(df, t); final["cv5"][lv] = r
        print(f"\n### {lv} ###")
        print(f"{'模型':<8}{'MAE':>16}{'R2':>16}{'Spearman':>16}{'TopK50':>16}")
        for m, d in r.items():
            print(f"{m:<8}{d['MAE'][0]:>9.3f}±{d['MAE'][1]:.3f}"
                  f"{d['R2'][0]:>8.3f}±{d['R2'][1]:.3f}"
                  f"{d['SP'][0]:>8.3f}±{d['SP'][1]:.3f}"
                  f"{d['TOP50'][0]:>8.3f}±{d['TOP50'][1]:.3f}")

    print("\n" + "="*80)
    print("B. 留一分辨率(LORO) GBDT —— 跨分辨率外推")
    print("="*80)
    for lv in LEVELS:
        t = f"gain_{lv}_mean"
        r = loro(df, t); final["loro"][lv] = r
        print(f"\n### {lv} ###")
        for hold, d in r.items():
            print(f"  {hold}: MAE={d['MAE']:.3f}  R2={d['R2']:.3f}  Spearman={d['SP']:.3f}  TopK50={d['TOP50']:.3f}")

    with open(os.path.join(OUT, "step4b_generalization.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print("\n已保存 step4b_generalization.json")


if __name__ == "__main__":
    main()
