"""
Step 3 + Step 4: 训练增益预测器并严格评估

模型(由简到繁):
  B0 全局均值(等价BONES"一张静态表"的最简形态)
  B1 按分辨率分组均值(BONES静态表更接近的形态)
  M1 岭回归(线性可解释基线)
  M2 HistGradientBoosting(GBDT, 主力)

划分(严格按段, 杜绝帧泄露):
  split=random : 每个分辨率内段随机 70/15/15
  split=temporal: 前70%段train / 中15%val / 后15%test(模拟流式顺序到来)

指标:
  回归: MAE / RMSE / R2 / Spearman
  决策: Top-k命中率(预测排序 vs 真实排序的重叠率), k=25%,50%
"""
import os
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
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


def split_segments(seg_ids, mode):
    seg_ids = sorted(seg_ids)
    n = len(seg_ids)
    if mode == "temporal":
        n_tr = int(n * 0.7); n_va = int(n * 0.85)
        return set(seg_ids[:n_tr]), set(seg_ids[n_tr:n_va]), set(seg_ids[n_va:])
    rng = np.random.RandomState(RNG)
    ids = seg_ids.copy(); rng.shuffle(ids)
    n_tr = int(n * 0.7); n_va = int(n * 0.85)
    return set(ids[:n_tr]), set(ids[n_tr:n_va]), set(ids[n_va:])


def topk_overlap(y_true, y_pred, k_ratio):
    """预测Top-k段 与 真实Top-k段 的重叠率(决策排序质量)"""
    n = len(y_true)
    k = max(1, int(round(n * k_ratio)))
    true_top = set(np.argsort(y_true)[-k:])
    pred_top = set(np.argsort(y_pred)[-k:])
    return len(true_top & pred_top) / k


def reg_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    if np.std(y_pred) < 1e-9:  # 常数预测, 排序相关未定义
        sp = np.nan
    else:
        sp = spearmanr(y_pred, y_true).statistic
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "Spearman": sp}


def evaluate_split(df, target, split_mode):
    # 逐分辨率划分再合并(保证每档都出现在train/test)
    train_idx, val_idx, test_idx = [], [], []
    for res in sorted(df["res"].unique()):
        sub = df[df["res"] == res]
        tr, va, te = split_segments(sub["seg"].unique().tolist(), split_mode)
        train_idx += sub[sub["seg"].isin(tr)].index.tolist()
        val_idx += sub[sub["seg"].isin(va)].index.tolist()
        test_idx += sub[sub["seg"].isin(te)].index.tolist()
    tr, va, te = df.loc[train_idx], df.loc[val_idx], df.loc[test_idx]

    Xtr, ytr = tr[FEATURES].values, tr[target].values
    Xte, yte = te[FEATURES].values, te[target].values

    results = {}
    # B0 全局均值
    pred = np.full(len(yte), ytr.mean())
    results["B0_global_mean"] = reg_metrics(yte, pred)
    # B1 按分辨率均值
    res_mean = tr.groupby("res_idx")[target].mean().to_dict()
    pred = te["res_idx"].map(res_mean).values
    results["B1_res_mean"] = reg_metrics(yte, pred)
    # M1 岭回归
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    ridge.fit(Xtr, ytr)
    results["M1_ridge"] = reg_metrics(yte, ridge.predict(Xte))
    # M2 GBDT (内部early stopping自动从训练集划分验证)
    gbdt = HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=500,
        l2_regularization=1.0, min_samples_leaf=8,
        early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=RNG)
    gbdt.fit(Xtr, ytr)
    pred_gbdt = gbdt.predict(Xte)
    results["M2_gbdt"] = reg_metrics(yte, pred_gbdt)

    # 决策指标(只对最强两个模型 + B1基线)
    for name, pred in [("B1_res_mean", te["res_idx"].map(res_mean).values),
                       ("M1_ridge", ridge.predict(Xte)),
                       ("M2_gbdt", pred_gbdt)]:
        results[name]["TopK25"] = topk_overlap(yte, pred, 0.25)
        results[name]["TopK50"] = topk_overlap(yte, pred, 0.50)

    # 特征重要性(GBDT)
    fi = dict(zip(FEATURES, gbdt.feature_importances_)) if hasattr(gbdt, "feature_importances_") else {}
    return results, fi


def main():
    df = pd.read_csv(os.path.join(OUT, "dataset_seg.csv"))
    all_out = {}
    for split in ["random", "temporal"]:
        print("=" * 78)
        print(f"划分方式: {split}")
        print("=" * 78)
        all_out[split] = {}
        for lv in LEVELS:
            target = f"gain_{lv}_mean"
            res, fi = evaluate_split(df, target, split)
            all_out[split][lv] = res
            print(f"\n### {lv} ###")
            header = f"{'模型':<16}{'MAE':>8}{'RMSE':>8}{'R2':>8}{'Spearman':>10}{'TopK25':>9}{'TopK50':>9}"
            print(header)
            for m, v in res.items():
                print(f"{m:<16}{v['MAE']:>8.3f}{v['RMSE']:>8.3f}{v['R2']:>8.3f}"
                      f"{v['Spearman']:>10.3f}{v.get('TopK25',np.nan):>9.3f}{v.get('TopK50',np.nan):>9.3f}")
            if lv == "sr_high":
                print("\n  GBDT特征重要性(sr_high):")
                for k, v in sorted(fi.items(), key=lambda x: -x[1]):
                    print(f"    {k:<22}{v:.4f}")

    with open(os.path.join(OUT, "step4_eval_results.json"), "w", encoding="utf-8") as f:
        json.dump(all_out, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存 step4_eval_results.json")


if __name__ == "__main__":
    main()
