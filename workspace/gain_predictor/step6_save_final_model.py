"""
Step 6 前置: 在全部段级数据上训练最终GBDT增益预测器并落盘, 供阶段二/仿真器集成加载
模型: 每个增强级别一个HistGradientBoosting; 同时保存特征清单与标准化前的岭回归对照
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = r"E:\dev\Code\Python\bones"
OUT = os.path.join(ROOT, "workspace", "gain_predictor")
FEATURES = ["res_idx", "motion_mean", "motion_std", "motion_max", "motion_q90",
            "texture_mean", "texture_std", "texture_max",
            "base_mean", "base_std", "scene_change_count", "high_motion_ratio"]
LEVELS = ["sr_low", "sr_medium", "sr_high"]
RNG = 42


def main():
    df = pd.read_csv(os.path.join(OUT, "dataset_seg.csv"))
    models = {}
    meta = {"features": FEATURES, "levels": LEVELS,
            "res_idx_map": {"240": 0, "360": 1, "480": 2, "720": 3, "1080": 4},
            "train_n": len(df), "note": "段级增益预测器, 输入LR侧内容特征, 输出该段SR后VMAF增益"}
    for lv in LEVELS:
        t = f"gain_{lv}_mean"
        m = HistGradientBoostingRegressor(
            max_depth=3, learning_rate=0.05, max_iter=500, l2_regularization=1.0,
            min_samples_leaf=8, early_stopping=True, validation_fraction=0.15,
            n_iter_no_change=30, random_state=RNG)
        m.fit(df[FEATURES].values, df[t].values)
        models[lv] = m
        # 训练集自检(非泛化, 仅确认拟合)
        pred = m.predict(df[FEATURES].values)
        r2 = 1 - np.sum((df[t].values-pred)**2)/np.sum((df[t].values-df[t].mean())**2)
        print(f"{lv}: 训练集拟合R2={r2:.3f}, n_iter={m.n_iter_}")

    joblib.dump(models, os.path.join(OUT, "gain_predictor_gbdt.pkl"))
    with open(os.path.join(OUT, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print("已保存 gain_predictor_gbdt.pkl + model_meta.json")


if __name__ == "__main__":
    main()
