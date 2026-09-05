"""
Step 8(修正版): 静态表口径增益预测器, 5折OOF生成与仿真器 vmaf_enh_seg 完全同形的
预测质量矩阵 (159段 × 5码率档 × 5方法[不增强+4增强方法])。

仿真器真实结构(经_parse_enhance核实):
  base_quality_vmaf   (4,159): 240/360/480/720 四档下载质量
  enhanced_quality    (3,4,159): 仅240/360/480三档 × 4增强方法; 720/1080增强=-inf
输出:
  pred_enh_oof.npy  OOF预测矩阵(本方法, 无乐观偏差)
  oracle_enh.npy    段级真值矩阵(上限)
  static_avg.npy    全局平均矩阵(BONES基线, 与 vmaf_enh_avg 逐元素一致)
  gain_predictor_static.pkl 全量模型
"""
import os, json
import numpy as np
import pandas as pd
import joblib
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = r"E:\dev\Code\Python\bones"
ENH = os.path.join(ROOT, "data/bbb/imdn_bbb_2080ti.json")
PF = os.path.join(ROOT, "data/bbb/per_frame_quality.json")
OUT = os.path.join(ROOT, "workspace/gain_predictor")
NUM_SEG, RES_N, METH_N = 159, 5, 5          # 5档码率, 不增强+4方法
ENH_RES = 3                                  # 仅前3档(240/360/480)有增强质量
N_METH = 4                                   # 4个增强方法
RNG = 42
FEATURES = ["motion_mean", "motion_std", "motion_max", "motion_q90",
            "texture_mean", "texture_std", "texture_max",
            "scene_change_count", "base_q"]
TARGETS = [f"gain_m{m+1}" for m in range(N_METH)]


def st(a, p):
    return {f"{p}_mean": np.mean(a), f"{p}_std": np.std(a), f"{p}_max": np.max(a),
            f"{p}_q90": np.quantile(a, 0.9)}


def build_df():
    enh = json.load(open(ENH)); pf = json.load(open(PF))
    base = np.array(enh["base_quality_vmaf"])      # (4,159)
    eq = np.array(enh["enhanced_quality_vmaf"])    # (3,4,159)
    cf = pf["content_features"]
    rows = []
    for seg in range(158):
        ck = str(seg)
        if ck not in cf:
            continue
        mo = np.array(cf[ck]["motion"], float); tx = np.array(cf[ck]["texture"], float)
        sc = np.array(cf[ck]["scene_change"], bool)
        feat = {"seg": seg}; feat.update(st(mo, "motion")); feat.update(st(tx, "texture"))
        feat["scene_change_count"] = int(sc.sum())
        for r in range(ENH_RES):                   # 只对有增强质量的3档建样本
            row = dict(feat); b = base[r][seg]; row["base_q"] = float(b); row["r"] = r
            for m in range(N_METH):
                row[f"gain_m{m+1}"] = float(eq[r][m][seg] - b)
            rows.append(row)
    return pd.DataFrame(rows), base, eq


def make_model():
    return HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=500, l2_regularization=1.0,
        min_samples_leaf=8, early_stopping=True, validation_fraction=0.15,
        n_iter_no_change=30, random_state=RNG)


def build_oracle(base, eq):
    """严格复现 _parse_enhance 的张量拼装, 得到 (159,5,5) 真值矩阵"""
    down = base.T.copy()                                   # (159,4)
    down = np.concatenate([down, np.full((NUM_SEG, 1), 100.0)], axis=1)  # (159,5)
    e = eq.transpose((2, 0, 1)).copy()                     # (159,3,4)
    e[e < 0] = -np.inf
    e = np.concatenate([e, np.full((NUM_SEG, RES_N-ENH_RES, N_METH), -np.inf)], axis=1)  # (159,5,4)
    m = np.concatenate([down[:, :, None], e], axis=2)      # (159,5,5)
    return m


def main():
    df, base, eq = build_df()
    segs = sorted(df["seg"].unique())
    oof = {t: np.zeros(len(df)) for t in TARGETS}
    full = {}
    cv_report = {}
    for t in TARGETS:
        for tri, tei in KFold(5, shuffle=True, random_state=RNG).split(segs):
            trs, tes = set(np.array(segs)[tri]), set(np.array(segs)[tei])
            tr = df[df["seg"].isin(trs)]; te = df[df["seg"].isin(tes)]
            mdl = make_model(); mdl.fit(tr[FEATURES].values, tr[t].values)
            oof[t][te.index] = mdl.predict(te[FEATURES].values)
        fm = make_model(); fm.fit(df[FEATURES].values, df[t].values); full[t] = fm
        # CV精度
        yt, yp = df[t].values, oof[t]
        r2 = 1-np.sum((yt-yp)**2)/np.sum((yt-yt.mean())**2)
        cv_report[t] = {"MAE": float(np.mean(np.abs(yt-yp))), "R2": float(r2),
                        "SP": float(spearmanr(yt, yp).statistic)}
    for t in TARGETS:
        d = cv_report[t]
        print(f"{t}: MAE={d['MAE']:.3f} R2={d['R2']:.3f} Spearman={d['SP']:.3f}")
    for t in TARGETS:
        df[t+"_oof"] = oof[t]

    oracle = build_oracle(base, eq)
    invalid = (oracle == -np.inf)
    static_avg = np.broadcast_to(
        np.nanmean(np.where(invalid, np.nan, oracle), axis=0),
        oracle.shape).copy()
    static_avg[invalid] = -np.inf        # 无效增强位与仿真器一致保持 -inf(非nan)
    pred = static_avg.copy()
    for _, row in df.iterrows():
        s, r = int(row["seg"]), int(row["r"])
        for m in range(N_METH):
            tv = eq[r][m][s]
            if tv < 0:
                continue
            pred[s, r, m+1] = row["base_q"] + row[f"gain_m{m+1}_oof"]
    pred[invalid] = -np.inf

    np.save(os.path.join(OUT, "pred_enh_oof.npy"), pred)
    np.save(os.path.join(OUT, "oracle_enh.npy"), oracle)
    np.save(os.path.join(OUT, "static_avg.npy"), static_avg)
    joblib.dump({"models": full, "features": FEATURES},
                os.path.join(OUT, "gain_predictor_static.pkl"))

    mask = (oracle[:, :ENH_RES, 1:] != -np.inf)
    err = np.abs(pred[:, :ENH_RES, 1:][mask]-oracle[:, :ENH_RES, 1:][mask])
    sp = spearmanr(pred[:, :ENH_RES, 1:][mask], oracle[:, :ENH_RES, 1:][mask]).statistic
    gavg = np.nanmean(np.where(oracle == -np.inf, np.nan, oracle), axis=0)
    print(f"\n矩阵{pred.shape}; OOF有效增强位 MAE={err.mean():.3f} Spearman={sp:.4f}; "
          f"static与全局avg最大差={np.nanmax(np.abs(static_avg[0]-gavg)):.2e}")
    print("static_avg[0](240档):", np.round(static_avg[0], 2))


if __name__ == "__main__":
    main()
