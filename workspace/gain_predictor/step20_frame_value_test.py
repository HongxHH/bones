"""
帧级价值数据检验（纯数据分析，不跑仿真）。

回答一个转向决策问题：段平均 VMAF 是否掩盖了足够大的帧级可利用差异，
使得"下到帧级 + 对最差帧敏感的质量口径"相比段级有显著更大的选择性增强价值。

两组分析：
  A. 段内帧增益离散度：段平均到底抹平了多少帧间差异（段内 std/极差/越界帧占比，段内 vs 段间方差）。
  B. 固定增强算力预算下，三种选帧策略在不同质量聚合口径上的总质量：
       - seg_oracle : 按段均值排序整段增强（段级能做到的上限，BONES 信息粒度）
       - frame_oracle: 按每帧真实增益排序选帧（帧级理论上限）
       - random     : 预算内随机选帧（等价无差别增强的期望）
     聚合口径：mean（BONES avg-VMAF）/ p10 / p5 / min（最差帧敏感）。
     价值 = frame_oracle 相对 seg_oracle / random 的提升，随预算变化。
数据：data/bbb/per_frame_quality.json，仅 240/360/480，帧增益 = sr_method[f]-base[f]。
"""
import json
import numpy as np
from collections import defaultdict

DATA = "data/bbb/per_frame_quality.json"
RES = [240, 360, 480]
METHODS = ["sr_low", "sr_medium", "sr_high"]
BUDGETS = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
AGG = ["mean", "p10", "p5", "min"]


def load_frames():
    d = json.load(open(DATA, encoding="utf-8"))
    res = d["results"]
    # frames[res][method][seg] = np.array(帧 VMAF)
    frames = {r: {m: {} for m in (["base"] + METHODS)} for r in RES}
    for key, arr in res.items():
        parts = key.split("_")
        r = int(parts[0])
        seg = int(parts[-1])
        method = "_".join(parts[1:-1])
        if r not in RES or method not in (["base"] + METHODS):
            continue
        frames[r][method][seg] = np.asarray(arr, dtype=float)
    return frames


def agg_quality(q, name):
    """q 为全部被观看帧的质量向量（增强后），按口径聚合为单一体验分。"""
    if name == "mean":
        return q.mean()
    if name == "p10":
        return np.percentile(q, 10)
    if name == "p5":
        return np.percentile(q, 5)
    if name == "min":
        return q.min()
    raise ValueError(name)


def analysis_a(frames):
    """段内帧增益离散度 + 段内/段间方差分解（帧级口径）。"""
    print("=" * 78)
    print("A. 段内帧增益离散度（帧增益 = sr_method[f] - base[f]）")
    print("=" * 78)
    out = {}
    for r in RES:
        base = frames[r]["base"]
        for m in METHODS:
            sr = frames[r][m]
            segs = sorted(set(base) & set(sr))
            within_std, within_range, beyond5 = [], [], []
            seg_means, all_gains = [], []
            for s in segs:
                b, e = base[s], sr[s]
                n = min(len(b), len(e))
                g = e[:n] - b[:n]
                g = g[np.isfinite(g)]
                if len(g) < 2:
                    continue
                within_std.append(g.std())
                within_range.append(g.max() - g.min())
                mu = g.mean()
                seg_means.append(mu)
                all_gains.append(g)
                beyond5.append((np.abs(g - mu) > 5).mean())
            all_gains = np.concatenate(all_gains)
            seg_means = np.asarray(seg_means)
            # 方差分解：总方差 = 段间(段均值方差) + 段内(各段方差平均)
            var_between = seg_means.var()
            var_within = np.mean([w ** 2 for w in within_std])
            ratio_within = var_within / (var_between + var_within)
            out[(r, m)] = dict(
                within_std=np.mean(within_std),
                within_range=np.mean(within_range),
                beyond5=np.mean(beyond5),
                var_between=var_between, var_within=var_within,
                within_share=ratio_within,
            )
            print(f"{r}p {m:9s}: 段内std均值={np.mean(within_std):5.2f}  "
                  f"段内极差均值={np.mean(within_range):6.2f}  "
                  f"|帧-段均值|>5的帧占比={np.mean(beyond5)*100:4.1f}%  "
                  f"段内方差占比={ratio_within*100:4.1f}%")
    return out


def build_gain_matrix(frames, r, m):
    """返回 (n_seg, n_frame) 的帧增益矩阵 G 与对齐后的 base 质量 B，仅保留两法都有的段。"""
    base, sr = frames[r]["base"], frames[r][m]
    segs = sorted(set(base) & set(sr))
    G, B = [], []
    for s in segs:
        b, e = base[s], sr[s]
        n = min(len(b), len(e))
        G.append(e[:n] - b[:n])
        B.append(b[:n])
    # 段内帧数应一致，截断到最短
    nf = min(len(x) for x in G)
    G = np.stack([g[:nf] for g in G])
    B = np.stack([b[:nf] for b in B])
    return G, B


def selection_value(G, B, budget, agg_name):
    """
    G,S: (S,F) 帧增益；B: (S,F) 未增强(base)质量。
    预算 = 允许增强的帧占比。返回不同策略增强后全体帧的聚合质量。
    策略：random / seg_gain(按段增益整段选) / frame_gain(按帧增益选) /
          frame_base(按帧base升序,最差帧优先,保护最差分位的近似最优)。
    """
    S, F = G.shape
    n_enh = int(round(budget * S * F))
    flat_gain = G.reshape(-1)
    flat_base = B.reshape(-1)

    def quality_of(mask):
        q = flat_base + np.where(mask, flat_gain, 0.0)
        return agg_quality(q, agg_name)

    base_q = agg_quality(flat_base, agg_name)

    # frame oracle（按平均增益选帧）
    mask_f = np.zeros(S * F, dtype=bool)
    mask_f[np.argsort(-flat_gain)[:n_enh]] = True
    q_frame_gain = quality_of(mask_f)

    # frame base（按 base 质量升序，最差帧优先——保护最差分位的近似最优）
    mask_fb = np.zeros(S * F, dtype=bool)
    mask_fb[np.argsort(flat_base)[:n_enh]] = True
    q_frame_base = quality_of(mask_fb)

    # seg oracle（按段均值增益排序整段选，段级粒度上限）
    seg_mean = G.mean(axis=1)
    mask_s = np.zeros((S, F), dtype=bool)
    remain = n_enh
    for si in np.argsort(-seg_mean):
        if remain <= 0:
            break
        take = min(F, remain)
        mask_s[si, :take] = True
        remain -= take
    q_seg = quality_of(mask_s.reshape(-1))

    # random 期望
    rng = np.random.default_rng(0)
    q_rand = []
    for _ in range(50):
        mask_r = np.zeros(S * F, dtype=bool)
        mask_r[rng.choice(S * F, n_enh, replace=False)] = True
        q_rand.append(quality_of(mask_r))
    q_rand = np.mean(q_rand)
    return base_q, q_rand, q_seg, q_frame_gain, q_frame_base


def analysis_b(frames):
    print("\n" + "=" * 78)
    print("B. 固定增强预算下，帧级 相对 段级 的额外价值（两种选帧准则）")
    print("=" * 78)
    summary = {}
    for r in RES:
        for m in METHODS:
            G, B = build_gain_matrix(frames, r, m)
            print(f"\n--- {r}p {m}（{G.shape[0]}段×{G.shape[1]}帧，base："
                  f"mean={B.mean():.2f} p5={np.percentile(B,5):.2f}）---")
            for bud in [0.3, 0.5, 0.7]:
                rec = {}
                line = [f"预算{bud*100:.0f}%"]
                for a in AGG:
                    bq, qr, qs, qfg, qfb = selection_value(G, B, bud, a)
                    rec[a] = dict(frame_gain_over_seg=qfg - qs,
                                  frame_base_over_seg=qfb - qs,
                                  seg_over_rand=qs - qr)
                    line.append(f"{a}:按增益帧>段{qfg-qs:+5.2f} 救差帧>段{qfb-qs:+5.2f}")
                summary[(r, m, bud)] = rec
                print("  " + "  ".join(line))
    return summary


def analysis_c(frames, content_features):
    """尺度对比 + 最差帧的段聚集度（判断'差'是段级还是帧级现象）。"""
    print("\n" + "=" * 78)
    print("C. 增益尺度对比 与 最差帧的段聚集度")
    print("=" * 78)
    scale = {}
    for r in RES:
        for m in METHODS:
            G, _ = build_gain_matrix(frames, r, m)
            between = G.mean(axis=1).std()
            within = G.std(axis=1).mean()
            scale[f"{r}_{m}"] = dict(between_std=float(between), within_std=float(within),
                                     ratio=float(within / between))
            print(f"{r}p {m:9s}: 段间std={between:5.2f} 段内std={within:4.2f} "
                  f"段内/段间={within/between:.3f}")
    cluster = {}
    for r in RES:
        base = frames[r]["base"]
        segs = sorted(base)
        Bs, SID = [], []
        for s in segs:
            Bs.append(base[s]); SID.append(np.full(len(base[s]), s))
        B = np.concatenate(Bs); SIDv = np.concatenate(SID)
        thr = np.percentile(B, 5)
        low_seg = SIDv[B <= thr]
        u, c = np.unique(low_seg, return_counts=True)
        # 聚集度：承载最差5%帧所需的最少段数占比；段内最差帧占比
        seg_share = len(u) / len(segs)
        intra = float(c.mean() / 120)
        cluster[str(r)] = dict(p5_thr=float(thr), n_seg_carry_p5=int(len(u)),
                               seg_share=float(seg_share), avg_p5frames_per_carrying_seg=float(c.mean()))
        print(f"{r}p: p5阈值={thr:5.2f}，最差5%帧由{len(u)}段承载（占段数{seg_share*100:.1f}%），"
              f"这些段平均有{intra*100:.0f}%的帧落在最差5%（高=整段塌陷而非零散差帧）")
    return scale, cluster


def main():
    d = json.load(open(DATA, encoding="utf-8"))
    frames = load_frames()
    a = analysis_a(frames)
    b = analysis_b(frames)
    scale, cluster = analysis_c(frames, d["content_features"])
    out = {
        "within": {f"{r}_{m}": {k: float(v) for k, v in dd.items()}
                   for (r, m), dd in a.items()},
        "scale": scale, "worst_frame_cluster": cluster,
    }
    with open("workspace/gain_predictor/step20_frame_value.json", "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n已保存 step20_frame_value.json")


if __name__ == "__main__":
    main()
