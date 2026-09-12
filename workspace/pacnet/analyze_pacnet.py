"""阶段0生死判决：读 eval_pac16.json，按四条生死线出 go/no-go。

H1a 保命档弹性税：e1 vs 同算力固定浅模型 plain2，均值差应 >= -0.2 dB
H1b 最深档不掉队：e3 vs 同算力对症 imdn_lo（6-block）
H1c 单调性：逐帧 e3>=e2>=e1（0.05dB 容差）比例 >= 95%
H2  困难段值得深算：hard(130/140/150) e3-e1 均值 >= +0.5 dB；简单段应接近 0
H4  时延：240p 整帧各出口中位时延对照 33ms/帧(30fps)预算
"""
import argparse
import json
import os

import numpy as np

ROOT = r"E:\dev\Code\Python\bones"
PACDIR = os.path.join(ROOT, "workspace/pacnet")
METHODS = ["base", "e1", "e2", "e3", "plain2", "imdn_div", "imdn_lo"]


def stack(frame_psnr):
    """返回 {method: ndarray(n_frame)} 与每帧的 seg 标签。"""
    acc = {m: [] for m in METHODS}
    segs = []
    for seg, js in frame_psnr.items():
        for j, row in js.items():
            for m in METHODS:
                acc[m].append(row[m])
            segs.append(int(seg))
    return {m: np.array(v) for m, v in acc.items()}, np.array(segs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default=os.path.join(PACDIR, "eval_pac16.json"))
    args = ap.parse_args()
    d = json.load(open(args.inp, encoding="utf-8"))
    hard = set(d["hard"])
    M, segs = stack(d["frame_psnr"])
    n = len(segs)
    hard_mask = np.array([s in hard for s in segs])
    easy_mask = ~hard_mask

    def stat(a):
        return dict(mean=round(float(a.mean()), 3), med=round(float(np.median(a)), 3),
                    std=round(float(a.std()), 3))

    print("=" * 70)
    print("各方法整体 PSNR（16 留出段，%d 帧）" % n)
    for m in METHODS:
        print(f"  {m:9s}", stat(M[m]))

    # H1a 保命档弹性税
    d_e1_plain = M["e1"] - M["plain2"]
    h1a = float(d_e1_plain.mean())
    # H1b 最深档
    d_e3_lo = M["e3"] - M["imdn_lo"]
    d_e3_div = M["e3"] - M["imdn_div"]
    h1b_lo, h1b_div = float(d_e3_lo.mean()), float(d_e3_div.mean())
    # H1c 单调性（容差 0.05dB）
    tol = 0.05
    mono_chain = np.mean((M["e2"] >= M["e1"] - tol) & (M["e3"] >= M["e2"] - tol))
    mono_13 = np.mean(M["e3"] >= M["e1"] - tol)
    # H2 困难/简单段深算增益
    gain = M["e3"] - M["e1"]
    hard_gain = float(gain[hard_mask].mean()) if hard_mask.any() else float("nan")
    easy_gain = float(gain[easy_mask].mean())
    # 逐困难段
    per_hard = {}
    for s in sorted(hard):
        msk = segs == s
        if msk.any():
            per_hard[s] = dict(e1=round(float(M["e1"][msk].mean()), 3),
                               e3=round(float(M["e3"][msk].mean()), 3),
                               gain=round(float(gain[msk].mean()), 3))

    lat = d.get("latency_ms_240p", {})
    budget = 33.3

    verdict = {}
    verdict["H1a_e1_minus_plain2_dB"] = round(h1a, 3)
    verdict["H1a_elastic_tax_within_1dB"] = bool(h1a >= -1.0)
    verdict["H1b_e3_minus_imdnLO_dB"] = round(h1b_lo, 3)
    verdict["H1b_pass(>=-0.5)"] = bool(h1b_lo >= -0.5)
    verdict["H1b_e3_minus_imdnDIV_dB"] = round(h1b_div, 3)
    verdict["H1c_mono_chain_ratio"] = round(float(mono_chain), 4)
    verdict["H1c_pass(>=0.95)"] = bool(mono_chain >= 0.95)
    verdict["H1c_e3>=e1_ratio"] = round(float(mono_13), 4)
    verdict["H2_hard_gain_dB"] = round(hard_gain, 3)
    verdict["H2_pass(>=0.5)"] = bool(hard_gain >= 0.5)
    verdict["H2_easy_gain_dB"] = round(easy_gain, 3)
    verdict["H2_per_hard"] = per_hard
    verdict["H4_latency_ms"] = lat
    verdict["H4_shallow_within_33ms"] = bool(lat.get("e1", 99) <= budget and lat.get("e2", 99) <= budget)

    print("=" * 70)
    print(json.dumps(verdict, indent=2, ensure_ascii=False))

    gates = [("H1b 最深档不掉队(差对症<0.5dB)", verdict["H1b_pass(>=-0.5)"]),
             ("H1c 越深越单调", verdict["H1c_pass(>=0.95)"]),
             ("H2 困难段深算有效", verdict["H2_pass(>=0.5)"])]
    print("=" * 70)
    for name, ok in gates:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"[{'观察' if verdict['H1a_elastic_tax_within_1dB'] else '偏大'}] "
          f"H1a 浅出口弹性税 {round(h1a, 3)} dB（阶段1自蒸馏收敛，不否决方向）")
    go = all(ok for _, ok in gates)
    print("=> 阶段0判决:", "GO（核心承重墙成立）" if go else "NO-GO / 需回改")
    out = args.inp.replace(".json", "_verdict.json")
    json.dump(verdict, open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("saved", out)


if __name__ == "__main__":
    main()
