"""Step 11: 收缩系数alpha扫描, 验证预测噪声假设并寻找最优激进程度"""
import os, sys, glob, json, argparse
import numpy as np
ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, ROOT)
os.chdir(os.path.join(ROOT, "control"))
from control.bones_selective import SelectiveBONES

MANI = "../data/bbb/bbb_video_info.json"; ENH = "../data/bbb/imdn_bbb_2080ti.json"
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def traces():
    t = glob.glob("../data/trace/3Glogs/*.json") + glob.glob("../data/trace/4Glogs/*.json")
    t += [p for p in glob.glob("../data/trace/*.json") if os.path.basename(p) != "dummy.json"]
    return sorted(t)


def args_for(tr, mode, alpha):
    return argparse.Namespace(
        log_path=os.path.join(ROOT, "workspace", "gain_predictor", "tmp_log.txt"),
        manifest_path=MANI, trace_path=tr, enhance_path=ENH, video_length=None,
        throughput="ewma", download_buffer=60, enhance_buffer=60,
        bandwidth_multiplier=1.0, speed_multiplier=1.0, verbose=False,
        gamma_p=10, V_multiplier=1.0, monitor=False, autotune=False,
        quick_start=False, quality_mode=mode, alpha=alpha)


def run(tr, mode, alpha=1.0):
    sim = SelectiveBONES(args_for(tr, mode, alpha)); sim.run(); rep = sim.report()
    hd, he = rep["download history"], rep["enhance history"]
    gpu = sum(float(sim.enhance_time[hd[i]][he[i]]) for i in range(len(hd)))
    return rep["qoe"], rep["average quality vmaf"], rep["average oscillation vmaf"], gpu


def main():
    TR = traces()
    # static 一次(等价alpha=0), oracle一次, pred各alpha
    res = {"static": [], "oracle": [], **{f"a{a}": [] for a in ALPHAS if a > 0}}
    for tr in TR:
        q, v, o, g = run(tr, "static"); res["static"].append((q, v, o, g))
        q, v, o, g = run(tr, "oracle"); res["oracle"].append((q, v, o, g))
        for a in ALPHAS:
            if a == 0: continue
            q, v, o, g = run(tr, "pred", a); res[f"a{a}"].append((q, v, o, g))

    def mean(key, idx): return float(np.mean([x[idx] for x in res[key]]))
    print(f"{'方案':<10}{'QoE':>10}{'VMAF':>10}{'振荡':>10}{'GPU_ms':>12}{'ΔQoE':>9}")
    base_q = mean("static", 0)
    order = ["static"] + [f"a{a}" for a in ALPHAS if a > 0] + ["oracle"]
    out = {}
    for k in order:
        q = mean(k, 0); v = mean(k, 1); o = mean(k, 2); g = mean(k, 3)
        out[k] = {"qoe": q, "vmaf": v, "osc": o, "gpu_ms": g}
        print(f"{k:<10}{q:>10.3f}{v:>10.3f}{o:>10.3f}{g:>12.0f}{q-base_q:>+9.3f}")
    # pred各alpha对static的QoE胜率
    print("\npred 各alpha 对 static 的QoE胜率:")
    sq = np.array([x[0] for x in res["static"]])
    for a in ALPHAS:
        if a == 0: continue
        pq = np.array([x[0] for x in res[f"a{a}"]])
        print(f"  alpha={a:<4} 胜率={np.mean(pq > sq)*100:.1f}%  ΔQoE均值={np.mean(pq-sq):+.3f}")
    json.dump(out, open(os.path.join(ROOT, "workspace/gain_predictor/step11_alpha_sweep.json"), "w"),
              indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
