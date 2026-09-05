"""
Step 10: 131条真实轨迹(86 3G + 40 4G + 5根目录)全量对比 static/oracle/pred
指标: QoE/VMAF/振荡/卡顿 + 增强段占比 + GPU增强总耗时(算力代理)
"""
import os, sys, glob, json, argparse, time
import numpy as np
ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, ROOT)
os.chdir(os.path.join(ROOT, "control"))
from control.bones_selective import SelectiveBONES

MANI = "../data/bbb/bbb_video_info.json"
ENH = "../data/bbb/imdn_bbb_2080ti.json"
MODES = ["static", "oracle", "pred"]


def collect_traces():
    traces = []
    traces += glob.glob("../data/trace/3Glogs/*.json")
    traces += glob.glob("../data/trace/4Glogs/*.json")
    for p in glob.glob("../data/trace/*.json"):
        if os.path.basename(p) != "dummy.json":
            traces.append(p)
    return sorted(traces)


def make_args(trace, mode):
    return argparse.Namespace(
        log_path=os.path.join(ROOT, "workspace", "gain_predictor", "tmp_log.txt"),
        manifest_path=MANI, trace_path=trace, enhance_path=ENH, video_length=None,
        throughput="ewma", download_buffer=60, enhance_buffer=60,
        bandwidth_multiplier=1.0, speed_multiplier=1.0, verbose=False,
        gamma_p=10, V_multiplier=1.0, monitor=False, autotune=False,
        quick_start=False, quality_mode=mode)


def run_once(trace, mode):
    sim = SelectiveBONES(make_args(trace, mode))
    sim.run()
    rep = sim.report()
    hd, he = rep["download history"], rep["enhance history"]
    # GPU增强总耗时(ms): 逐段按实际(下载档,增强方法)查enhance_time累加
    gpu_ms = sum(float(sim.enhance_time[hd[i]][he[i]]) for i in range(len(hd)))
    n_enh = sum(1 for x in he if x != 0)
    return {
        "qoe": float(rep["qoe"]), "vmaf": float(rep["average quality vmaf"]),
        "osc": float(rep["average oscillation vmaf"]),
        "freeze": float(rep["freeze time"]),
        "enh_ratio": n_enh / len(he), "gpu_ms": gpu_ms}


def main():
    traces = collect_traces()
    print(f"轨迹总数: {len(traces)}")
    results = {m: [] for m in MODES}
    t0 = time.time()
    for i, tr in enumerate(traces):
        row = {"trace": os.path.relpath(tr, "..").replace("\\", "/")}
        for m in MODES:
            row[m] = run_once(tr, m)
        for m in MODES:
            results[m].append(row[m])
        if (i+1) % 20 == 0:
            print(f"  已完成 {i+1}/{len(traces)}, 耗时{time.time()-t0:.0f}s")

    def agg(metric):
        return {m: float(np.mean([r[metric] for r in results[m]])) for m in MODES}

    summary = {k: agg(k) for k in ["qoe", "vmaf", "osc", "freeze", "enh_ratio", "gpu_ms"]}

    # pred/oracle 相对 static 的逐轨迹差值与胜率
    def pairdiff(a, b, metric):
        d = [results[a][i][metric]-results[b][i][metric] for i in range(len(traces))]
        win = sum(1 for x in d if x > 1e-6)/len(d)
        return float(np.mean(d)), float(np.std(d)), win
    summary["pred_vs_static"] = {k: dict(zip(["mean", "std", "win"], pairdiff("pred", "static", k)))
                                 for k in ["qoe", "vmaf", "osc", "enh_ratio", "gpu_ms"]}
    summary["oracle_vs_static"] = {k: dict(zip(["mean", "std", "win"], pairdiff("oracle", "static", k)))
                                   for k in ["qoe", "vmaf", "osc", "enh_ratio", "gpu_ms"]}

    # 分网络类型
    def net_of(t):
        if "3Glogs" in t: return "3G"
        if "4Glogs" in t: return "4G"
        return "root"
    for net in ["3G", "4G", "root"]:
        idx = [i for i, tr in enumerate(traces) if net_of(tr.replace("\\", "/")) == net]
        summary[f"net_{net}"] = {
            "n": len(idx),
            "qoe": {m: float(np.mean([results[m][i]["qoe"] for i in idx])) for m in MODES},
            "gpu_ms": {m: float(np.mean([results[m][i]["gpu_ms"] for i in idx])) for m in MODES}}

    out = os.path.join(ROOT, "workspace/gain_predictor/step10_sim_comparison.json")
    json.dump({"summary": summary, "per_trace": [
        {"trace": os.path.relpath(tr, "..").replace("\\", "/"),
         **{m: results[m][i] for m in MODES}} for i, tr in enumerate(traces)]},
        open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\n===== 全量均值 =====")
    print(f"{'指标':<10}{'static':>12}{'oracle':>12}{'pred':>12}")
    for k in ["qoe", "vmaf", "osc", "freeze", "enh_ratio", "gpu_ms"]:
        s = summary[k]
        print(f"{k:<10}{s['static']:>12.3f}{s['oracle']:>12.3f}{s['pred']:>12.3f}")
    print("\n===== pred 相对 static =====")
    for k, v in summary["pred_vs_static"].items():
        print(f"  {k:<10} Δ={v['mean']:+.3f}±{v['std']:.3f} 胜率={v['win']*100:.1f}%")
    print("\n===== oracle 相对 static(上限) =====")
    for k, v in summary["oracle_vs_static"].items():
        print(f"  {k:<10} Δ={v['mean']:+.3f}±{v['std']:.3f} 胜率={v['win']*100:.1f}%")
    print("\n分网络QoE:")
    for net in ["3G", "4G", "root"]:
        s = summary[f"net_{net}"]
        print(f"  {net}(n={s['n']}): static={s['qoe']['static']:.2f} oracle={s['qoe']['oracle']:.2f} pred={s['qoe']['pred']:.2f}")
    print(f"\n已保存 {out}, 总耗时{time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
