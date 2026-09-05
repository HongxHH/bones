"""Step 13: 解耦模式全量对比 + 下载码率一致性自检
static(BONES) | oracle/pred 联合 | oracle/pred 解耦(+alpha收缩)
关键自检: 解耦模式 download history 必须与 static 完全一致(freeze也应一致)"""
import os, sys, glob, json, argparse
import numpy as np
ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, ROOT); os.chdir(os.path.join(ROOT, "control"))
from control.bones_selective import SelectiveBONES

MANI = "../data/bbb/bbb_video_info.json"; ENH = "../data/bbb/imdn_bbb_2080ti.json"
CONFIGS = [
    ("static", "static", 1.0, False),
    ("oracle_joint", "oracle", 1.0, False),
    ("pred_joint", "pred", 1.0, False),
    ("oracle_dc", "oracle", 1.0, True),
    ("pred_dc_a1", "pred", 1.0, True),
    ("pred_dc_a075", "pred", 0.75, True),
    ("pred_dc_a05", "pred", 0.5, True),
]


def traces():
    t = glob.glob("../data/trace/3Glogs/*.json")+glob.glob("../data/trace/4Glogs/*.json")
    t += [p for p in glob.glob("../data/trace/*.json") if os.path.basename(p) != "dummy.json"]
    return sorted(t)


def mk(tr, mode, alpha, dc):
    return argparse.Namespace(log_path="../workspace/gain_predictor/tmp_log.txt",
        manifest_path=MANI, trace_path=tr, enhance_path=ENH, video_length=None,
        throughput="ewma", download_buffer=60, enhance_buffer=60,
        bandwidth_multiplier=1.0, speed_multiplier=1.0, verbose=False,
        gamma_p=10, V_multiplier=1.0, monitor=False, autotune=False,
        quick_start=False, quality_mode=mode, alpha=alpha, decouple=dc)


def run(tr, mode, alpha, dc):
    s = SelectiveBONES(mk(tr, mode, alpha, dc)); s.run(); r = s.report()
    hd, he = r["download history"], r["enhance history"]
    gpu = sum(float(s.enhance_time[hd[i]][he[i]]) for i in range(len(hd)))
    return {"qoe": r["qoe"], "vmaf": r["average quality vmaf"],
            "osc": r["average oscillation vmaf"], "freeze": r["freeze time"],
            "gpu": gpu, "down": list(hd), "enh": list(he)}


def main():
    TR = traces()
    allres = {name: [] for name, *_ in CONFIGS}
    dl_mismatch = 0
    for tr in TR:
        out = {}
        for name, mode, a, dc in CONFIGS:
            out[name] = run(tr, mode, a, dc)
        # 自检: 解耦模式下载码率序列与static一致
        for name in ["oracle_dc", "pred_dc_a1", "pred_dc_a075", "pred_dc_a05"]:
            if out[name]["down"] != out["static"]["down"]:
                dl_mismatch += 1
        for name in allres:
            allres[name].append(out[name])
    print(f"轨迹数={len(TR)}, 解耦下载码率不一致次数={dl_mismatch} (应为0)")

    keys = ["qoe", "vmaf", "osc", "freeze", "gpu"]
    base = {k: np.mean([r[k] for r in allres["static"]]) for k in keys}
    print(f"\n{'方案':<14}{'QoE':>9}{'ΔQoE':>8}{'VMAF':>8}{'振荡':>7}{'freeze_s':>10}{'GPU_ms':>10}{'胜率':>7}")
    sq = np.array([r["qoe"] for r in allres["static"]])
    summary = {}
    for name, *_ in CONFIGS:
        rr = allres[name]
        q = np.mean([r["qoe"] for r in rr]); v = np.mean([r["vmaf"] for r in rr])
        o = np.mean([r["osc"] for r in rr]); f = np.mean([r["freeze"] for r in rr])/1000
        g = np.mean([r["gpu"] for r in rr])
        qarr = np.array([r["qoe"] for r in rr]); win = np.mean(qarr > sq)*100
        summary[name] = {"qoe": q, "dqoe": q-base["qoe"], "vmaf": v, "osc": o,
                         "freeze_s": f, "gpu_ms": g, "win": win}
        print(f"{name:<14}{q:>9.3f}{q-base['qoe']:>+8.3f}{v:>8.3f}{o:>7.3f}{f:>10.2f}{g:>10.0f}{win:>6.1f}%")

    # 分网络
    def net(t): return "3G" if "3Glogs" in t else ("4G" if "4Glogs" in t else "root")
    nets = [net(t.replace("\\", "/")) for t in TR]
    print("\n分网络 ΔQoE(相对static):")
    for nb in ["3G", "4G", "root"]:
        idx = [i for i, x in enumerate(nets) if x == nb]
        line = f"  {nb}(n={len(idx)}): "
        for name, *_ in CONFIGS:
            d = np.mean([allres[name][i]["qoe"]-allres["static"][i]["qoe"] for i in idx])
            line += f"{name}={d:+.2f}  "
        print(line)

    json.dump({"summary": summary, "dl_mismatch": dl_mismatch},
              open(os.path.join(ROOT, "workspace/gain_predictor/step13_decouple.json"), "w"),
              indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
