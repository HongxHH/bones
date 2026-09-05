"""Step 14: 置信度门控(gating) margin扫描, 并采集决策优势gap分布"""
import os, sys, glob, json, argparse
import numpy as np
ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, ROOT); os.chdir(os.path.join(ROOT, "control"))
from control.bones_selective import SelectiveBONES

MANI = "../data/bbb/bbb_video_info.json"; ENH = "../data/bbb/imdn_bbb_2080ti.json"
MARGINS = [0.0, 0.3, 0.7, 1.5, 3.0, 6.0]


def traces():
    t = glob.glob("../data/trace/3Glogs/*.json")+glob.glob("../data/trace/4Glogs/*.json")
    t += [p for p in glob.glob("../data/trace/*.json") if os.path.basename(p) != "dummy.json"]
    return sorted(t)


def mk(tr, mode, gating, margin, alpha=1.0):
    return argparse.Namespace(log_path="../workspace/gain_predictor/tmp_log.txt",
        manifest_path=MANI, trace_path=tr, enhance_path=ENH, video_length=None,
        throughput="ewma", download_buffer=60, enhance_buffer=60,
        bandwidth_multiplier=1.0, speed_multiplier=1.0, verbose=False,
        gamma_p=10, V_multiplier=1.0, monitor=False, autotune=False,
        quick_start=False, quality_mode=mode, alpha=alpha, decouple=False,
        gating=gating, margin=margin)


def run(tr, mode, gating=False, margin=0.0, alpha=1.0):
    s = SelectiveBONES(mk(tr, mode, gating, margin, alpha)); s.run(); r = s.report()
    hd, he = r["download history"], r["enhance history"]
    gpu = sum(float(s.enhance_time[hd[i]][he[i]]) for i in range(len(hd)))
    return r["qoe"], r["average quality vmaf"], r["average oscillation vmaf"], gpu, s.gating_gaps


def main():
    TR = traces()
    stat = np.array([run(tr, "static")[:4] for tr in TR], float)
    orac = np.array([run(tr, "oracle")[:4] for tr in TR], float)
    res = {}
    allgaps = []
    for mg in MARGINS:
        rows, gaps = [], []
        for tr in TR:
            q, v, o, g, gp = run(tr, "pred", True, mg)
            rows.append((q, v, o, g)); gaps += gp
        res[mg] = np.array(rows, float)
        if mg == 0.0:
            allgaps = gaps
    gaps = np.array(allgaps)
    print("margin=0 决策翻转段的优势gap分布: n=%d" % len(gaps))
    if len(gaps):
        p = np.percentile(gaps, [10, 25, 50, 75, 90])
        print("  p10=%.5f p25=%.5f 中位=%.5f p75=%.5f p90=%.5f" % (p[0], p[1], p[2], p[3], p[4]))

    def line(name, M):
        dq = M[:, 0]-stat[:, 0]
        trim = np.sort(dq)[2:-2].mean()  # 去掉最差/最好各2条
        return ("%-14s QoE=%8.3f ΔQoE均值=%+.3f 中位=%+.3f 去极端=%+.3f 胜率=%.1f%% osc=%.3f GPU=%d" % (
            name, M[:, 0].mean(), dq.mean(), np.median(dq), trim,
            np.mean(dq > 1e-9)*100, M[:, 2].mean(), M[:, 3].mean()))
    print("\n" + line("static", stat))
    print(line("oracle_joint", orac))
    for mg in MARGINS:
        print(line(f"pred_mg={mg:g}", res[mg]))

    out = {"gap_percentile": np.percentile(gaps, [10, 25, 50, 75, 90]).tolist() if len(gaps) else [],
           "static_qoe": stat[:,0].mean(), "oracle_qoe": orac[:,0].mean(),
           "margin": {str(mg): {"qoe": res[mg][:,0].mean(),
                                "dqoe_mean": (res[mg][:,0]-stat[:,0]).mean(),
                                "dqoe_median": float(np.median(res[mg][:,0]-stat[:,0])),
                                "win": float(np.mean(res[mg][:,0]-stat[:,0] > 1e-9))}
                      for mg in MARGINS}}
    json.dump(out, open(os.path.join(ROOT, "workspace/gain_predictor/step14_gating.json"), "w"),
              indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
