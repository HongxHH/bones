"""Step 17: 收缩准入(以BONES平均增益为锚) x 阈值, 寻找"QoE持平、尽量省算力"的pred工作点
准入作用在向static平均收缩后的增益上: alpha越小越保守(只砍明确低于平均的组合)"""
import os, sys, glob, json, argparse
import numpy as np
ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, ROOT); os.chdir(os.path.join(ROOT, "control"))
from control.bones_selective import SelectiveBONES

MANI = "../data/bbb/bbb_video_info.json"; ENH = "../data/bbb/imdn_bbb_2080ti.json"
THRESH = [6, 8, 10, 12, 14]
ALPHAS = [0.3, 0.5, 0.7, 1.0]


def traces():
    t = glob.glob("../data/trace/3Glogs/*.json")+glob.glob("../data/trace/4Glogs/*.json")
    t += [p for p in glob.glob("../data/trace/*.json") if os.path.basename(p) != "dummy.json"]
    return sorted(t)


def mk(tr, mode, thresh, alpha):
    return argparse.Namespace(log_path="../workspace/gain_predictor/tmp_log.txt",
        manifest_path=MANI, trace_path=tr, enhance_path=ENH, video_length=None,
        throughput="ewma", download_buffer=60, enhance_buffer=60,
        bandwidth_multiplier=1.0, speed_multiplier=1.0, verbose=False,
        gamma_p=10, V_multiplier=1.0, monitor=False, autotune=False,
        quick_start=False, quality_mode=mode, alpha=alpha, decouple=False,
        gating=False, margin=0.0, budget_thresh=thresh)


def run(tr, mode, thresh, alpha):
    s = SelectiveBONES(mk(tr, mode, thresh, alpha)); s.run(); r = s.report()
    hd, he = r["download history"], r["enhance history"]
    gpu = sum(float(s.enhance_time[hd[i]][he[i]]) for i in range(len(hd)))
    return r["qoe"], gpu


def main():
    TR = traces()
    base = np.array([run(tr, "static", None, 1.0) for tr in TR], float)
    bq, bg = base[:, 0].mean(), base[:, 1].mean()
    # oracle参照(真值准入, 与alpha无关)
    orc = {}
    for th in THRESH:
        M = np.array([run(tr, "oracle", th, 1.0) for tr in TR], float)
        orc[th] = (M[:, 0].mean()-bq, (bg-M[:,1].mean())/bg*100, np.mean((M[:,0]-base[:,0])>1e-9))
    grid = {}
    for a in ALPHAS:
        for th in THRESH:
            M = np.array([run(tr, "pred", th, a) for tr in TR], float)
            dq = M[:,0]-base[:,0]
            grid[(a,th)] = (M[:,0].mean()-bq, (bg-M[:,1].mean())/bg*100,
                            np.mean(dq>1e-9), float(np.median(dq)))
    print("static QoE=%.3f\n" % bq)
    print("Oracle准入参照:")
    for th in THRESH:
        d,s,w = orc[th]; print(f"  th={th:<3} ΔQoE={d:+.3f} GPU省={s:5.1f}% 胜率={w*100:.1f}%")
    print("\nPred收缩准入网格 (ΔQoE / GPU省% / 胜率 / 中位Δ):")
    print(f"{'alpha':>6}" + "".join(f"{'th='+str(t):>22}" for t in THRESH))
    for a in ALPHAS:
        line = f"{a:>6}"
        for th in THRESH:
            d,s,w,m = grid[(a,th)]
            line += f"  {d:+5.2f}/{s:4.1f}%/{w*100:4.0f}%/{m:+4.2f}"
        print(line)
    json.dump({"base_qoe":bq,
               "oracle":{str(t):orc[t] for t in THRESH},
               "pred":{f"a{a}_t{th}":grid[(a,th)] for a in ALPHAS for th in THRESH}},
              open(os.path.join(ROOT,"workspace/gain_predictor/step17_shrink_budget.json"),"w"),
              indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
