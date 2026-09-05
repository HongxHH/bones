"""Step 16: 算力预算-质量帕累托扫描
static(BONES全量增强) vs oracle/pred 在不同增益准入阈值下:
阈值越高, 低增益(码率,增强)组合越被屏蔽, GPU算力越省。产出 算力节省%-ΔQoE 帕累托。"""
import os, sys, glob, json, argparse
import numpy as np
ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, ROOT); os.chdir(os.path.join(ROOT, "control"))
from control.bones_selective import SelectiveBONES

MANI = "../data/bbb/bbb_video_info.json"; ENH = "../data/bbb/imdn_bbb_2080ti.json"
# None=不做准入(等价原联合); 其后为增益阈值
THRESH = [None, 0, 2, 4, 6, 8, 10, 13, 16, 20, 25]


def traces():
    t = glob.glob("../data/trace/3Glogs/*.json")+glob.glob("../data/trace/4Glogs/*.json")
    t += [p for p in glob.glob("../data/trace/*.json") if os.path.basename(p) != "dummy.json"]
    return sorted(t)


def mk(tr, mode, thresh):
    return argparse.Namespace(log_path="../workspace/gain_predictor/tmp_log.txt",
        manifest_path=MANI, trace_path=tr, enhance_path=ENH, video_length=None,
        throughput="ewma", download_buffer=60, enhance_buffer=60,
        bandwidth_multiplier=1.0, speed_multiplier=1.0, verbose=False,
        gamma_p=10, V_multiplier=1.0, monitor=False, autotune=False,
        quick_start=False, quality_mode=mode, alpha=1.0, decouple=False,
        gating=False, margin=0.0, budget_thresh=thresh)


def run(tr, mode, thresh):
    s = SelectiveBONES(mk(tr, mode, thresh)); s.run(); r = s.report()
    hd, he = r["download history"], r["enhance history"]
    gpu = sum(float(s.enhance_time[hd[i]][he[i]]) for i in range(len(hd)))
    enh_ratio = np.mean([1 if x != 0 else 0 for x in he])
    return r["qoe"], r["average quality vmaf"], r["average oscillation vmaf"], gpu, enh_ratio


def main():
    TR = traces(); n = len(TR)
    stat = np.array([run(tr, "static", None) for tr in TR], float)
    base_gpu = stat[:, 3].mean(); base_qoe = stat[:, 0].mean()
    curves = {}
    for mode in ["oracle", "pred"]:
        rows = []
        for th in THRESH:
            M = np.array([run(tr, mode, th) for tr in TR], float)
            dq = M[:, 0]-stat[:, 0]
            rows.append({
                "thresh": ("none" if th is None else th),
                "qoe": float(M[:, 0].mean()), "dqoe": float(dq.mean()),
                "dqoe_median": float(np.median(dq)), "win": float(np.mean(dq > 1e-9)),
                "vmaf": float(M[:, 1].mean()), "osc": float(M[:, 2].mean()),
                "gpu": float(M[:, 3].mean()),
                "gpu_save_pct": float((base_gpu-M[:, 3].mean())/base_gpu*100),
                "enh_ratio": float(M[:, 4].mean())})
        curves[mode] = rows

    print("static基准: QoE=%.3f GPU=%.0fms 增强段占比=%.3f" % (
        base_qoe, base_gpu, stat[:, 4].mean()))
    for mode in ["oracle", "pred"]:
        print(f"\n===== {mode}_budget 帕累托 =====")
        print(f"{'thresh':>7}{'QoE':>9}{'ΔQoE':>8}{'胜率':>7}{'GPU节省%':>9}{'增强占比':>9}")
        for r in curves[mode]:
            print(f"{str(r['thresh']):>7}{r['qoe']:>9.3f}{r['dqoe']:>+8.3f}{r['win']*100:>6.1f}%{r['gpu_save_pct']:>9.1f}{r['enh_ratio']:>9.3f}")

    json.dump({"base_qoe": base_qoe, "base_gpu": base_gpu,
               "base_enh_ratio": float(stat[:, 4].mean()), "curves": curves},
              open(os.path.join(ROOT, "workspace/gain_predictor/step16_budget_pareto.json"), "w"),
              indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
