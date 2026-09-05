"""Step 19: 算力受限场景验证
核心假设: 充足算力下全量增强最优、选择性增强无意义; 算力稀缺(只能增强部分段)时,
BONES(static)按Lyapunov被动/盲目放弃增强, 而段级预测按增益把稀缺算力优先给高增益段。
扫描GPU增强速度倍率sm(越小越慢越稀缺), 观察pred/oracle相对static的ΔQoE是否随稀缺转正。"""
import os, sys, glob, json, argparse
import numpy as np
ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, ROOT); os.chdir(os.path.join(ROOT, "control"))
from control.bones_selective import SelectiveBONES

MANI = "../data/bbb/bbb_video_info.json"; ENH = "../data/bbb/imdn_bbb_2080ti.json"
SM_LIST = [1.0, 0.6, 0.4, 0.25]
MODES = ["static", "oracle", "pred"]


def traces():
    t = glob.glob("../data/trace/3Glogs/*.json")+glob.glob("../data/trace/4Glogs/*.json")
    t += [p for p in glob.glob("../data/trace/*.json") if os.path.basename(p) != "dummy.json"]
    return sorted(t)


def mk(tr, mode, sm):
    return argparse.Namespace(log_path="../workspace/gain_predictor/tmp_log.txt",
        manifest_path=MANI, trace_path=tr, enhance_path=ENH, video_length=None,
        throughput="ewma", download_buffer=60, enhance_buffer=60,
        bandwidth_multiplier=1.0, speed_multiplier=sm, verbose=False,
        gamma_p=10, V_multiplier=1.0, monitor=False, autotune=False,
        quick_start=False, quality_mode=mode, alpha=1.0, decouple=False,
        gating=False, margin=0.0, budget_thresh=None)


def run(tr, mode, sm):
    s = SelectiveBONES(mk(tr, mode, sm)); s.run(); r = s.report()
    hd, he = r["download history"], r["enhance history"]
    enh_ratio = np.mean([1 if x != 0 else 0 for x in he])
    return r["qoe"], r["average quality vmaf"], enh_ratio


def main():
    TR = traces()
    result = {}
    for sm in SM_LIST:
        block = {}
        for mode in MODES:
            M = np.array([run(tr, mode, sm) for tr in TR], float)
            block[mode] = M
        sq, oq, pq = block["static"][:,0], block["oracle"][:,0], block["pred"][:,0]
        result[sm] = {
            "static_qoe": sq.mean(), "oracle_dqoe": (oq-sq).mean(), "pred_dqoe": (pq-sq).mean(),
            "oracle_win": float(np.mean(oq>sq+1e-9)), "pred_win": float(np.mean(pq>sq+1e-9)),
            "static_enh": block["static"][:,2].mean(), "oracle_enh": block["oracle"][:,2].mean(),
            "pred_enh": block["pred"][:,2].mean(),
            "static_vmaf": block["static"][:,1].mean(), "pred_vmaf": block["pred"][:,1].mean()}
        r = result[sm]
        print(f"sm={sm:<4} staticQoE={r['static_qoe']:7.3f} | "
              f"oracle Δ={r['oracle_dqoe']:+.3f} 胜率{r['oracle_win']*100:4.1f}% | "
              f"pred Δ={r['pred_dqoe']:+.3f} 胜率{r['pred_win']*100:4.1f}% | "
              f"增强占比 static/oracle/pred={r['static_enh']:.3f}/{r['oracle_enh']:.3f}/{r['pred_enh']:.3f}")
    json.dump(result, open(os.path.join(ROOT,"workspace/gain_predictor/step19_compute_constrained.json"),"w"),
              indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
