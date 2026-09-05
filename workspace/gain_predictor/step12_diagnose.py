"""诊断: alpha=0.25 为何QoE暴跌; pred与static逐轨迹差异分布; 决策翻转分析"""
import os, sys, glob, argparse
import numpy as np
ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, ROOT); os.chdir(os.path.join(ROOT, "control"))
from control.bones_selective import SelectiveBONES

MANI = "../data/bbb/bbb_video_info.json"; ENH = "../data/bbb/imdn_bbb_2080ti.json"
TR = sorted(glob.glob("../data/trace/3Glogs/*.json")+glob.glob("../data/trace/4Glogs/*.json")
            + [p for p in glob.glob("../data/trace/*.json") if os.path.basename(p) != "dummy.json"])


def mk(tr, mode, a):
    return argparse.Namespace(log_path="../workspace/gain_predictor/tmp_log.txt",
        manifest_path=MANI, trace_path=tr, enhance_path=ENH, video_length=None,
        throughput="ewma", download_buffer=60, enhance_buffer=60,
        bandwidth_multiplier=1.0, speed_multiplier=1.0, verbose=False,
        gamma_p=10, V_multiplier=1.0, monitor=False, autotune=False,
        quick_start=False, quality_mode=mode, alpha=a)


def run(tr, mode, a=1.0):
    s = SelectiveBONES(mk(tr, mode, a)); s.run(); r = s.report()
    return r["qoe"], r["average quality vmaf"], r["average oscillation vmaf"], r["freeze time"]


worst = []
for tr in TR:
    sq, sv, so, sf = run(tr, "static")
    q25, _, _, f25 = run(tr, "pred", 0.25)
    q75, _, _, _ = run(tr, "pred", 0.75)
    worst.append((os.path.basename(tr), q25-sq, q75-sq, f25-sf, sq, q25))

worst.sort(key=lambda x: x[1])
print("=== alpha=0.25 相对static 最差的8条 ===")
print(f"{'trace':<22}{'dQoE@.25':>10}{'dQoE@.75':>10}{'dFreeze_ms':>12}{'QoE_static':>11}{'QoE_.25':>10}")
for w in worst[:8]:
    print(f"{w[0]:<22}{w[1]:>10.2f}{w[2]:>10.2f}{w[3]:>12.0f}{w[4]:>11.2f}{w[5]:>10.2f}")
d25 = np.array([w[1] for w in worst])
print(f"\nalpha=.25 ΔQoE: 中位数={np.median(d25):+.3f}, 均值={d25.mean():+.3f}, 最小={d25.min():+.2f}, 最大={d25.max():+.2f}")
print(f"ΔQoE<-1 的轨迹数: {(d25<-1).sum()}; <-5: {(d25<-5).sum()}")
