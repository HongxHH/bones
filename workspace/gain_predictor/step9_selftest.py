"""单轨迹自检: static线必须与原始BONES数值一致; 对比 static/oracle/pred 三线"""
import os, sys, argparse
import numpy as np
ROOT = r"E:\dev\Code\Python\bones"
sys.path.insert(0, ROOT)
os.chdir(os.path.join(ROOT, "control"))  # 仿真器内部相对路径以control为基准
from control.bones import BONES
from control.bones_selective import SelectiveBONES

MANI = "../data/bbb/bbb_video_info.json"
ENH = "../data/bbb/imdn_bbb_2080ti.json"


def make_args(trace, mode):
    return argparse.Namespace(
        log_path=os.path.join(ROOT, "workspace", "gain_predictor", "tmp_log.txt"),
        manifest_path=MANI, trace_path=trace, enhance_path=ENH, video_length=None,
        throughput="ewma", download_buffer=60, enhance_buffer=60,
        bandwidth_multiplier=1.0, speed_multiplier=1.0, verbose=False,
        gamma_p=10, V_multiplier=1.0, monitor=False, autotune=False,
        quick_start=False, quality_mode=mode)


def run(cls, trace, mode=None):
    args = make_args(trace, mode)
    sim = cls(args)
    sim.run()
    rep = sim.report()
    cnt = getattr(sim, "enh_level_count", None)
    return rep, (cnt.tolist() if cnt is not None else None)


for trace in ["../data/trace/3g_low.json", "../data/trace/4g_medium.json"]:
    print("=" * 78)
    print("轨迹:", trace)
    r_base, _ = run(BONES, trace, None)
    r_static, c_static = run(SelectiveBONES, trace, "static")
    r_oracle, c_oracle = run(SelectiveBONES, trace, "oracle")
    r_pred, c_pred = run(SelectiveBONES, trace, "pred")

    # 一致性自检
    dq = abs(r_base["qoe"] - r_static["qoe"])
    dv = abs(r_base["average quality vmaf"] - r_static["average quality vmaf"])
    print(f"[自检] 原始BONES vs Selective(static): |ΔQoE|={dq:.2e}, |ΔVMAF|={dv:.2e}  "
          f"{'✅一致' if dq < 1e-9 and dv < 1e-9 else '❌不一致!'}")

    def show(name, r, c):
        enh_ratio = 1 - c[0] / sum(c) if c and sum(c) > 0 else 0
        print(f"  {name:<8}: QoE={r['qoe']:7.3f} VMAF={r['average quality vmaf']:6.3f} "
              f"osc={r['average oscillation vmaf']:6.3f} freeze={r['freeze time']/1000:6.2f}s "
              f"增强占比={enh_ratio*100:5.1f}% level分布={c}")
    show("static", r_static, c_static)
    show("oracle", r_oracle, c_oracle)
    show("pred", r_pred, c_pred)
