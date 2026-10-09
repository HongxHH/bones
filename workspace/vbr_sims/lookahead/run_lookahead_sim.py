import sys
sys.path.append("..")
import argparse
import json
import numpy as np
from control.bones import BONES
from control.bones_ca import BONES_CA


def make_args(trace_path, enhance_path, manifest_path, **kwargs):
    defaults = dict(
        log_path="temp_log.txt",
        manifest_path=manifest_path,
        trace_path=trace_path,
        enhance_path=enhance_path,
        video_length=None,
        throughput="ewma",
        download_buffer=60,
        enhance_buffer=60,
        bandwidth_multiplier=1.0,
        speed_multiplier=1.0,
        verbose=False,
        gamma_p=10.0,
        V_multiplier=1.0,
        monitor=False,
        autotune=False,
        quick_start=False,
        content_aware=True,
        ca_mode='base_known',
        c_min=0.5,
        c_max=2.0,
        lookahead_N=1,
        smooth_alpha=0.0,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def clean_report(report):
    result = {}
    for k, v in report.items():
        if k in ["download history", "enhance history"]:
            continue
        if isinstance(v, (np.floating, np.integer)):
            result[k] = float(v)
        elif isinstance(v, np.ndarray):
            result[k] = v.tolist()
        else:
            result[k] = v
    return result


def run_sim(trace_path, enhance_path, manifest_path, lookahead_N=1, smooth_alpha=0.0, ca_mode='base_known'):
    args = make_args(trace_path, enhance_path, manifest_path,
                      lookahead_N=lookahead_N, smooth_alpha=smooth_alpha, ca_mode=ca_mode)
    sim = BONES_CA(args)
    sim.run()
    report = clean_report(sim.report())
    enh_hist = sim.report().get("enhance history", None)
    if enh_hist is not None and len(enh_hist) > 0:
        levels, counts = np.unique(enh_hist, return_counts=True)
        report["enhance_level_dist"] = {int(l): int(c) for l, c in zip(levels, counts)}
    return report


if __name__ == "__main__":
    manifest = "../data/bbb/bbb_video_info.json"
    enhance = "../data/bbb/imdn_bbb_3060ti.json"
    trace = "../data/trace/3Glogs/report.2010-09-13_1003CEST.json"

    results = {}

    # 1. BONES 基线（无内容感知）
    print("Running BONES baseline...")
    args = make_args(trace, enhance, manifest, content_aware=False)
    sim = BONES(args)
    sim.run()
    results['bones'] = clean_report(sim.report())

    # 2. CA-BONES 单段贪心（N=1）
    print("Running CA-BONES N=1...")
    results['ca_n1'] = run_sim(trace, enhance, manifest, lookahead_N=1)

    # 3. CA-BONES 多段前瞻 N=2
    print("Running CA-BONES N=2...")
    results['ca_n2'] = run_sim(trace, enhance, manifest, lookahead_N=2)

    # 4. CA-BONES 多段前瞻 N=3
    print("Running CA-BONES N=3...")
    results['ca_n3'] = run_sim(trace, enhance, manifest, lookahead_N=3)

    # 打印对比
    print("\n" + "="*80)
    print("前瞻规划仿真对比（stable3g 轨迹）")
    print("="*80)
    print(f"{'方法':<20} {'avg_vmaf':>10} {'QoE':>10} {'stall':>10} {'oscillation':>12}")
    print("-"*80)
    for name, r in results.items():
        vmaf = r.get('avg_bitrate', 0)  # 仿真器中可能叫别的名字
        qoe = r.get('QoE', r.get('qoe', 0))
        stall = r.get('avg_stall', r.get('stall', 0))
        osc = r.get('oscillation', r.get('vmaf_oscillation', 0))
        print(f"{name:<20} {vmaf:>10.2f} {qoe:>10.2f} {stall:>10.2f} {osc:>12.4f}")

    # 保存完整结果
    with open('lookahead_sim_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("\n完整结果已保存到 lookahead_sim_results.json")
