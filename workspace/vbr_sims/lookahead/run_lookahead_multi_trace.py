import sys
sys.path.append("..")
import argparse
import json
import numpy as np
import os
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


def run_sim(trace_path, enhance_path, manifest_path, lookahead_N=1, content_aware=True):
    args = make_args(trace_path, enhance_path, manifest_path,
                      lookahead_N=lookahead_N, content_aware=content_aware)
    if content_aware:
        sim = BONES_CA(args)
    else:
        sim = BONES(args)
    sim.run()
    report = clean_report(sim.report())
    return report


if __name__ == "__main__":
    manifest = "../data/bbb/bbb_video_info.json"
    enhance = "../data/bbb/imdn_bbb_3060ti.json"

    # 代表性轨迹：3G低带宽、4G移动、高带宽固定
    traces = [
        ("3G_stable",  "../data/trace/3Glogs/report.2010-09-13_1003CEST.json"),
        ("3G_var1",    "../data/trace/3Glogs/report.2010-09-14_1038CEST.json"),
        ("3G_var2",    "../data/trace/3Glogs/report.2011-02-10_1611CEST.json"),
        ("4G_bus",     "../data/trace/4Glogs/report_bus_0001.json"),
        ("4G_car",     "../data/trace/4Glogs/report_car_0001.json"),
        ("4G_bicycle", "../data/trace/4Glogs/report_bicycle_0001.json"),
        ("hd_fixed",   "../data/trace/hd_fs/trace0000.json"),
    ]

    all_results = {}

    for trace_name, trace_path in traces:
        if not os.path.exists(trace_path):
            print(f"[SKIP] {trace_name}: file not found")
            continue

        print(f"\n{'='*60}")
        print(f"Trace: {trace_name}")
        print(f"{'='*60}")

        trace_results = {}

        # 1. BONES 基线
        print("  Running BONES baseline...")
        r = run_sim(trace_path, enhance, manifest, lookahead_N=1, content_aware=False)
        trace_results['bones'] = r
        print(f"    QoE={r.get('qoe',0):.2f}  VMAF={r.get('average quality vmaf',0):.2f}  freeze={r.get('freeze time',0):.0f}ms")

        # 2. CA-BONES N=1
        print("  Running CA-BONES N=1...")
        r = run_sim(trace_path, enhance, manifest, lookahead_N=1, content_aware=True)
        trace_results['ca_n1'] = r
        print(f"    QoE={r.get('qoe',0):.2f}  VMAF={r.get('average quality vmaf',0):.2f}  freeze={r.get('freeze time',0):.0f}ms")

        # 3. CA-BONES N=2
        print("  Running CA-BONES N=2...")
        r = run_sim(trace_path, enhance, manifest, lookahead_N=2, content_aware=True)
        trace_results['ca_n2'] = r
        print(f"    QoE={r.get('qoe',0):.2f}  VMAF={r.get('average quality vmaf',0):.2f}  freeze={r.get('freeze time',0):.0f}ms")

        all_results[trace_name] = trace_results

    # 汇总对比表
    print(f"\n\n{'='*80}")
    print("多轨迹前瞻规划验证汇总")
    print(f"{'='*80}")
    print(f"{'轨迹':<14} {'方法':<10} {'QoE':>8} {'VMAF':>8} {'振荡':>8} {'卡顿(ms)':>10} {'卡顿次数':>8}")
    print("-"*80)

    for trace_name, trace_results in all_results.items():
        for method in ['bones', 'ca_n1', 'ca_n2']:
            if method not in trace_results:
                continue
            r = trace_results[method]
            qoe = r.get('qoe', 0)
            vmaf = r.get('average quality vmaf', 0)
            osc = r.get('average oscillation vmaf', 0)
            freeze = r.get('freeze time', 0)
            fevent = r.get('freeze event', 0)
            print(f"{trace_name:<14} {method:<10} {qoe:>8.2f} {vmaf:>8.2f} {osc:>8.2f} {freeze:>10.0f} {fevent:>8}")
        print()

    # 保存完整结果
    with open('lookahead_multi_trace_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print("完整结果已保存到 lookahead_multi_trace_results.json")
