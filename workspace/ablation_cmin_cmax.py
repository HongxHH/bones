import sys
sys.path.append("..")
import argparse
import json
import numpy as np
from control.bones_ca import BONES_CA


def make_args(trace_path, enhance_path, manifest_path, c_min, c_max):
    return argparse.Namespace(
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
        c_min=c_min,
        c_max=c_max,
    )


def run_one(trace_path, enhance_path, manifest_path, c_min, c_max):
    args = make_args(trace_path, enhance_path, manifest_path, c_min, c_max)
    sim = BONES_CA(args)
    sim.run()
    r = sim.report()
    return {
        'vmaf': float(r['average quality vmaf']),
        'qoe': float(r['qoe']),
        'freeze': float(r['freeze time']),
        'events': int(r['freeze event']),
    }


if __name__ == "__main__":
    manifest = "../data/bbb/bbb_video_info.json"
    enhance = "../data/bbb/imdn_bbb_3060ti.json"
    traces = [
        ("stable3g", "../data/trace/3Glogs/report.2010-09-13_1003CEST.json"),
        ("lowbw", "../data/trace/3Glogs/report.2010-09-13_1046CEST.json"),
        ("highvar", "../data/trace/3Glogs/report.2010-09-28_1003CEST.json"),
    ]

    # BONES baseline for reference
    from control.bones import BONES
    baselines = {}
    for tname, tpath in traces:
        args = argparse.Namespace(
            log_path="temp_log.txt", manifest_path=manifest, trace_path=tpath,
            enhance_path=enhance, video_length=None, throughput="ewma",
            download_buffer=60, enhance_buffer=60, bandwidth_multiplier=1.0,
            speed_multiplier=1.0, verbose=False, gamma_p=10.0, V_multiplier=1.0,
            monitor=False, autotune=False, quick_start=False,
        )
        sim = BONES(args)
        sim.run()
        r = sim.report()
        baselines[tname] = {'vmaf': float(r['average quality vmaf']), 'qoe': float(r['qoe'])}

    # Parameter grid
    c_mins = [0.5, 0.6, 0.7, 0.8, 0.9]
    c_maxs = [1.3, 1.5, 1.8, 2.0, 2.5]

    results = {}
    print(f"{'c_min':>6} {'c_max':>6} | {'stable3g dQoE':>14} {'lowbw dQoE':>12} {'highvar dQoE':>14} | {'avg dQoE':>10}")
    print("-" * 80)

    for c_min in c_mins:
        for c_max in c_maxs:
            if c_min >= c_max:
                continue
            key = f"{c_min}_{c_max}"
            results[key] = {'c_min': c_min, 'c_max': c_max, 'traces': {}}
            deltas = []
            for tname, tpath in traces:
                res = run_one(tpath, enhance, manifest, c_min, c_max)
                results[key]['traces'][tname] = res
                d_qoe = res['qoe'] - baselines[tname]['qoe']
                deltas.append(d_qoe)
            avg_dqoe = np.mean(deltas)
            results[key]['avg_dqoe'] = avg_dqoe
            print(f"{c_min:>6.1f} {c_max:>6.1f} | {deltas[0]:>14.2f} {deltas[1]:>12.2f} {deltas[2]:>14.2f} | {avg_dqoe:>10.2f}")

    # Find best
    best = max(results.values(), key=lambda x: x['avg_dqoe'])
    print(f"\nBest: c_min={best['c_min']}, c_max={best['c_max']}, avg_dQoE={best['avg_dqoe']:.2f}")
    for tname in traces:
        t = best['traces'][tname[0]]
        d = t['qoe'] - baselines[tname[0]]['qoe']
        print(f"  {tname[0]}: VMAF={t['vmaf']:.2f}, QoE={t['qoe']:.2f} (dQoE={d:+.2f}), freeze={t['freeze']:.0f}ms")

    with open("../workspace/cmin_cmax_ablation.json", "w") as f:
        json.dump({'baselines': baselines, 'results': results, 'best': best}, f, indent=2)
    print("\nSaved to workspace/cmin_cmax_ablation.json")
