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


def run_bones(trace_path, enhance_path, manifest_path):
    args = make_args(trace_path, enhance_path, manifest_path, content_aware=False)
    sim = BONES(args)
    sim.run()
    return clean_report(sim.report())


def run_ca(trace_path, enhance_path, manifest_path, ca_mode='base_known', c_min=0.5, c_max=2.0):
    args = make_args(trace_path, enhance_path, manifest_path,
                      content_aware=True, ca_mode=ca_mode, c_min=c_min, c_max=c_max)
    sim = BONES_CA(args)
    sim.run()
    report = clean_report(sim.report())
    if hasattr(sim, 'content_factors') and ca_mode in ('scalar', 'per_level', 'base_known'):
        cf = sim.content_factors
        if cf.ndim == 1:
            report["cf_mean"] = float(np.mean(cf))
        else:
            report["cf_per_level_mean"] = [float(np.mean(cf[:, e])) for e in range(cf.shape[1])]
    enh_hist = sim.report().get("enhance history", None)
    if enh_hist is not None and len(enh_hist) > 0:
        levels, counts = np.unique(enh_hist, return_counts=True)
        report["enhance_level_dist"] = {int(l): int(c) for l, c in zip(levels, counts)}
    return report


if __name__ == "__main__":
    manifest = "../data/bbb/bbb_video_info.json"
    enhance = "../data/bbb/imdn_bbb_3060ti.json"
    traces = [
        ("trace1_stable3g", "../data/trace/3Glogs/report.2010-09-13_1003CEST.json"),
        ("trace2_lowbw", "../data/trace/3Glogs/report.2010-09-13_1046CEST.json"),
        ("trace3_highvar", "../data/trace/3Glogs/report.2010-09-28_1003CEST.json"),
    ]

    methods = [
        ("BONES", lambda t: run_bones(t, enhance, manifest)),
        ("CA-scalar", lambda t: run_ca(t, enhance, manifest, ca_mode='scalar')),
        ("CA-per_level", lambda t: run_ca(t, enhance, manifest, ca_mode='per_level')),
        ("CA-base_known", lambda t: run_ca(t, enhance, manifest, ca_mode='base_known')),
        ("CA-oracle", lambda t: run_ca(t, enhance, manifest, ca_mode='oracle')),
    ]

    all_results = {}
    for trace_name, trace_path in traces:
        print(f"\n{'='*70}")
        print(f"Trace: {trace_name}")
        print(f"{'='*70}")
        all_results[trace_name] = {}

        for method_name, method_fn in methods:
            print(f"\n  [{method_name}]")
            res = method_fn(trace_path)
            all_results[trace_name][method_name] = res
            print(f"    VMAF={res['average quality vmaf']:.2f}, "
                  f"QoE={res['qoe']:.2f}, "
                  f"freeze={res['freeze time']:.0f}ms, "
                  f"events={res['freeze event']}")
            if 'enhance_level_dist' in res:
                print(f"    Enhance dist: {res['enhance_level_dist']}")

        bones = all_results[trace_name]["BONES"]
        print(f"\n  --- Delta vs BONES ---")
        for method_name in ["CA-scalar", "CA-per_level", "CA-base_known", "CA-oracle"]:
            res = all_results[trace_name][method_name]
            d_vmaf = res['average quality vmaf'] - bones['average quality vmaf']
            d_qoe = res['qoe'] - bones['qoe']
            d_freeze = res['freeze time'] - bones['freeze time']
            print(f"  {method_name:15s}: VMAF {d_vmaf:+.2f}, QoE {d_qoe:+.2f}, freeze {d_freeze:+.0f}ms")

    with open("../workspace/sim_comparison_v4.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\n\nResults saved to workspace/sim_comparison_v4.json")
