import sys
sys.path.append("..")
import argparse
import os
import json
import numpy as np
from control.bones import BONES

def run_one(trace_path, enhance_path, manifest_path, verbose=False):
    args = argparse.Namespace(
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
        verbose=verbose,
        gamma_p=10.0,
        V_multiplier=1.0,
        monitor=False,
        autotune=False,
        quick_start=False,
    )
    simulator = BONES(args)
    simulator.run()
    report = simulator.report()
    # Clean up non-serializable fields
    clean = {k: v for k, v in report.items() if k not in ["download history", "enhance history"]}
    return clean, report

if __name__ == "__main__":
    manifest = "../data/bbb/bbb_video_info.json"
    enhance = "../data/bbb/imdn_bbb_3060ti.json"
    traces = [
        ("trace1_stable3g", "../data/trace/3Glogs/report.2010-09-13_1003CEST.json"),
        ("trace2_lowbw", "../data/trace/3Glogs/report.2010-09-13_1046CEST.json"),
        ("trace3_highvar", "../data/trace/3Glogs/report.2010-09-28_1003CEST.json"),
    ]

    results = {}
    for name, trace in traces:
        print(f"Running {name}...")
        clean, full = run_one(trace, enhance, manifest)
        results[name] = clean
        print(f"  VMAF={clean['average quality vmaf']:.2f}, QoE={clean['qoe']:.2f}, "
              f"freeze={clean['freeze time']:.1f}ms, freeze_event={clean['freeze event']}")

    # Save results
    with open("../workspace/sim_bones_baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to workspace/sim_bones_baseline.json")
