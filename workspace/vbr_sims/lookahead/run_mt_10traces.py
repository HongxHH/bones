"""多轨迹对比：BONES_CA (SR-only) vs BONES_MT (multi-task)"""
import sys
sys.path.append('..')
import argparse
import json
import os
import numpy as np
from control.bones_ca import BONES_CA
from control.bones_mt import BONES_MT


def make_args(enhance_path, resource_aware=False, trace_path=None):
    a = argparse.Namespace(
        log_path='', manifest_path='../data/bbb/bbb_video_info.json',
        trace_path=trace_path, enhance_path=enhance_path,
        video_length=None, throughput='ewma',
        download_buffer=60, enhance_buffer=60, bandwidth_multiplier=1.0,
        speed_multiplier=1.0, verbose=False, gamma_p=10.0, V_multiplier=1.0,
        monitor=False, autotune=False, quick_start=False,
        content_aware=True, c_min=0.5, c_max=2.0, ca_mode='base_known',
        ca_latency=False, lookahead_N=1, smooth_alpha=0.0,
    )
    if resource_aware:
        a.resource_aware = True
        a.content_aware_task = True
        a.gpu_util_penalty = 1.0
        a.gpu_mem_limit_mb = 8000
        a.max_allowed_util = 0.60
        a.motion_threshold = 0.4
    return a


def run_sim(scheduler_cls, enhance_path, trace_path, resource_aware=False):
    args = make_args(enhance_path, resource_aware, trace_path)
    sim = scheduler_cls(args)
    sim.run()
    report = sim.report()
    result = {
        'avg_vmaf': float(report['average quality vmaf']),
        'freeze_time': float(report['freeze time']),
        'freeze_events': int(report['freeze event']),
        'qoe': float(report['qoe']),
        'avg_bandwidth': float(report['average bandwidth']),
    }
    if hasattr(sim, 'get_task_combo_stats'):
        ts = sim.get_task_combo_stats()
        result['task_combos'] = {k: v['count'] for k, v in ts.items()
                                  if k in ['none', 'sr', 'sr_deblur', 'deblur', 'sr_deblur_fi']}
        result['avg_gpu_util'] = float(ts.get('avg_gpu_utilization', 0))
    return result


if __name__ == '__main__':
    trace_dir = '../data/trace/hd_fs'
    trace_files = sorted([f for f in os.listdir(trace_dir) if f.endswith('.json')])[:10]

    enhance_ca = '../data/bbb/imdn_bbb_3060ti.json'
    enhance_mt = '../data/bbb/multitask_bbb_rtx5080.json'

    all_results = {}
    for i, tf in enumerate(trace_files):
        trace_path = os.path.join(trace_dir, tf)
        name = tf.replace('.json', '')
        print(f"[{i+1}/{len(trace_files)}] {name}...", end=' ', flush=True)

        ca_res = run_sim(BONES_CA, enhance_ca, trace_path)
        mt_res = run_sim(BONES_MT, enhance_mt, trace_path, resource_aware=True)

        delta_vmaf = mt_res['avg_vmaf'] - ca_res['avg_vmaf']
        delta_qoe = mt_res['qoe'] - ca_res['qoe']
        print(f"CA={ca_res['avg_vmaf']:.1f} MT={mt_res['avg_vmaf']:.1f} "
              f"(dVMAF={delta_vmaf:+.1f}, dQoE={delta_qoe:+.1f})")

        all_results[name] = {'ca': ca_res, 'mt': mt_res}

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY (10 traces)")
    print(f"{'='*70}")
    for metric in ['avg_vmaf', 'qoe', 'freeze_time', 'freeze_events']:
        ca_vals = [r['ca'][metric] for r in all_results.values()]
        mt_vals = [r['mt'][metric] for r in all_results.values()]
        print(f"  {metric:<20}: CA={np.mean(ca_vals):.2f}+/-{np.std(ca_vals):.2f}  "
              f"MT={np.mean(mt_vals):.2f}+/-{np.std(mt_vals):.2f}  "
              f"delta={np.mean(mt_vals)-np.mean(ca_vals):+.2f}")

    # Task combo distribution
    print(f"\n  Task Combo Distribution (BONES_MT, averaged):")
    combo_totals = {'none': 0, 'sr': 0, 'sr_deblur': 0, 'deblur': 0, 'sr_deblur_fi': 0}
    total_segs = 0
    for r in all_results.values():
        tc = r['mt'].get('task_combos', {})
        for k in combo_totals:
            combo_totals[k] += tc.get(k, 0)
        total_segs += sum(tc.values())
    for k, v in combo_totals.items():
        print(f"    {k:<15}: {v:>5} ({v/total_segs*100:>5.1f}%)")

    # Save
    with open('../workspace/mt_comparison_10traces.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to mt_comparison_10traces.json")
    print("DONE")
