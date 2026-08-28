"""
多任务增强对比仿真：BONES_CA (SR-only) vs BONES_MT (SR+Deblur+FI)

对比指标：
  - 平均 VMAF (QoE)
  - 卡顿率 / 启动延迟
  - 码率选择分布
  - 任务组合使用分布
  - GPU 资源利用率
"""
import sys
sys.path.append("..")
import argparse
import json
import os
import numpy as np
from control.bones_ca import BONES_CA
from control.bones_mt import BONES_MT


def make_base_args(trace_path, enhance_path, manifest_path):
    return argparse.Namespace(
        log_path="",
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
        c_min=0.5,
        c_max=2.0,
        ca_mode='base_known',
        ca_latency=False,
        lookahead_N=1,
        smooth_alpha=0.0,
    )


def run_ca_bones(trace_path, enhance_path, manifest_path):
    """BONES_CA with SR-only enhance data (baseline)"""
    args = make_base_args(trace_path, enhance_path, manifest_path)
    sim = BONES_CA(args)
    sim.run()
    report = sim.report()
    result = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
              for k, v in report.items() if k not in ["download history", "enhance history"]}
    # Enhance level distribution
    enh_hist = report.get("enhance history", None)
    if enh_hist is not None and len(enh_hist) > 0:
        levels, counts = np.unique(enh_hist, return_counts=True)
        result["enhance_level_dist"] = {int(l): int(c) for l, c in zip(levels, counts)}
    return result


def run_mt_bones(trace_path, enhance_path, manifest_path, resource_aware=True, content_aware_task=True):
    """BONES_MT with multi-task enhance data"""
    args = make_base_args(trace_path, enhance_path, manifest_path)
    args.resource_aware = resource_aware
    args.content_aware_task = content_aware_task
    args.gpu_util_penalty = 0.3
    args.gpu_mem_limit_mb = 8000
    args.max_allowed_util = 0.90
    args.motion_threshold = 0.4
    sim = BONES_MT(args)
    sim.run()
    report = sim.report()
    result = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v
              for k, v in report.items() if k not in ["download history", "enhance history"]}
    # Task combo distribution
    task_stats = sim.get_task_combo_stats()
    result["task_combo_stats"] = task_stats
    # Enhance history (task combos)
    enh_hist = report.get("enhance history", None)
    if enh_hist is not None and len(enh_hist) > 0:
        levels, counts = np.unique(enh_hist, return_counts=True)
        result["enhance_level_dist"] = {int(l): int(c) for l, c in zip(levels, counts)}
    return result


def print_comparison(name, ca_res, mt_res):
    """打印对比结果"""
    print(f"\n{'='*70}")
    print(f"Trace: {name}")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'BONES_CA (SR)':>15} {'BONES_MT (MT)':>15} {'Delta':>10}")
    print(f"{'-'*70}")

    key_metrics = [
        ('avg_vmaf', 'Avg VMAF'),
        ('avg_bitrate', 'Avg Bitrate (kbps)'),
        ('stall_count', 'Stall Count'),
        ('stall_duration', 'Stall Duration (s)'),
        ('startup_delay', 'Startup Delay (s)'),
        ('vmaf_switch_count', 'VMAF Switches'),
    ]

    for key, label in key_metrics:
        ca_val = ca_res.get(key, 0)
        mt_val = mt_res.get(key, 0)
        if isinstance(ca_val, (int, float)) and isinstance(mt_val, (int, float)):
            delta = mt_val - ca_val
            print(f"{label:<30} {ca_val:>15.2f} {mt_val:>15.2f} {delta:>+10.2f}")
        else:
            print(f"{label:<30} {str(ca_val):>15} {str(mt_val):>15} {'':>10}")

    # Task combo distribution
    print(f"\n  Task Combo Distribution (BONES_MT):")
    task_stats = mt_res.get('task_combo_stats', {})
    for combo in ['none', 'sr', 'sr_deblur', 'deblur', 'sr_deblur_fi']:
        if combo in task_stats:
            s = task_stats[combo]
            print(f"    {combo:<15}: {s['count']:>4} segments ({s['ratio']*100:>5.1f}%)")
    if 'avg_gpu_utilization' in task_stats:
        print(f"    {'Avg GPU Util':<15}: {task_stats['avg_gpu_utilization']*100:.1f}%")


if __name__ == "__main__":
    manifest = "../data/bbb/bbb_video_info.json"
    enhance_ca = "../data/bbb/imdn_bbb_3060ti.json"
    enhance_mt = "../data/bbb/multitask_bbb_rtx5080.json"

    # Use hd_fs traces
    trace_dir = "../data/trace/hd_fs"
    trace_files = sorted([f for f in os.listdir(trace_dir) if f.endswith('.json')])[:5]  # First 5 traces

    all_results = {}
    for trace_file in trace_files:
        trace_path = os.path.join(trace_dir, trace_file)
        name = trace_file.replace('.json', '')

        print(f"\nRunning trace: {name}...")

        print("  BONES_CA (SR-only)...")
        ca_res = run_ca_bones(trace_path, enhance_ca, manifest)

        print("  BONES_MT (multi-task, resource-aware)...")
        mt_res = run_mt_bones(trace_path, enhance_mt, manifest, resource_aware=True, content_aware_task=True)

        print_comparison(name, ca_res, mt_res)
        all_results[name] = {'ca': ca_res, 'mt': mt_res}

    # Summary across traces
    print(f"\n{'='*70}")
    print("SUMMARY ACROSS ALL TRACES")
    print(f"{'='*70}")

    for metric in ['avg_vmaf', 'stall_count', 'stall_duration']:
        ca_vals = [r['ca'].get(metric, 0) for r in all_results.values()]
        mt_vals = [r['mt'].get(metric, 0) for r in all_results.values()]
        print(f"  {metric:<25}: CA={np.mean(ca_vals):.2f}±{np.std(ca_vals):.2f}  "
              f"MT={np.mean(mt_vals):.2f}±{np.std(mt_vals):.2f}  "
              f"delta={np.mean(mt_vals)-np.mean(ca_vals):+.2f}")

    # Save results
    output_file = "../workspace/mt_comparison_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")
    print("DONE")
