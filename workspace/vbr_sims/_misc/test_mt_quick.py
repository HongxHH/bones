import sys
sys.path.append('..')
import argparse
from control.bones_mt import BONES_MT
from control.bones_ca import BONES_CA
import numpy as np


def make_args(enhance_path, resource_aware=False):
    a = argparse.Namespace(
        log_path='', manifest_path='../data/bbb/bbb_video_info.json',
        trace_path='../data/trace/hd_fs/trace0000.json',
        enhance_path=enhance_path, video_length=None, throughput='ewma',
        download_buffer=60, enhance_buffer=60, bandwidth_multiplier=1.0,
        speed_multiplier=1.0, verbose=False, gamma_p=10.0, V_multiplier=1.0,
        monitor=False, autotune=False, quick_start=False,
        content_aware=True, c_min=0.5, c_max=2.0, ca_mode='base_known',
        ca_latency=False, lookahead_N=1, smooth_alpha=0.0,
    )
    if resource_aware:
        a.resource_aware = True
        a.content_aware_task = True
        a.gpu_util_penalty = 0.3
        a.gpu_mem_limit_mb = 8000
        a.max_allowed_util = 0.90
        a.motion_threshold = 0.4
    return a


def print_result(name, report, task_stats=None):
    print(f"  {name}:")
    print(f"    avg VMAF: {report['average quality vmaf']:.2f}")
    print(f"    freeze time: {report['freeze time']:.2f}s")
    print(f"    freeze events: {report['freeze event']}")
    print(f"    QoE: {report['qoe']:.2f}")
    if task_stats:
        combos = {k: v['count'] for k, v in task_stats.items()
                  if k in ['none', 'sr', 'sr_deblur', 'deblur', 'sr_deblur_fi']}
        print(f"    Task combos: {combos}")
        if 'avg_gpu_utilization' in task_stats:
            print(f"    Avg GPU util: {task_stats['avg_gpu_utilization']*100:.1f}%")


# BONES_CA baseline
print("=== BONES_CA (SR-only, original data) ===")
sim_ca = BONES_CA(make_args('../data/bbb/imdn_bbb_3060ti.json'))
sim_ca.run()
print_result("BONES_CA", sim_ca.report())

# BONES_MT without resource awareness
print("\n=== BONES_MT (multi-task, NO resource awareness) ===")
sim_mt1 = BONES_MT(make_args('../data/bbb/multitask_bbb_rtx5080.json', resource_aware=False))
sim_mt1.run()
print_result("BONES_MT (no RA)", sim_mt1.report(), sim_mt1.get_task_combo_stats())

# BONES_MT with resource awareness
print("\n=== BONES_MT (multi-task, WITH resource awareness) ===")
sim_mt2 = BONES_MT(make_args('../data/bbb/multitask_bbb_rtx5080.json', resource_aware=True))
sim_mt2.run()
print_result("BONES_MT (RA)", sim_mt2.report(), sim_mt2.get_task_combo_stats())

print("\nDONE")
