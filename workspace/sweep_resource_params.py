import sys; sys.path.append('..')
import argparse, numpy as np
from control.bones_mt import BONES_MT


def run(penalty, max_util, trace='../data/trace/hd_fs/trace0000.json'):
    a = argparse.Namespace(
        log_path='', manifest_path='../data/bbb/bbb_video_info.json',
        trace_path=trace, enhance_path='../data/bbb/multitask_bbb_rtx5080.json',
        video_length=None, throughput='ewma', download_buffer=60, enhance_buffer=60,
        bandwidth_multiplier=1.0, speed_multiplier=1.0, verbose=False,
        gamma_p=10.0, V_multiplier=1.0, monitor=False, autotune=False, quick_start=False,
        content_aware=True, c_min=0.5, c_max=2.0, ca_mode='base_known',
        ca_latency=False, lookahead_N=1, smooth_alpha=0.0,
        resource_aware=True, content_aware_task=True,
        gpu_util_penalty=penalty, gpu_mem_limit_mb=8000,
        max_allowed_util=max_util, motion_threshold=0.4,
    )
    sim = BONES_MT(a); sim.run()
    r = sim.report(); ts = sim.get_task_combo_stats()
    tc = {k: v['count'] for k, v in ts.items() if k in ['none', 'sr', 'sr_deblur', 'deblur', 'sr_deblur_fi']}
    return r['average quality vmaf'], r['qoe'], tc, ts.get('avg_gpu_utilization', 0)


print("Resource awareness parameter sweep:")
print(f"{'penalty':>8} {'max_util':>9} {'VMAF':>7} {'QoE':>7} {'GPU%':>6}  combos")
for p in [0.0, 0.3, 0.5, 1.0, 2.0, 5.0]:
    for u in [0.90, 0.75, 0.60]:
        vmaf, qoe, tc, gpu = run(p, u)
        print(f"{p:>8.1f} {u:>9.2f} {vmaf:>7.2f} {qoe:>7.2f} {gpu*100:>6.1f}  {tc}")
print("DONE")
