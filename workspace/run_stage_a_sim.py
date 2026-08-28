"""
阶段A 仿真器验证：对比 BONES / CA-base_known / CA-predicted / CA-oracle
"""
import sys
sys.path.append("..")
import argparse
import json
import numpy as np
from control.bones import BONES
from control.bones_ca import BONES_CA


def make_args(trace_path, **kwargs):
    defaults = dict(
        log_path="temp_log.txt",
        manifest_path="../data/bbb/bbb_video_info.json",
        trace_path=trace_path,
        enhance_path="../data/bbb/imdn_bbb_3060ti.json",
        video_length=None, throughput="ewma", download_buffer=40, enhance_buffer=60,
        bandwidth_multiplier=1.0, speed_multiplier=1.0, verbose=False,
        gamma_p=10.0, V_multiplier=1.0, monitor=False, autotune=False, quick_start=True,
        content_aware=True, ca_mode="base_known", c_min=0.7, c_max=1.8,
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def run_sim(controller_cls, trace_path, **kwargs):
    args = make_args(trace_path, **kwargs)
    sim = controller_cls(args)
    sim.run()
    r = sim.report()
    return {
        'qoe': r.get('qoe'),
        'vmaf': r.get('average quality vmaf'),
        'oscillation': r.get('average oscillation vmaf'),
        'freeze_time': r.get('freeze time'),
        'freeze_event': r.get('freeze event'),
        'enhance_hist': r.get('enhance history'),
    }


if __name__ == "__main__":
    traces = {
        'stable3g': '../player/data/3Glogs/report.2010-09-13_1003CEST.json',
        'lowbw': '../player/data/3Glogs/report.2010-09-13_1046CEST.json',
        'highvar': '../player/data/3Glogs/report.2010-09-28_1003CEST.json',
    }

    configs = [
        ('BONES', BONES, dict(content_aware=False)),
        ('CA-base_known', BONES_CA, dict(content_aware=True, ca_mode='base_known', c_min=0.7, c_max=1.8)),
        ('CA-predicted', BONES_CA, dict(content_aware=True, ca_mode='predicted', c_min=0.7, c_max=1.8)),
        ('CA-joint', BONES_CA, dict(content_aware=True, ca_mode='predicted', c_min=0.7, c_max=1.8, ca_latency=True)),
        ('CA-oracle', BONES_CA, dict(content_aware=True, ca_mode='oracle', c_min=0.7, c_max=1.8)),
    ]

    results = {}
    for trace_name, trace_path in traces.items():
        results[trace_name] = {}
        for config_name, ctrl_cls, kwargs in configs:
            print(f"Running {config_name} on {trace_name}...", flush=True)
            try:
                r = run_sim(ctrl_cls, trace_path, **kwargs)
                results[trace_name][config_name] = r
                print(f"  QoE={r['qoe']:.2f}, VMAF={r['vmaf']:.2f}, freeze={r['freeze_event']}", flush=True)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                import traceback
                traceback.print_exc()
                results[trace_name][config_name] = {'error': str(e)}

    # 汇总
    print("\n" + "=" * 90)
    print("阶段A 仿真器验证结果汇总")
    print("=" * 90)
    print(f"{'Trace':<12} {'Method':<18} {'QoE':>8} {'VMAF':>8} {'Osc':>8} {'Freeze':>8} {'dQoE':>8}")
    print("-" * 90)

    for trace_name in traces:
        bones_qoe = results[trace_name]['BONES'].get('qoe')
        for config_name, _, _ in configs:
            r = results[trace_name][config_name]
            qoe = r.get('qoe', 0) or 0
            vmaf = r.get('vmaf', 0) or 0
            osc = r.get('oscillation', 0) or 0
            freeze = r.get('freeze_event', 0) or 0
            dqoe = (qoe - bones_qoe) if (qoe and bones_qoe) else 0
            print(f"{trace_name:<12} {config_name:<18} {qoe:>8.2f} {vmaf:>8.2f} {osc:>8.2f} {freeze:>8.0f} {dqoe:>+8.2f}")
        print()

    # 平均 dQoE
    print("=" * 90)
    print("3条轨迹平均 dQoE (vs BONES)")
    print("=" * 90)
    for config_name, _, _ in configs[1:]:
        dqoes = []
        for trace_name in traces:
            bones_qoe = results[trace_name]['BONES'].get('qoe')
            ca_qoe = results[trace_name][config_name].get('qoe')
            if bones_qoe and ca_qoe:
                dqoes.append(ca_qoe - bones_qoe)
        if dqoes:
            print(f"  {config_name:<18}: avg dQoE = {np.mean(dqoes):+.2f} (std={np.std(dqoes):.2f}, min={min(dqoes):+.2f}, max={max(dqoes):+.2f})")

    # 保存结果（去掉 enhance_hist 以节省空间）
    save_results = {}
    for trace_name in traces:
        save_results[trace_name] = {}
        for config_name in [c[0] for c in configs]:
            r = results[trace_name][config_name].copy()
            r.pop('enhance_hist', None)
            save_results[trace_name][config_name] = r
    json.dump(save_results, open('../workspace/stage_a_sim_results.json', 'w'), indent=2)
    print("\n结果已保存到 workspace/stage_a_sim_results.json")
