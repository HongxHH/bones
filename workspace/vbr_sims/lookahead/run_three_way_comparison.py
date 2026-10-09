"""
三方案对比实验：
1. 纯BONES：段级固定增强，下载→增强→播放（串行）
2. BONES+帧级DP：BONES码率决策 + 帧级DP段内增强分配
3. BONES+增量传输：下载→播放（基础质量），后台渐进增强高优先级帧

在4条网络轨迹上对比。
"""
import sys
import os
import json
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.bones import BONES
from control.bones_frame_level import BONESFrameLevel
from control.bones_incremental import BONESIncremental


def make_args(trace_path, enhance_path, log_path):
    args = argparse.Namespace()
    args.log_path = log_path
    args.manifest_path = r"E:\dev\Code\Python\bones\data\bbb\bbb_video_info.json"
    args.trace_path = trace_path
    args.enhance_path = enhance_path
    args.video_length = None
    args.throughput = "ewma"
    args.download_buffer = 60
    args.enhance_buffer = 60
    args.bandwidth_multiplier = 1.0
    args.speed_multiplier = 1.0
    args.verbose = False
    args.gamma_p = 10.0
    args.V_multiplier = 1.0
    args.monitor = False
    args.autotune = False
    args.quick_start = False
    args.per_frame_path = r"E:\dev\Code\Python\bones\data\bbb\per_frame_quality.json"
    return args


def run_sim(sim_class, args, label):
    try:
        sim = sim_class(args)
        sim.run()
        report = sim.report()
        print(f"  [{label}] VMAF={report['average quality vmaf']:.2f}, "
              f"QoE={report['qoe']:.2f}, "
              f"freeze={report['freeze time']:.0f}ms, "
              f"osc={report['average oscillation vmaf']:.2f}")
        if 'enhanced frame ratio' in report:
            print(f"         增强帧比例={report['enhanced frame ratio']*100:.1f}%, "
                  f"高优先级命中率={report.get('high priority enhanced ratio', 0)*100:.1f}%")
        return report
    except Exception as e:
        print(f"  [{label}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    trace_dir = r"E:\dev\Code\Python\bones\data\trace"
    enhance_path = r"E:\dev\Code\Python\bones\data\bbb\imdn_bbb_3060ti.json"
    log_dir = r"E:\dev\Code\Python\bones\workspace\sim_logs_v2"
    os.makedirs(log_dir, exist_ok=True)

    traces = ["3g_low", "4g_medium", "wifi_high", "volatile"]
    all_results = {}

    for trace_name in traces:
        trace_path = os.path.join(trace_dir, f"{trace_name}.json")
        print(f"\n{'='*70}")
        print(f"轨迹: {trace_name}")
        print(f"{'='*70}")

        # 1. 纯BONES
        args1 = make_args(trace_path, enhance_path, os.path.join(log_dir, f"bones_{trace_name}.log"))
        r1 = run_sim(BONES, args1, "BONES")

        # 2. BONES+帧级DP
        args2 = make_args(trace_path, enhance_path, os.path.join(log_dir, f"fl_{trace_name}.log"))
        r2 = run_sim(BONESFrameLevel, args2, "BONES+FrameLevel")

        # 3. BONES+增量传输
        args3 = make_args(trace_path, enhance_path, os.path.join(log_dir, f"inc_{trace_name}.log"))
        r3 = run_sim(BONESIncremental, args3, "BONES+Incremental")

        if r1 and r2 and r3:
            print(f"\n  --- 对比 ---")
            print(f"  BONES:          VMAF={r1['average quality vmaf']:.2f}, QoE={r1['qoe']:.2f}")
            print(f"  +帧级DP:        VMAF={r2['average quality vmaf']:.2f} ({r2['average quality vmaf']-r1['average quality vmaf']:+.2f}), "
                  f"QoE={r2['qoe']:.2f} ({r2['qoe']-r1['qoe']:+.2f})")
            print(f"  +增量传输:      VMAF={r3['average quality vmaf']:.2f} ({r3['average quality vmaf']-r1['average quality vmaf']:+.2f}), "
                  f"QoE={r3['qoe']:.2f} ({r3['qoe']-r1['qoe']:+.2f})")
            print(f"  增量增强帧比例: {r3['enhanced frame ratio']*100:.1f}%, "
                  f"高优先级命中率: {r3.get('high priority enhanced ratio', 0)*100:.1f}%")

            all_results[trace_name] = {
                "bones": {"vmaf": r1['average quality vmaf'], "qoe": r1['qoe'],
                          "freeze": r1['freeze time'], "osc": r1['average oscillation vmaf']},
                "frame_level": {"vmaf": r2['average quality vmaf'], "qoe": r2['qoe'],
                                "freeze": r2['freeze time'], "osc": r2['average oscillation vmaf']},
                "incremental": {"vmaf": r3['average quality vmaf'], "qoe": r3['qoe'],
                                "freeze": r3['freeze time'], "osc": r3['average oscillation vmaf'],
                                "enh_frame_ratio": r3['enhanced frame ratio'],
                                "high_prio_ratio": r3.get('high priority enhanced ratio', 0)},
            }

    # 汇总
    print(f"\n{'='*70}")
    print("汇总：三方案对比")
    print(f"{'='*70}")
    print(f"{'轨迹':<12} {'BONES VMAF':<12} {'+FL VMAF':<11} {'+INC VMAF':<11} "
          f"{'BONES QoE':<11} {'+FL QoE':<10} {'+INC QoE':<10} {'INC增强率':<10}")
    print("-" * 90)
    for trace_name in traces:
        if trace_name in all_results:
            r = all_results[trace_name]
            print(f"{trace_name:<12} {r['bones']['vmaf']:<12.2f} "
                  f"{r['frame_level']['vmaf']:<11.2f} {r['incremental']['vmaf']:<11.2f} "
                  f"{r['bones']['qoe']:<11.2f} {r['frame_level']['qoe']:<10.2f} "
                  f"{r['incremental']['qoe']:<10.2f} {r['incremental']['enh_frame_ratio']*100:<10.1f}")

    # 保存
    output_path = r"E:\dev\Code\Python\bones\workspace\three_way_comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
