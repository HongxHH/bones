"""
四方案对比实验：
1. 纯BONES
2. BONES+帧级DP
3. BONES+增量传输
4. BONES+联合方案（帧级DP+增量传输）
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
from control.bones_combined import BONESCombined


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
              f"QoE={report['qoe']:.2f}, osc={report['average oscillation vmaf']:.2f}")
        if 'enhanced frame ratio' in report:
            print(f"         增强率={report['enhanced frame ratio']*100:.1f}%, "
                  f"高优先级命中={report.get('high priority enhanced ratio', 0)*100:.1f}%")
        return report
    except Exception as e:
        print(f"  [{label}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    trace_dir = r"E:\dev\Code\Python\bones\data\trace"
    enhance_path = r"E:\dev\Code\Python\bones\data\bbb\imdn_bbb_3060ti.json"
    log_dir = r"E:\dev\Code\Python\bones\workspace\sim_logs_v3"
    os.makedirs(log_dir, exist_ok=True)

    traces = ["3g_low", "4g_medium", "wifi_high", "volatile"]
    all_results = {}

    for trace_name in traces:
        trace_path = os.path.join(trace_dir, f"{trace_name}.json")
        print(f"\n{'='*70}")
        print(f"轨迹: {trace_name}")
        print(f"{'='*70}")

        r1 = run_sim(BONES, make_args(trace_path, enhance_path, os.path.join(log_dir, f"bones_{trace_name}.log")), "BONES")
        r2 = run_sim(BONESFrameLevel, make_args(trace_path, enhance_path, os.path.join(log_dir, f"fl_{trace_name}.log")), "+FrameLevel")
        r3 = run_sim(BONESIncremental, make_args(trace_path, enhance_path, os.path.join(log_dir, f"inc_{trace_name}.log")), "+Incremental")
        r4 = run_sim(BONESCombined, make_args(trace_path, enhance_path, os.path.join(log_dir, f"comb_{trace_name}.log")), "+Combined")

        if all([r1, r2, r3, r4]):
            print(f"\n  --- QoE对比 ---")
            print(f"  BONES:      {r1['qoe']:.2f}")
            print(f"  +FrameLevel: {r2['qoe']:.2f} ({r2['qoe']-r1['qoe']:+.2f})")
            print(f"  +Incremental: {r3['qoe']:.2f} ({r3['qoe']-r1['qoe']:+.2f})")
            print(f"  +Combined:   {r4['qoe']:.2f} ({r4['qoe']-r1['qoe']:+.2f})")

            all_results[trace_name] = {
                "bones": {"vmaf": r1['average quality vmaf'], "qoe": r1['qoe'], "osc": r1['average oscillation vmaf']},
                "frame_level": {"vmaf": r2['average quality vmaf'], "qoe": r2['qoe'], "osc": r2['average oscillation vmaf']},
                "incremental": {"vmaf": r3['average quality vmaf'], "qoe": r3['qoe'], "osc": r3['average oscillation vmaf'],
                                "enh_ratio": r3['enhanced frame ratio']},
                "combined": {"vmaf": r4['average quality vmaf'], "qoe": r4['qoe'], "osc": r4['average oscillation vmaf'],
                             "enh_ratio": r4['enhanced frame ratio'], "high_prio_ratio": r4.get('high priority enhanced ratio', 0)},
            }

    # 汇总
    print(f"\n{'='*70}")
    print("汇总：四方案QoE对比")
    print(f"{'='*70}")
    print(f"{'轨迹':<12} {'BONES':<8} {'+FL':<8} {'+INC':<8} {'+COMB':<8} "
          f"{'COMB增益':<10} {'COMB增强率':<11} {'高优先级命中':<12}")
    print("-" * 85)
    for trace_name in traces:
        if trace_name in all_results:
            r = all_results[trace_name]
            print(f"{trace_name:<12} {r['bones']['qoe']:<8.2f} {r['frame_level']['qoe']:<8.2f} "
                  f"{r['incremental']['qoe']:<8.2f} {r['combined']['qoe']:<8.2f} "
                  f"{r['combined']['qoe']-r['bones']['qoe']:<+10.2f} "
                  f"{r['combined']['enh_ratio']*100:<11.1f} {r['combined']['high_prio_ratio']*100:<12.1f}")

    # 保存
    output_path = r"E:\dev\Code\Python\bones\workspace\four_way_comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
