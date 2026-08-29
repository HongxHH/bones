"""
对比实验：纯BONES vs BONES+帧级DP

在4条网络轨迹上对比：
- 纯BONES：段级固定增强
- BONES+帧级DP：BONES码率决策 + 帧级DP增强分配

指标：平均VMAF、QoE、卡顿时间、增强帧比例、GPU使用
"""
import sys
import os
import json
import argparse
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from control.bones import BONES
from control.bones_frame_level import BONESFrameLevel


def make_args(trace_path, enhance_path, log_path):
    """创建仿真参数"""
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


def run_simulation(sim_class, args, label):
    """运行一次仿真"""
    try:
        sim = sim_class(args)
        sim.run()
        report = sim.report()
        print(f"  [{label}] VMAF={report['average quality vmaf']:.2f}, "
              f"QoE={report['qoe']:.2f}, "
              f"freeze={report['freeze time']:.0f}ms, "
              f"osc={report['average oscillation vmaf']:.2f}")
        return report
    except Exception as e:
        print(f"  [{label}] ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    trace_dir = r"E:\dev\Code\Python\bones\data\trace"
    enhance_path = r"E:\dev\Code\Python\bones\data\bbb\imdn_bbb_3060ti.json"
    log_dir = r"E:\dev\Code\Python\bones\workspace\sim_logs"
    os.makedirs(log_dir, exist_ok=True)

    traces = ["3g_low", "4g_medium", "wifi_high", "volatile"]

    all_results = {}

    for trace_name in traces:
        trace_path = os.path.join(trace_dir, f"{trace_name}.json")
        print(f"\n{'='*60}")
        print(f"轨迹: {trace_name}")
        print(f"{'='*60}")

        # 纯BONES
        args_bones = make_args(trace_path, enhance_path,
                                os.path.join(log_dir, f"bones_{trace_name}.log"))
        report_bones = run_simulation(BONES, args_bones, "BONES")

        # BONES+帧级DP
        args_fl = make_args(trace_path, enhance_path,
                             os.path.join(log_dir, f"fl_{trace_name}.log"))
        report_fl = run_simulation(BONESFrameLevel, args_fl, "BONES+FrameLevel")

        if report_bones and report_fl:
            vmaf_gain = report_fl['average quality vmaf'] - report_bones['average quality vmaf']
            qoe_gain = report_fl['qoe'] - report_bones['qoe']
            enh_ratio = report_fl.get('enhanced frame ratio', 0)
            level_dist = report_fl.get('level distribution', {})

            print(f"\n  --- 对比 ---")
            print(f"  VMAF增益: {vmaf_gain:+.2f}")
            print(f"  QoE增益:  {qoe_gain:+.2f}")
            print(f"  增强帧比例: {enh_ratio*100:.1f}%")
            print(f"  级别分布: base={level_dist.get(0,0)*100:.0f}%, "
                  f"low={level_dist.get(1,0)*100:.0f}%, "
                  f"medium={level_dist.get(2,0)*100:.0f}%, "
                  f"high={level_dist.get(3,0)*100:.0f}%")

            all_results[trace_name] = {
                "bones": {
                    "vmaf": report_bones['average quality vmaf'],
                    "qoe": report_bones['qoe'],
                    "freeze": report_bones['freeze time'],
                    "oscillation": report_bones['average oscillation vmaf'],
                },
                "frame_level": {
                    "vmaf": report_fl['average quality vmaf'],
                    "qoe": report_fl['qoe'],
                    "freeze": report_fl['freeze time'],
                    "oscillation": report_fl['average oscillation vmaf'],
                    "enhanced_frame_ratio": enh_ratio,
                    "level_distribution": {k: float(v) for k, v in level_dist.items()},
                },
                "gain": {
                    "vmaf": vmaf_gain,
                    "qoe": qoe_gain,
                }
            }

    # 汇总
    print(f"\n{'='*60}")
    print("汇总：BONES+帧级DP vs 纯BONES")
    print(f"{'='*60}")
    print(f"{'轨迹':<14} {'BONES VMAF':<12} {'FL VMAF':<10} {'VMAF增益':<10} "
          f"{'BONES QoE':<11} {'FL QoE':<9} {'QoE增益':<9}")
    print("-" * 80)
    for trace_name in traces:
        if trace_name in all_results:
            r = all_results[trace_name]
            print(f"{trace_name:<14} {r['bones']['vmaf']:<12.2f} "
                  f"{r['frame_level']['vmaf']:<10.2f} {r['gain']['vmaf']:<+10.2f} "
                  f"{r['bones']['qoe']:<11.2f} {r['frame_level']['qoe']:<9.2f} "
                  f"{r['gain']['qoe']:<+9.2f}")

    # 保存结果
    output_path = r"E:\dev\Code\Python\bones\workspace\bones_vs_framelevel_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n结果已保存到 {output_path}")


if __name__ == "__main__":
    main()
