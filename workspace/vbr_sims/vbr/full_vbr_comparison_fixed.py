"""
全轨迹VBR验证（修复1080p bug版）

对比三种VBR策略（二值、平方根、4档）在131条真实网络轨迹上的
BONES_QoE和MD_QoE，修复1080p处理bug（CBR和VBR都用实际VMAF）。
"""
import sys
sys.path.append(r"E:\dev\Code\Python\bones")
import os
import json
import argparse
import numpy as np
import random
from pathlib import Path
from control import BONES
from md_qoe import calculate_md_qoe, calculate_bones_qoe


def run_bones_get_history(trace_path, manifest_path, enhance_path):
    """运行BONES仿真，返回决策历史和逐段数据"""
    np.random.seed(0)
    random.seed(0)

    log_path = rf"E:\dev\Code\Python\bones\workspace\sim_log_full_vbr_fixed.txt"

    from config import BONESConfig
    config = BONESConfig()
    parser = argparse.ArgumentParser()
    args = config.add_args(parser)
    args.method = "BONES"
    args.log_path = log_path
    args.trace_path = trace_path
    args.manifest_path = manifest_path
    args.enhance_path = enhance_path
    args.video_length = None
    args.throughput = "ewma"
    args.download_buffer = 60
    args.enhance_buffer = 60
    args.bandwidth_multiplier = 1.0
    args.speed_multiplier = 1.0
    args.know_seg = False
    args.verbose = False

    simulator = BONES(args)
    simulator.run()

    # 获取逐段数据
    history_down = simulator.buffer_download.history
    num_seg = len(history_down)

    # 逐段VMAF（CBR，未应用VBR提升）
    quality_vmaf_cbr = np.zeros(num_seg)
    for i in range(num_seg):
        action_down = history_down[i]
        action_enh = simulator.buffer_enhance.history[i] if i < len(simulator.buffer_enhance.history) else 0
        quality_vmaf_cbr[i] = simulator.vmaf_enh_seg[i][action_down][action_enh]

    # 逐段卡顿时间（简化：用总卡顿时间平均分配）
    total_freeze = simulator.buffer_download.idle_time
    freeze_time_per_seg = np.full(num_seg, total_freeze / num_seg)

    return {
        "history_down": history_down,
        "quality_vmaf_cbr": quality_vmaf_cbr,
        "freeze_time_per_seg": freeze_time_per_seg,
        "total_freeze": total_freeze,
        "num_seg": num_seg,
    }


def apply_vbr_improvement(quality_vmaf_cbr, history_down, vbr_improvements, cbr_base_vmafs):
    """应用VBR提升量，同时修正CBR的1080p质量（CBR和VBR都用实际VMAF）"""
    num_seg = len(quality_vmaf_cbr)
    quality_cbr_corrected = quality_vmaf_cbr.copy()
    quality_vmaf_vbr = quality_vmaf_cbr.copy()

    for i in range(num_seg):
        action_down = history_down[i]

        # 1080p特殊处理：CBR和VBR都用实际CBR VMAF作基准
        if action_down == 4 and action_down in cbr_base_vmafs and i < len(cbr_base_vmafs[action_down]):
            actual_cbr = cbr_base_vmafs[action_down][i]
            if actual_cbr > 0:
                quality_cbr_corrected[i] = actual_cbr
                quality_vmaf_vbr[i] = actual_cbr

        # 应用VBR提升量
        if action_down in vbr_improvements and i < len(vbr_improvements[action_down]):
            improvement = vbr_improvements[action_down][i]
            quality_vmaf_vbr[i] += improvement

    return quality_cbr_corrected, quality_vmaf_vbr


def load_vbr_data(intermediate_dir):
    """加载VBR提升量数据和实际CBR VMAF基准"""
    intermediate_dir = Path(intermediate_dir)
    improvements = {}
    cbr_base_vmafs = {}

    res_to_idx = {"240p": 0, "360p": 1, "480p": 2, "720p": 3, "1080p": 4}

    for data_file in intermediate_dir.glob("*_data.json"):
        with open(data_file) as f:
            data = json.load(f)
        res_name = data_file.stem.replace("_data", "")
        bitrate_idx = res_to_idx.get(res_name, -1)
        if bitrate_idx < 0:
            continue

        cbr_vmafs = np.array(data['cbr_vmafs'])
        vbr_vmafs = np.array(data['vbr_vmafs'])
        imp = vbr_vmafs - cbr_vmafs
        imp[cbr_vmafs <= 0] = 0
        imp[vbr_vmafs <= 0] = 0
        improvements[bitrate_idx] = imp
        cbr_base_vmafs[bitrate_idx] = cbr_vmafs

    return improvements, cbr_base_vmafs


def main():
    # 配置
    manifest_path = r"E:\dev\Code\Python\bones\data\bbb\bbb_video_info.json"
    enhance_path = r"E:\dev\Code\Python\bones\data\bbb\imdn_bbb_2080ti.json"
    trace_root = r"E:\dev\Code\Python\bones\data\trace"

    # 三种VBR策略的数据目录
    vbr_strategies = {
        "二值分配": r"E:\dev\Code\Python\bones\data\bbb_vbr_integrated\intermediate",
        "平方根分配": r"E:\dev\Code\Python\bones\data\bbb_vbr_sqrt\intermediate",
        "4档分配": r"E:\dev\Code\Python\bones\data\bbb_vbr_4stair\intermediate",
    }

    # 选择全部真实网络轨迹
    trace_files = []
    for trace_dir in ["3Glogs", "4Glogs", ""]:
        dir_path = os.path.join(trace_root, trace_dir) if trace_dir else trace_root
        if os.path.isdir(dir_path):
            for f in sorted(os.listdir(dir_path)):
                if f.endswith(".json"):
                    trace_type = trace_dir if trace_dir else "root"
                    trace_files.append((trace_type, os.path.join(dir_path, f)))

    print(f"全轨迹VBR验证（修复1080p bug版）")
    print(f"轨迹数: {len(trace_files)}")
    print(f"策略: {list(vbr_strategies.keys())}")
    print()

    # 加载三种VBR策略的数据
    vbr_data = {}
    for name, data_dir in vbr_strategies.items():
        print(f"加载{name}数据...")
        improvements, cbr_base_vmafs = load_vbr_data(data_dir)
        vbr_data[name] = {"improvements": improvements, "cbr_base_vmafs": cbr_base_vmafs}
    print()

    # 对每条轨迹运行仿真并计算两种QoE
    all_results = []

    for idx, (trace_type, trace_path) in enumerate(trace_files):
        trace_name = os.path.basename(trace_path)
        print(f"[{idx+1}/{len(trace_files)}] [{trace_type}] {trace_name}")

        try:
            # 运行BONES仿真（CBR决策）
            sim_data = run_bones_get_history(trace_path, manifest_path, enhance_path)

            trace_result = {
                "trace_name": trace_name,
                "trace_type": trace_type,
                "strategies": {},
            }

            # 对每种VBR策略计算QoE
            for vbr_name, data in vbr_data.items():
                # 应用VBR提升量（同时修正CBR的1080p质量）
                quality_cbr_corrected, quality_vmaf_vbr = apply_vbr_improvement(
                    sim_data["quality_vmaf_cbr"].copy(),
                    sim_data["history_down"],
                    data["improvements"],
                    data["cbr_base_vmafs"]
                )

                # 计算CBR和VBR的QoE
                cbr_bones = calculate_bones_qoe(quality_cbr_corrected, sim_data["freeze_time_per_seg"])
                cbr_md = calculate_md_qoe(quality_cbr_corrected, sim_data["freeze_time_per_seg"])
                vbr_bones = calculate_bones_qoe(quality_vmaf_vbr, sim_data["freeze_time_per_seg"])
                vbr_md = calculate_md_qoe(quality_vmaf_vbr, sim_data["freeze_time_per_seg"])

                trace_result["strategies"][vbr_name] = {
                    "cbr_bones_qoe": cbr_bones["qoe"],
                    "cbr_md_qoe": cbr_md["md_qoe"],
                    "vbr_bones_qoe": vbr_bones["qoe"],
                    "vbr_md_qoe": vbr_md["md_qoe"],
                    "bones_qoe_diff": vbr_bones["qoe"] - cbr_bones["qoe"],
                    "md_qoe_diff": vbr_md["md_qoe"] - cbr_md["md_qoe"],
                    "vmaf_diff": vbr_bones["avg_vmaf"] - cbr_bones["avg_vmaf"],
                    "oscillation_diff": vbr_bones["oscillation"] - cbr_bones["oscillation"],
                }

            all_results.append(trace_result)

            # 打印该轨迹的结果
            for vbr_name in vbr_strategies:
                s = trace_result["strategies"][vbr_name]
                print(f"  {vbr_name}: BONES_QoE={s['bones_qoe_diff']:+.2f}, "
                      f"MD_QoE={s['md_qoe_diff']:+.2f}, "
                      f"VMAF={s['vmaf_diff']:+.2f}, 振荡={s['oscillation_diff']:+.2f}")

        except Exception as e:
            print(f"  错误: {e}")
        print()

    # 汇总对比
    print("=" * 90)
    print("汇总对比（131条全轨迹，修复1080p bug后）")
    print("=" * 90)
    print(f"{'策略':<15} {'BONES_QoE提升':<15} {'MD_QoE提升':<15} {'VMAF提升':<12} {'振荡变化':<12} {'胜率':<10}")
    print("-" * 85)

    for vbr_name in vbr_strategies:
        avg_bones_diff = np.mean([r["strategies"][vbr_name]["bones_qoe_diff"] for r in all_results])
        avg_md_diff = np.mean([r["strategies"][vbr_name]["md_qoe_diff"] for r in all_results])
        avg_vmaf_diff = np.mean([r["strategies"][vbr_name]["vmaf_diff"] for r in all_results])
        avg_osc_diff = np.mean([r["strategies"][vbr_name]["oscillation_diff"] for r in all_results])
        win_rate = np.mean([1 if r["strategies"][vbr_name]["bones_qoe_diff"] > 0 else 0 for r in all_results]) * 100

        print(f"{vbr_name:<15} {avg_bones_diff:<+15.2f} {avg_md_diff:<+15.2f} "
              f"{avg_vmaf_diff:<+12.2f} {avg_osc_diff:<+12.2f} {win_rate:<10.1f}%")

    # 按网络类型汇总
    print("\n按网络类型汇总（BONES_QoE提升）:")
    for trace_type in ["3Glogs", "4Glogs", "root"]:
        type_results = [r for r in all_results if r["trace_type"] == trace_type]
        if not type_results:
            continue
        print(f"  {trace_type} ({len(type_results)}条):")
        for vbr_name in vbr_strategies:
            avg_bones_diff = np.mean([r["strategies"][vbr_name]["bones_qoe_diff"] for r in type_results])
            avg_md_diff = np.mean([r["strategies"][vbr_name]["md_qoe_diff"] for r in type_results])
            print(f"    {vbr_name}: BONES_QoE={avg_bones_diff:+.2f}, MD_QoE={avg_md_diff:+.2f}")

    # 保存结果
    output_path = r"E:\dev\Code\Python\bones\workspace\full_vbr_comparison_fixed.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
