"""
用MD-QoE重新评估三种VBR策略（二值、平方根、4档）

在少量轨迹上测试，对比BONES QoE和MD-QoE下的方案排名。
"""
import sys
sys.path.append(r"E:\dev\Code\Python\bones")
import os
import json
import argparse
import numpy as np
import random
from control import BONES
from md_qoe import calculate_md_qoe, calculate_bones_qoe


def run_bones_get_history(trace_path, manifest_path, enhance_path):
    """运行BONES仿真，返回决策历史和逐段数据"""
    np.random.seed(0)
    random.seed(0)

    log_path = rf"E:\dev\Code\Python\bones\workspace\sim_log_md_qoe.txt"

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

    # 逐段卡顿时间
    freeze_time_per_seg = np.zeros(num_seg)
    # 从simulator的下载历史中获取卡顿时间
    # 简化：用总卡顿时间平均分配（实际应该逐段记录）
    total_freeze = simulator.buffer_download.idle_time
    # 更精确的方式：从network trace和下载时间计算
    # 这里简化处理，用平均分配
    freeze_time_per_seg[:] = total_freeze / num_seg

    return {
        "history_down": history_down,
        "quality_vmaf_cbr": quality_vmaf_cbr,
        "freeze_time_per_seg": freeze_time_per_seg,
        "total_freeze": total_freeze,
        "num_seg": num_seg,
    }


def apply_vbr_improvement(quality_vmaf_cbr, history_down, vbr_improvements, cbr_base_vmafs):
    """应用VBR提升量，得到VBR后的逐段VMAF

    注意：CBR和VBR的1080p质量都用实际VMAF（而不是BONES假设的100），
    这样对比才公平。
    """
    num_seg = len(quality_vmaf_cbr)
    quality_vmaf_vbr = quality_vmaf_cbr.copy()

    for i in range(num_seg):
        action_down = history_down[i]

        # 1080p特殊处理：CBR和VBR都用实际CBR VMAF作基准
        if action_down == 4 and action_down in cbr_base_vmafs and i < len(cbr_base_vmafs[action_down]):
            actual_cbr = cbr_base_vmafs[action_down][i]
            if actual_cbr > 0:
                quality_vmaf_vbr[i] = actual_cbr
                # 同时修改CBR的质量（保持一致）
                quality_vmaf_cbr[i] = actual_cbr

        # 应用VBR提升量
        if action_down in vbr_improvements and i < len(vbr_improvements[action_down]):
            improvement = vbr_improvements[action_down][i]
            quality_vmaf_vbr[i] += improvement

    return quality_vmaf_cbr, quality_vmaf_vbr


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


from pathlib import Path


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

    # 选择少量测试轨迹（3G和4G各5条）
    test_traces = []
    for trace_dir in ["3Glogs", "4Glogs"]:
        dir_path = os.path.join(trace_root, trace_dir)
        files = sorted([f for f in os.listdir(dir_path) if f.endswith(".json")])[:5]
        for f in files:
            test_traces.append((trace_dir, os.path.join(dir_path, f)))

    print(f"测试轨迹数: {len(test_traces)}")
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

    for trace_type, trace_path in test_traces:
        trace_name = os.path.basename(trace_path)
        print(f"处理轨迹: [{trace_type}] {trace_name}")

        # 运行BONES仿真（CBR决策）
        sim_data = run_bones_get_history(trace_path, manifest_path, enhance_path)

        # 计算CBR的QoE
        cbr_bones = calculate_bones_qoe(sim_data["quality_vmaf_cbr"], sim_data["freeze_time_per_seg"])
        cbr_md = calculate_md_qoe(sim_data["quality_vmaf_cbr"], sim_data["freeze_time_per_seg"])

        trace_result = {
            "trace_name": trace_name,
            "trace_type": trace_type,
            "cbr_bones_qoe": cbr_bones["qoe"],
            "cbr_md_qoe": cbr_md["md_qoe"],
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

            # 重新计算CBR的QoE（用修正后的1080p质量）
            cbr_bones = calculate_bones_qoe(quality_cbr_corrected, sim_data["freeze_time_per_seg"])
            cbr_md = calculate_md_qoe(quality_cbr_corrected, sim_data["freeze_time_per_seg"])

            vbr_bones = calculate_bones_qoe(quality_vmaf_vbr, sim_data["freeze_time_per_seg"])
            vbr_md = calculate_md_qoe(quality_vmaf_vbr, sim_data["freeze_time_per_seg"])

            trace_result["strategies"][vbr_name] = {
                "bones_qoe": vbr_bones["qoe"],
                "bones_qoe_diff": vbr_bones["qoe"] - cbr_bones["qoe"],
                "md_qoe": vbr_md["md_qoe"],
                "md_qoe_diff": vbr_md["md_qoe"] - cbr_md["md_qoe"],
                "avg_vmaf": vbr_bones["avg_vmaf"],
                "oscillation": vbr_bones["oscillation"],
                "perceived_quality": vbr_md["perceived_quality"],
                "temporal_smoothness": vbr_md["temporal_smoothness"],
                "cbr_avg_vmaf": cbr_bones["avg_vmaf"],
                "cbr_oscillation": cbr_bones["oscillation"],
            }

        all_results.append(trace_result)

        # 打印该轨迹的结果
        print(f"  CBR: BONES_QoE={cbr_bones['qoe']:.2f}, MD_QoE={cbr_md['md_qoe']:.2f}")
        for vbr_name in vbr_strategies:
            s = trace_result["strategies"][vbr_name]
            print(f"  {vbr_name}: BONES_QoE={s['bones_qoe']:.2f} ({s['bones_qoe_diff']:+.2f}), "
                  f"MD_QoE={s['md_qoe']:.2f} ({s['md_qoe_diff']:+.2f}), "
                  f"VMAF={s['avg_vmaf']:.2f}, 振荡={s['oscillation']:.2f}")
        print()

    # 汇总对比
    print("=" * 80)
    print("汇总对比（10条测试轨迹）")
    print("=" * 80)
    print(f"{'策略':<15} {'BONES_QoE提升':<15} {'MD_QoE提升':<15} {'VMAF提升':<12} {'振荡变化':<12}")
    print("-" * 75)

    for vbr_name in vbr_strategies:
        avg_bones_diff = np.mean([r["strategies"][vbr_name]["bones_qoe_diff"] for r in all_results])
        avg_md_diff = np.mean([r["strategies"][vbr_name]["md_qoe_diff"] for r in all_results])
        avg_vmaf_diff = np.mean([r["strategies"][vbr_name]["avg_vmaf"] - r["strategies"][vbr_name]["cbr_avg_vmaf"] for r in all_results])
        avg_osc_diff = np.mean([r["strategies"][vbr_name]["oscillation"] - r["strategies"][vbr_name]["cbr_oscillation"] for r in all_results])

        print(f"{vbr_name:<15} {avg_bones_diff:<+15.2f} {avg_md_diff:<+15.2f} "
              f"{avg_vmaf_diff:<+12.2f} {avg_osc_diff:<+12.2f}")

    # 保存结果
    output_path = r"E:\dev\Code\Python\bones\workspace\md_qoe_comparison.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
