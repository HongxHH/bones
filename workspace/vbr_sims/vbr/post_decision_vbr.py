"""
决策后应用VBR提升：BONES用原始CBR数据做ABR决策，决策完成后对每段质量应用VBR提升量

这是最干净的验证方式：ABR决策、文件大小、下载时间、缓冲区行为都保持CBR不变，
只有最终播放质量得到VBR提升，纯粹衡量VBR编码对QoE的贡献。
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


def run_bones_get_history(trace_path, manifest_path, enhance_path):
    """运行BONES仿真，返回决策历史和原始报告"""
    np.random.seed(0)
    random.seed(0)

    log_path = rf"E:\dev\Code\Python\bones\workspace\sim_log_post_vbr.txt"

    args = argparse.Namespace(
        method="BONES",
        log_path=log_path,
        trace_path=trace_path,
        manifest_path=manifest_path,
        enhance_path=enhance_path,
        video_length=None,
        throughput="ewma",
        download_buffer=60,
        enhance_buffer=60,
        bandwidth_multiplier=1.0,
        speed_multiplier=1.0,
        know_seg=False,
        verbose=False,
    )

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
    report = simulator.report()

    # 将numpy数组转换为标量
    for key in report:
        if isinstance(report[key], np.ndarray):
            report[key] = float(report[key].flatten()[0]) if report[key].size > 0 else 0
        elif isinstance(report[key], (np.floating, np.integer)):
            report[key] = float(report[key])

    return report, simulator


def apply_vbr_and_recalculate(simulator, vbr_improvements, cbr_base_vmafs, high_motion_segments):
    """
    在BONES决策完成后，对每段质量应用VBR提升量，重新计算QoE

    vbr_improvements: dict {bitrate_idx: np.array of shape (num_segments,)}
    cbr_base_vmafs: dict {bitrate_idx: np.array of shape (num_segments,)} 实际测量的CBR VMAF基准
    """
    history_down = simulator.buffer_download.history
    history_enh = simulator.buffer_enhance.history
    num_seg = simulator.seg_size.shape[0]

    # 原始质量（不应用VBR）
    quality_vmaf_orig = np.zeros((num_seg,))
    # VBR提升后的质量
    quality_vmaf_vbr = np.zeros((num_seg,))

    for i in range(num_seg):
        action_down = history_down[i]
        action_enh = history_enh[i]

        # 原始质量
        # 对于1080p（action_down==4），BONES将其设为满分100，但实际CBR VMAF是94.31
        # 使用实际CBR VMAF作为基准，更准确反映用户体验
        if action_down == 4 and action_down in cbr_base_vmafs and cbr_base_vmafs[action_down][i] > 0:
            orig_q = cbr_base_vmafs[action_down][i]
        else:
            orig_q = simulator.vmaf_enh_seg[i][action_down][action_enh]
        quality_vmaf_orig[i] = orig_q

        # VBR提升后的质量
        improvement = 0
        if action_down in vbr_improvements:
            improvement = vbr_improvements[action_down][i]
        quality_vmaf_vbr[i] = orig_q + improvement

    # 重新计算指标（原始）
    oscillation_orig = 0
    for i in range(1, num_seg):
        oscillation_orig += abs(quality_vmaf_orig[i] - quality_vmaf_orig[i - 1])
    oscillation_orig /= (num_seg - 1)
    avg_quality_orig = np.mean(quality_vmaf_orig)

    # 重新计算指标（VBR）
    oscillation_vbr = 0
    for i in range(1, num_seg):
        oscillation_vbr += abs(quality_vmaf_vbr[i] - quality_vmaf_vbr[i - 1])
    oscillation_vbr /= (num_seg - 1)
    avg_quality_vbr = np.mean(quality_vmaf_vbr)

    # 卡顿和带宽不变（决策不变）
    freeze_time = simulator.buffer_download.idle_time
    avg_freeze = freeze_time / num_seg
    total_bandwidth = np.sum([simulator.seg_size[i][history_down[i]] for i in range(num_seg)])
    avg_bandwidth = total_bandwidth / num_seg / simulator.seg_time

    # QoE公式：qoe = quality - oscillation - avg_freeze / 10
    qoe_orig = avg_quality_orig - oscillation_orig - avg_freeze / 10
    qoe_vbr = avg_quality_vbr - oscillation_vbr - avg_freeze / 10

    return {
        "original": {
            "average quality vmaf": float(avg_quality_orig),
            "average oscillation vmaf": float(oscillation_orig),
            "freeze time": float(freeze_time),
            "average bandwidth": float(avg_bandwidth),
            "qoe": float(qoe_orig),
        },
        "vbr": {
            "average quality vmaf": float(avg_quality_vbr),
            "average oscillation vmaf": float(oscillation_vbr),
            "freeze time": float(freeze_time),
            "average bandwidth": float(avg_bandwidth),
            "qoe": float(qoe_vbr),
        },
        "diff": {
            "average quality vmaf": float(avg_quality_vbr - avg_quality_orig),
            "average oscillation vmaf": float(oscillation_vbr - oscillation_orig),
            "freeze time": 0.0,
            "average bandwidth": 0.0,
            "qoe": float(qoe_vbr - qoe_orig),
        },
        "download_history": history_down,
        "enhance_history": history_enh,
    }


def load_vbr_improvements(intermediate_dir, num_segments=159):
    """加载VBR提升量数据和实际CBR VMAF基准"""
    intermediate_dir = Path(intermediate_dir)
    improvements = {}
    cbr_base_vmafs = {}

    for data_file in intermediate_dir.glob("*_data.json"):
        with open(data_file) as f:
            data = json.load(f)
        bitrate_idx = data['bitrate_idx']
        cbr_vmafs = np.array(data['cbr_vmafs'])
        vbr_vmafs = np.array(data['vbr_vmafs'])
        imp = vbr_vmafs - cbr_vmafs
        # 处理测量失败的段（提升量为0）
        imp[cbr_vmafs <= 0] = 0
        imp[vbr_vmafs <= 0] = 0
        improvements[bitrate_idx] = imp
        cbr_base_vmafs[bitrate_idx] = cbr_vmafs

        valid_cbr = cbr_vmafs[cbr_vmafs > 0]
        print(f"  码率档{bitrate_idx} ({data['resolution']}): "
              f"CBR基准={np.mean(valid_cbr):.2f}, "
              f"平均提升={np.mean(imp[imp != 0]):.2f}, "
              f"有效段数={np.sum(imp != 0)}/{num_segments}")

    return improvements, cbr_base_vmafs


def main():
    # 配置
    manifest_path = r"E:\dev\Code\Python\bones\data\bbb\bbb_video_info.json"
    enhance_path = r"E:\dev\Code\Python\bones\data\bbb\imdn_bbb_2080ti.json"
    features_path = r"E:\dev\Code\Python\bones\data\bbb\content_features_bbb.json"
    intermediate_dir = r"E:\dev\Code\Python\bones\data\bbb_vbr_integrated\intermediate"
    trace_root = r"E:\dev\Code\Python\bones\data\trace"

    num_segments = 159

    # 选择全部真实网络轨迹（3Glogs + 4Glogs + 根目录）
    trace_files = []
    for trace_dir in ["3Glogs", "4Glogs", ""]:
        dir_path = os.path.join(trace_root, trace_dir) if trace_dir else trace_root
        if os.path.isdir(dir_path):
            for f in sorted(os.listdir(dir_path)):
                if f.endswith(".json"):
                    trace_type = trace_dir if trace_dir else "root"
                    trace_files.append((trace_type, os.path.join(dir_path, f)))

    print(f"共加载 {len(trace_files)} 条网络轨迹")
    print()

    print("=" * 80)
    print("决策后应用VBR提升：纯净QoE验证")
    print("=" * 80)
    print(f"ABR决策使用原始CBR数据，决策完成后应用VBR提升量")
    print(f"文件大小、下载时间、缓冲区行为、卡顿都保持CBR不变")
    print()

    # 加载VBR提升量和实际CBR VMAF基准
    print("加载VBR提升量数据...")
    vbr_improvements, cbr_base_vmafs = load_vbr_improvements(intermediate_dir, num_segments)
    print()

    # 加载高运动段信息
    with open(features_path) as f:
        features = json.load(f)
    motion = np.array([f["motion_intensity"] for f in features["features"][:num_segments]])
    sorted_idx = np.argsort(-motion)
    num_high = int(num_segments * 0.60)
    high_motion_segments = set(sorted_idx[:num_high].tolist())

    all_results = []

    for trace_type, trace_path in trace_files:
        trace_name = os.path.basename(trace_path)
        print(f"处理轨迹: [{trace_type}] {trace_name}")

        # 运行BONES，获取决策历史
        print(f"  运行BONES仿真（CBR决策）...")
        orig_report, simulator = run_bones_get_history(trace_path, manifest_path, enhance_path)

        # 决策后应用VBR提升，重新计算QoE
        print(f"  决策后应用VBR提升...")
        result = apply_vbr_and_recalculate(simulator, vbr_improvements, cbr_base_vmafs, high_motion_segments)

        all_results.append({
            "trace_type": trace_type,
            "trace_name": trace_name,
            **result["diff"],
            "original": result["original"],
            "vbr": result["vbr"],
        })

        print(f"    原始: QoE={result['original']['qoe']:.2f}, VMAF={result['original']['average quality vmaf']:.2f}, "
              f"振荡={result['original']['average oscillation vmaf']:.2f}, 卡顿={result['original']['freeze time']:.0f}ms")
        print(f"    VBR:  QoE={result['vbr']['qoe']:.2f}, VMAF={result['vbr']['average quality vmaf']:.2f}, "
              f"振荡={result['vbr']['average oscillation vmaf']:.2f}, 卡顿={result['vbr']['freeze time']:.0f}ms")
        print(f"    差异: QoE={result['diff']['qoe']:+.2f}, VMAF={result['diff']['average quality vmaf']:+.2f}, "
              f"振荡={result['diff']['average oscillation vmaf']:+.2f}")
        print()

    # 汇总
    print("=" * 80)
    print("汇总对比")
    print("=" * 80)
    print(f"{'轨迹':<35} {'类型':<5} {'QoE差异':<10} {'VMAF差异':<10} {'振荡差异':<10}")
    print("-" * 75)
    for r in all_results:
        print(f"{r['trace_name']:<35} {r['trace_type']:<5} {r['qoe']:<+10.2f} "
              f"{r['average quality vmaf']:<+10.2f} {r['average oscillation vmaf']:<+10.2f}")

    # 按类型汇总
    print()
    print("按网络类型汇总:")
    for trace_type in ["4G", "3G"]:
        type_results = [r for r in all_results if r["trace_type"] == trace_type]
        if not type_results:
            continue
        avg_qoe = np.mean([r["qoe"] for r in type_results])
        avg_vmaf = np.mean([r["average quality vmaf"] for r in type_results])
        avg_osc = np.mean([r["average oscillation vmaf"] for r in type_results])
        print(f"  {trace_type}: QoE={avg_qoe:+.2f}, VMAF={avg_vmaf:+.2f}, 振荡={avg_osc:+.2f}")

    # 总体汇总
    avg_qoe = np.mean([r["qoe"] for r in all_results])
    avg_vmaf = np.mean([r["average quality vmaf"] for r in all_results])
    avg_osc = np.mean([r["average oscillation vmaf"] for r in all_results])
    print(f"\n总体: QoE={avg_qoe:+.2f}, VMAF={avg_vmaf:+.2f}, 振荡={avg_osc:+.2f}")

    # 保存结果
    output_path = r"E:\dev\Code\Python\bones\workspace\post_decision_vbr_full_traces.json"
    save_results = []
    for r in all_results:
        save_r = {k: v for k, v in r.items() if k not in ["original", "vbr"]}
        save_results.append(save_r)
    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n结果已保存到: {output_path}")

    # VBR胜率统计
    win_count = sum(1 for r in all_results if r["qoe"] > 0)
    total_count = len(all_results)
    print(f"\nVBR胜率: {win_count}/{total_count} = {win_count/total_count*100:.1f}%")

    # 按QoE提升幅度统计
    large_gain = sum(1 for r in all_results if r["qoe"] > 1.0)
    medium_gain = sum(1 for r in all_results if 0 < r["qoe"] <= 1.0)
    small_loss = sum(1 for r in all_results if -0.5 <= r["qoe"] <= 0)
    large_loss = sum(1 for r in all_results if r["qoe"] < -0.5)
    print(f"QoE提升>1.0: {large_gain}条 ({large_gain/total_count*100:.1f}%)")
    print(f"QoE提升0~1.0: {medium_gain}条 ({medium_gain/total_count*100:.1f}%)")
    print(f"QoE变化-0.5~0: {small_loss}条 ({small_loss/total_count*100:.1f}%)")
    print(f"QoE下降>0.5: {large_loss}条 ({large_loss/total_count*100:.1f}%)")


if __name__ == "__main__":
    main()
