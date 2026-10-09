"""
CBR vs VBR 端到端QoE对比实验

使用BONES算法，在相同的网络轨迹上分别用CBR和VBR的manifest/enhance文件运行仿真，对比QoE。
"""
import sys
sys.path.append(r"E:\dev\Code\Python\bones")
import os
import json
import argparse
import numpy as np
import random
from control import BONES


def run_single(trace_path, manifest_path, enhance_path, method="BONES", log_suffix=""):
    """运行单次仿真，返回报告"""
    np.random.seed(0)
    random.seed(0)

    log_path = rf"E:\dev\Code\Python\bones\workspace\sim_log_{log_suffix}.txt"

    args = argparse.Namespace(
        method=method,
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

    # 添加BONES特定参数
    from config import BONESConfig
    config = BONESConfig()
    parser = argparse.ArgumentParser()
    args = config.add_args(parser)
    # 重新设置基础参数
    args.method = method
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
    return report


def main():
    # 配置
    cbr_manifest = r"E:\dev\Code\Python\bones\data\bbb\bbb_video_info.json"
    cbr_enhance = r"E:\dev\Code\Python\bones\data\bbb\imdn_bbb_2080ti.json"
    # 纯净VBR：manifest保持CBR不变（文件大小不变），只提升enhance中的VMAF
    vbr_manifest = r"E:\dev\Code\Python\bones\data\bbb\bbb_video_info.json"
    vbr_enhance = r"E:\dev\Code\Python\bones\data\bbb_vbr_pure\imdn_bbb_2080ti_pure_vbr.json"

    trace_root = r"E:\dev\Code\Python\bones\data\trace"

    # 选择代表性轨迹（4G和3G各3条）
    trace_files = [
        ("4G", os.path.join(trace_root, "4Glogs", "report_bicycle_0001.json")),
        ("4G", os.path.join(trace_root, "4Glogs", "report_bus_0001.json")),
        ("4G", os.path.join(trace_root, "4Glogs", "report_bus_0002.json")),
        ("3G", os.path.join(trace_root, "3Glogs", "report.2010-09-13_1003CEST.json")),
        ("3G", os.path.join(trace_root, "3Glogs", "report.2010-09-13_1046CEST.json")),
        ("3G", os.path.join(trace_root, "3Glogs", "report.2010-09-14_1038CEST.json")),
    ]

    print("=" * 80)
    print("CBR vs VBR 端到端QoE对比实验（BONES算法）")
    print("=" * 80)
    print(f"CBR manifest: {cbr_manifest}")
    print(f"VBR manifest: {vbr_manifest}")
    print(f"测试轨迹数: {len(trace_files)}")
    print()

    all_results = []

    for trace_type, trace_path in trace_files:
        trace_name = os.path.basename(trace_path)
        print(f"处理轨迹: [{trace_type}] {trace_name}")

        # 运行CBR
        print(f"  运行CBR...")
        cbr_report = run_single(trace_path, cbr_manifest, cbr_enhance, log_suffix=f"cbr_{trace_name}")
        print(f"    QoE={cbr_report['qoe']:.2f}, VMAF={cbr_report['average quality vmaf']:.2f}, "
              f"卡顿={cbr_report['freeze time']:.0f}ms, 带宽={cbr_report['average bandwidth']:.0f}kbps")

        # 运行VBR
        print(f"  运行VBR...")
        vbr_report = run_single(trace_path, vbr_manifest, vbr_enhance, log_suffix=f"vbr_{trace_name}")
        print(f"    QoE={vbr_report['qoe']:.2f}, VMAF={vbr_report['average quality vmaf']:.2f}, "
              f"卡顿={vbr_report['freeze time']:.0f}ms, 带宽={vbr_report['average bandwidth']:.0f}kbps")

        # 计算差异
        diff = {
            "trace_type": trace_type,
            "trace_name": trace_name,
            "qoe_diff": vbr_report["qoe"] - cbr_report["qoe"],
            "vmaf_diff": vbr_report["average quality vmaf"] - cbr_report["average quality vmaf"],
            "oscillation_diff": vbr_report["average oscillation vmaf"] - cbr_report["average oscillation vmaf"],
            "freeze_diff": vbr_report["freeze time"] - cbr_report["freeze time"],
            "bandwidth_diff": vbr_report["average bandwidth"] - cbr_report["average bandwidth"],
            "cbr": cbr_report,
            "vbr": vbr_report,
        }
        all_results.append(diff)

        print(f"  差异: QoE={diff['qoe_diff']:+.2f}, VMAF={diff['vmaf_diff']:+.2f}, "
              f"卡顿={diff['freeze_diff']:+.0f}ms, 带宽={diff['bandwidth_diff']:+.0f}kbps")
        print()

    # 汇总
    print("=" * 80)
    print("汇总对比")
    print("=" * 80)
    print(f"{'轨迹':<35} {'类型':<5} {'QoE差异':<10} {'VMAF差异':<10} {'卡顿差异':<12} {'带宽差异':<12}")
    print("-" * 85)
    for r in all_results:
        print(f"{r['trace_name']:<35} {r['trace_type']:<5} {r['qoe_diff']:<+10.2f} "
              f"{r['vmaf_diff']:<+10.2f} {r['freeze_diff']:<+12.0f} {r['bandwidth_diff']:<+12.0f}")

    # 按类型汇总
    print()
    print("按网络类型汇总:")
    for trace_type in ["4G", "3G"]:
        type_results = [r for r in all_results if r["trace_type"] == trace_type]
        if not type_results:
            continue
        avg_qoe = np.mean([r["qoe_diff"] for r in type_results])
        avg_vmaf = np.mean([r["vmaf_diff"] for r in type_results])
        avg_freeze = np.mean([r["freeze_diff"] for r in type_results])
        avg_bw = np.mean([r["bandwidth_diff"] for r in type_results])
        print(f"  {trace_type}: QoE={avg_qoe:+.2f}, VMAF={avg_vmaf:+.2f}, "
              f"卡顿={avg_freeze:+.0f}ms, 带宽={avg_bw:+.0f}kbps")

    # 总体汇总
    avg_qoe = np.mean([r["qoe_diff"] for r in all_results])
    avg_vmaf = np.mean([r["vmaf_diff"] for r in all_results])
    avg_freeze = np.mean([r["freeze_diff"] for r in all_results])
    avg_bw = np.mean([r["bandwidth_diff"] for r in all_results])
    print(f"\n总体: QoE={avg_qoe:+.2f}, VMAF={avg_vmaf:+.2f}, "
          f"卡顿={avg_freeze:+.0f}ms, 带宽={avg_bw:+.0f}kbps")

    # 保存结果
    output_path = r"E:\dev\Code\Python\bones\workspace\cbr_vs_pure_vbr_results.json"
    # 移除不可序列化的内容
    save_results = []
    for r in all_results:
        save_r = {k: v for k, v in r.items() if k not in ["cbr", "vbr"]}
        save_r["cbr_qoe"] = r["cbr"]["qoe"]
        save_r["vbr_qoe"] = r["vbr"]["qoe"]
        save_results.append(save_r)
    with open(output_path, "w") as f:
        json.dump(save_results, f, indent=2)
    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
