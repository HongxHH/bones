"""
PAC-Net 多轨迹批量对比实验

对选定的 3G/4G trace，依次跑 PACNet 和基线 imdn_always，
完成后汇总 QoE/VMAF/stall 等指标对比表。

用法：python run_multi_trace.py
"""
import os
import sys
import json
import subprocess
import time
from pathlib import Path

PLAYER_DIR = r"E:\dev\Code\Python\bones\player"
OUTPUT_DIR = os.path.join(PLAYER_DIR, "output", "data")
PYTHON = r"E:\dev\env\miniconda3\envs\SR\python.exe"
LOG_DIR = r"E:\dev\Code\Python\bones\workspace\pacnet\multi_trace_logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 选定的 trace（3G 3条不同日期 + 4G 3条不同场景）
TRACES = [
    ("3G", "report.2010-09-13_1003CEST.json", "data/3Glogs/report.2010-09-13_1003CEST.json"),
    ("3G", "report.2010-09-14_1038CEST.json", "data/3Glogs/report.2010-09-14_1038CEST.json"),
    ("3G", "report.2010-09-21_1001CEST.json", "data/3Glogs/report.2010-09-21_1001CEST.json"),
    ("4G", "report_bicycle_0001.json", "data/4Glogs/report_bicycle_0001.json"),
    ("4G", "report_bus_0001.json", "data/4Glogs/report_bus_0001.json"),
    ("4G", "report_car_0001.json", "data/4Glogs/report_car_0001.json"),
]

ENHANCERS = [
    ("pacnet", "PAC-Net"),
    ("imdn_always", "基线IMDN"),
]


def run_experiment(trace_path, enhancer, tag):
    """跑一次播放器实验，返回结果文件路径"""
    cmd = [
        PYTHON, "play.py",
        "--mod_nes", "bones",
        "--mod_enhancer", enhancer,
        "--mod_renderer", "headless",
        "--mod_analyzer", "data_collector",
        "--input", "data/video_mp4box/bbb.mpd",
        "--trace", trace_path,
    ]
    log_file = os.path.join(LOG_DIR, f"{tag}.log")
    err_file = os.path.join(LOG_DIR, f"{tag}.err")
    print(f"  Running: {tag}")
    t0 = time.time()
    with open(log_file, "w") as lf, open(err_file, "w") as ef:
        proc = subprocess.run(cmd, cwd=PLAYER_DIR, stdout=lf, stderr=ef, timeout=1800)
    elapsed = time.time() - t0
    print(f"    Done in {elapsed:.0f}s, returncode={proc.returncode}")

    # 找最新的结果文件
    results = list(Path(OUTPUT_DIR).glob(f"bones_{enhancer}_*{tag.split('_')[-1]}*.json"))
    if not results:
        # 用更宽松的匹配
        results = sorted(Path(OUTPUT_DIR).glob(f"bones_{enhancer}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if results:
        return str(results[0])
    return None


def read_result(result_path):
    """读取结果JSON，提取关键指标"""
    if not result_path or not os.path.exists(result_path):
        return None
    with open(result_path, encoding="utf-8") as f:
        data = json.load(f)
    return {
        "qoe": data.get("QOE"),
        "avg_vmaf": data.get("avg_vmaf"),
        "oscillation": data.get("oscillation_vmaf"),
        "num_stall": data.get("num_stall"),
        "dur_stall": data.get("dur_stall"),
        "num_switches": data.get("num_quality_switches"),
        "fast_complete": data.get("fast_complete_segments_count"),
        "avg_bitrate": data.get("avg_bitrate"),
    }


def main():
    all_results = {}

    for net_type, trace_name, trace_path in TRACES:
        trace_tag = trace_name.replace(".json", "")
        print(f"\n{'='*60}")
        print(f"Trace: {net_type} / {trace_name}")
        print(f"{'='*60}")

        for enhancer, enh_name in ENHANCERS:
            tag = f"{net_type}_{trace_tag}_{enhancer}"
            result_path = run_experiment(trace_path, enhancer, tag)
            metrics = read_result(result_path)
            all_results[tag] = {
                "net_type": net_type,
                "trace": trace_name,
                "enhancer": enhancer,
                "enh_name": enh_name,
                "metrics": metrics,
                "result_file": result_path,
            }
            if metrics:
                print(f"    QoE={metrics['qoe']:.2f} VMAF={metrics['avg_vmaf']:.2f} "
                      f"stall={metrics['num_stall']}({metrics['dur_stall']:.1f}s) "
                      f"switches={metrics['num_switches']} fast={metrics['fast_complete']}")
            else:
                print(f"    FAILED to read result")

    # 汇总对比表
    print(f"\n{'='*80}")
    print("多轨迹汇总对比")
    print(f"{'='*80}")
    print(f"{'网络':<6} {'Trace':<35} {'指标':<12} {'基线':>10} {'PACNet':>10} {'差值':>10}")
    print("-" * 80)

    summary = {"3G": {"wins": 0, "total": 0, "qoe_diff": []},
               "4G": {"wins": 0, "total": 0, "qoe_diff": []}}

    for net_type, trace_name, _ in TRACES:
        trace_tag = trace_name.replace(".json", "")
        base_key = f"{net_type}_{trace_tag}_imdn_always"
        pac_key = f"{net_type}_{trace_tag}_pacnet"
        base = all_results.get(base_key, {}).get("metrics")
        pac = all_results.get(pac_key, {}).get("metrics")
        if not base or not pac:
            continue
        for metric, label in [("qoe", "QoE"), ("avg_vmaf", "VMAF"),
                               ("dur_stall", "stall(s)"), ("num_switches", "switches")]:
            bv = base[metric]
            pv = pac[metric]
            diff = pv - bv if bv is not None and pv is not None else None
            if diff is not None:
                print(f"{net_type:<6} {trace_name:<35} {label:<12} {bv:>10.2f} {pv:>10.2f} {diff:>+10.2f}")
        print()
        # 胜率统计
        if pac["qoe"] > base["qoe"]:
            summary[net_type]["wins"] += 1
        summary[net_type]["total"] += 1
        summary[net_type]["qoe_diff"].append(pac["qoe"] - base["qoe"])

    print(f"\n=== 胜率统计 ===")
    for net_type in ["3G", "4G"]:
        s = summary[net_type]
        if s["total"] > 0:
            avg_diff = sum(s["qoe_diff"]) / len(s["qoe_diff"])
            print(f"{net_type}: {s['wins']}/{s['total']} 胜, 平均QoE差值={avg_diff:+.2f}")

    # 保存汇总
    output_path = r"E:\dev\Code\Python\bones\workspace\pacnet\multi_trace_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存到 {output_path}")


if __name__ == "__main__":
    main()
