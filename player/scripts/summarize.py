#!/usr/bin/env python3
"""
实验结果汇总与对比脚本
用法:
  python scripts/summarize.py --dir output/data --format markdown
  python scripts/summarize.py --dir output/data --format csv --output results.csv
  python scripts/summarize.py --dir output/data --group-by mod_nes
"""
import argparse
import json
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# 核心指标列表（按显示顺序）
CORE_METRICS = [
    "QOE",
    "avg_vmaf",
    "oscillation_vmaf",
    "avg_freeze",
    "num_stall",
    "dur_stall",
    "num_quality_switches",
    "enhancer_abort_total",
    "fast_complete_saved_frames_total",
    "fast_complete_segments_count",
    "fast_complete_quality_drop_avg",
]

# 从文件名或 config_snapshot 中提取的标识字段
IDENTIFIER_FIELDS = [
    "mod_nes",
    "mod_enhancer",
    "use_fast_complete",
    "input",
    "trace",
]


def load_result_files(directory: str) -> List[Dict[str, Any]]:
    """加载目录下所有结果 JSON 文件"""
    results = []
    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"ERROR: Directory not found: {directory}", file=sys.stderr)
        return results

    for f in sorted(dir_path.glob("*.json")):
        # 跳过配置归档和运行日志文件
        if f.name.startswith("config_") or f.name.startswith("run_log_"):
            continue
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
            data["_filename"] = f.name
            data["_filepath"] = str(f)
            results.append(data)
        except Exception as e:
            print(f"WARNING: Failed to load {f.name}: {e}", file=sys.stderr)
    return results


def extract_identifiers(data: Dict[str, Any]) -> Dict[str, str]:
    """从结果数据中提取标识信息（算法、trace、视频等）"""
    identifiers = {}
    config = data.get("environment", {}).get("config_snapshot", {})

    for field in IDENTIFIER_FIELDS:
        # 优先从 config_snapshot 取
        if field in config:
            val = config[field]
            if field in ("input", "trace"):
                val = os.path.basename(str(val))
            identifiers[field] = str(val)
        # 其次从顶层取
        elif field in data:
            identifiers[field] = str(data[field])
        else:
            identifiers[field] = "?"

    return identifiers


def extract_metrics(data: Dict[str, Any]) -> Dict[str, Any]:
    """从结果数据中提取核心指标"""
    metrics = {}
    for m in CORE_METRICS:
        val = data.get(m, None)
        if isinstance(val, float):
            metrics[m] = round(val, 4)
        else:
            metrics[m] = val
    return metrics


def extract_gpu_info(data: Dict[str, Any]) -> Dict[str, Any]:
    """提取 GPU 和环境信息"""
    env = data.get("environment", {})
    gpu = data.get("gpu_metrics", {})
    startup = data.get("startup_metrics", {})
    return {
        "gpu_name": env.get("gpu_name", "?"),
        "torch_version": env.get("torch_version", "?"),
        "duration_seconds": env.get("duration_seconds", "?"),
        "peak_memory_mb": gpu.get("peak_memory_mb", "?"),
        "time_to_first_play": startup.get("time_to_first_play", "?"),
    }


def group_results(results: List[Dict[str, Any]], group_by: str) -> Dict[str, List[Dict[str, Any]]]:
    """按指定字段分组"""
    groups = {}
    for data in results:
        identifiers = extract_identifiers(data)
        key = identifiers.get(group_by, "?")
        groups.setdefault(key, []).append(data)
    return groups


def compute_stats(values: List[float]) -> Tuple[float, Optional[float]]:
    """计算均值和标准差"""
    if not values:
        return (0.0, None)
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    return (round(mean, 4), round(std, 4) if std is not None else None)


def format_markdown(results: List[Dict[str, Any]], show_gpu: bool = False) -> str:
    """生成 Markdown 格式表格"""
    if not results:
        return "No results found."

    lines = []
    # 表头
    header = ["#", "Algorithm", "Enhancer", "Fast", "Video", "Trace"]
    header += [m for m in CORE_METRICS if any(r.get(m) is not None for r in results)]
    if show_gpu:
        header += ["PeakMem(MB)", "Startup(s)", "Duration(s)"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for i, data in enumerate(results, 1):
        ids = extract_identifiers(data)
        metrics = extract_metrics(data)
        gpu = extract_gpu_info(data) if show_gpu else {}

        row = [
            str(i),
            ids.get("mod_nes", "?"),
            ids.get("mod_enhancer", "?"),
            ids.get("use_fast_complete", "?"),
            ids.get("input", "?"),
            ids.get("trace", "?"),
        ]
        for m in CORE_METRICS:
            if any(r.get(m) is not None for r in results):
                val = metrics.get(m, "?")
                row.append(str(val) if val is not None else "?")
        if show_gpu:
            row.append(str(gpu.get("peak_memory_mb", "?")))
            row.append(str(gpu.get("time_to_first_play", "?")))
            row.append(str(gpu.get("duration_seconds", "?")))
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def format_csv(results: List[Dict[str, Any]], show_gpu: bool = False) -> str:
    """生成 CSV 格式"""
    if not results:
        return "No results found."

    import csv
    import io

    output = io.StringIO()
    header = ["Algorithm", "Enhancer", "Fast", "Video", "Trace"]
    header += CORE_METRICS
    if show_gpu:
        header += ["PeakMem_MB", "Startup_s", "Duration_s", "GPU"]
    writer = csv.writer(output)
    writer.writerow(header)

    for data in results:
        ids = extract_identifiers(data)
        metrics = extract_metrics(data)
        gpu = extract_gpu_info(data) if show_gpu else {}
        row = [
            ids.get("mod_nes", "?"),
            ids.get("mod_enhancer", "?"),
            ids.get("use_fast_complete", "?"),
            ids.get("input", "?"),
            ids.get("trace", "?"),
        ]
        for m in CORE_METRICS:
            row.append(metrics.get(m, ""))
        if show_gpu:
            row.append(gpu.get("peak_memory_mb", ""))
            row.append(gpu.get("time_to_first_play", ""))
            row.append(gpu.get("duration_seconds", ""))
            row.append(gpu.get("gpu_name", ""))
        writer.writerow(row)

    return output.getvalue()


def format_grouped_stats(results: List[Dict[str, Any]], group_by: str) -> str:
    """生成分组统计（均值±标准差）"""
    groups = group_results(results, group_by)
    lines = []
    lines.append(f"### Grouped by: {group_by}")
    lines.append("")

    # 表头
    header = [group_by, "count"] + CORE_METRICS
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")

    for key, group in sorted(groups.items()):
        row = [key, str(len(group))]
        for m in CORE_METRICS:
            values = [r.get(m) for r in group if r.get(m) is not None and isinstance(r.get(m), (int, float))]
            if values:
                mean, std = compute_stats(values)
                if std is not None and len(values) > 1:
                    row.append(f"{mean:.2f}±{std:.2f}")
                else:
                    row.append(f"{mean:.2f}")
            else:
                row.append("?")
        lines.append("| " + " | ".join(row) + " |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="实验结果汇总与对比脚本")
    parser.add_argument("--dir", required=True, help="结果文件目录")
    parser.add_argument("--format", choices=["markdown", "csv", "console"], default="markdown", help="输出格式")
    parser.add_argument("--output", default=None, help="输出文件路径（默认打印到控制台）")
    parser.add_argument("--group-by", default=None, help="按指定字段分组统计（如 mod_nes, use_fast_complete）")
    parser.add_argument("--show-gpu", action="store_true", help="显示 GPU 和环境信息列")
    args = parser.parse_args()

    # 加载结果
    results = load_result_files(args.dir)
    if not results:
        print("No result files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(results)} result files from {args.dir}", file=sys.stderr)

    # 生成输出
    if args.group_by:
        output = format_grouped_stats(results, args.group_by)
    elif args.format == "markdown":
        output = format_markdown(results, show_gpu=args.show_gpu)
    elif args.format == "csv":
        output = format_csv(results, show_gpu=args.show_gpu)
    else:  # console
        output = format_markdown(results, show_gpu=args.show_gpu)

    # 输出
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(output)
        print(f"Results saved to: {args.output}", file=sys.stderr)
    else:
        print(output)


if __name__ == "__main__":
    main()
