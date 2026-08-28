"""
对比实验运行脚本

运行四种 NES 控制器并对比结果：
1. BONES（基线）
2. Content-Aware BONES（内容感知质量/延迟预测）
3. DRL NES（DRL 联合决策，无内容特征）
4. DRL NES + Content（完整方法：DRL + 内容特征）

用法:
    python scripts/run_comparison.py --trace data/3Glogs/report.2010-09-13_1003CEST.json
    python scripts/run_comparison.py --methods bones content_aware_bones --quick
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 实验配置
EXPERIMENT_CONFIGS = {
    "bones": {
        "name": "BONES (基线)",
        "mod_nes": "bones",
        "description": "原始 BONES Lyapunov 优化，静态全局质量/延迟表",
    },
    "content_aware_bones": {
        "name": "Content-Aware BONES",
        "mod_nes": "content_aware_bones",
        "description": "BONES + 内容感知质量/延迟预测",
    },
    "drl_nes": {
        "name": "DRL NES",
        "mod_nes": "drl_nes",
        "description": "DRL 联合决策（DQN，内容特征状态输入）",
    },
}


def run_experiment(method: str, config: dict, args) -> Dict:
    """
    运行单个实验

    Args:
        method: 方法名称
        config: 实验配置
        args: 命令行参数

    Returns:
        实验结果字典
    """
    print(f"\n{'='*60}")
    print(f"运行实验: {config['name']}")
    print(f"  描述: {config['description']}")
    print(f"  方法: {config['mod_nes']}")
    print(f"{'='*60}")

    # 构建命令
    cmd = [
        sys.executable, "play.py",
        "--mod_nes", config["mod_nes"],
        "--mod_enhancer", "imdn_always",
        "--mod_renderer", "opengl",
        "--mod_analyzer", "data_collector",
        "--input", args.input,
        "--trace", args.trace,
    ]

    # 添加额外参数
    if "extra_args" in config:
        cmd.extend(config["extra_args"])

    # 运行
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            capture_output=True,
            text=True,
            timeout=args.timeout,
        )
        elapsed = time.time() - start_time

        if result.returncode != 0:
            print(f"  实验失败 (returncode={result.returncode})")
            print(f"  stderr: {result.stderr[-500:]}")
            return {
                "method": method,
                "name": config["name"],
                "success": False,
                "error": result.stderr[-500:],
                "elapsed": elapsed,
            }

        print(f"  实验完成，耗时 {elapsed:.1f}s")

        # 查找最新的结果文件
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "data")
        result_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
        result_files.sort(key=lambda f: os.path.getmtime(os.path.join(output_dir, f)), reverse=True)

        if result_files:
            latest_result = os.path.join(output_dir, result_files[0])
            with open(latest_result, 'r') as f:
                data = json.load(f)
            print(f"  结果文件: {result_files[0]}")
            return {
                "method": method,
                "name": config["name"],
                "success": True,
                "elapsed": elapsed,
                "result_file": result_files[0],
                "data": data,
            }
        else:
            print("  警告：未找到结果文件")
            return {
                "method": method,
                "name": config["name"],
                "success": False,
                "error": "No result file found",
                "elapsed": elapsed,
            }

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"  实验超时 ({elapsed:.1f}s > {args.timeout}s)")
        return {
            "method": method,
            "name": config["name"],
            "success": False,
            "error": f"Timeout after {elapsed:.1f}s",
            "elapsed": elapsed,
        }
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  实验异常: {e}")
        return {
            "method": method,
            "name": config["name"],
            "success": False,
            "error": str(e),
            "elapsed": elapsed,
        }


def extract_metrics(result: Dict) -> Dict:
    """从实验结果中提取关键指标"""
    if not result.get("success"):
        return {"method": result["name"], "success": False}

    data = result.get("data", {})
    metrics = {
        "method": result["name"],
        "success": True,
        "QoE": data.get("qoe", data.get("QoE", "N/A")),
        "avg_vmaf": data.get("avg_vmaf", data.get("average_vmaf", "N/A")),
        "rebuffer_count": data.get("rebuffer_count", data.get("num_rebuffer", "N/A")),
        "rebuffer_duration": data.get("rebuffer_duration", "N/A"),
        "quality_switches": data.get("quality_switches", "N/A"),
        "startup_delay": data.get("startup_metrics", {}).get("startup_delay", "N/A") if isinstance(data.get("startup_metrics"), dict) else "N/A",
        "gpu_avg_power": data.get("gpu_metrics", {}).get("avg_power_w", "N/A") if isinstance(data.get("gpu_metrics"), dict) else "N/A",
        "gpu_peak_memory": data.get("gpu_metrics", {}).get("peak_memory_mb", "N/A") if isinstance(data.get("gpu_metrics"), dict) else "N/A",
        "elapsed_s": round(result.get("elapsed", 0), 1),
    }
    return metrics


def print_comparison_table(all_metrics: List[Dict]):
    """打印对比表格"""
    print(f"\n{'='*80}")
    print("对比实验结果汇总")
    print(f"{'='*80}")

    # 表头
    headers = ["方法", "QoE", "avg VMAF", "卡顿次数", "质量切换", "启动延迟(s)", "GPU功耗(W)", "耗时(s)"]
    print(f"{'方法':<35} {'QoE':>8} {'avgVMAF':>8} {'卡顿':>6} {'切换':>6} {'启动':>8} {'功耗':>8} {'耗时':>8}")
    print("-" * 95)

    for m in all_metrics:
        if not m.get("success"):
            print(f"{m['method']:<35} {'FAILED':>8}")
            continue
        print(f"{m['method']:<35} "
              f"{str(m.get('QoE', 'N/A')):>8} "
              f"{str(m.get('avg_vmaf', 'N/A')):>8} "
              f"{str(m.get('rebuffer_count', 'N/A')):>6} "
              f"{str(m.get('quality_switches', 'N/A')):>6} "
              f"{str(m.get('startup_delay', 'N/A')):>8} "
              f"{str(m.get('gpu_avg_power', 'N/A')):>8} "
              f"{str(m.get('elapsed_s', 'N/A')):>8}")

    print("=" * 95)


def save_comparison_report(all_results: List[Dict], all_metrics: List[Dict], args):
    """保存对比报告"""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "comparison")
    os.makedirs(report_dir, exist_ok=True)

    report_file = os.path.join(report_dir, f"comparison_{timestamp}.json")
    report = {
        "timestamp": timestamp,
        "trace": args.trace,
        "video": args.input,
        "methods": [r["method"] for r in all_results],
        "metrics": all_metrics,
        "raw_results": all_results,
    }

    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\n对比报告已保存: {report_file}")
    return report_file


def main():
    parser = argparse.ArgumentParser(description="NES 控制器对比实验")
    parser.add_argument("--methods", nargs="+", default=["bones", "content_aware_bones"],
                        help="要运行的方法列表（默认: bones content_aware_bones）")
    parser.add_argument("--input", type=str, default="data/video_mp4box/bbb.mpd",
                        help="视频 MPD 文件")
    parser.add_argument("--trace", type=str, default="data/3Glogs/report.2010-09-13_1003CEST.json",
                        help="网络 trace 文件")
    parser.add_argument("--timeout", type=int, default=1800,
                        help="单个实验超时时间（秒），默认 30 分钟")
    parser.add_argument("--quick", action="store_true",
                        help="快速模式（仅运行前两个方法）")

    args = parser.parse_args()

    if args.quick:
        args.methods = ["bones", "content_aware_bones"]

    # 验证方法
    valid_methods = [m for m in args.methods if m in EXPERIMENT_CONFIGS]
    invalid_methods = [m for m in args.methods if m not in EXPERIMENT_CONFIGS]
    if invalid_methods:
        print(f"警告：以下方法不存在，将跳过: {invalid_methods}")
        print(f"可用方法: {list(EXPERIMENT_CONFIGS.keys())}")

    if not valid_methods:
        print("错误：没有有效的方法")
        return

    print(f"将运行 {len(valid_methods)} 个实验:")
    for m in valid_methods:
        print(f"  - {EXPERIMENT_CONFIGS[m]['name']}")

    # 运行实验
    all_results = []
    all_metrics = []

    for method in valid_methods:
        config = EXPERIMENT_CONFIGS[method]
        result = run_experiment(method, config, args)
        all_results.append(result)
        metrics = extract_metrics(result)
        all_metrics.append(metrics)

    # 打印对比表格
    print_comparison_table(all_metrics)

    # 保存报告
    report_file = save_comparison_report(all_results, all_metrics, args)

    print(f"\n全部实验完成！报告: {report_file}")


if __name__ == "__main__":
    main()
