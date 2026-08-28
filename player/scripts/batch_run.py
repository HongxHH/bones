#!/usr/bin/env python3
"""
批量实验运行脚本
用法: python scripts/batch_run.py --config experiments.json
"""
import argparse
import copy
import datetime
import json
import os
import subprocess
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_config(config_path: str) -> Dict[str, Any]:
    """加载实验配置文件（JSON 格式）"""
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_input_files(config: Dict[str, Any], player_dir: str) -> List[str]:
    """检查所有输入文件是否存在，返回缺失文件列表"""
    missing = []
    video = config.get("video", "")
    if video and not os.path.exists(os.path.join(player_dir, video)):
        missing.append(f"video: {video}")
    for trace in config.get("traces", []):
        if not os.path.exists(os.path.join(player_dir, trace)):
            missing.append(f"trace: {trace}")
    return missing


def build_experiment_list(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """生成实验列表（algorithms × traces 的笛卡尔积）"""
    experiments = []
    algorithms = config.get("algorithms", [])
    traces = config.get("traces", [])
    video = config.get("video", "")
    common_args = config.get("common_args", {})

    for algo in algorithms:
        for trace in traces:
            exp = copy.deepcopy(common_args)
            exp.update(algo)
            exp["input"] = video
            exp["trace"] = trace
            experiments.append(exp)
    return experiments


def build_command(exp: Dict[str, Any], conda_env: Optional[str], player_dir: str) -> List[str]:
    """构建实验运行命令"""
    args = []
    for key, value in exp.items():
        if isinstance(value, bool):
            if value:
                args.append(f"--{key}")
        else:
            args.append(f"--{key}")
            args.append(str(value))

    if conda_env:
        cmd = ["conda", "run", "-n", conda_env, "--no-capture-output", "python", "play.py"] + args
    else:
        cmd = [sys.executable, "play.py"] + args
    return cmd


def run_single_experiment(
    exp: Dict[str, Any],
    index: int,
    total: int,
    config: Dict[str, Any],
    player_dir: str,
) -> Dict[str, Any]:
    """运行单个实验，包含重试逻辑"""
    conda_env = config.get("conda_env", None)
    retry = config.get("retry", 2)
    timeout = config.get("timeout", 1800)
    output_dir = config.get("output_dir", "output/data")

    exp_label = f"{exp.get('mod_nes', '?')}_{exp.get('mod_enhancer', '?')}_fast-{exp.get('use_fast_complete', '?')}"
    trace_label = os.path.basename(exp.get("trace", "?"))

    result = {
        "index": index,
        "label": exp_label,
        "trace": trace_label,
        "status": "unknown",
        "duration": None,
        "result_file": None,
        "error": None,
        "attempts": 0,
    }

    for attempt in range(1, retry + 1):
        result["attempts"] = attempt
        print(f"\n[{index}/{total}] Attempt {attempt}/{retry}: {exp_label} | trace={trace_label}")

        cmd = build_command(exp, conda_env, player_dir)
        print(f"  Command: {' '.join(cmd)}")

        start_time = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=player_dir,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
            duration = time.time() - start_time
            result["duration"] = round(duration, 1)

            if proc.returncode != 0:
                result["status"] = "failed"
                result["error"] = f"Exit code {proc.returncode}"
                # 打印最后 30 行 stderr 便于调试
                stderr_lines = proc.stderr.strip().split("\n")[-30:]
                if stderr_lines:
                    print(f"  STDERR (last 30 lines):")
                    for line in stderr_lines:
                        print(f"    {line}")
                if attempt < retry:
                    print(f"  Retrying...")
                    continue
                else:
                    break

            # 检查结果文件是否生成
            # 结果文件名包含 mod_nes, mod_enhancer, fast 等信息
            data_dir = os.path.join(player_dir, output_dir)
            result_files = []
            if os.path.exists(data_dir):
                for f in os.listdir(data_dir):
                    if f.endswith(".json") and exp.get("mod_nes", "") in f and exp.get("mod_enhancer", "") in f:
                        fpath = os.path.join(data_dir, f)
                        result_files.append((fpath, os.path.getmtime(fpath)))

            if result_files:
                # 取最新的文件
                latest = max(result_files, key=lambda x: x[1])
                result["result_file"] = latest[0]
                result["status"] = "success"
                print(f"  SUCCESS in {duration:.1f}s -> {os.path.basename(latest[0])}")
                break
            else:
                result["status"] = "no_result"
                result["error"] = "No result JSON file found"
                print(f"  WARNING: No result file found after {duration:.1f}s")
                if attempt < retry:
                    print(f"  Retrying...")
                    continue
                else:
                    break

        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            result["duration"] = round(duration, 1)
            result["status"] = "timeout"
            result["error"] = f"Timeout after {timeout}s"
            print(f"  TIMEOUT after {timeout}s")
            if attempt < retry:
                print(f"  Retrying...")
                continue
            else:
                break

        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)
            print(f"  ERROR: {e}")
            if attempt < retry:
                print(f"  Retrying...")
                continue
            else:
                break

    return result


def main():
    parser = argparse.ArgumentParser(description="批量实验运行脚本")
    parser.add_argument("--config", required=True, help="实验配置文件路径（JSON）")
    parser.add_argument("--player-dir", default=None, help="player 目录路径（默认脚本所在目录的上一级）")
    args = parser.parse_args()

    # 确定 player 目录
    if args.player_dir:
        player_dir = os.path.abspath(args.player_dir)
    else:
        player_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    print(f"Player directory: {player_dir}")
    print(f"Config file: {args.config}")

    # 加载配置
    config = load_config(args.config)

    # 检查输入文件
    missing = check_input_files(config, player_dir)
    if missing:
        print(f"\nERROR: Missing input files:")
        for m in missing:
            print(f"  - {m}")
        sys.exit(1)

    # 生成实验列表
    experiments = build_experiment_list(config)
    total = len(experiments)
    print(f"\nTotal experiments: {total}")
    for i, exp in enumerate(experiments, 1):
        print(f"  [{i}] {exp.get('mod_nes', '?')} + {exp.get('mod_enhancer', '?')} fast={exp.get('use_fast_complete', '?')} | trace={os.path.basename(exp.get('trace', '?'))}")

    # 创建输出目录并归档配置
    output_dir = config.get("output_dir", "output/data/batch")
    # 如果 output_dir 不是绝对路径，相对于 player_dir
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(player_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # 归档配置文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config_archive = os.path.join(output_dir, f"config_{timestamp}.json")
    with open(config_archive, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    print(f"\nConfig archived to: {config_archive}")

    # 依次运行实验
    print(f"\n{'='*60}")
    print("Starting batch experiments...")
    print(f"{'='*60}")

    results = []
    batch_start = time.time()

    for i, exp in enumerate(experiments, 1):
        # 将输出目录注入到实验参数中
        exp_with_output = copy.deepcopy(exp)
        exp_with_output["metric_output"] = output_dir
        result = run_single_experiment(exp_with_output, i, total, config, player_dir)
        results.append(result)

    batch_duration = time.time() - batch_start

    # 打印汇总
    print(f"\n{'='*60}")
    print(f"Batch complete! Total time: {batch_duration:.1f}s")
    print(f"{'='*60}")
    print(f"{'#':<4} {'Status':<10} {'Duration':<10} {'Algorithm':<30} {'Trace':<25}")
    print("-" * 80)

    success_count = 0
    for r in results:
        status_symbol = "OK" if r["status"] == "success" else r["status"].upper()
        if r["status"] == "success":
            success_count += 1
        print(f"{r['index']:<4} {status_symbol:<10} {r['duration'] or '-':<10} {r['label']:<30} {r['trace']:<25}")
        if r["error"]:
            print(f"     Error: {r['error']}")

    print(f"\nSuccess: {success_count}/{total}")

    # 保存运行记录
    run_log = os.path.join(output_dir, f"run_log_{timestamp}.json")
    with open(run_log, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "total": total,
            "success": success_count,
            "batch_duration": round(batch_duration, 1),
            "results": results,
        }, f, indent=4, ensure_ascii=False)
    print(f"Run log saved to: {run_log}")

    # 如果有失败的实验，返回非零退出码
    if success_count < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
