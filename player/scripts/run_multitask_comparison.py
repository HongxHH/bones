"""
多任务增强对比实验运行脚本

对比三种配置：
1. BONES + IMDN (SR-only, 基线)
2. CA-BONES + IMDN (内容感知SR)
3. MT-BONES + MultiTask (多任务增强: SR+Deblur)

用法:
    python scripts/run_multitask_comparison.py --trace ../data/trace/hd_fs/trace0000.json
    python scripts/run_multitask_comparison.py --methods mt_bones --quick
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
        "name": "BONES (SR-only baseline, no fast)",
        "mod_nes": "bones",
        "mod_enhancer": "imdn_always",
        "description": "原始 BONES + IMDN 超分（单任务，关闭fast）",
        "config_file": "config_bones_no_fast.json",
    },
    "ca_bones": {
        "name": "CA-BONES (content-aware SR, no fast)",
        "mod_nes": "ca_bones",
        "mod_enhancer": "imdn_always",
        "description": "内容感知 BONES + IMDN 超分（关闭fast）",
        "extra_args": ["--content_aware", "True", "--ca_mode", "base_known"],
        "config_file": "config_bones_no_fast.json",
    },
    "mt_bones": {
        "name": "MT-BONES (SR+USM, fast enabled)",
        "mod_nes": "mt_bones",
        "mod_enhancer": "multitask",
        "description": "多任务 BONES + MultiTaskEnhancer (SR+USM锐化，开启fast)",
        "config_file": "config_mt_bones.json",
    },
}


def run_experiment(method: str, config: dict, args) -> Dict:
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"Running: {config['name']}")
    print(f"  NES: {config['mod_nes']}, Enhancer: {config['mod_enhancer']}")
    print(f"{'='*60}")

    # 构建命令
    cmd = [
        sys.executable, "play.py",
        "--mod_nes", config["mod_nes"],
        "--mod_enhancer", config["mod_enhancer"],
        "--mod_renderer", "headless",  # 无头模式，不渲染画面
        "--mod_analyzer", "data_collector",
        "--input", args.input,
        "--trace", args.trace,
        "--run_dir", f"output/mt_run_{method}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    ]

    if "config_file" in config:
        cmd.extend(["--config", config["config_file"]])

    if "extra_args" in config:
        cmd.extend(config["extra_args"])

    # 运行
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=args.timeout,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        elapsed = time.time() - start_time
        print(f"  Completed in {elapsed:.1f}s, return code: {result.returncode}")

        # 保存日志
        log_dir = f"output/logs"
        os.makedirs(log_dir, exist_ok=True)
        with open(f"{log_dir}/mt_{method}.log", "w", encoding="utf-8") as f:
            f.write(result.stdout)
            if result.stderr:
                f.write("\n--- STDERR ---\n")
                f.write(result.stderr)

        return {
            "method": method,
            "name": config["name"],
            "elapsed": elapsed,
            "returncode": result.returncode,
            "success": result.returncode == 0,
        }
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {args.timeout}s")
        return {"method": method, "name": config["name"], "success": False, "error": "timeout"}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {"method": method, "name": config["name"], "success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Multi-task enhancement comparison")
    parser.add_argument("--input", default="../data/video_mp4box/bbb.mpd",
                        help="MPD file path")
    parser.add_argument("--trace", default="../data/trace/hd_fs/trace0000.json",
                        help="Network trace file path")
    parser.add_argument("--methods", nargs="+", default=list(EXPERIMENT_CONFIGS.keys()),
                        choices=list(EXPERIMENT_CONFIGS.keys()),
                        help="Methods to run")
    parser.add_argument("--timeout", type=int, default=900,
                        help="Timeout per experiment in seconds")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: only run mt_bones")
    args = parser.parse_args()

    if args.quick:
        args.methods = ["mt_bones"]

    print(f"Multi-task Enhancement Comparison")
    print(f"  Input: {args.input}")
    print(f"  Trace: {args.trace}")
    print(f"  Methods: {args.methods}")
    print(f"  Timeout: {args.timeout}s per run")

    results = []
    for method in args.methods:
        if method not in EXPERIMENT_CONFIGS:
            print(f"Unknown method: {method}, skipping")
            continue
        result = run_experiment(method, EXPERIMENT_CONFIGS[method], args)
        results.append(result)

    # 汇总
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for r in results:
        status = "OK" if r.get("success") else "FAIL"
        print(f"  {r['name']:<40} {status:>5}  {r.get('elapsed', 0):.1f}s")

    # 保存结果
    output_file = "output/mt_comparison_results.json"
    os.makedirs("output", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
