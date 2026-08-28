"""
多 trace 对比实验脚本

在多个网络 trace 上对比 BONES 基线和 Conservative Content-Aware BONES，
验证方法的泛化性。

用法:
    python scripts/run_multitrace.py
    python scripts/run_multitrace.py --methods bones conservative_ca_bones
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# 实验配置：多样化的网络 trace
TRACES = [
    {
        "name": "3G_stable",
        "path": "data/3Glogs/report.2010-09-13_1003CEST.json",
        "desc": "3G 中高带宽稳定 (mean=1448kbps, std=403)",
    },
    {
        "name": "3G_low_var",
        "path": "data/3Glogs/report.2010-09-13_1046CEST.json",
        "desc": "3G 低带宽高波动 (mean=571kbps, std=642)",
    },
    {
        "name": "3G_high_var",
        "path": "data/3Glogs/report.2010-09-28_1003CEST.json",
        "desc": "3G 中带宽极高波动 (mean=1392kbps, std=857)",
    },
]

METHODS = [
    {"name": "bones", "mod_nes": "bones", "desc": "BONES 基线"},
    {"name": "conservative_ca", "mod_nes": "conservative_ca_bones", "desc": "保守内容感知 BONES"},
]


def run_experiment(method: dict, trace: dict, args) -> dict:
    """运行单个实验"""
    print(f"\n{'='*60}")
    print(f"实验: {method['desc']} | {trace['desc']}")
    print(f"{'='*60}")

    cmd = [
        sys.executable, "play.py",
        "--mod_nes", method["mod_nes"],
        "--mod_enhancer", "imdn_always",
        "--mod_renderer", "opengl",
        "--mod_analyzer", "data_collector",
        "--input", args.input,
        "--trace", trace["path"],
    ]

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
            print(f"  stderr: {result.stderr[-300:]}")
            return {"method": method["name"], "trace": trace["name"], "success": False, "error": result.stderr[-300:], "elapsed": elapsed}

        print(f"  实验完成，耗时 {elapsed:.1f}s")

        # 查找最新结果文件
        output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "data")
        result_files = [f for f in os.listdir(output_dir) if f.endswith('.json')]
        result_files.sort(key=lambda f: os.path.getmtime(os.path.join(output_dir, f)), reverse=True)

        if result_files:
            latest = os.path.join(output_dir, result_files[0])
            with open(latest) as f:
                data = json.load(f)
            print(f"  结果: avg_vmaf={data.get('avg_vmaf', 'N/A'):.2f}, QoE={data.get('QOE', 'N/A'):.2f}, "
                  f"stalls={data.get('num_stall', 'N/A')}, stall_dur={data.get('dur_stall', 'N/A'):.1f}s")
            return {
                "method": method["name"], "trace": trace["name"], "success": True,
                "elapsed": elapsed, "result_file": result_files[0], "data": data,
            }
        else:
            return {"method": method["name"], "trace": trace["name"], "success": False, "error": "No result file", "elapsed": elapsed}

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"  实验超时 ({elapsed:.1f}s)")
        return {"method": method["name"], "trace": trace["name"], "success": False, "error": f"Timeout after {elapsed:.1f}s", "elapsed": elapsed}
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"  实验异常: {e}")
        return {"method": method["name"], "trace": trace["name"], "success": False, "error": str(e), "elapsed": elapsed}


def print_summary(results: list):
    """打印汇总表格"""
    print(f"\n{'='*100}")
    print("多 Trace 对比实验汇总")
    print(f"{'='*100}")
    print(f"{'方法':<20} {'Trace':<15} {'avgVMAF':>8} {'QoE':>8} {'卡顿':>6} {'卡顿(s)':>8} {'完成率':>8} {'GPU显存':>8}")
    print("-" * 100)

    for r in results:
        if not r.get("success"):
            print(f"{r['method']:<20} {r['trace']:<15} {'FAILED':>8}")
            continue
        d = r["data"]
        segs = d.get("segments", [])
        completed = sum(1 for s in segs if s.get("is_enhance"))
        completion_rate = completed / len(segs) * 100 if segs else 0
        gpu_mem = d.get("gpu_metrics", {}).get("peak_memory_mb", "N/A")
        print(f"{r['method']:<20} {r['trace']:<15} "
              f"{d.get('avg_vmaf', 0):>8.2f} {d.get('QOE', 0):>8.2f} "
              f"{d.get('num_stall', 0):>6} {d.get('dur_stall', 0):>8.1f} "
              f"{completion_rate:>7.1f}% {str(gpu_mem):>8}")

    print("=" * 100)


def main():
    parser = argparse.ArgumentParser(description="多 Trace 对比实验")
    parser.add_argument("--input", type=str, default="data/video_mp4box/bbb.mpd")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--methods", nargs="+", default=["bones", "conservative_ca"])
    args = parser.parse_args()

    # 筛选方法
    methods = [m for m in METHODS if m["name"] in args.methods]
    if not methods:
        print(f"错误：没有有效的方法。可用: {[m['name'] for m in METHODS]}")
        return

    print(f"将运行 {len(methods)} 方法 x {len(TRACES)} trace = {len(methods)*len(TRACES)} 个实验")

    results = []
    for method in methods:
        for trace in TRACES:
            # 清理运行目录
            run_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "istream_run_dir")
            if os.path.exists(run_dir):
                import shutil
                shutil.rmtree(run_dir, ignore_errors=True)

            result = run_experiment(method, trace, args)
            results.append(result)

    print_summary(results)

    # 保存报告
    report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "multitrace")
    os.makedirs(report_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    report_file = os.path.join(report_dir, f"multitrace_{timestamp}.json")
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n报告已保存: {report_file}")


if __name__ == "__main__":
    main()
