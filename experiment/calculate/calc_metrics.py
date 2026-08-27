import argparse
import json
import os
import re
import subprocess
from typing import Dict, List


METRIC_FILTERS = {
    "psnr": ("[distorted][reference]psnr", r"average:([\d\.]+)"),
    "ssim": ("[distorted][reference]ssim", r"All:([\d\.]+)"),
    "vmaf": ("[distorted][reference]libvmaf=n_threads=4", r"VMAF score: ([\d\.]+)"),
}


def build_command(
    reference: str,
    distorted: str,
    filter_cmd: str,
    width: int | None,
    height: int | None,
    rescale: bool,
) -> List[str]:
    reference = os.path.abspath(reference)
    distorted = os.path.abspath(distorted)
    if rescale:
        if not (width and height):
            raise ValueError("--width/--height 必须在启用重采样时提供")
        filter_complex = ";".join(
            [
                f"[0:v]scale={width}:{height}:flags=bicubic,setpts=PTS-STARTPTS[reference]",
                f"[1:v]scale={width}:{height}:flags=bicubic,setpts=PTS-STARTPTS[distorted]",
                filter_cmd,
            ]
        )
    else:
        filter_complex = ";".join(
            [
                "[0:v]setpts=PTS-STARTPTS[reference_raw]",
                "[1:v]setpts=PTS-STARTPTS[distorted_raw]",
                "[distorted_raw][reference_raw]scale2ref=flags=bicubic[distorted][reference]",
                filter_cmd,
            ]
        )

    return [
        "ffmpeg",
        "-i",
        reference,
        "-i",
        distorted,
        "-lavfi",
        filter_complex,
        "-f",
        "null",
        "-",
    ]


def run_metric(
    reference: str,
    distorted: str,
    metric: str,
    width: int | None,
    height: int | None,
    rescale: bool,
) -> float:
    filter_cmd, pattern = METRIC_FILTERS[metric]
    command = build_command(reference, distorted, filter_cmd, width, height, rescale)
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        output = exc.stdout.strip() if exc.stdout else "无输出"
        raise RuntimeError(f"运行 {metric} 失败，ffmpeg 输出：\n{output}") from exc
    match = re.search(pattern, completed.stdout)
    if not match:
        raise RuntimeError(f"Failed to parse {metric} from ffmpeg output.")
    return float(match.group(1))


def calculate_metrics(
    reference: str,
    distorted: str,
    metrics: List[str],
    width: int | None,
    height: int | None,
    rescale: bool,
) -> Dict[str, float]:
    results: Dict[str, float] = {}
    for metric in metrics:
        metric = metric.lower()
        if metric not in METRIC_FILTERS:
            raise ValueError(f"Unsupported metric '{metric}'. Supported: {list(METRIC_FILTERS)}")
        results[metric] = run_metric(reference, distorted, metric, width, height, rescale)
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate quality metrics (PSNR/SSIM/VMAF) between two videos."
    )
    parser.add_argument("--reference", default=r"E:\dev\Code\Python\bones\enhance\IMDN\dataset\bbb\output_2160p.mp4", help="Path to ground-truth video.")
    parser.add_argument("--distorted", default=r"E:\dev\Code\Python\bones\player\output\bones_inter0_nofast\render_20251124_172813.mp4", help="Path to rendered/enhanced video.")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["psnr", "ssim", "vmaf"],
        help="Metrics to compute.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1920,
        help="Target width for rescaling (used when --rescale is set).",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1080,
        help="Target height for rescaling (used when --rescale is set).",
    )
    parser.add_argument(
        "--no-rescale",
        default=False,
        action="store_true",
        help="Disable rescaling the distorted video before comparison.",
    )
    parser.add_argument(
        "--output",
        help="Optional path to store JSON results. Prints to stdout if omitted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rescale = not args.no_rescale
    results = calculate_metrics(
        reference= args.reference,
        distorted= args.distorted,
        metrics=args.metrics,
        width=args.width,
        height=args.height,
        rescale=rescale,
    )
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4)
    print(json.dumps(results, indent=4))


if __name__ == "__main__":
    main()

# python experiment/calculate/calc_metrics.py --reference GT.mp4 --distorted render.mp4 --metrics psnr vmaf --output metrics.json


# 修复后如果希望在 1080p 精度下对 4K 参考与 1080p 渲染进行对比，保持默认参数即可；若要在原始 4K 分辨率计算，改传 --width 3840 --height 2160 或直接加 --no-rescale（此时 scale2ref 会把失真流放大到引用流大小）。