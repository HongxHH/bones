"""
预计算视频内容特征脚本

用法:
    python scripts/precompute_content_features.py --video data/video_mp4box/output_1080p.mp4 --output data/content_features/bbb_features.json
"""
import argparse
import sys
import os

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from istream_player.utils.content_features import precompute_video_features


def main():
    parser = argparse.ArgumentParser(description="预计算视频内容特征")
    parser.add_argument("--video", type=str,
                        default="data/video_mp4box/output_1080p.mp4",
                        help="视频文件路径")
    parser.add_argument("--output", type=str,
                        default="data/content_features/bbb_features.json",
                        help="输出 JSON 文件路径")
    parser.add_argument("--segment-duration", type=float, default=4.0,
                        help="段时长（秒）")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="目标帧率")
    parser.add_argument("--max-segments", type=int, default=None,
                        help="最大段数（None 表示全部）")
    parser.add_argument("--downscale-width", type=int, default=480,
                        help="特征计算降采样宽度（默认 480p）")
    parser.add_argument("--frame-sample-interval", type=int, default=4,
                        help="运动/场景特征帧采样间隔（默认每4帧取1帧）")

    args = parser.parse_args()

    precompute_video_features(
        video_path=args.video,
        output_path=args.output,
        segment_duration=args.segment_duration,
        fps=args.fps,
        max_segments=args.max_segments,
        downscale_width=args.downscale_width,
        frame_sample_interval=args.frame_sample_interval,
    )


if __name__ == "__main__":
    main()
