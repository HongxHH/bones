from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from .app import create_app
from .config import ServerConfig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Prism control plane (DASH origin + experiment jobs)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--public-base-url", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--ffmpeg", default=None)
    args = parser.parse_args(argv)

    config = ServerConfig(host=args.host, port=args.port)
    if args.public_base_url:
        config.public_base_url = args.public_base_url
    else:
        bind_host = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
        config.public_base_url = f"http://{bind_host}:{args.port}"
    if args.data_dir:
        config.data_dir = Path(args.data_dir)
    if args.ffmpeg:
        config.ffmpeg_bin = args.ffmpeg
        config.ffprobe_bin = args.ffmpeg.replace("ffmpeg", "ffprobe")

    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
