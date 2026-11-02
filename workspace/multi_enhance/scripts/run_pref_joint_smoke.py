# -*- coding: utf-8 -*-
"""Quick pref_joint smoke: one short-ish BBB×0921 run with balanced knobs."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PLAYER = REPO / "player"
PY = sys.executable


def main():
    metric = PLAYER / "output" / "data" / "pref_joint" / "smoke_balanced"
    metric.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY,
        "play.py",
        "--mod_nes",
        "pref_joint",
        "--mod_enhancer",
        "multitask",
        "--mod_renderer",
        "headless",
        "--mod_downloader",
        "trace",
        "--mod_status",
        "off",
        "--latency_adapt",
        "false",
        "--use_fast_complete",
        "true",
        "--user_courage",
        "balanced",
        "--user_taste",
        "balanced",
        "--metric_output",
        str(metric),
        "-i",
        "data/video_mp4box/bbb.mpd",
        "-t",
        "data/3Glogs/report.2010-09-21_0742CEST.json",
    ]
    print(" ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd, cwd=str(PLAYER)))


if __name__ == "__main__":
    main()
