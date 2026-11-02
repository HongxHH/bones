# -*- coding: utf-8 -*-
"""
阶段3四臂串行：always_greedy + IMDN + burn K + 指定 recovery。
结果写到 player/output/data/phase3/ 与本目录 phase3_summary.json。
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(r"E:\dev\Code\Python\bones")
PLAYER = REPO / "player"
OUT_DIR = PLAYER / "output" / "data" / "phase3"
HERE = Path(__file__).resolve().parent
PY = Path(r"E:\dev\env\miniconda3\envs\SR\python.exe")
FF = None  # not needed

ARMS = [
    {
        "id": "brute_C3",
        "fixed_enhance_action": 3,
        "recovery_mode": "none",
        "use_fast_complete": "false",
        "extra": [],
    },
    {
        "id": "bicubic_tail_C3",
        "fixed_enhance_action": 3,
        "recovery_mode": "bicubic",
        "use_fast_complete": "true",
        "extra": [],
    },
    {
        "id": "fallback_F1_C3",
        "fixed_enhance_action": 3,
        "recovery_mode": "fallback",
        "use_fast_complete": "true",
        "extra": ["--fallback_enhance_action", "1"],
    },
    {
        "id": "whole_F1",
        "fixed_enhance_action": 1,
        "recovery_mode": "none",
        "use_fast_complete": "false",
        "extra": [],
    },
]

TRACES = [
    "data/3Glogs/report.2010-09-13_1003CEST.json",
    "data/3Glogs/report.2010-09-14_2303CEST.json",
    "data/3Glogs/report.2010-09-21_0742CEST.json",
]


def trace_tag(trace: str) -> str:
    return Path(trace).stem.replace("report.", "")[:20]


def find_newest_result(before: float, arm_id: str) -> Path | None:
    """Analyzer writes always_greedy_imdn_always_*.json under metric_output."""
    if not OUT_DIR.exists():
        return None
    cands = []
    for p in OUT_DIR.glob("*.json"):
        if p.name.startswith("phase3_"):
            continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if mtime >= before - 1:
            cands.append(p)
    if not cands:
        return None
    cands.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return cands[0]


def summarize(path: Path) -> dict:
    d = json.load(open(path, encoding="utf-8"))
    segs = d.get("segments") or []
    fc = sum(1 for s in segs if s.get("fast_complete"))
    abort = sum(1 for s in segs if not s.get("is_enhance") and (s.get("enhance_action") or 0) > 0)
    wasted = sum(int(s.get("wasted_enhanced_cnt") or 0) for s in segs)
    enh = sum(int(s.get("enhanced_cnt") or 0) for s in segs if s.get("enhanced_cnt") is not None)
    fb = 0
    # frame levels / fallback only if present in raw analyzer — may be missing
    return {
        "path": str(path),
        "QOE": d.get("QOE"),
        "avg_vmaf": d.get("avg_vmaf"),
        "avg_freeze": d.get("avg_freeze"),
        "dur_stall": d.get("dur_stall"),
        "num_stall": d.get("num_stall"),
        "fast_complete_segments_count": d.get("fast_complete_segments_count"),
        "fast_complete_saved_frames_total": d.get("fast_complete_saved_frames_total"),
        "abort_wasted_frames_total": d.get("abort_wasted_frames_total"),
        "enhancer_abort_total": d.get("enhancer_abort_total"),
        "segments": len(segs),
        "fc_segs_recount": fc,
        "abort_segs_recount": abort,
        "wasted_recount": wasted,
        "enhanced_frames_sum": enh,
    }


def run_one(arm: dict, trace: str, burn_k: float) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"{arm['id']}__{trace_tag(trace)}"
    env = os.environ.copy()
    env["PACNET_BURN_K"] = str(burn_k)
    env["OMP_NUM_THREADS"] = "4"
    env["OPENBLAS_NUM_THREADS"] = "4"
    env["MKL_NUM_THREADS"] = "4"
    env["PYTHONPATH"] = str(PLAYER)
    # avoid interactive opencv if possible
    cmd = [
        str(PY), "play.py",
        "--mod_nes", "always_greedy",
        "--mod_enhancer", "imdn",
        "--mod_renderer", "headless",
        "--mod_downloader", "trace",
        "--mod_status", "off",
        "--fixed_enhance_action", str(arm["fixed_enhance_action"]),
        "--recovery_mode", arm["recovery_mode"],
        "--use_fast_complete", arm["use_fast_complete"],
        "--latency_adapt", "false",
        "--trace", trace,
        "--metric_output", str(OUT_DIR),
        "--run_dir", str(PLAYER / "output" / "istream_run_dir" / tag),
        "-i", "data/video_mp4box/bbb.mpd",
    ] + arm["extra"]
    log_path = HERE / "tmp" / f"phase3_{tag}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n=== START {tag} K={burn_k} ===", flush=True)
    print(" ".join(cmd), flush=True)
    t0 = time.time()
    before = t0
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd, cwd=str(PLAYER), env=env,
            stdout=logf, stderr=subprocess.STDOUT,
        )
    elapsed = time.time() - t0
    result_path = find_newest_result(before, arm["id"])
    row = {
        "arm": arm["id"],
        "trace": trace,
        "burn_k": burn_k,
        "exit_code": proc.returncode,
        "elapsed_s": round(elapsed, 1),
        "log": str(log_path),
        "ok": proc.returncode == 0 and result_path is not None,
    }
    if result_path is not None:
        # rename for clarity
        dest = OUT_DIR / f"phase3_{tag}.json"
        try:
            if result_path.resolve() != dest.resolve():
                dest.write_bytes(result_path.read_bytes())
        except OSError:
            dest = result_path
        row["summary"] = summarize(dest if dest.exists() else result_path)
        row["result"] = str(dest if dest.exists() else result_path)
    print(f"=== END {tag} rc={proc.returncode} {elapsed:.0f}s ok={row['ok']} ===", flush=True)
    return row


def main():
    burn_k = float(os.environ.get("PHASE3_BURN_K", "5"))
    only_trace = os.environ.get("PHASE3_TRACE")  # optional single
    only_arm = os.environ.get("PHASE3_ARM")
    traces = [only_trace] if only_trace else TRACES
    arms = [a for a in ARMS if (not only_arm or a["id"] == only_arm)]
    runs = []
    for trace in traces:
        for arm in arms:
            runs.append(run_one(arm, trace, burn_k))
            # persist partial
            summary = {
                "measured_at": datetime.now().isoformat(timespec="seconds"),
                "burn_k": burn_k,
                "runs": runs,
            }
            out = HERE / "phase3_summary.json"
            json.dump(summary, open(out, "w", encoding="utf-8"), indent=2)
    print("WROTE", HERE / "phase3_summary.json", flush=True)


if __name__ == "__main__":
    main()
