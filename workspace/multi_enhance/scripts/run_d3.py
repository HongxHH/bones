# -*- coding: utf-8 -*-
"""D3：1～2 条 3G 轨迹，tool_greedy vs fixed sr_high vs none。"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent  # multi_enhance/
REPO = WS.parents[1]  # bones/
PLAYER = REPO / "player"
OUT_DIR = WS / "results" / "runs"
PY = sys.executable

TRACES = {
    "0913": "data/3Glogs/report.2010-09-13_1003CEST.json",
    "0921": "data/3Glogs/report.2010-09-21_0742CEST.json",
    "0914": "data/3Glogs/report.2010-09-14_2303CEST.json",
}

ARMS = {
    "tool_greedy": {
        "mod_nes": "tool_greedy",
        "mod_enhancer": "multitask",
    },
    "fixed_sr_high": {
        "mod_nes": "always_greedy",
        "mod_enhancer": "imdn",
        "fixed_enhance_action": "3",
    },
    "none": {
        "mod_nes": "always_greedy",
        "mod_enhancer": "imdn",
        "fixed_enhance_action": "0",
    },
}


def build_cmd(arm: dict, trace: str, metric_dir: Path) -> list[str]:
    metric_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY,
        "play.py",
        "--mod_nes",
        arm["mod_nes"],
        "--mod_enhancer",
        arm["mod_enhancer"],
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
        "--metric_output",
        str(metric_dir),
        "-i",
        "data/video_mp4box/bbb.mpd",
        "-t",
        trace,
    ]
    if "fixed_enhance_action" in arm:
        cmd += ["--fixed_enhance_action", arm["fixed_enhance_action"]]
    return cmd


def extract_metrics(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    qoe = raw.get("QOE", raw.get("QoE", raw.get("qoe")))
    stall = raw.get("dur_stall", raw.get("stall_time", raw.get("stall")))
    from collections import Counter

    tools = Counter()
    for s in raw.get("segments", []):
        tid = s.get("tool_id")
        if tid is None:
            tid = s.get("enhance_level")
        tools[str(tid)] += 1
    return {
        "result_file": str(path.relative_to(REPO)).replace("\\", "/"),
        "QOE": qoe,
        "avg_vmaf": raw.get("avg_vmaf"),
        "dur_stall": stall,
        "num_stall": raw.get("num_stall"),
        "n_segments": len(raw.get("segments", [])),
        "tool_or_level_hist": dict(tools),
    }


def run_one(arm_name: str, trace_key: str, timeout: int, metric_root: Path) -> dict:
    arm = ARMS[arm_name]
    trace = TRACES[trace_key]
    metric_dir = metric_root / f"{trace_key}_{arm_name}"
    cmd = build_cmd(arm, trace, metric_dir)
    before = {p for p in metric_dir.glob("*.json")} if metric_dir.exists() else set()
    t0 = time.time()
    print(f"\n=== {arm_name} @ {trace_key} ===", flush=True)
    print(" ", " ".join(cmd), flush=True)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PLAYER),
            timeout=timeout,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        rc = proc.returncode
        err_tail = (proc.stderr or "")[-2000:]
        out_tail = (proc.stdout or "")[-2000:]
    except subprocess.TimeoutExpired as e:
        rc = -9
        err_tail = f"timeout after {timeout}s"
        out_tail = (e.stdout or "")[-2000:] if isinstance(e.stdout, str) else ""
    dur = time.time() - t0
    newest = None
    if metric_dir.exists():
        cands = sorted(metric_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in cands:
            if p not in before:
                newest = p
                break
        if newest is None and cands:
            newest = cands[0]
    row = {
        "arm": arm_name,
        "trace": trace_key,
        "trace_path": trace,
        "returncode": rc,
        "duration_s": round(dur, 1),
        "ok": rc == 0,
    }
    if newest:
        row.update(extract_metrics(newest))
    else:
        row["result_file"] = None
    row["stdout_tail"] = out_tail
    row["stderr_tail"] = err_tail
    print(
        f"  rc={rc} dur={dur:.0f}s file={row.get('result_file')} "
        f"QOE={row.get('QOE')} stall={row.get('dur_stall')}",
        flush=True,
    )
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", default=["0913"], choices=list(TRACES))
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_root = PLAYER / "output" / "data" / "d3"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rows = []
    for tr in args.traces:
        for arm in args.arms:
            rows.append(run_one(arm, tr, args.timeout, metric_root))
            (OUT_DIR / f"d3_partial_{stamp}.json").write_text(
                json.dumps(
                    [{k: v for k, v in r.items() if k not in ("stdout_tail", "stderr_tail")} for r in rows],
                    indent=2,
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

    slim = [{k: v for k, v in r.items() if k not in ("stdout_tail", "stderr_tail")} for r in rows]
    summary = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "traces": args.traces,
        "arms": args.arms,
        "rows": slim,
        "rows_with_logs": rows,
    }
    out = OUT_DIR / f"d3_summary_{stamp}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    for r in slim:
        print(
            f"  {r['arm']:14s} {r['trace']} QOE={r.get('QOE')} "
            f"stall={r.get('dur_stall')} ok={r.get('ok')}"
        )


if __name__ == "__main__":
    main()
