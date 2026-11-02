# -*- coding: utf-8 -*-
"""D5：BBB/TOS 消融 — ours / none / fixed_high / no_hd / no_slack / no_fc。

全部臂默认 multitask + 同一套 multitask_* 质量表，保证查表 QoE 口径一致。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent
REPO = WS.parents[1]
PLAYER = REPO / "player"
OUT_DIR = WS / "results" / "runs"
PY = sys.executable

TRACES = {
    "0913": "data/3Glogs/report.2010-09-13_1003CEST.json",
    "0921": "data/3Glogs/report.2010-09-21_0742CEST.json",
}

VIDEOS = {
    "bbb": {
        "input": "data/video_mp4box/bbb.mpd",
        "extra": [],
    },
    "tos": {
        "input": "data/tears_of_steel/tos.mpd",
        "extra": ["--config", "tos.yaml"],
    },
}

# All arms use multitask so analyzer QOE shares multitask_*_quality.json
ARMS = {
    "ours": {
        "mod_nes": "tool_greedy",
        "mod_enhancer": "multitask",
        "use_fast_complete": "true",
    },
    "none": {
        "mod_nes": "always_greedy",
        "mod_enhancer": "multitask",
        "use_fast_complete": "true",
        "fixed_enhance_action": "0",
    },
    "fixed_high": {
        "mod_nes": "always_greedy",
        "mod_enhancer": "multitask",
        "use_fast_complete": "true",
        "fixed_enhance_action": "3",
    },
    "no_hd": {
        "mod_nes": "tool_greedy",
        "mod_enhancer": "multitask",
        "use_fast_complete": "true",
        "tool_mask_hd": "true",
    },
    "no_slack": {
        "mod_nes": "tool_greedy",
        "mod_enhancer": "multitask",
        "use_fast_complete": "true",
        "tool_ignore_slack": "true",
    },
    "no_fc": {
        "mod_nes": "tool_greedy",
        "mod_enhancer": "multitask",
        "use_fast_complete": "false",
    },
}


def build_cmd(arm: dict, video: str, trace: str, metric_dir: Path) -> list[str]:
    metric_dir.mkdir(parents=True, exist_ok=True)
    v = VIDEOS[video]
    cmd = [PY, "play.py"]
    cmd += v["extra"]
    cmd += [
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
        arm.get("use_fast_complete", "true"),
        "--metric_output",
        str(metric_dir),
        "-i",
        v["input"],
        "-t",
        trace,
    ]
    for key in ("fixed_enhance_action", "tool_mask_hd", "tool_ignore_slack"):
        if key in arm:
            cmd += [f"--{key}", arm[key]]
    return cmd


def extract_metrics(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    tools = Counter()
    downs = Counter()
    for s in raw.get("segments", []):
        tid = s.get("tool_id")
        if tid is None:
            tid = s.get("enhance_level", s.get("enhance_action"))
        tools[str(tid)] += 1
        downs[str(s.get("download_action"))] += 1
    return {
        "result_file": str(path.relative_to(REPO)).replace("\\", "/"),
        "QOE": raw.get("QOE", raw.get("QoE")),
        "avg_vmaf": raw.get("avg_vmaf"),
        "dur_stall": raw.get("dur_stall"),
        "num_stall": raw.get("num_stall"),
        "n_segments": len(raw.get("segments", [])),
        "tool_or_level_hist": dict(tools),
        "download_hist": dict(downs),
        "fast_complete_segments": raw.get("fast_complete_segments_count"),
    }


def run_one(
    arm_name: str, video: str, trace_key: str, timeout: int, metric_root: Path
) -> dict:
    arm = ARMS[arm_name]
    trace = TRACES[trace_key]
    metric_dir = metric_root / f"{video}_{trace_key}_{arm_name}"
    cmd = build_cmd(arm, video, trace, metric_dir)
    before = {p for p in metric_dir.glob("*.json")} if metric_dir.exists() else set()
    t0 = time.time()
    print(f"\n=== {arm_name} @ {video}/{trace_key} ===", flush=True)
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
        err_tail = (proc.stderr or "")[-2500:]
        out_tail = (proc.stdout or "")[-2500:]
    except subprocess.TimeoutExpired as e:
        rc = -9
        err_tail = f"timeout after {timeout}s"
        out_tail = (e.stdout or "")[-2500:] if isinstance(e.stdout, str) else ""
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
        "video": video,
        "trace": trace_key,
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
        f"  rc={rc} dur={dur:.0f}s QOE={row.get('QOE')} stall={row.get('dur_stall')} "
        f"file={row.get('result_file')}",
        flush=True,
    )
    if rc != 0:
        print(f"  stderr_tail:\n{err_tail[-1000:]}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="bbb", choices=list(VIDEOS))
    ap.add_argument("--traces", nargs="+", default=["0913"], choices=list(TRACES))
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    ap.add_argument("--timeout", type=int, default=4500)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metric_root = PLAYER / "output" / "data" / "d5"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rows = []
    for tr in args.traces:
        for arm in args.arms:
            rows.append(run_one(arm, args.video, tr, args.timeout, metric_root))
            slim = [
                {k: v for k, v in r.items() if k not in ("stdout_tail", "stderr_tail")}
                for r in rows
            ]
            (OUT_DIR / f"d5_partial_{stamp}.json").write_text(
                json.dumps(slim, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
            )

    slim = [
        {k: v for k, v in r.items() if k not in ("stdout_tail", "stderr_tail")} for r in rows
    ]
    summary = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "video": args.video,
        "traces": args.traces,
        "arms": args.arms,
        "note": "all arms multitask Q-table for fair lookup QoE",
        "rows": slim,
        "rows_with_logs": rows,
    }
    out = OUT_DIR / f"d5_summary_{stamp}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    for r in slim:
        print(
            f"  {r['arm']:12s} QOE={r.get('QOE')} stall={r.get('dur_stall')} ok={r.get('ok')}"
        )


if __name__ == "__main__":
    main()
