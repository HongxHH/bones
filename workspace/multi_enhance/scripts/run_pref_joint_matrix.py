# -*- coding: utf-8 -*-
"""BBB x 0921 pref_joint courage/taste matrix (PARAMS_JOINT_V1 §7)."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PLAYER = REPO / "player"
WS = Path(__file__).resolve().parents[1]
OUT = WS / "results" / "runs"
PY = sys.executable

TRACE = "data/3Glogs/report.2010-09-21_0742CEST.json"
INPUT = "data/video_mp4box/bbb.mpd"

# A already smoked; re-include for one canonical summary if --all
ARMS = [
    ("A", "balanced", "balanced"),
    ("B", "conservative", "balanced"),
    ("C", "aggressive", "balanced"),
    ("D", "balanced", "sharp"),
    ("E", "balanced", "clean"),
]

TOOL = {0: "none", 1: "L1", 2: "L2", 3: "L3", 4: "x2", 5: "rst"}


def tool_id(s: dict) -> int:
    t = s.get("tool_id")
    if t is None:
        t = s.get("enhance_action")
    return int(t) if t is not None else 0


def summarize(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    segs = data.get("segments") or []
    return {
        "QOE": data.get("QOE"),
        "dur_stall": data.get("dur_stall"),
        "avg_vmaf": data.get("avg_vmaf"),
        "fast_complete_segments": data.get("fast_complete_segments_count"),
        "tools": dict(Counter(TOOL[tool_id(s)] for s in segs)),
        "download": dict(Counter(int(s.get("download_action") or 0) for s in segs)),
        "result_file": str(path.relative_to(REPO)).replace("\\", "/"),
    }


def run_arm(tag: str, courage: str, taste: str, timeout: int) -> dict:
    metric = PLAYER / "output" / "data" / "pref_joint" / f"0921_{tag}_{courage}_{taste}"
    metric.mkdir(parents=True, exist_ok=True)
    before = set(metric.glob("*.json"))
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
        courage,
        "--user_taste",
        taste,
        "--metric_output",
        str(metric),
        "-i",
        INPUT,
        "-t",
        TRACE,
    ]
    print(f"\n=== {tag} courage={courage} taste={taste} ===", flush=True)
    print(" ", " ".join(cmd), flush=True)
    t0 = time.time()
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
        err = (proc.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        rc = -9
        err = f"timeout {timeout}s"
    dur = time.time() - t0
    newest = None
    cands = sorted(metric.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in cands:
        if p not in before:
            newest = p
            break
    if newest is None and cands:
        newest = cands[0]
    row = {
        "tag": tag,
        "courage": courage,
        "taste": taste,
        "returncode": rc,
        "ok": rc == 0,
        "duration_s": round(dur, 1),
    }
    if newest and rc == 0:
        row.update(summarize(newest))
    else:
        row["stderr_tail"] = err
    print(
        f"  rc={rc} dur={dur:.0f}s QOE={row.get('QOE')} stall={row.get('dur_stall')} "
        f"FC={row.get('fast_complete_segments')} tools={row.get('tools')}",
        flush=True,
    )
    return row


def main():
    timeout = 4500
    only = None
    args = sys.argv[1:]
    if args and args[0].isdigit():
        timeout = int(args[0])
        args = args[1:]
    if args:
        only = set(a.upper() for a in args)

    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rows = []

    # Attach prior smoke A if skipping re-run
    smoke = PLAYER / "output" / "data" / "pref_joint" / "smoke_balanced"
    smoke_files = sorted(smoke.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    for tag, courage, taste in ARMS:
        if only and tag not in only:
            continue
        if tag == "A" and not only and smoke_files:
            row = {
                "tag": "A",
                "courage": "balanced",
                "taste": "balanced",
                "returncode": 0,
                "ok": True,
                "duration_s": None,
                "reused_smoke": True,
            }
            row.update(summarize(smoke_files[0]))
            print(f"\n=== A reused smoke === QOE={row['QOE']} stall={row['dur_stall']}", flush=True)
            rows.append(row)
            continue
        rows.append(run_arm(tag, courage, taste, timeout))
        (OUT / f"pref_joint_matrix_partial_{stamp}.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    # Reference rows from D5 canonical (not re-run)
    d5 = WS / "results" / "canonical" / "d5_bbb_0921.json"
    refs = []
    if d5.is_file():
        d5data = json.loads(d5.read_text(encoding="utf-8"))
        for r in d5data.get("rows") or []:
            if r.get("arm") in ("ours", "none", "no_slack", "fixed_high"):
                refs.append(
                    {
                        "tag": f"ref_{r['arm']}",
                        "arm": r["arm"],
                        "QOE": r.get("QOE"),
                        "dur_stall": r.get("dur_stall"),
                        "fast_complete_segments": r.get("fast_complete_segments"),
                        "tools": r.get("tool_or_level_hist"),
                    }
                )

    summary = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "video": "bbb",
        "trace": "0921",
        "note": "pref_joint matrix; refs from d5_bbb_0921",
        "rows": rows,
        "refs": refs,
    }
    out = OUT / f"pref_joint_matrix_{stamp}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    for r in rows:
        print(
            f"  {r['tag']} {r['courage']}/{r['taste']}: "
            f"QOE={r.get('QOE')} stall={r.get('dur_stall')} FC={r.get('fast_complete_segments')}"
        )


if __name__ == "__main__":
    main()
