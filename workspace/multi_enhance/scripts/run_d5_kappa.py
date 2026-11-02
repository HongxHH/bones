# -*- coding: utf-8 -*-
"""BBB x 0921: tool_greedy kappa sweep (0, 0.5) vs cached κ=1 / no_slack."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
WS = HERE.parent
REPO = WS.parents[1]
PLAYER = REPO / "player"
OUT = WS / "results" / "runs"
PY = sys.executable

TRACE = "data/3Glogs/report.2010-09-21_0742CEST.json"
INPUT = "data/video_mp4box/bbb.mpd"
KAPPAS = [0.0, 0.5]


def main():
    timeout = 4500
    if len(sys.argv) > 1:
        timeout = int(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    rows = []
    for k in KAPPAS:
        tag = f"bbb_0921_kappa_{str(k).replace('.', 'p')}"
        metric = PLAYER / "output" / "data" / "d5" / tag
        metric.mkdir(parents=True, exist_ok=True)
        before = set(metric.glob("*.json"))
        cmd = [
            PY,
            "play.py",
            "--mod_nes",
            "tool_greedy",
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
            "--tool_slack_kappa",
            str(k),
            "--metric_output",
            str(metric),
            "-i",
            INPUT,
            "-t",
            TRACE,
        ]
        print(f"\n=== kappa={k} ===", flush=True)
        print(" ", " ".join(cmd), flush=True)
        t0 = time.time()
        proc = subprocess.run(
            cmd,
            cwd=str(PLAYER),
            timeout=timeout,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        dur = time.time() - t0
        newest = None
        cands = sorted(metric.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        for p in cands:
            if p not in before:
                newest = p
                break
        if newest is None and cands:
            newest = cands[0]
        row = {"kappa": k, "returncode": proc.returncode, "duration_s": round(dur, 1), "ok": proc.returncode == 0}
        if newest:
            data = json.loads(newest.read_text(encoding="utf-8"))
            row.update(
                {
                    "QOE": data.get("QOE"),
                    "dur_stall": data.get("dur_stall"),
                    "avg_vmaf": data.get("avg_vmaf"),
                    "fast_complete_segments": data.get("fast_complete_segments_count"),
                    "result_file": str(newest.relative_to(REPO)).replace("\\", "/"),
                }
            )
        print(
            f"  rc={proc.returncode} dur={dur:.0f}s QOE={row.get('QOE')} stall={row.get('dur_stall')}",
            flush=True,
        )
        if proc.returncode != 0:
            print((proc.stderr or "")[-1500:], flush=True)
        rows.append(row)
        (OUT / f"d5_kappa_partial_{stamp}.json").write_text(
            json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    summary = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "video": "bbb",
        "trace": "0921",
        "note": "kappa sweep; compare to canonical d5_bbb_0921 ours/no_slack",
        "rows": rows,
    }
    out = OUT / f"d5_kappa_{stamp}.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
