"""Golden-path acceptance for Prism control plane (no GUI attach)."""

from __future__ import annotations

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# prism/scripts -> repo root
ROOT = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:8080/v1"
BBB = ROOT / "player" / "data" / "video_mp4box"


def http(method: str, path: str, body: dict | None = None, timeout: float = 120.0):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(f"{BASE}{path}", data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def main() -> int:
    print("1) health")
    health = http("GET", "/health")
    assert health.get("ok") and health.get("product") == "Prism", health
    print("   ok", health.get("product"))

    print("2) system + BBB preset")
    sysinfo = http("GET", "/system")
    preset = (sysinfo.get("presets") or {}).get("bbb_dash") or {}
    assert preset.get("exists"), preset
    print("   bbb", preset.get("path"))

    print("3) start run (identity, wait)")
    spec = {
        "apiVersion": "prism.lab/v1",
        "kind": "ExperimentSpec",
        "metadata": {"name": "accept-bbb"},
        "spec": {
            "content": {"source": f"path:{BBB}"},
            "arms": [
                {
                    "id": "a",
                    "session": {"policy": {"allowed_representation_ids": [0, 1]}},
                    "player": {"mod_renderer": "headless", "mod_status": "off"},
                },
            ],
            "measure": [{"plugin": "noop"}],
            "execution": {"arm_mode": "serial"},
        },
    }
    run = http("POST", "/runs", {"spec": spec, "attach_players": False, "wait": True}, timeout=180)
    assert run["status"] == "running", run
    rid = run["id"]
    print("   run", rid, "arms", [a["id"] for a in run["arms"]])

    print("4) telemetry samples")
    for i in range(4):
        http(
            "POST",
            "/telemetry",
            {
                "run_id": rid,
                "arm_id": "a",
                "sample": {
                    "ts": time.time() + i,
                    "state": "BUFFERING" if i == 1 else "PLAYING",
                    "position": float(i),
                    "download_buffer_s": 2.0,
                    "bandwidth_bps": 1.2e6,
                    "download_action": {"0": 0},
                },
            },
        )
    series = http("GET", f"/runs/{rid}/telemetry?arm_id=a&limit=10")
    assert len(series["series"]["a"]) >= 4
    print("   samples", len(series["series"]["a"]))

    print("5) stop + measure jobs")
    stopped = http("POST", f"/runs/{rid}/stop")
    assert stopped["status"] == "stopped"
    mids = (stopped.get("metrics") or {}).get("measure_job_ids") or []
    assert mids, stopped.get("metrics")
    print("   measure", mids)

    print("6) csv export")
    req = urllib.request.Request(f"{BASE}/runs/{rid}/export?format=csv")
    with urllib.request.urlopen(req, timeout=30) as resp:
        csv_text = resp.read().decode("utf-8")
    assert "rebuffer_s" in csv_text and rid in csv_text
    print("   csv bytes", len(csv_text))

    print("ACCEPT OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except urllib.error.URLError as exc:
        print(
            "ACCEPT FAIL: control plane not reachable — start with: python -m prism serve --port 8080",
            file=sys.stderr,
        )
        print(exc, file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        print("ACCEPT FAIL:", exc, file=sys.stderr)
        raise SystemExit(1)
