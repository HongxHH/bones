from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional

from prism.paths import repo_root

log = logging.getLogger("prism.attach")

REPO_ROOT = repo_root()
PLAYER_DIR = REPO_ROOT / "player"
PLAYER_LOG_DIR = REPO_ROOT / "prism" / "var" / "player_logs"
STATUS_PORT_BASE = 8765


def player_cmd(
    mpd_url: str,
    player: dict[str, Any],
    status_port: int,
    *,
    run_id: str | None = None,
    arm_id: str | None = None,
    session_id: str | None = None,
    experiment_id: str | None = None,
    server_url: str | None = None,
    frame_shm: str | None = None,
    frame_archive: str | None = None,
) -> list[str]:
    play_py = PLAYER_DIR / "play.py"
    trace = str(player.get("trace") or "").strip()
    downloader = str(player.get("mod_downloader") or "tcp")
    if trace and downloader in ("tcp", "auto", ""):
        downloader = "trace"
    if not trace and downloader == "trace":
        downloader = "tcp"
    args = [
        sys.executable,
        str(play_py),
        "-i",
        mpd_url,
        "--mod_downloader",
        downloader,
        "--mod_renderer",
        str(player.get("mod_renderer") or "opencv"),
        "--mod_status",
        str(player.get("mod_status") or "status"),
        "--status_port",
        str(status_port),
    ]
    nes = player.get("mod_nes") or "none"
    # "none" is not a player module. The default bones table assumes a 5-rung
    # ladder and crashes on the 3-representation BBB preset, so play with a
    # controller that follows the actual MPD.
    if not nes or nes == "none":
        nes = "always_greedy"
    args.extend(["--mod_nes", str(nes)])
    enhancer = str(player.get("mod_enhancer") or "imdn_always").strip() or "imdn_always"
    args.extend(["--mod_enhancer", enhancer])
    if trace:
        args.extend(["--trace", trace])
    if server_url:
        args.extend(["--status_server_url", server_url])
    if run_id:
        args.extend(["--telemetry_run_id", run_id])
    if arm_id:
        args.extend(["--telemetry_arm_id", arm_id])
    if session_id:
        args.extend(["--telemetry_session_id", session_id])
    if experiment_id:
        args.extend(["--telemetry_experiment_id", experiment_id])
    args.extend(["--telemetry_interval_ms", str(player.get("telemetry_interval_ms") or 500)])
    sys_metrics = player.get("collect_system_metrics", True)
    args.extend(["--collect_system_metrics", "true" if sys_metrics else "false"])
    shm = (frame_shm or os.environ.get("PRISM_FRAME_SHM") or "").strip()
    if shm:
        args.extend(["--frame_shm", shm])
    archive = (frame_archive or "").strip()
    if archive:
        args.extend(["--frame_archive", archive])
    return args


def attach_url() -> Optional[str]:
    return os.environ.get("PRISM_ATTACH_URL") or None


def spawn_player(cmd: list[str], *, log_path: Path | None = None) -> int:
    """Spawn via remote agent if PRISM_ATTACH_URL set, else local subprocess."""
    url = attach_url()
    if url:
        return _spawn_via_agent(url.rstrip("/"), cmd)
    return _spawn_local(cmd, log_path=log_path)


def kill_pid(pid: int) -> None:
    if pid <= 0:
        return
    url = attach_url()
    if url:
        _kill_via_agent(url.rstrip("/"), pid)
        return
    _kill_local(pid)


def pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
                check=False,
            )
            return str(pid) in (out.stdout or "")
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def tail_log(path: Path | None, *, max_chars: int = 1200) -> str:
    if not path or not path.is_file():
        return ""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def _spawn_local(cmd: list[str], *, log_path: Path | None = None) -> int:
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    stdout = subprocess.DEVNULL
    stderr = subprocess.DEVNULL
    log_fh = None
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w", encoding="utf-8", errors="replace")
        stdout = log_fh
        stderr = subprocess.STDOUT
    proc = subprocess.Popen(
        cmd,
        cwd=str(PLAYER_DIR),
        env=os.environ.copy(),
        stdout=stdout,
        stderr=stderr,
        creationflags=creationflags,
    )
    # Child keeps the fd; close our handle so the file is not locked forever.
    if log_fh is not None:
        try:
            log_fh.close()
        except Exception:
            pass
    return int(proc.pid)


def _kill_local(pid: int) -> None:
    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
        else:
            os.kill(pid, 15)
    except Exception as exc:
        log.warning("failed to stop pid %s: %s", pid, exc)


def _spawn_via_agent(base: str, cmd: list[str]) -> int:
    payload = json.dumps({"cmd": cmd, "cwd": str(PLAYER_DIR)}).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/spawn",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(f"attach agent unreachable at {base}: {exc}") from exc
    if not data.get("ok"):
        raise RuntimeError(data.get("error") or "attach agent spawn failed")
    return int(data["pid"])


def _kill_via_agent(base: str, pid: int) -> None:
    payload = json.dumps({"pid": pid}).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/kill",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            resp.read()
    except Exception as exc:
        log.warning("attach agent kill failed pid=%s: %s", pid, exc)
