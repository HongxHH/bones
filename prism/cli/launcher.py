from __future__ import annotations

import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

from prism import PRODUCT_NAME, __version__
from prism.paths import repo_root

REPO_ROOT = repo_root()


def health_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/v1/health"


def lab_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/lab/"


def check_health(host: str, port: int, timeout_s: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(health_url(host, port), timeout=timeout_s) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def wait_health(host: str, port: int, timeout_s: float) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if check_health(host, port, timeout_s=1.0):
            return True
        time.sleep(0.25)
    return False


def spawn_server(bind: str, port: int) -> subprocess.Popen:
    env = os.environ.copy()
    cmd = [sys.executable, "-m", "prism", "serve", "--host", bind, "--port", str(port)]
    # DETACHED on Windows so closing the launcher terminal later is less surprising;
    # we still keep the handle to wait optionally.
    creationflags = 0
    if sys.platform == "win32":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return subprocess.Popen(
        cmd,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )


def ensure_control_plane(
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    bind: str = "0.0.0.0",
    start_server: bool = True,
    timeout_s: float = 30.0,
) -> tuple[int, subprocess.Popen | None]:
    """Return (exit_code, child_process_or_None). exit_code 0 means healthy."""
    if check_health(host, port):
        print(f"control plane already up at {health_url(host, port)}")
        return 0, None
    if not start_server:
        print(
            f"error: control plane not reachable at {health_url(host, port)}",
            file=sys.stderr,
        )
        return 1, None
    print(f"starting control plane on {bind}:{port} …")
    child = spawn_server(bind, port)
    if not wait_health(host, port, timeout_s):
        print(
            f"error: control plane did not become healthy within {timeout_s:.0f}s\n"
            f"  tried {health_url(host, port)}\n"
            f"  try: python -m prism serve --port {port}",
            file=sys.stderr,
        )
        return 1, child
    print("control plane ready")
    return 0, child


def launch_lab(
    *,
    host: str = "127.0.0.1",
    port: int = 8080,
    bind: str = "0.0.0.0",
    open_browser: bool = True,
    start_server: bool = True,
    timeout_s: float = 30.0,
    wait: bool = True,
) -> int:
    print(f"{PRODUCT_NAME} {__version__}")
    code, child = ensure_control_plane(
        host=host,
        port=port,
        bind=bind,
        start_server=start_server,
        timeout_s=timeout_s,
    )
    if code != 0:
        return code

    url = lab_url(host, port)
    print(f"lab: {url}")
    if open_browser:
        webbrowser.open(url)

    if wait and child is not None:
        print("Prism spawned the server; press Ctrl+C to stop this launcher (server keeps running).")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            print("\nlauncher exit (server left running)")
            return 0
    return 0
