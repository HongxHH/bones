"""Local attach agent — owns player subprocesses so the control plane need not.

Run:  python -m prism agent --port 8079
Then: set PRISM_ATTACH_URL=http://127.0.0.1:8079
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse

log = logging.getLogger("prism.agent")

_PROCS: dict[int, subprocess.Popen] = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args: Any) -> None:
        log.info("%s - " + fmt, self.address_string(), *args)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def _write_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("", "/health"):
            self._write_json(200, {"ok": True, "product": "Prism", "role": "attach-agent", "pids": list(_PROCS)})
            return
        self._write_json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path.rstrip("/")
        try:
            data = self._read_json()
        except Exception as exc:
            self._write_json(400, {"ok": False, "error": str(exc)})
            return
        if path.endswith("/spawn") or path == "/spawn":
            cmd = data.get("cmd") or []
            cwd = data.get("cwd") or os.getcwd()
            if not isinstance(cmd, list) or not cmd:
                self._write_json(400, {"ok": False, "error": "cmd list required"})
                return
            try:
                creationflags = 0
                if sys.platform == "win32":
                    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                proc = subprocess.Popen(
                    [str(c) for c in cmd],
                    cwd=str(cwd),
                    env=os.environ.copy(),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creationflags,
                )
                _PROCS[proc.pid] = proc
                self._write_json(200, {"ok": True, "pid": proc.pid})
            except Exception as exc:
                self._write_json(500, {"ok": False, "error": str(exc)})
            return
        if path.endswith("/kill") or path == "/kill":
            pid = int(data.get("pid") or 0)
            if pid <= 0:
                self._write_json(400, {"ok": False, "error": "pid required"})
                return
            try:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
                else:
                    os.kill(pid, 15)
                _PROCS.pop(pid, None)
                self._write_json(200, {"ok": True, "pid": pid})
            except Exception as exc:
                self._write_json(500, {"ok": False, "error": str(exc)})
            return
        self._write_json(404, {"ok": False, "error": "not found"})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="prism agent")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8079)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        log.error("refusing to bind attach agent on non-loopback host %s", args.host)
        return 2
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log.info("Prism attach agent on http://%s:%s", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("agent stop")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
