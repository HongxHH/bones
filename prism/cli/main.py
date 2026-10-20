from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

from prism import PRODUCT_NAME

from .launcher import ensure_control_plane, launch_lab


def _http(method: str, url: str, body: dict | None = None, *, raw: bool = False):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            payload = resp.read()
            if raw:
                return payload
            text = payload.decode("utf-8")
            return json.loads(text) if text else {}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"HTTP {exc.code}: {detail}") from exc


def _http_json(method: str, url: str, body: dict | None = None) -> dict:
    return _http(method, url, body, raw=False)  # type: ignore[return-value]


def _ensure(args: argparse.Namespace) -> int:
    code, _ = ensure_control_plane(
        host=args.host,
        port=args.port,
        bind=getattr(args, "bind", "0.0.0.0"),
        start_server=not getattr(args, "no_server", False),
        timeout_s=getattr(args, "timeout", 30.0),
    )
    return code


def cmd_desk(args: argparse.Namespace) -> int:
    from prism.desk.app import main as desk_main

    argv = [
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--bind",
        args.bind,
        "--timeout",
        str(args.timeout),
    ]
    if args.no_server:
        argv.append("--no-server")
    return desk_main(argv)


def cmd_lab(args: argparse.Namespace) -> int:
    return launch_lab(
        host=args.host,
        port=args.port,
        bind=args.bind,
        open_browser=not args.no_browser,
        start_server=not args.no_server,
        timeout_s=args.timeout,
    )


def cmd_serve(args: argparse.Namespace) -> int:
    from prism.control.main import main as serve_main

    argv = ["--host", args.host, "--port", str(args.port)]
    if getattr(args, "public_base_url", None):
        argv.extend(["--public-base-url", args.public_base_url])
    if getattr(args, "data_dir", None):
        argv.extend(["--data-dir", str(args.data_dir)])
    if getattr(args, "ffmpeg", None):
        argv.extend(["--ffmpeg", args.ffmpeg])
    serve_main(argv)
    return 0


def cmd_agent(args: argparse.Namespace) -> int:
    from .agent import main as agent_main

    return agent_main(["--host", args.host, "--port", str(args.port)])


def cmd_run(args: argparse.Namespace) -> int:
    from prism.control.experiment_spec import load_experiment_spec

    if _ensure(args) != 0:
        return 1
    spec = load_experiment_spec(Path(args.spec))
    url = f"http://{args.host}:{args.port}/v1/runs"
    payload = {
        "spec": spec.model_dump(mode="json"),
        "attach_players": True,
    }
    run = _http_json("POST", url, payload)
    print(json.dumps(run, indent=2, ensure_ascii=False))
    print(f"\n{PRODUCT_NAME} run {run.get('id')} status={run.get('status')}")
    for arm in run.get("arms") or []:
        print(f"  arm {arm.get('id')}: {arm.get('mpd_url')}")
    return 0 if run.get("status") in ("running", "stopped") else 1


def cmd_status(args: argparse.Namespace) -> int:
    if _ensure(args) != 0:
        return 1
    base = f"http://{args.host}:{args.port}/v1"
    run = _http_json("GET", f"{base}/runs/{args.run_id}")
    metrics = _http_json("GET", f"{base}/runs/{args.run_id}/metrics")
    print(f"{PRODUCT_NAME} run {run.get('id')}  status={run.get('status')}  name={run.get('name')}")
    print(f"  experiment={run.get('experiment_id')}  mode={run.get('arm_mode')}  active={run.get('active_arm_id')}")
    for arm in run.get("arms") or []:
        m = (metrics.get("arms") or {}).get(arm.get("id")) or arm.get("metrics") or {}
        print(
            f"  arm {arm.get('id')}: {arm.get('status')}  "
            f"rebuffer={m.get('rebuffer_s')}s  "
            f"bw={m.get('avg_bandwidth_bps')}  "
            f"samples={m.get('samples')}"
        )
        if arm.get("mpd_url"):
            print(f"    {arm.get('mpd_url')}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    if _ensure(args) != 0:
        return 1
    base = f"http://{args.host}:{args.port}/v1"
    if not args.run_id:
        raise SystemExit("export requires run_id")
    url = f"{base}/runs/{args.run_id}/export?format=csv"
    default_name = f"{args.run_id}.csv"
    raw = _http("GET", url, raw=True)
    out = Path(args.output) if args.output else Path(default_name)
    out.write_bytes(raw)  # type: ignore[arg-type]
    print(f"wrote {out} ({len(raw)} bytes)")  # type: ignore[arg-type]
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    if _ensure(args) != 0:
        return 1
    run = _http_json("POST", f"http://{args.host}:{args.port}/v1/runs/{args.run_id}/stop")
    print(json.dumps({"id": run.get("id"), "status": run.get("status"), "metrics": run.get("metrics")}, indent=2))
    return 0


def _add_server_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--bind", default="0.0.0.0")
    p.add_argument("--no-server", action="store_true")
    p.add_argument("--timeout", type=float, default=30.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="prism",
        description="Prism — experiment lab and run CLI.",
    )
    sub = parser.add_subparsers(dest="command")

    p_desk = sub.add_parser("desk", help="open the Prism window (Lab + picture pane)")
    _add_server_flags(p_desk)
    p_desk.set_defaults(func=cmd_desk)

    p_lab = sub.add_parser("lab", help="open Lab in the system browser")
    _add_server_flags(p_lab)
    p_lab.add_argument("--no-browser", action="store_true")
    p_lab.set_defaults(func=cmd_lab)

    p_run = sub.add_parser("run", help="start a run from YAML/JSON spec")
    p_run.add_argument("spec", help="path to ExperimentSpec (.yaml/.json)")
    _add_server_flags(p_run)
    p_run.set_defaults(func=cmd_run)

    p_status = sub.add_parser("status", help="show run status and metrics")
    p_status.add_argument("run_id")
    _add_server_flags(p_status)
    p_status.set_defaults(func=cmd_status)

    p_export = sub.add_parser("export", help="export run CSV")
    p_export.add_argument("run_id")
    p_export.add_argument("-o", "--output", help="output file path")
    _add_server_flags(p_export)
    p_export.set_defaults(func=cmd_export)

    p_stop = sub.add_parser("stop", help="stop a run")
    p_stop.add_argument("run_id")
    _add_server_flags(p_stop)
    p_stop.set_defaults(func=cmd_stop)

    p_serve = sub.add_parser("serve", help="run the control plane (API / Lab / Origin)")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=8080)
    p_serve.add_argument("--public-base-url", default=None)
    p_serve.add_argument("--data-dir", default=None)
    p_serve.add_argument("--ffmpeg", default=None)
    p_serve.set_defaults(func=cmd_serve)

    p_agent = sub.add_parser("agent", help="local attach agent (spawn/kill players)")
    p_agent.add_argument("--host", default="127.0.0.1")
    p_agent.add_argument("--port", type=int, default=8079)
    p_agent.set_defaults(func=cmd_agent)

    if argv is None:
        argv = sys.argv[1:]
    if not argv or argv[0].startswith("-"):
        argv = ["desk", *argv]

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
