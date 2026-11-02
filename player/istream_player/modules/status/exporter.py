"""StatusExporter: local HTTP/WS console + event fan-in.

Playback aggregation lives in ``StatusSnapshot``; control-plane reporting in
``TelemetryPublisher``. This module wires listeners, serves the console API,
and runs the broadcast / telemetry tick loop.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from aiohttp import ClientSession, WSMsgType, web

from istream_player.config.config import PlayerConfig
from istream_player.core.buffer import BufferEventListener
from istream_player.core.bw_meter import BandwidthMeter, BandwidthUpdateListener
from istream_player.core.enhancer import Enhancer, EnhancerEventListener
from istream_player.core.module import Module, ModuleOption
from istream_player.core.mpd_provider import MPDProvider
from istream_player.core.player import Player, PlayerEventListener
from istream_player.core.renderer import Renderer
from istream_player.core.scheduler import Scheduler, SchedulerEventListener
from istream_player.models import State
from istream_player.models.mpd_objects import Segment
from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.modules.status.snapshot import StatusSnapshot
from istream_player.modules.status.telemetry_publisher import TelemetryPublisher


@ModuleOption("off")
class StatusOff(Module):
    async def setup(self, config: PlayerConfig, *args) -> None:
        return


@ModuleOption(
    "status",
    requires=[Player, Scheduler, DownloadBufferImpl, EnhanceBufferImpl, Enhancer, BandwidthMeter, MPDProvider, Renderer],
)
class StatusExporter(
    Module,
    PlayerEventListener,
    SchedulerEventListener,
    BufferEventListener,
    EnhancerEventListener,
    BandwidthUpdateListener,
):
    log = logging.getLogger("StatusExporter")

    def __init__(self) -> None:
        super().__init__()
        self._snap = StatusSnapshot()
        self._telemetry = TelemetryPublisher(
            get_snapshot=self.snapshot,
            get_config=lambda: self._snap.config,
            get_session=self._proxy_session,
            server_base=self._server_base,
            log=self.log,
        )
        self._ws: set[web.WebSocketResponse] = set()
        self._runner: Optional[web.AppRunner] = None
        self._proxy: Optional[ClientSession] = None
        self._closed = asyncio.Event()
        self._dirty = False
        self._last_broadcast = 0.0
        self._min_interval_s = 0.25
        self._launched: List[dict] = []

    # --- convenience mirrors used by HTTP handlers / launch helpers ---
    @property
    def player(self) -> Optional[Player]:
        return self._snap.player

    @property
    def config(self) -> Optional[PlayerConfig]:
        return self._snap.config

    @property
    def mpd_provider(self) -> Optional[MPDProvider]:
        return self._snap.mpd_provider

    async def setup(
        self,
        config: PlayerConfig,
        player: Player,
        scheduler: Scheduler,
        download_buffer: DownloadBufferImpl,
        enhance_buffer: EnhanceBufferImpl,
        enhancer: Enhancer,
        bw_meter: BandwidthMeter,
        mpd_provider: MPDProvider,
        renderer: Renderer,
        **kwargs,
    ) -> None:
        self._snap.bind(
            config=config,
            player=player,
            scheduler=scheduler,
            download_buffer=download_buffer,
            enhance_buffer=enhance_buffer,
            enhancer=enhancer,
            bw_meter=bw_meter,
            mpd_provider=mpd_provider,
            renderer=renderer,
        )
        player.add_listener(self)
        scheduler.add_listener(self)
        download_buffer.add_listener(self)
        enhance_buffer.add_listener(self)
        enhancer.add_listener(self)
        bw_meter.add_listener(self)

    def snapshot(self) -> dict:
        return self._snap.build()

    async def on_state_change(self, position: float, old_state: State, new_state: State):
        self._snap.note_state(new_state)
        if new_state == State.END:
            # Write the analysis file before END telemetry. Prism may stop the
            # process as soon as it sees END, and cleanup() would then be skipped.
            try:
                from istream_player.modules.analyzer.analyzer import flush_analyzers
                flush_analyzers()
            except Exception as exc:
                self.log.warning("analysis flush before END failed: %s", exc)
            try:
                await self._telemetry.force_report()
            except Exception as exc:
                self.log.warning("final telemetry on END failed: %s", exc)
            self._closed.set()
        await self._broadcast(urgent=True)

    async def on_position_change(self, position):
        self._mark_dirty()

    async def on_segment_playback_start(self, segments: Dict[int, Segment]):
        self._snap.note_playback_start(segments)
        await self._broadcast(urgent=True)

    async def on_segment_download_complete(self, index: int, segments, stats):
        self._mark_dirty()

    async def on_buffer_level_change(self, buffer_level: float):
        self._mark_dirty()

    async def on_enhancement_start(self, segments: Dict[int, Segment]):
        self._snap.note_enhancement_start(segments)
        self._mark_dirty()

    async def on_bandwidth_update(self, bw: float) -> None:
        self._mark_dirty()

    def _mark_dirty(self) -> None:
        self._dirty = True

    async def _broadcast(self, *, urgent: bool = False) -> None:
        if not self._ws:
            self._dirty = False
            return
        now = time.monotonic()
        if not urgent and now - self._last_broadcast < self._min_interval_s:
            self._dirty = True
            return
        payload = json.dumps(self.snapshot(), ensure_ascii=False)
        stale = []
        for ws in list(self._ws):
            try:
                await ws.send_str(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            self._ws.discard(ws)
        self._last_broadcast = now
        self._dirty = False

    async def run(self) -> None:
        app = web.Application()
        app.router.add_get("/api/status", self._handle_status)
        app.router.add_get("/api/config", self._handle_config)
        app.router.add_post("/api/pause", self._handle_pause)
        app.router.add_post("/api/resume", self._handle_resume)
        app.router.add_post("/api/compare", self._handle_compare)
        app.router.add_post("/api/stop", self._handle_stop)
        app.router.add_post("/api/switch_source", self._handle_switch_source)
        app.router.add_post("/api/launch", self._handle_launch)
        app.router.add_get("/api/launches", self._handle_launches)
        app.router.add_get("/api/server/{path:.*}", self._handle_server_get)
        app.router.add_post("/api/server/{path:.*}", self._handle_server_post)
        app.router.add_patch("/api/server/{path:.*}", self._handle_server_patch)
        app.router.add_get("/ws", self._handle_ws)
        console_dir = Path(__file__).resolve().parents[3] / "console"
        if console_dir.exists():
            app.router.add_get("/", self._handle_index)
            app.router.add_static("/static", console_dir, show_index=False)
        else:
            app.router.add_get("/", self._handle_index)
        runner = web.AppRunner(app)
        await runner.setup()
        host = getattr(self.config, "status_host", "127.0.0.1")
        port = int(getattr(self.config, "status_port", 8765))
        site = web.TCPSite(runner, host, port)
        await site.start()
        self._runner = runner
        self.log.info("Status console on http://%s:%s", host, port)
        try:
            while not self._closed.is_set():
                try:
                    await asyncio.wait_for(self._closed.wait(), timeout=0.25)
                except asyncio.TimeoutError:
                    if self._dirty:
                        await self._broadcast()
                    await self._telemetry.maybe_report()
        except asyncio.CancelledError:
            raise
        finally:
            await self.cleanup()

    async def cleanup(self) -> None:
        if self._proxy is not None:
            await self._proxy.close()
            self._proxy = None
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None

    async def _handle_status(self, request: web.Request) -> web.Response:
        return web.json_response(self.snapshot())

    async def _handle_config(self, request: web.Request) -> web.Response:
        return web.json_response(self.snapshot().get("config") or {})

    async def _handle_pause(self, request: web.Request) -> web.Response:
        if self.player is not None:
            self.player.pause()
        await self._broadcast(urgent=True)
        return web.json_response(self.snapshot())

    async def _handle_resume(self, request: web.Request) -> web.Response:
        if self.player is not None:
            self.player.resume()
        await self._broadcast(urgent=True)
        return web.json_response(self.snapshot())

    async def _handle_compare(self, request: web.Request) -> web.Response:
        """Turn the bicubic-baseline wipe on or off. Later frames pick it up."""
        try:
            body = await request.json()
        except Exception:
            body = {}
        enabled = bool((body or {}).get("enabled"))
        if self.config is not None:
            self.config.compare_view = enabled
        await self._broadcast(urgent=True)
        snap = self.snapshot()
        snap["compare_view"] = enabled
        return web.json_response(snap)

    async def _handle_stop(self, request: web.Request) -> web.Response:
        if self.player is not None:
            self.player.stop()
        await self._broadcast(urgent=True)
        return web.json_response(self.snapshot())

    async def _handle_switch_source(self, request: web.Request) -> web.Response:
        """U2: switch current player MPD URL and force refresh (dynamic sessions follow next)."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "JSON body required"}, status=400)
        url = body.get("url") or body.get("mpd_url")
        if not url or self.mpd_provider is None:
            return web.json_response({"ok": False, "error": "url required / mpd_provider missing"}, status=400)
        self.mpd_provider.mpd_url = url  # type: ignore[attr-defined]
        if self.config is not None:
            self.config.input = url
        try:
            await self.mpd_provider.update(force=True)  # type: ignore[call-arg]
        except TypeError:
            await self.mpd_provider.update()
        await self._broadcast(urgent=True)
        return web.json_response({"ok": True, "input": url, "status": self.snapshot()})

    async def _handle_launch(self, request: web.Request) -> web.Response:
        """U2: spawn a new player process with the given MPD URL."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "JSON body required"}, status=400)
        url = body.get("url") or body.get("mpd_url")
        if not url:
            return web.json_response({"ok": False, "error": "url required"}, status=400)
        player_dir = Path(__file__).resolve().parents[3]
        play_py = player_dir / "play.py"
        args = [
            sys.executable,
            str(play_py),
            "-i",
            url,
            "--mod_downloader",
            body.get("mod_downloader", "tcp"),
            "--mod_renderer",
            body.get("mod_renderer", "headless"),
            "--mod_status",
            body.get("mod_status", "off"),
        ]
        if body.get("mod_nes"):
            args.extend(["--mod_nes", str(body["mod_nes"])])
        if body.get("extra_args"):
            args.extend([str(a) for a in body["extra_args"]])
        env = os.environ.copy()
        try:
            proc = subprocess.Popen(
                args,
                cwd=str(player_dir),
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            return web.json_response({"ok": False, "error": str(exc), "cmd": args}, status=500)
        record = {"pid": proc.pid, "cmd": args, "url": url, "at": time.time()}
        self._launched.append(record)
        return web.json_response({"ok": True, **record, "player_cmd": " ".join(args)})

    async def _handle_launches(self, request: web.Request) -> web.Response:
        return web.json_response({"launches": self._launched})

    async def _handle_index(self, request: web.Request) -> web.Response:
        console = Path(__file__).resolve().parents[3] / "console" / "index.html"
        if console.exists():
            return web.FileResponse(console)
        return web.Response(text="BONES status API. Missing player/console/index.html", content_type="text/plain")

    async def _handle_ws(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._ws.add(ws)
        await ws.send_str(json.dumps(self.snapshot(), ensure_ascii=False))
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT and msg.data == "ping":
                    await ws.send_str("pong")
                elif msg.type in (WSMsgType.CLOSE, WSMsgType.ERROR):
                    break
        finally:
            self._ws.discard(ws)
        return ws

    def _server_base(self) -> str:
        return getattr(self.config, "status_server_url", "http://127.0.0.1:8080").rstrip("/")

    async def _proxy_session(self) -> ClientSession:
        if self._proxy is None or self._proxy.closed:
            self._proxy = ClientSession()
        return self._proxy

    async def _handle_server_get(self, request: web.Request) -> web.Response:
        return await self._proxy_server(request, "GET")

    async def _handle_server_post(self, request: web.Request) -> web.Response:
        return await self._proxy_server(request, "POST")

    async def _handle_server_patch(self, request: web.Request) -> web.Response:
        return await self._proxy_server(request, "PATCH")

    async def _proxy_server(self, request: web.Request, method: str) -> web.Response:
        path = request.match_info.get("path", "")
        override = request.headers.get("X-Bones-Server") or request.query.get("server")
        base = (override or self._server_base()).rstrip("/")
        url = f"{base}/v1/{path}"
        if request.query_string:
            url = f"{url}?{request.query_string}"
        body = await request.read()
        session = await self._proxy_session()
        try:
            async with session.request(
                method,
                url,
                data=body,
                headers={"Content-Type": request.headers.get("Content-Type", "application/json")},
            ) as resp:
                payload = await resp.read()
                return web.Response(body=payload, status=resp.status, content_type=resp.content_type)
        except Exception as exc:
            return web.json_response(
                {"ok": False, "error": str(exc), "hint": "Start bones server or set status_server_url"},
                status=502,
            )
