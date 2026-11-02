"""Periodic L0 telemetry POST to the control plane.

Thin publisher: takes a snapshot callable, projects via telemetry_schema,
retries briefly, never raises into the playback loop.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Optional

from istream_player.config.config import PlayerConfig
from istream_player.modules.status.telemetry_schema import project_sample

GetSnapshot = Callable[[], dict[str, Any]]
GetSession = Callable[[], Awaitable[Any]]


class TelemetryPublisher:
    def __init__(
        self,
        *,
        get_snapshot: GetSnapshot,
        get_config: Callable[[], Optional[PlayerConfig]],
        get_session: GetSession,
        server_base: Callable[[], str],
        log: Optional[logging.Logger] = None,
    ) -> None:
        self._get_snapshot = get_snapshot
        self._get_config = get_config
        self._get_session = get_session
        self._server_base = server_base
        self._log = log or logging.getLogger("TelemetryPublisher")
        self._last_telemetry = 0.0
        self._last_state: Optional[str] = None
        self._post_task: Optional[asyncio.Task] = None

    def enabled(self) -> bool:
        config = self._get_config()
        if config is None:
            return False
        return bool(getattr(config, "telemetry_run_id", "") and getattr(config, "telemetry_arm_id", ""))

    async def maybe_report(self) -> None:
        if not self.enabled():
            return
        config = self._get_config()
        assert config is not None
        interval_ms = int(getattr(config, "telemetry_interval_ms", 500) or 500)
        now = time.monotonic()
        if now - self._last_telemetry < max(0.1, interval_ms / 1000.0):
            return
        if self._post_task is not None and not self._post_task.done():
            return
        # Reserve the slot, then post on a later loop turn so playback can continue.
        self._last_telemetry = time.monotonic()
        task = asyncio.create_task(self._post_snapshot(), name="telemetry-post")
        task.add_done_callback(self._log_task_error)
        self._post_task = task

    def _log_task_error(self, task: asyncio.Task) -> None:
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            self._log.warning("telemetry post task failed: %s", exc)

    async def flush(self) -> None:
        """Wait for the in-flight background post. Swallows its error so END can still send."""
        task = self._post_task
        if task is None or task.done():
            return
        try:
            await task
        except Exception as exc:
            self._log.warning("telemetry post task failed: %s", exc)

    async def force_report(self) -> None:
        """Post one sample even if the interval has not elapsed (e.g. END).

        Waits out any background post first so the END sample is the only writer.
        """
        if not self.enabled():
            return
        await self.flush()
        await self._post_snapshot()
        self._last_telemetry = time.monotonic()

    async def _post_snapshot(self) -> None:
        config = self._get_config()
        assert config is not None
        snap = self._get_snapshot()
        state_name = str(snap.get("state") or "")
        event = None
        if self._last_state is not None and state_name != self._last_state:
            event = f"state:{self._last_state}->{state_name}"
        self._last_state = state_name
        snap_for_tel = dict(snap)
        snap_for_tel["enhanced_ready_inventory"] = snap.get("enhanced_ready_inventory") or []
        snap_for_tel["enhance_pending_inventory"] = snap.get("enhance_pending_inventory") or []
        if snap_for_tel.get("segment_index") is None:
            snap_for_tel["segment_index"] = snap_for_tel.get("playing_segment")
        sample = project_sample(snap_for_tel, ts=time.time(), event=event)
        body = {
            "run_id": getattr(config, "telemetry_run_id", ""),
            "arm_id": getattr(config, "telemetry_arm_id", ""),
            "session_id": getattr(config, "telemetry_session_id", "") or None,
            "sample": sample,
        }
        url = f"{self._server_base().rstrip('/')}/v1/telemetry"
        session = await self._get_session()
        last_err: Optional[str] = None
        for attempt in range(3):
            try:
                async with session.post(url, json=body) as resp:
                    if resp.status < 400:
                        last_err = None
                        break
                    text = await resp.text()
                    last_err = f"{resp.status}: {text[:200]}"
            except Exception as exc:
                last_err = str(exc)
            await asyncio.sleep(0.05 * (attempt + 1))
        if last_err:
            self._log.warning("telemetry post failed after retries: %s", last_err)
        self._last_telemetry = time.monotonic()
