import asyncio
import logging
import sys
from asyncio import create_subprocess_exec
from asyncio.subprocess import PIPE
from typing import Dict

from istream_player.config.config import PlayerConfig
from istream_player.core.renderer import Renderer
from istream_player.core.downloader import (DownloadEventListener,
                                            DownloadManager)
from istream_player.core.module import Module, ModuleOption
from istream_player.core.mpd_provider import MPDProvider
from istream_player.core.player import Player, PlayerEventListener
from istream_player.models import Segment, State


from OpenGL.GLUT import *
from istream_player.modules.decoder import DecoderNvCodec
import numpy as np
import time


@ModuleOption("headless", requires=[Player], default=True)
class HeadlessRenderer(Module, Renderer, PlayerEventListener):
    log = logging.getLogger("Headless Renderer")

    def __init__(self) -> None:
        super().__init__()
        self.task_start = None
        self.task_total = None
        self.current_segment_index = 0

    async def setup(self, config: PlayerConfig, player: Player):
        player.add_listener(self)
        self.player = player

    async def on_segment_playback_start(self, segments: Dict[int, Segment]):
        for as_idx in segments:
            segment = segments[as_idx]
            # 从 URL 解析段索引（与 opencv renderer 一致）
            try:
                self.current_segment_index = int(segment.url.split('segment_')[1].split('.')[0])
            except (IndexError, ValueError):
                pass
            self.task_start = time.perf_counter()
            self.task_total = segment.duration
            await self._sleep_playback(segment.duration)

    async def _sleep_playback(self, duration: float):
        elapsed = 0.0
        step = 0.05
        while elapsed < duration:
            player = getattr(self, "player", None)
            if player is not None and player.state == State.END:
                return
            if player is not None and (player.state == State.PAUSED or getattr(player, "_paused", False)):
                await asyncio.sleep(step)
                self.task_start = time.perf_counter() - elapsed
                continue
            await asyncio.sleep(step)
            elapsed += step

    def remain_task(self):
        if self.task_start is None:
            return 0
        return max(self.task_total - (time.perf_counter() - self.task_start), 0)

    def render_fps(self) -> float:
        # Headless sleeps by duration; no real frame presentation clock.
        return 0.0

    def get_current_render_segment_index(self):
        return self.current_segment_index

