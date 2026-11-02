# -*- coding: utf-8 -*-
"""Segment-ahead tool picker: buffer ABR download + feasible argmax-Q enhance.

DESIGN / RESEARCH: workspace/multi_enhance/ (Ours-v0).
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np
from istream_player.config.config import PlayerConfig
from istream_player.core.bw_meter import BandwidthMeter
from istream_player.core.enhancer import Enhancer
from istream_player.core.module import Module, ModuleOption
from istream_player.core.nes import NESController
from istream_player.core.scheduler import Scheduler
from istream_player.models import AdaptationSet
from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.modules.enhancer.tool_ids import (
    LEGAL_BY_DOWNLOAD,
    TOOL_NONE,
    TOOL_RESTORE_SHARP,
    TOOL_SR_X2_MED,
    tool_name,
)


@ModuleOption(
    "tool_greedy",
    requires=[DownloadBufferImpl, EnhanceBufferImpl, Enhancer, Scheduler, BandwidthMeter],
    default=False,
)
class ToolGreedyController(Module, NESController):
    log = logging.getLogger("ToolGreedyController")

    def __init__(self):
        super().__init__()
        self.reservoir = 0.375
        self.upper_reservoir = 0.9
        self.enhance_safety = 0.9
        self.seg_time = None
        self.first_download = True

    async def setup(
        self,
        config: PlayerConfig,
        download_buffer: DownloadBufferImpl,
        enhance_buffer: EnhanceBufferImpl,
        enhancer: Enhancer,
        scheduler: Scheduler,
        bw_meter: BandwidthMeter,
        **kwargs,
    ):
        self.config = config
        self.buffer_size = config.buffer_duration * 1000
        self.download_buffer = download_buffer
        self.enhance_buffer = enhance_buffer
        self.enhancer = enhancer
        self.scheduler = scheduler
        self.bw_meter = bw_meter
        # D5 ablation / slack tuning
        self.tool_ignore_slack = bool(getattr(config, "tool_ignore_slack", False))
        self.tool_mask_hd = bool(getattr(config, "tool_mask_hd", False))
        self.tool_slack_kappa = float(getattr(config, "tool_slack_kappa", 1.0))
        self.last_pick_debug: dict | None = None
        if self.tool_ignore_slack or self.tool_mask_hd or self.tool_slack_kappa != 1.0:
            self.log.info(
                f"tool_greedy ablation ignore_slack={self.tool_ignore_slack} "
                f"mask_hd={self.tool_mask_hd} slack_kappa={self.tool_slack_kappa}"
            )

    def update_selection(
        self, adaptation_sets: Dict[int, AdaptationSet], index: int
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        if self.seg_time is None:
            self.seg_time = self.scheduler.mpd_provider.mpd.max_segment_duration * 1000

        download_actions = {}
        enhance_actions = {}
        for adaptation_set in adaptation_sets.values():
            if self.first_download:
                self.first_download = False
                d, e = 0, TOOL_NONE
            else:
                d, e = self._decide(adaptation_set, index)
            download_actions[adaptation_set.id] = d
            enhance_actions[adaptation_set.id] = e
            self.log.info(
                f"tool_greedy seg={index} down={d} tool={e}({tool_name(e)})"
            )
        return download_actions, enhance_actions

    def _decide(self, adaptation_set: AdaptationSet, index: int):
        bitrate = [
            representation.bandwidth
            for representation in adaptation_set.representations.values()
        ]
        self.bitrate = np.array(bitrate)
        self.buff_down = self.download_buffer.buffer_level(continuous=True) * 1000
        self.buff_enh = self.enhance_buffer.buffer_level(continuous=True) * 1000

        self.threshold = self.bitrate - self.bitrate[0]
        if self.threshold[-1] <= 0:
            self.threshold = np.linspace(0, 1, len(self.bitrate)) * (
                self.upper_reservoir - self.reservoir
            )
        else:
            self.threshold = (self.threshold / self.threshold[-1]) * (
                self.upper_reservoir - self.reservoir
            )
        self.threshold = (self.threshold + self.reservoir) * self.buffer_size

        action_down = self._buffer_control()
        action_enh = self._pick_tool(action_down, index)
        return action_down, action_enh

    def _buffer_control(self) -> int:
        buffer_level = self.buff_down
        num_action_down = self.bitrate.shape[0]
        action_down = 0
        if buffer_level > self.threshold[-1]:
            action_down = num_action_down - 1
        else:
            for action_download in range(num_action_down):
                if buffer_level < self.threshold[action_download]:
                    action_down = action_download
                    break
        return action_down

    def _pick_tool(self, action_down: int, index: int) -> int:
        enhance_time = self.enhancer.get_latency_table(segment_index=index)
        quality = self.enhancer.get_quality_table()
        if enhance_time is None or quality is None:
            return TOOL_NONE

        # get_latency_table returns segment-seconds; convert like buffer_greedy
        enhance_time = np.asarray(enhance_time, dtype=float) * 1000.0 / self.enhance_safety
        quality = np.asarray(quality, dtype=float).copy()

        r = min(action_down, enhance_time.shape[0] - 1, quality.shape[0] - 1)
        n_tools = min(enhance_time.shape[1], quality.shape[1])
        legal = LEGAL_BY_DOWNLOAD.get(action_down, {TOOL_NONE})

        # D5 no_hd: hide HD tools from argmax via −inf Q
        if getattr(self, "tool_mask_hd", False):
            for rr in range(quality.shape[0]):
                if quality.shape[1] > TOOL_SR_X2_MED:
                    quality[rr, TOOL_SR_X2_MED] = -np.inf
                if quality.shape[1] > TOOL_RESTORE_SHARP:
                    quality[rr, TOOL_RESTORE_SHARP] = -np.inf

        # slack (ms): buff_down - buff_enh - kappa*seg_time
        # kappa=1 → 现行（需约 1 段 download−enhance 裕量）；kappa=0 → 只比相对缓冲差与 T
        kappa = float(getattr(self, "tool_slack_kappa", 1.0))
        slack_ms = self.buff_down - self.buff_enh - kappa * self.seg_time
        ignore_slack = getattr(self, "tool_ignore_slack", False)

        best_q = float("-inf")
        best_t = TOOL_NONE
        rejected: list[str] = []
        # none quality = base at this rung if present, else 0
        none_q = float(quality[r, TOOL_NONE]) if n_tools > 0 else 0.0
        if np.isfinite(none_q):
            best_q = none_q

        for t in range(n_tools):
            if t not in legal:
                continue
            q = float(quality[r, t])
            if not np.isfinite(q) or q <= -1e8:
                continue
            t_ms = float(enhance_time[r, t])
            if t != TOOL_NONE and not ignore_slack:
                if t_ms >= self.seg_time:
                    rejected.append(f"{t}:t>=seg")
                    continue
                if slack_ms < t_ms:
                    rejected.append(f"{t}:slack<{t_ms:.0f}")
                    continue
            if q > best_q:
                best_q = q
                best_t = t

        self.last_pick_debug = {
            "buff_down_ms": float(self.buff_down),
            "buff_enh_ms": float(self.buff_enh),
            "seg_ms": float(self.seg_time),
            "kappa": kappa,
            "slack_ms": float(slack_ms),
            "action_down": int(action_down),
            "picked": int(best_t),
            "rejected": rejected[:12],
        }
        return int(best_t)
