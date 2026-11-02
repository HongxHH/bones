# -*- coding: utf-8 -*-
"""Preference-aware joint download×enhance picker (multi_enhance joint v1).

Docs: workspace/multi_enhance/docs/DESIGN_JOINT_V1.md
      workspace/multi_enhance/docs/PARAMS_JOINT_V1.md

Enumerates legal (download, tool) pairs and maximizes:
  score = Q + taste_bonus - risk_weight * tightness * risk_scale
Feasibility uses planned completion time (table + floors), not fast-complete.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    tool_name,
)

# Defaults mirror priors/joint_v1_params.json (used if file missing).
_DEFAULT_PARAMS: Dict[str, Any] = {
    "defaults": {"user_courage": "balanced", "user_taste": "balanced"},
    "constants": {
        "enhance_safety": 0.9,
        "enhance_max_seg": 1.5,
        "bw_floor_bps": 50000.0,
        "risk_scale": 10.0,
        "first_segment_force_none": True,
    },
    "courage": {
        "conservative": {"danger_seg": 2.0, "relax_seg": 0.0, "risk_weight": 1.0},
        "balanced": {"danger_seg": 1.0, "relax_seg": 0.25, "risk_weight": 0.5},
        "aggressive": {"danger_seg": 0.5, "relax_seg": 0.5, "risk_weight": 0.2},
    },
    "taste_bonus": {
        "balanced": {"sr": 0.0, "restore": 0.0, "none": 0.0},
        "sharp": {"sr": 3.0, "restore": 0.0, "none": 0.0},
        "clean": {"sr": 0.0, "restore": 4.0, "none": 0.0},
    },
    "tool_kind": {"0": "none", "1": "sr", "2": "sr", "3": "sr", "4": "sr", "5": "restore"},
    "plan_floor_s": {"0": 0.0, "1": 0.4, "2": 0.8, "3": 2.0, "4": 1.2, "5": 0.3},
}


def default_params_path() -> Path:
    # player/istream_player/modules/nes/pref_joint.py → repo root = parents[4]
    repo = Path(__file__).resolve().parents[4]
    return repo / "workspace" / "multi_enhance" / "priors" / "joint_v1_params.json"


def load_joint_params(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or default_params_path()
    if p.is_file():
        data = json.loads(p.read_text(encoding="utf-8"))
        data.pop("_metadata", None)
        return data
    return json.loads(json.dumps(_DEFAULT_PARAMS))


def plan_enhance_s(
    t_table_s: float,
    tool: int,
    params: Dict[str, Any],
) -> float:
    const = params["constants"]
    safety = float(const["enhance_safety"])
    floors = params["plan_floor_s"]
    floor = float(floors.get(str(tool), floors.get(tool, 0.0)))
    if not np.isfinite(t_table_s) or t_table_s >= 1e5:
        table_term = floor
    else:
        table_term = float(t_table_s) / max(safety, 1e-6)
    return max(table_term, floor)


def taste_bonus_for(tool: int, taste: str, params: Dict[str, Any]) -> float:
    kind = params["tool_kind"].get(str(tool), "none")
    table = params["taste_bonus"].get(taste) or params["taste_bonus"]["balanced"]
    return float(table.get(kind, 0.0))


def score_pair(
    *,
    q: float,
    download_s: float,
    enhance_s: float,
    buff_down_s: float,
    buff_enh_s: float,
    t_seg_s: float,
    courage: str,
    taste: str,
    tool: int,
    params: Dict[str, Any],
) -> Tuple[bool, float, Dict[str, float]]:
    """Return (feasible, score, debug)."""
    const = params["constants"]
    c = params["courage"].get(courage) or params["courage"]["balanced"]
    danger_s = float(c["danger_seg"]) * t_seg_s
    relax_s = float(c["relax_seg"]) * t_seg_s
    risk_weight = float(c["risk_weight"])
    risk_scale = float(const["risk_scale"])
    enhance_max = float(const["enhance_max_seg"]) * t_seg_s

    budget_s = (buff_down_s - buff_enh_s) - danger_s + relax_s
    dbg = {
        "download_s": download_s,
        "enhance_s": enhance_s,
        "budget_s": budget_s,
        "danger_s": danger_s,
        "relax_s": relax_s,
    }

    if enhance_s > enhance_max + 1e-9:
        return False, float("-inf"), {**dbg, "reject": "enhance_max"}
    if download_s + enhance_s > budget_s + 1e-9:
        return False, float("-inf"), {**dbg, "reject": "budget"}

    tightness = (download_s + enhance_s) / max(budget_s, 1e-3)
    bonus = taste_bonus_for(tool, taste, params)
    risk_penalty = risk_weight * tightness * risk_scale
    score = float(q) + bonus - risk_penalty
    dbg.update(
        {
            "tightness": tightness,
            "taste_bonus": bonus,
            "risk_penalty": risk_penalty,
            "score": score,
        }
    )
    return True, score, dbg


def pick_joint(
    *,
    quality: np.ndarray,
    latency: np.ndarray,
    bitrates_bps: np.ndarray,
    bw_bps: float,
    buff_down_s: float,
    buff_enh_s: float,
    t_seg_s: float,
    courage: str,
    taste: str,
    params: Dict[str, Any],
) -> Tuple[int, int, Dict[str, Any]]:
    """Enumerate legal pairs; return (download_action, tool_id, debug)."""
    const = params["constants"]
    bw = max(float(bw_bps), float(const["bw_floor_bps"]))
    n_down = int(min(len(bitrates_bps), quality.shape[0], latency.shape[0]))
    n_tools = int(min(quality.shape[1], latency.shape[1]))

    best_sc: Optional[float] = None
    best_d, best_t = 0, TOOL_NONE
    best_dbg: Dict[str, Any] = {}
    considered = 0
    rejected = 0

    for d in range(n_down):
        legal = LEGAL_BY_DOWNLOAD.get(d, {TOOL_NONE})
        download_s = float(bitrates_bps[d]) * t_seg_s / bw
        for t in range(n_tools):
            if t not in legal:
                continue
            q = float(quality[d, t])
            if not np.isfinite(q) or q <= -1e8:
                continue
            considered += 1
            enh_s = plan_enhance_s(float(latency[d, t]), t, params)
            ok, sc, dbg = score_pair(
                q=q,
                download_s=download_s,
                enhance_s=enh_s,
                buff_down_s=buff_down_s,
                buff_enh_s=buff_enh_s,
                t_seg_s=t_seg_s,
                courage=courage,
                taste=taste,
                tool=t,
                params=params,
            )
            if not ok:
                rejected += 1
                continue
            # Higher score wins; tie → lower download, then lighter tool
            take = False
            if best_sc is None:
                take = True
            elif sc > best_sc + 1e-12:
                take = True
            elif abs(sc - best_sc) <= 1e-12 and (d < best_d or (d == best_d and t < best_t)):
                take = True
            if take:
                best_sc = sc
                best_d, best_t = d, t
                best_dbg = dbg

    if best_sc is None:
        return (
            0,
            TOOL_NONE,
            {"fallback": True, "considered": considered, "rejected": rejected},
        )
    return (
        int(best_d),
        int(best_t),
        {
            "fallback": False,
            "considered": considered,
            "rejected": rejected,
            "score": best_sc,
            **best_dbg,
        },
    )


@ModuleOption(
    "pref_joint",
    requires=[DownloadBufferImpl, EnhanceBufferImpl, Enhancer, Scheduler, BandwidthMeter],
    default=False,
)
class PrefJointController(Module, NESController):
    log = logging.getLogger("PrefJointController")

    def __init__(self):
        super().__init__()
        self.seg_time_s: Optional[float] = None
        self.first_download = True
        self.last_pick_debug: Optional[dict] = None
        self.params = load_joint_params()
        self.courage = "balanced"
        self.taste = "balanced"

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
        self.download_buffer = download_buffer
        self.enhance_buffer = enhance_buffer
        self.enhancer = enhancer
        self.scheduler = scheduler
        self.bw_meter = bw_meter

        params_path = getattr(config, "joint_params_path", None)
        if params_path:
            self.params = load_joint_params(Path(params_path))
        else:
            self.params = load_joint_params()

        defaults = self.params.get("defaults", {})
        self.courage = str(
            getattr(config, "user_courage", None) or defaults.get("user_courage") or "balanced"
        )
        self.taste = str(
            getattr(config, "user_taste", None) or defaults.get("user_taste") or "balanced"
        )
        if self.courage not in self.params["courage"]:
            self.log.warning(f"unknown user_courage={self.courage}, fallback balanced")
            self.courage = "balanced"
        if self.taste not in self.params["taste_bonus"]:
            self.log.warning(f"unknown user_taste={self.taste}, fallback balanced")
            self.taste = "balanced"

        self.log.info(
            f"pref_joint courage={self.courage} taste={self.taste} "
            f"params={default_params_path() if not params_path else params_path}"
        )

    def update_selection(
        self, adaptation_sets: Dict[int, AdaptationSet], index: int
    ) -> Tuple[Dict[int, int], Dict[int, int]]:
        if self.seg_time_s is None:
            self.seg_time_s = float(
                self.scheduler.mpd_provider.mpd.max_segment_duration
            )

        download_actions: Dict[int, int] = {}
        enhance_actions: Dict[int, int] = {}
        force_none = bool(
            self.params["constants"].get("first_segment_force_none", True)
        )

        for adaptation_set in adaptation_sets.values():
            if self.first_download and force_none:
                self.first_download = False
                d, e = 0, TOOL_NONE
                self.last_pick_debug = {"cold_start": True}
            else:
                self.first_download = False
                d, e = self._decide(adaptation_set, index)
            download_actions[adaptation_set.id] = d
            enhance_actions[adaptation_set.id] = e
            self.log.info(
                f"pref_joint seg={index} down={d} tool={e}({tool_name(e)}) "
                f"courage={self.courage} taste={self.taste}"
            )
        return download_actions, enhance_actions

    def _decide(self, adaptation_set: AdaptationSet, index: int) -> Tuple[int, int]:
        bitrates = np.array(
            [rep.bandwidth for rep in adaptation_set.representations.values()],
            dtype=float,
        )
        quality = self.enhancer.get_quality_table()
        latency = self.enhancer.get_latency_table(segment_index=index)
        if quality is None or latency is None:
            return 0, TOOL_NONE

        quality = np.asarray(quality, dtype=float)
        latency = np.asarray(latency, dtype=float)
        buff_down_s = float(self.download_buffer.buffer_level(continuous=True))
        buff_enh_s = float(self.enhance_buffer.buffer_level(continuous=True))
        try:
            bw = float(self.bw_meter.bandwidth)
        except Exception:
            bw = float(self.params["constants"]["bw_floor_bps"])
        if not np.isfinite(bw) or bw <= 0:
            bw = float(self.params["constants"]["bw_floor_bps"])

        d, t, dbg = pick_joint(
            quality=quality,
            latency=latency,
            bitrates_bps=bitrates,
            bw_bps=bw,
            buff_down_s=buff_down_s,
            buff_enh_s=buff_enh_s,
            t_seg_s=float(self.seg_time_s or 4.0),
            courage=self.courage,
            taste=self.taste,
            params=self.params,
        )
        self.last_pick_debug = {
            "buff_down_s": buff_down_s,
            "buff_enh_s": buff_enh_s,
            "bw_bps": bw,
            "courage": self.courage,
            "taste": self.taste,
            "picked_down": d,
            "picked_tool": t,
            **dbg,
        }
        return d, t
