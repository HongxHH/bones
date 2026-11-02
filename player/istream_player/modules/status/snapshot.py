"""Playback/runtime status snapshot builder.

Owns mutable playback fields, inventory scanning, stall + host metrics.
StatusExporter remains the event/HTTP shell and calls ``build()``.
"""

from __future__ import annotations

import time
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from istream_player.config.config import PlayerConfig
from istream_player.core.bw_meter import BandwidthMeter
from istream_player.core.enhancer import Enhancer
from istream_player.core.mpd_provider import MPDProvider
from istream_player.core.player import Player
from istream_player.core.renderer import Renderer
from istream_player.core.scheduler import Scheduler
from istream_player.models import State
from istream_player.models.mpd_objects import Segment
from istream_player.modules.download_buffer import DownloadBufferImpl
from istream_player.modules.enhance_buffer import EnhanceBufferImpl
from istream_player.modules.status.runtime_metrics import RuntimeMetrics
from istream_player.modules.status.stall_tracker import StallTracker

_INVENTORY_CAP = 32
_MODEL_LABELS = {
    "imdn_always": "IMDN",
    "imdn": "IMDN",
    "multitask": "Multitask",
    "pacnet": "PACNet",
    "off": "off",
}


def jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.name
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class StatusSnapshot:
    def __init__(self, *, runtime: Optional[RuntimeMetrics] = None) -> None:
        self.player: Optional[Player] = None
        self.scheduler: Optional[Scheduler] = None
        self.download_buffer: Optional[DownloadBufferImpl] = None
        self.enhance_buffer: Optional[EnhanceBufferImpl] = None
        self.enhancer: Optional[Enhancer] = None
        self.bw_meter: Optional[BandwidthMeter] = None
        self.mpd_provider: Optional[MPDProvider] = None
        self.renderer: Optional[Renderer] = None
        self.config: Optional[PlayerConfig] = None

        self.segment_index: Optional[int] = None
        self.download_action: Dict[int, int] = {}
        self.enhance_action: Dict[int, int] = {}
        self.playing_enhanced: Optional[bool] = None
        self.playing_enhance_level: Optional[int] = None
        self.playing_width: Optional[int] = None
        self.playing_height: Optional[int] = None
        self.last_enhanced_segment: Optional[int] = None
        self.enhance_busy: bool = False
        self.enhancing_segments: Dict[int, Segment] = {}

        self.stalls = StallTracker()
        # Start disabled until bind() applies collect_system_metrics.
        self.runtime = (
            runtime
            if runtime is not None
            else RuntimeMetrics(interval_s=1.0, enabled=False)
        )

    def bind(
        self,
        *,
        config: PlayerConfig,
        player: Player,
        scheduler: Scheduler,
        download_buffer: DownloadBufferImpl,
        enhance_buffer: EnhanceBufferImpl,
        enhancer: Enhancer,
        bw_meter: BandwidthMeter,
        mpd_provider: MPDProvider,
        renderer: Renderer,
    ) -> None:
        self.config = config
        self.player = player
        self.scheduler = scheduler
        self.download_buffer = download_buffer
        self.enhance_buffer = enhance_buffer
        self.enhancer = enhancer
        self.bw_meter = bw_meter
        self.mpd_provider = mpd_provider
        self.renderer = renderer
        self.runtime.set_enabled(bool(getattr(config, "collect_system_metrics", True)))

    def note_state(self, new_state: State) -> None:
        self.stalls.note_state(new_state)
        if new_state == State.END:
            self.stalls.close_open()

    def note_playback_start(self, segments: Dict[int, Segment]) -> None:
        if not segments:
            return
        first = next(iter(segments.values()))
        try:
            self.segment_index = int(Path(str(first.url)).stem.split("_")[-1])
        except Exception:
            self.segment_index = getattr(self.player, "_current_index", None)
        self.download_action = {k: getattr(v, "download_action", None) for k, v in segments.items()}
        self.enhance_action = {k: getattr(v, "enhance_action", None) for k, v in segments.items()}
        has_frames = getattr(first, "decode_data", None) is not None
        enhanced = bool(getattr(first, "is_enhance", False) or has_frames)
        level = getattr(first, "enhance_action", None)
        self.playing_enhanced = enhanced
        self.playing_enhance_level = level if enhanced else 0
        self._remember_picture(first)

    def note_enhancement_start(self, segments: Dict[int, Segment]) -> None:
        self.enhancing_segments = dict(segments) if segments else {}

    def _playing_segment(self) -> Optional[int]:
        """Segment the player is currently on (on-screen / being rendered).

        Prefer ``player._current_index``: it is set *before* listener fan-out, so
        it stays correct for the whole segment even while OpenCV is inside
        ``on_segment_playback_start``. Renderer index is only a fallback — if
        StatusExporter runs before Renderer, the renderer's index is still the
        previous segment until that handler starts.
        """
        if self.player is not None:
            cur = getattr(self.player, "_current_index", None)
            if cur is not None:
                try:
                    return int(cur)
                except (TypeError, ValueError):
                    pass
        renderer = self.renderer
        if renderer is not None:
            getter = getattr(renderer, "get_current_render_segment_index", None)
            if callable(getter):
                live = getter()
                if live is not None:
                    try:
                        return int(live)
                    except (TypeError, ValueError):
                        pass
        return self.segment_index

    def _playing_picture(self) -> tuple[Optional[bool], Optional[int], Optional[int], Optional[int]]:
        """Enhanced flag / level / WxH for the segment currently being rendered."""
        enhanced = self.playing_enhanced
        level = self.playing_enhance_level
        width = self.playing_width
        height = self.playing_height
        renderer = self.renderer
        if renderer is not None and bool(getattr(renderer, "is_playing", False)):
            has_frames = bool(getattr(renderer, "_frame_enhanced", None))
            is_enh = bool(getattr(renderer, "is_enhance", False) or has_frames)
            enhanced = is_enh
            raw_level = getattr(renderer, "current_segment_enhance_action", None)
            level = raw_level if is_enh else 0
            size = getattr(renderer, "display_size", None)
            if callable(size):
                try:
                    wh = size()
                    if wh and len(wh) >= 2:
                        width, height = int(wh[0]), int(wh[1])
                except Exception:
                    pass
        return enhanced, level, width, height

    def build(self) -> dict[str, Any]:
        t0 = time.perf_counter()
        download_level = 0.0
        enhance_level = 0.0
        enhance_queue = 0.0
        try:
            if self.download_buffer is not None:
                download_level = float(self.download_buffer.buffer_level(continuous=True))
        except Exception:
            pass
        try:
            if self.enhance_buffer is not None:
                enhance_queue = float(self.enhance_buffer.buffer_level(continuous=False))
                enhance_level = float(self.enhance_buffer.buffer_level(continuous=True))
        except Exception:
            pass
        actions: dict[str, Any] = {}
        if self.scheduler is not None:
            actions["download"] = getattr(self.scheduler, "_current_download_actions", None) or self.download_action
            actions["enhance"] = getattr(self.scheduler, "_current_enhance_actions", None) or self.enhance_action
        cfg: dict[str, Any] = {}
        if self.config is not None:
            cfg = {
                k: jsonable(v)
                for k, v in self.config.__dict__.items()
                if k.startswith("mod_") or k in ("input", "trace", "video_dataset", "buffer_duration")
            }
        state = self.player.state if self.player is not None else State.IDLE
        downloading = getattr(self.scheduler, "_index", None) if self.scheduler is not None else None
        enhancing = self._refresh_enhancing()
        ready_inv = self._enhanced_ready_inventory()
        pending_inv = self._enhance_pending_inventory(enhancing)
        enhanced_ready = sum(float(row.get("duration_s") or 0.0) for row in ready_inv)
        playing_seg = self._playing_segment()
        playing_enhanced, playing_level, playing_w, playing_h = self._playing_picture()
        playing_download = None
        if self.download_action:
            try:
                playing_download = next(iter(self.download_action.values()))
            except StopIteration:
                playing_download = None
        state_name = state.name if isinstance(state, Enum) else str(state)
        stall_active = state == State.BUFFERING or state_name == "BUFFERING"
        runtime: dict[str, Any] = {}
        t_rt = time.perf_counter()
        try:
            runtime = self.runtime.sample()
        except Exception:
            runtime = {}
        runtime_ms = (time.perf_counter() - t_rt) * 1000.0
        payload = {
            "state": state_name,
            "paused": bool(getattr(self.player, "_paused", state == State.PAUSED)),
            "position": getattr(self.player, "_position", 0.0) if self.player else 0.0,
            "segment_index": playing_seg,
            "playing_segment": playing_seg,
            "enhancing_segment": enhancing,
            "enhance_busy": self.enhance_busy,
            "downloading_segment": downloading,
            "playing_width": playing_w,
            "playing_height": playing_h,
            "playing_download_action": playing_download,
            "download_buffer_s": download_level,
            "enhance_pending_s": enhance_level,
            "enhance_queue_s": enhance_queue,
            "enhance_buffer_s": enhance_level,
            "enhanced_ready_s": enhanced_ready,
            "enhanced_ready_inventory": ready_inv,
            "enhance_pending_inventory": pending_inv,
            "render_fps": self._render_fps(),
            "stall_active": stall_active,
            "stall_s_total": round(self.stalls.total_s(), 3),
            "frame_archive_dropped": self._archive_dropped(),
            "cpu_pct": runtime.get("cpu_pct"),
            "cpu_pct_raw": runtime.get("cpu_pct_raw"),
            "cpu_system_pct": runtime.get("cpu_system_pct"),
            "rss_mb": runtime.get("rss_mb"),
            "mem_pct": runtime.get("mem_pct"),
            "mem_total_mb": runtime.get("mem_total_mb"),
            "gpu_util_pct": runtime.get("gpu_util_pct"),
            "gpu_mem_used_mb": runtime.get("gpu_mem_used_mb"),
            "gpu_mem_total_mb": runtime.get("gpu_mem_total_mb"),
            "gpu_mem_pct": runtime.get("gpu_mem_pct"),
            "gpu_proc_mem_mb": runtime.get("gpu_proc_mem_mb"),
            "gpu_proc_mem_pct": runtime.get("gpu_proc_mem_pct"),
            "gpu_proc_mem_src": runtime.get("gpu_proc_mem_src"),
            "bandwidth_bps": getattr(self.bw_meter, "bandwidth", 0) if self.bw_meter else 0,
            "download_action": jsonable(actions.get("download") or {}),
            "enhance_action": jsonable(actions.get("enhance") or {}),
            "playing_enhanced": playing_enhanced,
            "playing_enhance_level": playing_level,
            "compare_view": bool(getattr(self.config, "compare_view", False)) if self.config is not None else False,
            "facts": self._experiment_facts(),
            "config": cfg,
        }
        total_ms = (time.perf_counter() - t0) * 1000.0
        if total_ms >= 8.0 or runtime_ms >= 1.0:
            note = getattr(self.renderer, "note_loop_block", None) if self.renderer is not None else None
            if callable(note):
                note({
                    "kind": "loop",
                    "part": "snapshot",
                    "total_ms": round(total_ms, 2),
                    "runtime_ms": round(runtime_ms, 2),
                    "buf_s": round(float(download_level), 3),
                    "seg": playing_seg,
                })
        return payload

    def _experiment_facts(self) -> Optional[dict]:
        enhancer = self.enhancer
        report = getattr(enhancer, "experiment_facts", None) if enhancer is not None else None
        if not callable(report):
            return None
        try:
            facts = report()
        except Exception:
            return None
        return facts if isinstance(facts, dict) else None

    def _refresh_enhancing(self) -> Optional[int]:
        """Return the in-flight enhance index, or None when idle.

        ``last_enhanced_segment`` is kept for debugging, but must not leak into
        ``enhancing_segment`` after work finishes (sticky-HUD bug).
        """
        current = getattr(self.enhancer, "_current_enhancing_index", None) if self.enhancer is not None else None
        self.enhance_busy = current is not None
        if current is not None:
            self.last_enhanced_segment = int(current)
            return self.last_enhanced_segment
        self.enhancing_segments = {}
        return None

    def _archive_dropped(self) -> int:
        renderer = self.renderer
        archive = getattr(renderer, "frame_archive", None) if renderer is not None else None
        if archive is None:
            return 0
        try:
            return int(getattr(archive, "dropped", 0) or 0)
        except (TypeError, ValueError):
            return 0

    def _render_fps(self) -> Optional[float]:
        renderer = self.renderer
        if renderer is None:
            return None
        try:
            if hasattr(renderer, "render_fps"):
                return round(float(renderer.render_fps()), 2)
            fps = getattr(renderer, "current_fps", None)
            if isinstance(fps, (int, float)):
                return round(float(fps), 2)
        except Exception:
            return None
        return None

    def _model_label(self) -> str:
        raw = ""
        if self.config is not None:
            raw = str(getattr(self.config, "mod_enhancer", "") or "")
        key = raw.strip().lower()
        if key in _MODEL_LABELS:
            return _MODEL_LABELS[key]
        return raw or "—"

    @staticmethod
    def _first_segment(group: Any) -> Optional[Segment]:
        if not isinstance(group, dict) or not group:
            return None
        return next(iter(group.values()))

    def _segment_row(
        self,
        index: int,
        segment: Segment,
        *,
        status: Optional[str] = None,
        remain_s: Optional[float] = None,
    ) -> dict[str, Any]:
        level = getattr(segment, "enhance_action", None)
        latency = getattr(segment, "enhance_latency", None)
        row: dict[str, Any] = {
            "index": int(index),
            "duration_s": float(getattr(segment, "duration", 0.0) or 0.0),
            "enhance_level": int(level) if level is not None else None,
            "repr_id": getattr(segment, "repr_id", None),
            "model": self._model_label(),
        }
        if latency is not None:
            try:
                row["enhance_latency_s"] = float(latency)
            except (TypeError, ValueError):
                row["enhance_latency_s"] = None
        if status is not None:
            row["status"] = status
        if remain_s is not None:
            row["remain_s"] = float(remain_s)
        return row

    def _enhanced_ready_inventory(self) -> list[dict[str, Any]]:
        buffer = self.download_buffer
        if buffer is None:
            return []
        out: list[dict[str, Any]] = []
        try:
            groups = getattr(buffer, "_segments", {}) or {}
            for index in sorted(groups.keys()):
                segment = self._first_segment(groups.get(index))
                if segment is None:
                    continue
                has_frames = getattr(segment, "decode_data", None) is not None
                if not (has_frames or bool(getattr(segment, "is_enhance", False))):
                    continue
                out.append(self._segment_row(int(index), segment, status="ready"))
                if len(out) >= _INVENTORY_CAP:
                    break
        except Exception:
            return out
        return out

    def _enhance_pending_inventory(self, enhancing: Optional[int]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        seen: set[int] = set()
        remain = None
        if self.enhancer is not None and self.enhance_busy:
            try:
                remain = float(self.enhancer.remain_task())
            except Exception:
                remain = None
        if self.enhance_busy and enhancing is not None and self.enhancing_segments:
            segment = self._first_segment(self.enhancing_segments)
            if segment is not None:
                out.append(
                    self._segment_row(
                        int(enhancing),
                        segment,
                        status="enhancing",
                        remain_s=remain,
                    )
                )
                seen.add(int(enhancing))
        buffer = self.enhance_buffer
        if buffer is None:
            return out
        try:
            groups = getattr(buffer, "_segments", {}) or {}
            for index in sorted(groups.keys()):
                idx = int(index)
                if idx in seen:
                    continue
                segment = self._first_segment(groups.get(index))
                if segment is None:
                    continue
                out.append(self._segment_row(idx, segment, status="queued"))
                seen.add(idx)
                if len(out) >= _INVENTORY_CAP:
                    break
        except Exception:
            return out
        return out

    def _remember_picture(self, segment: Segment) -> None:
        self.playing_width = None
        self.playing_height = None
        if self.scheduler is None:
            return
        try:
            adapt = self.scheduler.adaptation_sets[segment.as_id]
            rep = adapt.representations[segment.repr_id]
            self.playing_width = int(rep.width)
            self.playing_height = int(rep.height)
        except Exception:
            return
