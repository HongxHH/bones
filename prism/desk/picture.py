"""Right-hand picture pane. Full-resolution frames, no second player window."""

from __future__ import annotations

import json
import threading
import urllib.parse
import urllib.request

from PySide6.QtCore import QPoint, QRect, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QPainter
from PySide6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

_STAT_KEYS = (
    "play",
    "enhance_seg",
    "lead",
    "download_seg",
    "picture",
    "rung",
    "fps",
    "stall",
    "down",
    "ready",
    "pending",
    "bw",
)
_STATE_NAME = {
    "READY": "播放",
    "BUFFERING": "缓冲",
    "PAUSED": "暂停",
    "END": "结束",
    "IDLE": "等待",
}

from prism.frame_sink import BASE_NAME, NAME, FrameSink


def _levels(obj, out: list[str]) -> None:
    if obj is None or isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        out.append(str(int(obj)))
        return
    if isinstance(obj, str):
        if obj:
            out.append(obj)
        return
    if isinstance(obj, dict):
        for value in obj.values():
            _levels(value, out)
    elif isinstance(obj, list):
        for value in obj:
            _levels(value, out)


def _enhance_text(action) -> str:
    found: list[str] = []
    _levels(action, found)
    if not found:
        return "增强 —"
    unique = list(dict.fromkeys(found))
    return "增强 " + ",".join(unique[:4])


def _picture_enhance(status: dict) -> str:
    """Only the segment on screen. Never the scheduler's next enhance_action."""
    if status.get("playing_enhanced") is True:
        level = status.get("playing_enhance_level")
        if level in (None, "", 0, "0"):
            return "已增强"
        return "增强 " + str(level)
    if status.get("playing_enhanced") is False:
        return "未增强"
    return "—"


def _blank_facts(mode: str = "等待") -> dict:
    facts = {key: "—" for key in _STAT_KEYS}
    facts["mode"] = mode
    facts["tones"] = {}
    return facts


def _idx(value) -> str:
    if value is None or value == "":
        return "—"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return str(value)


def _seconds(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.1f}s"
    except (TypeError, ValueError):
        return "—"


def _mbps(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value) / 1e6:.1f}"
    except (TypeError, ValueError):
        return "—"


def _fps(value) -> str:
    if value is None:
        return "—"
    try:
        return f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "—"


def _tone_fps(value, *, target: float = 24.0) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return ""
    if amount <= 0:
        return "warn"
    if amount + 0.5 < target:
        return "warn"
    return "ok"


def _lead(play, enhancing) -> tuple[str, str]:
    if play is None or enhancing is None:
        return "—", ""
    try:
        delta = int(enhancing) - int(play)
    except (TypeError, ValueError):
        return "—", ""
    if delta > 0:
        return f"+{delta}", "ok"
    if delta < 0:
        return str(delta), "warn"
    return "0", ""


def _rung(width, height, fallback=None) -> str:
    try:
        if width and height:
            return f"{int(width)}×{int(height)}"
    except (TypeError, ValueError):
        pass
    if fallback is None or fallback == "":
        return "—"
    return str(fallback)


def _tone_seconds(value, *, good: float = 4.0, low: float = 0.5) -> str:
    try:
        amount = float(value or 0)
    except (TypeError, ValueError):
        return ""
    if amount >= good:
        return "ok"
    if amount < low:
        return "warn"
    return ""


def _pending_s(status: dict):
    if status.get("enhance_pending_s") is not None:
        return status.get("enhance_pending_s")
    if status.get("enhance_queue_s") is not None:
        return status.get("enhance_queue_s")
    return status.get("enhance_buffer_s")


def _ready_s(status: dict):
    if status.get("enhanced_ready_s") is not None:
        return status.get("enhanced_ready_s")
    return None


def _enhance_seg_text(status: dict) -> tuple[str, str]:
    index = status.get("enhancing_segment")
    if index is None:
        return "—", ""
    if status.get("enhance_busy"):
        return _idx(index), "ok"
    return _idx(index), ""


def _facts_from_status(status: dict, paused: bool) -> dict:
    play = status.get("playing_segment")
    if play is None:
        play = status.get("segment_index")
    enhancing = status.get("enhancing_segment")
    lead, lead_tone = _lead(play, enhancing)
    picture = _picture_enhance(status) if status else "—"
    picture_tone = ""
    if picture == "未增强":
        picture_tone = "warn"
    elif picture.startswith("增强 ") or picture == "已增强":
        picture_tone = "ok"
    ready = _ready_s(status)
    pending = _pending_s(status)
    enhance_seg, enhance_seg_tone = _enhance_seg_text(status)
    state = "暂停" if paused else _STATE_NAME.get(str(status.get("state") or ""), "播放")
    if play is None and enhancing is None and state == "播放":
        state = "等待"
    facts = _blank_facts(state)
    facts.update(
        {
            "play": _idx(play),
            "enhance_seg": enhance_seg,
            "lead": lead,
            "download_seg": _idx(status.get("downloading_segment")),
            "picture": picture,
            "rung": _rung(status.get("playing_width"), status.get("playing_height")),
            "fps": _fps(status.get("render_fps")),
            "stall": _seconds(status.get("stall_s_total")),
            "down": _seconds(status.get("download_buffer_s")),
            "ready": _seconds(ready),
            "pending": _seconds(pending),
            "bw": _mbps(status.get("bandwidth_bps")),
            "tones": {
                "lead": lead_tone,
                "picture": picture_tone,
                "fps": _tone_fps(status.get("render_fps")),
                "stall": "warn" if status.get("stall_active") else "",
                "ready": _tone_seconds(ready),
                "pending": "ok" if status.get("enhance_busy") else _tone_seconds(pending, good=8.0, low=0.0),
                "enhance_seg": enhance_seg_tone,
            },
        }
    )
    return facts


def _facts_from_frame(row: dict) -> dict:
    enhanced = bool(row.get("enhanced"))
    level = row.get("enhance_level")
    if enhanced and level not in (None, "", 0, "0"):
        picture, tone = f"增强 {level}", "ok"
    elif enhanced:
        picture, tone = "已增强", "ok"
    else:
        picture, tone = "未增强", "warn"
    rung = row.get("representation")
    facts = _blank_facts("回看")
    facts.update(
        {
            "play": _idx(row.get("segment")),
            "enhance_seg": "—",
            "lead": _idx(row.get("frame")),
            "picture": picture,
            "rung": "—" if rung is None else f"档 {rung}",
            "down": _seconds(row.get("download_buffer_s")),
            "ready": _seconds(row.get("enhanced_ready_s")),
            "pending": _seconds(row.get("enhance_pending_s") if row.get("enhance_pending_s") is not None else row.get("enhance_buffer_s")),
            "bw": "—",
            "tones": {"picture": tone},
            "labels": {"lead": "帧"},
        }
    )
    return facts


class _Stat(QWidget):
    def __init__(self, name: str) -> None:
        super().__init__()
        self.setObjectName("hudStat")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(1)
        self._name = QLabel(name)
        self._name.setObjectName("hudName")
        self._value = QLabel("—")
        self._value.setObjectName("hudValue")
        layout.addWidget(self._name)
        layout.addWidget(self._value)

    def set_name(self, name: str) -> None:
        self._name.setText(name)

    def set_value(self, text: str, tone: str = "") -> None:
        self._value.setText(text)
        name = "hudValue" + tone.capitalize() if tone else "hudValue"
        if self._value.objectName() != name:
            self._value.setObjectName(name)
            self._value.style().unpolish(self._value)
            self._value.style().polish(self._value)


def _read_focus(host: str, port: int, previous: int) -> dict | None:
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/v1/system/picture", timeout=0.4) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None
    if not data.get("active"):
        return {"focus": False}
    seq = int(data.get("seq") or 0)
    payload = {
        "focus": True,
        "seq": seq,
        "line": f"{data.get('arm_id')}  ·  帧 {data.get('frame')}",
        "port": 0,
        "paused": False,
    }
    if seq == previous:
        return payload
    run_id = data.get("run_id")
    arm_id = data.get("arm_id")
    frame = int(data.get("frame") or 0)
    query = urllib.parse.quote(str(arm_id), safe="")
    base = f"http://{host}:{port}/v1/runs/{run_id}/frames/{frame}"
    try:
        with urllib.request.urlopen(f"{base}?arm_id={query}", timeout=0.6) as resp:
            row = json.loads(resp.read().decode("utf-8"))
        payload["facts"] = _facts_from_frame(row)
    except Exception:
        pass
    try:
        with urllib.request.urlopen(f"{base}/image?arm_id={query}", timeout=1.5) as resp:
            payload["jpeg"] = resp.read()
    except Exception:
        payload["jpeg"] = b""
    try:
        with urllib.request.urlopen(f"{base}/base?arm_id={query}", timeout=1.5) as resp:
            payload["base_jpeg"] = resp.read()
    except Exception:
        payload["base_jpeg"] = b""
    return payload


def _pick_live_run(runs: list) -> dict | None:
    live = [item for item in runs if item.get("status") == "running"]
    if not live:
        return None
    # Prefer the newest run so stale "running" rows from earlier sessions lose.
    live.sort(key=lambda item: str(item.get("started_at") or item.get("updated_at") or ""), reverse=True)
    for run in live:
        arms = run.get("arms") or []
        arm_id = run.get("active_arm_id") or ""
        arm = next((item for item in arms if item.get("id") == arm_id), None)
        if arm is None:
            arm = next((item for item in arms if item.get("pid")), arms[0] if arms else None)
        status_port = int((arm or {}).get("status_port") or 0)
        if not status_port or not (arm or {}).get("pid"):
            continue
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{status_port}/api/status", timeout=0.4) as resp:
                status = json.loads(resp.read().decode("utf-8"))
        except Exception:
            continue
        return {"run": run, "arm": arm, "arm_id": (arm or {}).get("id") or arm_id, "status": status, "port": status_port}
    return {"run": live[0], "arm": None, "arm_id": "", "status": None, "port": 0}


def _fetch_hud(host: str, port: int) -> dict:
    empty = {"arm": "", "facts": _blank_facts(), "paused": False, "port": 0}
    try:
        with urllib.request.urlopen(f"http://{host}:{port}/v1/runs", timeout=0.6) as resp:
            runs = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return empty
    picked = _pick_live_run(runs or [])
    if not picked:
        return empty
    status = picked.get("status") or {}
    paused = bool(status.get("paused")) if status else False
    facts = _facts_from_status(status, paused) if status else _blank_facts("播放")
    return {
        "arm": picked.get("arm_id") or "臂",
        "facts": facts,
        "paused": paused,
        "port": int(picked.get("port") or 0),
        "compare": bool(status.get("compare_view")) if status else False,
    }


class _HudWorker(QThread):
    updated = Signal(object)

    def __init__(self, host: str, port: int) -> None:
        super().__init__()
        self._host = host
        self._port = port
        self._stop = False
        self._focus_seq = -1

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        while not self._stop:
            focus = _read_focus(self._host, self._port, self._focus_seq)
            if focus and focus.get("focus"):
                self._focus_seq = int(focus.get("seq") or 0)
                self.updated.emit(focus)
            else:
                self._focus_seq = -1
                live = _fetch_hud(self._host, self._port)
                live["focus"] = False
                self.updated.emit(live)
            for _ in range(8):
                if self._stop:
                    return
                self.msleep(50)


class PicturePane(QWidget):
    def __init__(self, host: str = "127.0.0.1", port: int = 8080) -> None:
        super().__init__()
        self.setObjectName("stage")
        self._sink = FrameSink(NAME, create=True)
        self._sink.open()
        self._base_sink = FrameSink(BASE_NAME, create=True)
        self._base_sink.open()
        self._seq = 0
        self._base_seq = 0
        self._image: QImage | None = None
        self._base_image: QImage | None = None
        self._compare = False
        self._compare_touch = False
        self._compare_posting = False
        self._wipe = 0.5
        self._wipe_drag = False
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._hint = "播放后在此显示"
        self._status_port = 0
        self._paused = False
        self._hold = False
        self._pixel = False
        self._pan = QPoint(0, 0)
        self._drag: QPoint | None = None
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._poll)
        self._timer.start(33)
        self._bar = QWidget(self)
        self._bar.setObjectName("hud")
        outer = QHBoxLayout(self._bar)
        outer.setContentsMargins(8, 0, 10, 0)
        outer.setSpacing(0)
        self._mode = QLabel("等待")
        self._mode.setObjectName("hudMode")
        self._mode.setAlignment(Qt.AlignmentFlag.AlignCenter)
        outer.addWidget(self._mode)
        self._stats: dict[str, _Stat] = {}
        names = (
            ("play", "播放段"),
            ("enhance_seg", "增强段"),
            ("lead", "领先"),
            ("download_seg", "下载段"),
            ("picture", "画面"),
            ("rung", "源分辨率"),
            ("fps", "FPS"),
            ("stall", "卡顿"),
            ("down", "下载缓冲"),
            ("ready", "已增强就绪"),
            ("pending", "待增强"),
            ("bw", "带宽"),
        )
        for index, (key, name) in enumerate(names):
            # Separators before 画面 / FPS / 下载缓冲 groups.
            if index in (4, 6, 8):
                rule = QWidget()
                rule.setObjectName("hudRule")
                rule.setFixedWidth(1)
                outer.addWidget(rule)
            stat = _Stat(name)
            self._stats[key] = stat
            outer.addWidget(stat)
        outer.addStretch(1)
        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        buttons.setContentsMargins(8, 0, 0, 0)
        self._compare_btn = QPushButton("对比")
        self._compare_btn.clicked.connect(self._toggle_compare)
        self._pixel_btn = QPushButton("1:1")
        self._pixel_btn.clicked.connect(self._toggle_pixel)
        self._pause = QPushButton("暂停")
        self._pause.clicked.connect(self._toggle_pause)
        self._live = QPushButton("直播")
        self._live.clicked.connect(self._go_live)
        self._live.hide()
        self._fold = QPushButton("隐藏")
        self._fold.clicked.connect(self._hide_hud)
        buttons.addWidget(self._compare_btn)
        buttons.addWidget(self._pixel_btn)
        buttons.addWidget(self._pause)
        buttons.addWidget(self._live)
        buttons.addWidget(self._fold)
        outer.addLayout(buttons)
        self._chip = QPushButton("信息", self)
        self._chip.setObjectName("hudChip")
        self._chip.clicked.connect(self._show_hud)
        self._chip.hide()
        self._worker = _HudWorker(host, port)
        self._worker.updated.connect(self._apply_hud)
        self._worker.start()

    def close_sink(self) -> None:
        self._timer.stop()
        self._worker.stop()
        self._worker.wait(800)
        self._sink.close()
        self._base_sink.close()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._place_hud()

    def _hud_height(self) -> int:
        if not self._bar.isVisible():
            return 0
        return 72

    def _place_hud(self) -> None:
        height = self._hud_height()
        self._bar.setGeometry(0, max(0, self.height() - height), self.width(), height)
        self._chip.setGeometry(self.width() - 64, 10, 52, 26)

    def _apply_hud(self, info: object) -> None:
        if not isinstance(info, dict):
            return
        if not info.get("focus"):
            self._status_port = int(info.get("port") or 0)
        self._sync_compare(info.get("compare"))
        if info.get("focus"):
            self._hold = True
            self._pause.hide()
            self._live.show()
            if "facts" in info:
                self._show_facts(info.get("facts") or _blank_facts("回看"))
            if "jpeg" in info:
                image = QImage.fromData(info.get("jpeg") or b"", "JPEG")
                if not image.isNull():
                    self._image = image
                self._set_scrub_base(info.get("base_jpeg") or b"")
                self.update()
            return
        was_hold = self._hold
        self._hold = False
        self._live.hide()
        self._pause.show()
        self._paused = bool(info.get("paused"))
        self._show_facts(info.get("facts") or _blank_facts())
        self._pause.setText("继续" if self._paused else "暂停")
        self._pause.setEnabled(self._status_port > 0)
        if was_hold:
            self._seq = 0
            self._base_seq = 0
            self._base_image = None

    def _sync_compare(self, reported) -> None:
        if self._compare_touch:
            if isinstance(reported, bool) and reported == self._compare:
                self._compare_touch = False
            else:
                self._push_compare()
            return
        if not isinstance(reported, bool) or self._compare == reported:
            return
        self._compare = reported
        if not reported:
            self._base_image = None
        self._style_compare()

    def _style_compare(self) -> None:
        name = "hudOn" if self._compare else "hudBtn"
        if self._compare_btn.objectName() != name:
            self._compare_btn.setObjectName(name)
            self._compare_btn.style().unpolish(self._compare_btn)
            self._compare_btn.style().polish(self._compare_btn)

    def _set_scrub_base(self, payload: bytes) -> None:
        if not self._compare or not payload:
            self._base_image = None
            return
        image = QImage.fromData(payload, "JPEG")
        self._base_image = None if image.isNull() else image

    def _toggle_compare(self) -> None:
        self._compare = not self._compare
        self._compare_touch = True
        if not self._compare:
            self._base_image = None
            self._wipe_drag = False
        self._style_compare()
        self.setFocus()
        self.update()
        self._push_compare()

    def _push_compare(self) -> None:
        port = self._status_port
        if port <= 0 or self._compare_posting:
            return
        self._compare_posting = True
        enabled = self._compare

        def go() -> None:
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/api/compare",
                    data=json.dumps({"enabled": enabled}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=1.0).read()
            except Exception:
                return
            finally:
                self._compare_posting = False

        threading.Thread(target=go, daemon=True).start()

    def _show_facts(self, facts: dict) -> None:
        self._mode.setText(str(facts.get("mode") or "等待"))
        tone = str(facts.get("mode") or "")
        mode_name = "hudModeWarn" if tone in ("暂停", "缓冲", "回看") else "hudMode"
        if self._mode.objectName() != mode_name:
            self._mode.setObjectName(mode_name)
            self._mode.style().unpolish(self._mode)
            self._mode.style().polish(self._mode)
        labels = {"lead": "领先"}
        labels.update(facts.get("labels") or {})
        tones = facts.get("tones") or {}
        for key, stat in self._stats.items():
            if key in labels:
                stat.set_name(labels[key])
            stat.set_value(str(facts.get(key) or "—"), str(tones.get(key) or ""))

    def _hide_hud(self) -> None:
        self._bar.hide()
        self._chip.show()

    def _show_hud(self) -> None:
        self._chip.hide()
        self._bar.show()

    def _toggle_pause(self) -> None:
        port = self._status_port
        if port <= 0:
            return
        path = "/api/resume" if self._paused else "/api/pause"

        def go() -> None:
            try:
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}{path}",
                    data=b"{}",
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urllib.request.urlopen(req, timeout=1.0).read()
            except Exception:
                return

        threading.Thread(target=go, daemon=True).start()

    def _go_live(self) -> None:
        host = self._worker._host
        port = self._worker._port

        def go() -> None:
            try:
                req = urllib.request.Request(
                    f"http://{host}:{port}/v1/system/picture",
                    method="DELETE",
                )
                urllib.request.urlopen(req, timeout=1.0).read()
            except Exception:
                return

        threading.Thread(target=go, daemon=True).start()

    def _toggle_pixel(self) -> None:
        self._pixel = not self._pixel
        self._pixel_btn.setText("适应" if self._pixel else "1:1")
        self._center_pan()
        self.update()

    def _pixel_scale(self) -> float:
        return max(1.0, float(self.devicePixelRatioF()))

    def _center_pan(self) -> None:
        image = self._image
        if image is None:
            self._pan = QPoint(0, 0)
            return
        view_w, view_h = self._view_pixels()
        self._pan = QPoint(
            max(0, (image.width() - view_w) // 2),
            max(0, (image.height() - view_h) // 2),
        )

    def _view_pixels(self) -> tuple[int, int]:
        scale = self._pixel_scale()
        return int(self.width() * scale), int(max(0, self.height() - self._hud_height()) * scale)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self._compare and self._image is not None:
            if not self._pixel or self._near_wipe(event.position().x()):
                self._wipe_drag = True
                self._set_wipe(event.position().x())
                self.setFocus()
                return
        if self._pixel and event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position().toPoint()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._wipe_drag:
            self._set_wipe(event.position().x())
            return
        if self._drag is None:
            return
        point = event.position().toPoint()
        delta = point - self._drag
        self._drag = point
        scale = self._pixel_scale()
        self._pan = QPoint(
            self._pan.x() - int(delta.x() * scale),
            self._pan.y() - int(delta.y() * scale),
        )
        self.update()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._wipe_drag = False
        self._drag = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self._compare and event.key() in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            step = -0.02 if event.key() == Qt.Key.Key_Left else 0.02
            self._wipe = min(1.0, max(0.0, self._wipe + step))
            self.update()
            return
        super().keyPressEvent(event)

    def set_status(self, text: str) -> None:
        self._hint = text
        self.update()

    def _poll(self) -> None:
        if self._hold:
            return
        try:
            got = self._sink.read()
        except Exception:
            return
        if got is None:
            return
        seq, frame = got
        if seq == self._seq:
            return
        self._seq = seq
        h, w, _ = frame.shape
        image = QImage(frame.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._image = image.copy()
        self._take_live_base(frame.shape[:2])
        self.update()

    def _take_live_base(self, shape: tuple[int, int]) -> None:
        if not self._compare:
            self._base_image = None
            return
        try:
            got = self._base_sink.read()
        except Exception:
            self._base_image = None
            return
        if got is None:
            self._base_image = None
            return
        seq, frame = got
        if seq == self._base_seq or frame.shape[:2] != shape:
            self._base_image = None
            return
        self._base_seq = seq
        h, w, _ = frame.shape
        image = QImage(frame.data, w, h, w * 3, QImage.Format.Format_RGB888)
        self._base_image = image.copy()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#0e1a22"))
        view = QRect(0, 0, self.width(), max(0, self.height() - self._hud_height()))
        if self._image is None or self._image.isNull():
            painter.setPen(QColor(238, 243, 245, 140))
            font = QFont("Segoe UI", 16)
            painter.setFont(font)
            painter.drawText(view, Qt.AlignmentFlag.AlignCenter, "画面\n" + self._hint)
            painter.end()
            return
        if self._showing_wipe():
            self._draw_wipe(painter, self._base_image, self._image, view)
        else:
            self._draw_frame(painter, self._image, view)
        painter.end()

    def _showing_wipe(self) -> bool:
        base = self._base_image
        image = self._image
        return bool(
            self._compare
            and base is not None
            and image is not None
            and not base.isNull()
            and base.size() == image.size()
        )

    def _draw_frame(self, painter: QPainter, image: QImage, dest: QRect) -> None:
        if dest.width() <= 0 or dest.height() <= 0:
            return
        if self._pixel:
            src = self._source_rect(image, dest)
            painter.drawImage(QRect(dest.x(), dest.y(), src.width(), src.height()), image, src)
        else:
            painter.drawImage(self._letterbox(image.size(), dest), image)

    def _source_rect(self, image: QImage, dest: QRect) -> QRect:
        scale = self._pixel_scale()
        x = min(max(0, self._pan.x()), max(0, image.width() - 1))
        y = min(max(0, self._pan.y()), max(0, image.height() - 1))
        width = min(max(1, int(dest.width() * scale)), image.width() - x)
        height = min(max(1, int(dest.height() * scale)), image.height() - y)
        return QRect(x, y, width, height)

    def _frame_dest(self, image: QImage, view: QRect) -> tuple[QRect, QRect | None]:
        if self._pixel:
            src = self._source_rect(image, view)
            return QRect(view.x(), view.y(), src.width(), src.height()), src
        return self._letterbox(image.size(), view), None

    def _set_wipe(self, x: float) -> None:
        image = self._image
        view = QRect(0, 0, self.width(), max(0, self.height() - self._hud_height()))
        if image is None or image.isNull():
            span = max(1, view.width())
            self._wipe = min(1.0, max(0.0, x / span))
        else:
            dest, _src = self._frame_dest(image, view)
            span = max(1, dest.width())
            self._wipe = min(1.0, max(0.0, (x - dest.x()) / span))
        self.update()

    def _near_wipe(self, x: float) -> bool:
        image = self._image
        if image is None:
            return False
        view = QRect(0, 0, self.width(), max(0, self.height() - self._hud_height()))
        dest, _src = self._frame_dest(image, view)
        split = dest.x() + int(dest.width() * self._wipe)
        return abs(x - split) <= 8

    def _draw_wipe(self, painter: QPainter, base: QImage, enhanced: QImage, view: QRect) -> None:
        dest, src = self._frame_dest(enhanced, view)
        if dest.width() <= 0 or dest.height() <= 0:
            return
        split = int((src.width() if src is not None else dest.width()) * self._wipe)
        split = min(max(0, split), src.width() if src is not None else dest.width())
        if src is None:
            painter.save()
            painter.setClipRect(QRect(dest.x(), dest.y(), split, dest.height()))
            painter.drawImage(dest, base)
            painter.restore()
            painter.save()
            painter.setClipRect(QRect(dest.x() + split, dest.y(), max(0, dest.width() - split), dest.height()))
            painter.drawImage(dest, enhanced)
            painter.restore()
            line_x = dest.x() + split
        else:
            left = QRect(src.x(), src.y(), split, src.height())
            right = QRect(src.x() + split, src.y(), max(0, src.width() - split), src.height())
            painter.drawImage(QRect(dest.x(), dest.y(), left.width(), dest.height()), base, left)
            painter.drawImage(QRect(dest.x() + split, dest.y(), right.width(), dest.height()), enhanced, right)
            line_x = dest.x() + split
        painter.setPen(QColor("#eef3f5"))
        painter.drawLine(line_x, dest.y(), line_x, dest.bottom())

    def _letterbox(self, size, dest: QRect) -> QRect:
        if size.width() <= 0 or size.height() <= 0 or dest.width() <= 0 or dest.height() <= 0:
            return QRect()
        scale = min(dest.width() / size.width(), dest.height() / size.height())
        width = max(1, int(size.width() * scale))
        height = max(1, int(size.height() * scale))
        x = dest.x() + (dest.width() - width) // 2
        y = dest.y() + (dest.height() - height) // 2
        return QRect(x, y, width, height)
