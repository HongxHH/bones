"""Prism window: Lab on the left, live picture on the right."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request

from prism.cli.launcher import ensure_control_plane, lab_url
from prism.desk.picture import PicturePane
from prism.desk.scrub import split_left
from prism.frame_sink import NAME

from PySide6.QtCore import Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWidgets import QApplication, QMainWindow, QSplitter


class _PlaneWorker(QThread):
    finished_ok = Signal(bool, str)

    def __init__(self, host: str, port: int, bind: str, start_server: bool, timeout_s: float) -> None:
        super().__init__()
        self.host = host
        self.port = port
        self.bind = bind
        self.start_server = start_server
        self.timeout_s = timeout_s
        self.child: subprocess.Popen | None = None

    def run(self) -> None:
        code, child = ensure_control_plane(
            host=self.host,
            port=self.port,
            bind=self.bind,
            start_server=self.start_server,
            timeout_s=self.timeout_s,
        )
        self.child = child
        self.finished_ok.emit(code == 0, lab_url(self.host, self.port))


class PrismWindow(QMainWindow):
    def __init__(self, host: str, port: int, bind: str, start_server: bool, timeout_s: float) -> None:
        super().__init__()
        self.setWindowTitle("Prism")
        self.resize(*_initial_window_size())
        self._owned: subprocess.Popen | None = None
        self._review = False
        self._user_resized = False
        self._applying_split = False

        self._host = host
        self._port = port
        self._web = QWebEngineView()
        self._web.setMinimumWidth(360)
        self._stage = PicturePane(host, port)
        self._split = QSplitter(Qt.Orientation.Horizontal)
        self._split.addWidget(self._web)
        self._split.addWidget(self._stage)
        self._split.setStretchFactor(0, 0)
        self._split.setStretchFactor(1, 1)
        self._split.setChildrenCollapsible(False)
        self._split.setHandleWidth(6)
        self._split.splitterMoved.connect(self._on_splitter_moved)
        self.setCentralWidget(self._split)
        self.statusBar().hide()
        self._split_timer = QTimer(self)
        self._split_timer.timeout.connect(self._sync_split)
        self._split_timer.start(1000)

        self._worker = _PlaneWorker(host, port, bind, start_server, timeout_s)
        self._worker.finished_ok.connect(self._on_plane)
        self._worker.start()

    def _on_plane(self, ok: bool, url: str) -> None:
        self._owned = self._worker.child
        if not ok:
            self.statusBar().show()
            self.statusBar().showMessage("控制面未就绪")
            self._stage.set_status("控制面未就绪")
            return
        _register_frame_sink(self._host, self._port)
        self._web.setUrl(QUrl(url + ("&" if "?" in url else "?") + "shell=1"))
        self._layout_split()

    def showEvent(self, event) -> None:  # noqa: N802
        super().showEvent(event)
        self._layout_split()

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if not self._user_resized:
            self._layout_split()

    def _on_splitter_moved(self, _pos: int, _index: int) -> None:
        if self._applying_split:
            return
        self._user_resized = True

    def _sync_split(self) -> None:
        page = self._web.page()
        if page is None:
            return
        page.runJavaScript(
            "(document.body && document.body.dataset.review) || ''",
            self._apply_review,
        )

    def _apply_review(self, value) -> None:
        review = value == "compare"
        if review == self._review:
            return
        self._review = review
        # Entering or leaving compare overrides a manual drag.
        self._user_resized = False
        self._layout_split()

    def _layout_split(self) -> None:
        if self._user_resized:
            return
        total = self._split.width()
        if total < 200:
            return
        left = split_left(total, review=self._review)
        self._applying_split = True
        try:
            self._split.setSizes([left, max(1, total - left)])
        finally:
            self._applying_split = False

    def closeEvent(self, event) -> None:  # noqa: N802
        child = self._owned
        self._owned = None
        if child is not None and child.poll() is None and child.pid:
            _kill_tree(int(child.pid))
        self._stage.close_sink()
        super().closeEvent(event)


def _initial_window_size() -> tuple[int, int]:
    screen = QApplication.primaryScreen()
    if screen is None:
        return 1920, 1080
    avail = screen.availableGeometry()
    return min(1920, max(1280, avail.width())), min(1080, max(800, avail.height()))


def _register_frame_sink(host: str, port: int) -> None:
    body = json.dumps({"name": NAME}).encode("utf-8")
    req = urllib.request.Request(
        f"http://{host}:{port}/v1/system/frame-sink",
        data=body,
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            resp.read()
    except Exception as exc:
        print(f"frame sink register failed: {exc}", file=sys.stderr)


def _kill_tree(pid: int) -> None:
    if pid <= 0:
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, check=False)
        return
    try:
        os.kill(pid, 15)
    except OSError:
        pass


def _stylesheet() -> str:
    return """
    QMainWindow, QWidget { background: #eef3f5; color: #0e1a22; }
    QStatusBar { background: #f7fafb; color: #5a6b76; }
    QSplitter::handle { background: rgba(14, 26, 34, 0.18); width: 6px; }
    #stage { background: #0e1a22; }
    #hud {
      background: rgba(14, 26, 34, 0.92);
      border-top: 2px solid #0d6e6a;
    }
    #hudStat, #hudName, #hudValue, #hudValueOk, #hudValueWarn, #hudMode, #hudModeWarn {
      background: transparent;
    }
    #hudMode, #hudModeWarn {
      color: #8fe0da;
      font-size: 12px;
      font-weight: 700;
      min-width: 44px;
      padding: 0 4px;
    }
    #hudModeWarn { color: #f0c08a; }
    #hudName {
      color: rgba(238, 243, 245, 0.48);
      font-size: 11px;
    }
    #hudValue, #hudValueOk, #hudValueWarn {
      color: #eef3f5;
      font-size: 16px;
      font-family: "Cascadia Mono", "Consolas", monospace;
    }
    #hudValueOk { color: #8fe0da; }
    #hudValueWarn { color: #f0c08a; }
    #hudRule { background: rgba(238, 243, 245, 0.16); }
    #hud QPushButton, #hudChip {
      color: #eef3f5;
      background: rgba(238, 243, 245, 0.12);
      border: 0;
      padding: 6px 12px;
    }
    #hud QPushButton#hudOn { background: #0d6e6a; }
    #hud QPushButton:hover, #hudChip:hover { background: rgba(13, 110, 106, 0.55); }
    #scrub {
      background: rgba(14, 26, 34, 0.94);
      border-top: 1px solid rgba(143, 224, 218, 0.18);
    }
    #scrubMeta {
      color: rgba(238, 243, 245, 0.78);
      background: transparent;
      font-size: 11px;
      letter-spacing: 0.01em;
    }
    #scrub QPushButton {
      color: #eef3f5;
      background: rgba(238, 243, 245, 0.10);
      border: 1px solid rgba(238, 243, 245, 0.14);
      border-radius: 4px;
      padding: 3px 10px;
      font-size: 11px;
      min-height: 22px;
    }
    #scrub QPushButton:hover {
      background: rgba(13, 110, 106, 0.55);
      border-color: rgba(143, 224, 218, 0.35);
    }
    #scrubSlider {
      background: transparent;
      padding: 0 2px;
    }
    #scrubSlider::groove:horizontal {
      height: 6px;
      background: rgba(238, 243, 245, 0.14);
      border: 1px solid rgba(238, 243, 245, 0.08);
      border-radius: 3px;
    }
    #scrubSlider::sub-page:horizontal {
      background: #0d6e6a;
      border-radius: 3px;
    }
    #scrubSlider::add-page:horizontal {
      background: transparent;
      border-radius: 3px;
    }
    #scrubSlider::handle:horizontal {
      width: 14px;
      height: 14px;
      margin: -5px 0;
      background: #8fe0da;
      border: 2px solid #eef3f5;
      border-radius: 8px;
    }
    #scrubSlider::handle:horizontal:hover {
      background: #b7f0eb;
    }
    #scrubSlider::handle:horizontal:pressed {
      background: #0d6e6a;
      border-color: #8fe0da;
    }
    #scrubSlider:disabled {
      opacity: 0.45;
    }
    """


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="shell", description="Prism window")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--no-server", action="store_true")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)

    os.environ["PRISM_FRAME_SHM"] = NAME
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)
    app = QApplication(sys.argv)
    app.setApplicationName("Prism")
    app.setStyleSheet(_stylesheet())
    window = PrismWindow(
        host=args.host,
        port=args.port,
        bind=args.bind,
        start_server=not args.no_server,
        timeout_s=args.timeout,
    )
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
