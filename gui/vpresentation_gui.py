#!/usr/bin/env python3
"""VPresentation GUI (PySide6).

Drives the persistent Rust `serve` engine (candle + Metal). Because a full
pose render takes a few seconds, live posing renders on demand in a worker
thread, and the idle blink is done by precomputing an eyes-open / eyes-closed
frame pair and alternating them on a timer (no per-frame inference).

Run:  .venv/bin/python gui/vpresentation_gui.py [character.png]
"""
import os
import sys
import subprocess
import tempfile
from pathlib import Path

from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QSlider, QPushButton, QCheckBox,
    QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
)

REPO = Path(__file__).resolve().parent.parent
SERVE_BIN = REPO / "target" / "release" / "serve"
NUM_PARAMS = 45

# (label, pose-index, min, max) for the exposed sliders.
CONTROLS = [
    ("Head X", 39, -1.0, 1.0),
    ("Head Y", 40, -1.0, 1.0),
    ("Neck Z", 41, -1.0, 1.0),
    ("Body Y", 42, -1.0, 1.0),
    ("Wink L", 12, 0.0, 1.0),
    ("Wink R", 13, 0.0, 1.0),
    ("Mouth (aaa)", 26, 0.0, 1.0),
    ("Breathing", 44, 0.0, 1.0),
]
EYE_WINK = (12, 13)


class Engine:
    """Wraps the persistent Rust serve process (blocking request/response)."""

    def __init__(self, character: str):
        self.proc = subprocess.Popen(
            [str(SERVE_BIN), character],
            cwd=str(REPO),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            text=True, bufsize=1,
        )
        line = self.proc.stdout.readline().strip()
        if not line.startswith("READY"):
            raise RuntimeError(f"engine failed to start: {line}")
        self.device = line.split(" ", 1)[1] if " " in line else "?"
        self._tmp = tempfile.mkdtemp(prefix="vpres_")
        self._n = 0

    def render(self, pose) -> str:
        self._n += 1
        out = os.path.join(self._tmp, f"frame_{self._n}.png")
        nums = ",".join(f"{v:.5f}" for v in pose)
        self.proc.stdin.write(f"{out};{nums}\n")
        self.proc.stdin.flush()
        resp = self.proc.stdout.readline().strip()
        if not resp.startswith("OK"):
            raise RuntimeError(resp)
        return out  # "OK <path> <ms>"  -> path is out

    def close(self):
        try:
            self.proc.stdin.write("quit\n")
            self.proc.stdin.flush()
        except Exception:
            pass


class RenderWorker(QThread):
    """Renders poses off the UI thread. Coalesces to the latest request."""
    done = Signal(str, object)  # path, tag

    def __init__(self, engine: Engine):
        super().__init__()
        self.engine = engine
        self._pending = None
        self._running = True

    def request(self, pose, tag):
        self._pending = (list(pose), tag)

    def run(self):
        while self._running:
            job = self._pending
            if job is None:
                self.msleep(20)
                continue
            self._pending = None
            pose, tag = job
            try:
                path = self.engine.render(pose)
                self.done.emit(path, tag)
            except Exception as e:  # noqa
                print("render error:", e, file=sys.stderr)

    def stop(self):
        self._running = False


class MainWindow(QWidget):
    def __init__(self, character: str):
        super().__init__()
        self.setWindowTitle("VPresentation — THA4 (candle + Metal)")
        self.pose = [0.0] * NUM_PARAMS

        self.engine = Engine(character)
        self.worker = RenderWorker(self.engine)
        self.worker.done.connect(self.on_rendered)
        self.worker.start()

        # --- image view ---
        self.view = QLabel("rendering…")
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setFixedSize(512, 512)
        self.view.setStyleSheet("background:#d9d9d9;border:1px solid #999;")

        # --- controls ---
        form = QFormLayout()
        self.sliders = {}
        for label, idx, lo, hi in CONTROLS:
            s = QSlider(Qt.Horizontal)
            s.setMinimum(0)
            s.setMaximum(100)
            s.setValue(int((0.0 - lo) / (hi - lo) * 100))
            s.valueChanged.connect(self.on_slider)
            self.sliders[idx] = (s, lo, hi)
            form.addRow(label, s)
        ctrl_box = QGroupBox("Pose")
        ctrl_box.setLayout(form)

        self.render_btn = QPushButton("Render")
        self.render_btn.clicked.connect(self.render_current)
        self.blink_chk = QCheckBox("Auto-blink")
        self.blink_chk.toggled.connect(self.toggle_blink)
        self.status = QLabel("device: %s" % self.engine.device)

        right = QVBoxLayout()
        right.addWidget(ctrl_box)
        right.addWidget(self.render_btn)
        right.addWidget(self.blink_chk)
        right.addWidget(self.status)
        right.addStretch(1)

        root = QHBoxLayout(self)
        root.addWidget(self.view)
        root.addLayout(right)

        # blink state
        self._open_px = None
        self._closed_px = None
        self._blink_timer = QTimer(self)
        self._blink_timer.timeout.connect(self._blink_tick)
        self._blink_phase = 0

        QTimer.singleShot(100, self.render_current)  # initial frame

    # --- slider handling ---
    def on_slider(self):
        for idx, (s, lo, hi) in self.sliders.items():
            self.pose[idx] = lo + (hi - lo) * (s.value() / 100.0)

    def render_current(self):
        self.status.setText("rendering…")
        self.worker.request(self.pose, "live")

    def on_rendered(self, path, tag):
        px = QPixmap(path)
        if tag == "live":
            self.view.setPixmap(px)
            self.status.setText("device: %s — ready" % self.engine.device)
        elif tag == "blink_open":
            self._open_px = px
            self.view.setPixmap(px)
        elif tag == "blink_closed":
            self._closed_px = px
            self.status.setText("auto-blink ready")

    # --- auto blink ---
    def toggle_blink(self, on):
        if on:
            self.status.setText("preparing blink frames…")
            self.worker.request(self.pose, "blink_open")
            closed = list(self.pose)
            closed[EYE_WINK[0]] = 1.0
            closed[EYE_WINK[1]] = 1.0
            self.worker.request(closed, "blink_closed")
            self._blink_timer.start(120)
        else:
            self._blink_timer.stop()
            if self._open_px:
                self.view.setPixmap(self._open_px)

    def _blink_tick(self):
        if not self._open_px or not self._closed_px:
            return
        # Mostly open; brief closed every ~2.4s for a natural blink.
        self._blink_phase = (self._blink_phase + 1) % 20
        px = self._closed_px if self._blink_phase in (0, 1) else self._open_px
        self.view.setPixmap(px)

    def closeEvent(self, e):
        self.worker.stop()
        self.worker.wait(500)
        self.engine.close()
        super().closeEvent(e)


def main():
    character = sys.argv[1] if len(sys.argv) > 1 else str(REPO / "data/images/lambda_00.png")
    if not SERVE_BIN.exists():
        print(f"build the engine first: cargo build --release -p tha4 --bin serve", file=sys.stderr)
        sys.exit(1)
    app = QApplication(sys.argv)
    win = MainWindow(character)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
