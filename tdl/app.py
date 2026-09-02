"""PySide6 + PyVista viewer: frames, timeline scrubbing, velocity limits."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_API", "pyside6")

import numpy as np
from PySide6.QtCore import QElapsedTimer, Qt, QTimer
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from pyvistaqt import QtInteractor

from tdl.io import load_task
from tdl.multi import MultiTrajectory, build_multi_trajectory
from tdl.schema import Limits, Task, TowerLight, TowerLightState
from tdl.viz import KIND_COLORS, SYSTEM_TCP_COLORS, TcpActor, add_static_scene


class IoLedRow(QWidget):
    def __init__(self, name: str, description: str = ""):
        super().__init__()
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 2, 0, 2)
        self.led = QLabel()
        self.led.setFixedSize(14, 14)
        self.label = QLabel(name)
        self.label.setStyleSheet("font-weight: 600;")
        row.addWidget(self.led)
        row.addWidget(self.label, stretch=1)
        if description:
            self.setToolTip(description)
        self.set_on(False)

    def set_on(self, on: bool) -> None:
        if on:
            self.led.setStyleSheet(
                "background-color: #00E676; border-radius: 7px; border: 1px solid #69F0AE;"
            )
        else:
            self.led.setStyleSheet(
                "background-color: #37474F; border-radius: 7px; border: 1px solid #546E7A;"
            )


class TowerLightStack(QWidget):
    def __init__(self, name: str, light: TowerLight):
        super().__init__()
        self._states = list(light.states)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)
        title = QLabel(name)
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)
        stack = QVBoxLayout()
        stack.setSpacing(3)
        self._lamps: list[tuple[TowerLightState, QLabel]] = []
        for state in reversed(self._states):
            row = QHBoxLayout()
            lamp = QLabel()
            lamp.setFixedSize(16, 16)
            row.addWidget(lamp)
            label = QLabel(state.name)
            label.setStyleSheet("color: #90A4AE;")
            row.addWidget(label, stretch=1)
            stack.addLayout(row)
            self._lamps.append((state, lamp))
        layout.addLayout(stack)
        self._status = QLabel("—")
        self._status.setStyleSheet("color: #78909C; font-size: 11px;")
        layout.addWidget(self._status)
        self.set_state(None)

    @staticmethod
    def _dim_color(color: str, factor: float = 0.28) -> str:
        hex_color = color.lstrip("#")
        if len(hex_color) != 6:
            return "#37474F"
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"#{int(r * factor):02x}{int(g * factor):02x}{int(b * factor):02x}"

    def set_state(self, index: int | None) -> None:
        for i, (state, lamp) in enumerate(self._lamps):
            actual_idx = len(self._states) - 1 - i
            active = index is not None and actual_idx == index
            fill = state.color if active else self._dim_color(state.color)
            border = state.color if active else self._dim_color(state.color, 0.45)
            lamp.setStyleSheet(
                f"background-color: {fill}; border-radius: 8px; border: 1px solid {border};"
            )
        if index is None or not 0 <= index < len(self._states):
            self._status.setText("—")
        else:
            self._status.setText(self._states[index].name)


def _example_path() -> Path:
    here = Path(__file__).resolve().parent.parent / "examples" / "pick_and_place.yaml"
    return here


class Viewer(QMainWindow):
    def __init__(self, path: Path | None = None):
        super().__init__()
        self.setWindowTitle("Task Definition Language")
        self.resize(1400, 880)
        self.task: Task | None = None
        self.multi_traj: MultiTrajectory | None = None
        self._path_names: list[str] = []
        self._playing = False
        self._t = 0.0
        self._play_t0 = 0.0
        self._elapsed = QElapsedTimer()
        self.io_panel: QGroupBox | None = None
        self.io_leds: dict[str, IoLedRow] = {}
        self.tower_panel: QGroupBox | None = None
        self.tower_widgets: dict[str, TowerLightStack] = {}
        self._closing = False

        root = QWidget()
        self.setCentralWidget(root)
        split = QSplitter(Qt.Orientation.Horizontal)
        layout = QHBoxLayout(root)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(split)

        view_wrap = QWidget()
        view_layout = QVBoxLayout(view_wrap)
        view_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter = QtInteractor(view_wrap)
        view_layout.addWidget(self.plotter.interactor, stretch=1)
        view_layout.addLayout(self._timeline_bar())
        split.addWidget(view_wrap)

        split.addWidget(self._side_panel())
        split.setStretchFactor(0, 4)
        split.setStretchFactor(1, 1)

        self.tcps: dict[str, TcpActor] = {}
        self.timer = QTimer(self)
        self.timer.setInterval(16)
        self.timer.timeout.connect(self._tick)

        if path is None:
            path = _example_path()
        if path.exists():
            self.load_file(path)
        else:
            self.statusBar().showMessage("Open a .yaml or .json task file")

    def closeEvent(self, event: QCloseEvent) -> None:
        self._closing = True
        if hasattr(self, "timer"):
            self.timer.stop()
        if hasattr(self, "tcps"):
            self.tcps.clear()
        plotter = self.plotter
        if plotter is not None:
            try:
                if not getattr(plotter, "_closed", True):
                    plotter.close()
            except Exception:
                pass
            self.plotter = None
        super().closeEvent(event)

    def _timeline_bar(self) -> QHBoxLayout:
        bar = QHBoxLayout()
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setRange(0, 1000)
        self.time_slider.valueChanged.connect(self._slider_moved)
        self.time_label = QLabel("0.00 s")
        self.seg_label = QLabel("—")
        self.seg_label.setMinimumWidth(180)
        bar.addWidget(self.play_btn)
        bar.addWidget(self.time_slider, stretch=1)
        bar.addWidget(self.time_label)
        bar.addWidget(self.seg_label)
        return bar

    def _side_panel(self) -> QWidget:
        panel = QWidget()
        v = QVBoxLayout(panel)

        file_row = QHBoxLayout()
        self.file_label = QLabel("No file")
        self.file_label.setWordWrap(True)
        open_btn = QPushButton("Open…")
        open_btn.clicked.connect(self.open_file)
        file_row.addWidget(self.file_label, stretch=1)
        file_row.addWidget(open_btn)
        v.addLayout(file_row)

        self.name_label = QLabel("")
        self.name_label.setStyleSheet("font-weight: 600; font-size: 15px;")
        v.addWidget(self.name_label)

        self.joints_label = QLabel("")
        self.joints_label.setWordWrap(True)
        self.joints_label.setStyleSheet("color: #90A4AE;")
        v.addWidget(self.joints_label)

        limits = QGroupBox("Velocity limits")
        form = QFormLayout(limits)
        self.spin_lin = QDoubleSpinBox()
        self.spin_lin.setRange(0.01, 5.0)
        self.spin_lin.setSingleStep(0.05)
        self.spin_lin.setSuffix(" m/s")
        self.spin_ang = QDoubleSpinBox()
        self.spin_ang.setRange(1.0, 720.0)
        self.spin_ang.setSingleStep(5.0)
        self.spin_ang.setSuffix(" deg/s")
        self.spin_app = QDoubleSpinBox()
        self.spin_app.setRange(0.005, 2.0)
        self.spin_app.setSingleStep(0.01)
        self.spin_app.setSuffix(" m/s")
        for spin in (self.spin_lin, self.spin_ang, self.spin_app):
            spin.valueChanged.connect(self._limits_changed)
        form.addRow("Free space", self.spin_lin)
        form.addRow("Angular", self.spin_ang)
        form.addRow("Approach", self.spin_app)
        v.addWidget(limits)

        self.io_panel = QGroupBox("I/O")
        self.io_layout = QVBoxLayout(self.io_panel)
        self.io_layout.setContentsMargins(8, 8, 8, 8)
        self.io_layout.setSpacing(2)
        v.addWidget(self.io_panel)

        self.tower_panel = QGroupBox("Tower lights")
        self.tower_layout = QVBoxLayout(self.tower_panel)
        self.tower_layout.setContentsMargins(8, 8, 8, 8)
        self.tower_layout.setSpacing(6)
        v.addWidget(self.tower_panel)

        self.follow = QCheckBox("Reset camera on load")
        self.follow.setChecked(True)
        v.addWidget(self.follow)

        v.addWidget(QLabel("Sequence"))
        self.seq_list = QListWidget()
        self.seq_list.itemClicked.connect(self._jump_to_item)
        v.addWidget(self.seq_list, stretch=1)

        hint = QLabel("Drag in the 3D view to inspect. Scrub the timeline to move the tool frame along the motion.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #78909C;")
        v.addWidget(hint)
        return panel

    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open task",
            str(Path.cwd()),
            "Task files (*.yaml *.yml *.json)",
        )
        if path:
            self.load_file(Path(path))

    def load_file(self, path: Path) -> None:
        self.task = load_task(path)
        self.file_label.setText(path.name)
        self.name_label.setText(self.task.name)
        self.setWindowTitle(f"{self.task.name} — Task Definition Language")
        joints = ", ".join(f"{j:g}" for j in self.task.robot.rest.joints)
        unit = "deg" if self.task.degrees else "rad"
        self.joints_label.setText(f"rest joints ({unit}): [{joints}]")
        self._block_limit_signals(True)
        self.spin_lin.setValue(self.task.limits.linear)
        ang = self.task.limits.angular if self.task.degrees else float(np.rad2deg(self.task.limits.angular))
        self.spin_ang.setValue(ang)
        self.spin_app.setValue(self.task.limits.approach_linear)
        self._block_limit_signals(False)
        self._rebuild(reset_camera=True)
        self.statusBar().showMessage(f"Loaded {path}")

    def current_limits(self) -> Limits:
        ang = self.spin_ang.value()
        if self.task and not self.task.degrees:
            ang = float(np.deg2rad(ang))
        return Limits(
            linear=self.spin_lin.value(),
            angular=ang,
            approach_linear=self.spin_app.value(),
        )

    def _block_limit_signals(self, blocked: bool) -> None:
        for spin in (self.spin_lin, self.spin_ang, self.spin_app):
            spin.blockSignals(blocked)

    def _limits_changed(self) -> None:
        if self.task is None:
            return
        frac = self.time_slider.value() / max(self.time_slider.maximum(), 1)
        self._rebuild(reset_camera=False)
        self.time_slider.setValue(int(frac * self.time_slider.maximum()))

    def _rebuild(self, reset_camera: bool) -> None:
        if self._closing or self.plotter is None:
            return
        assert self.task is not None
        self.multi_traj = build_multi_trajectory(self.task, self.current_limits())
        self.plotter.clear()
        add_static_scene(self.plotter, self.task, self.multi_traj)
        self.tcps = {}
        for name in self.multi_traj.systems:
            label = name or "main"
            color = SYSTEM_TCP_COLORS.get(name, "#FFECB3")
            self.tcps[name] = TcpActor(self.plotter, name=label, ball_color=color)
        self._fill_sequence()
        self._rebuild_io_panel()
        self._rebuild_tower_panel()
        if reset_camera and self.follow.isChecked():
            self.plotter.view_isometric()
            self.plotter.reset_camera()
        self._t = 0.0
        self._sync_slider()
        self._apply_time(0.0)
        self.plotter.render()

    def _fill_sequence(self) -> None:
        self.seq_list.clear()
        if self.multi_traj is None:
            return
        for seg in self.multi_traj.segments:
            item = QListWidgetItem(f"{seg.t0:5.2f}–{seg.t1:5.2f}s   {seg.label}")
            item.setData(Qt.ItemDataRole.UserRole, seg.t0)
            color = KIND_COLORS.get(seg.kind, "#FFFFFF")
            item.setForeground(Qt.GlobalColor.white)
            item.setToolTip(seg.kind)
            item.setData(Qt.ItemDataRole.UserRole + 1, color)
            self.seq_list.addItem(item)

    def _rebuild_io_panel(self) -> None:
        assert self.io_panel is not None
        while self.io_layout.count():
            item = self.io_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.io_leds.clear()
        if self.task is None or not self.task.io:
            self.io_panel.hide()
            return
        self.io_panel.show()
        for name in sorted(self.task.io):
            sig = self.task.io[name]
            row = IoLedRow(name, sig.description)
            self.io_layout.addWidget(row)
            self.io_leds[name] = row
        self.io_layout.addStretch(1)

    def _rebuild_tower_panel(self) -> None:
        assert self.tower_panel is not None
        while self.tower_layout.count():
            item = self.tower_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.tower_widgets.clear()
        if self.task is None or not self.task.tower_lights:
            self.tower_panel.hide()
            return
        self.tower_panel.show()
        for name in sorted(self.task.tower_lights):
            widget = TowerLightStack(name, self.task.tower_lights[name])
            self.tower_layout.addWidget(widget)
            self.tower_widgets[name] = widget
        self.tower_layout.addStretch(1)

    def _jump_to_item(self, item: QListWidgetItem) -> None:
        t0 = float(item.data(Qt.ItemDataRole.UserRole))
        if self.multi_traj is None:
            return
        self._playing = False
        self.play_btn.setText("Play")
        self.timer.stop()
        frac = t0 / max(self.multi_traj.duration, 1e-6)
        self.time_slider.setValue(int(frac * self.time_slider.maximum()))

    def toggle_play(self) -> None:
        if self.multi_traj is None:
            return
        self._playing = not self._playing
        self.play_btn.setText("Pause" if self._playing else "Play")
        if self._playing:
            if self._t >= self.multi_traj.duration - 1e-9:
                self._t = 0.0
                self._sync_slider()
                self._apply_time(self._t)
            self._play_t0 = self._t
            self._elapsed.start()
            self.timer.start()
        else:
            self.timer.stop()

    def _tick(self) -> None:
        if self._closing or self.plotter is None:
            return
        if self.multi_traj is None:
            return
        t = self._play_t0 + self._elapsed.elapsed() / 1000.0
        if t >= self.multi_traj.duration:
            t = self.multi_traj.duration
            self._playing = False
            self.play_btn.setText("Play")
            self.timer.stop()
        self._t = t
        self._sync_slider()
        self._apply_time(t)

    def _sync_slider(self) -> None:
        if self.multi_traj is None:
            return
        value = int(round((self._t / max(self.multi_traj.duration, 1e-9)) * self.time_slider.maximum()))
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(value)
        self.time_slider.blockSignals(False)

    def _slider_moved(self, value: int) -> None:
        if self.multi_traj is None:
            return
        self._t = (value / max(self.time_slider.maximum(), 1)) * self.multi_traj.duration
        if self._playing:
            self._play_t0 = self._t
            self._elapsed.restart()
        self._apply_time(self._t)

    def _apply_time(self, t: float) -> None:
        if self._closing or self.plotter is None or self.multi_traj is None or not self.tcps:
            return
        states = self.multi_traj.at_time(t)
        parts = []
        for name, tcp in self.tcps.items():
            pose, kind, label = states[name]
            tcp.set_pose(pose)
            tag = name or "main"
            parts.append(f"{tag}: {label or kind}")
        self.time_label.setText(f"{t:.2f} / {self.multi_traj.duration:.2f} s")
        primary = next(iter(states.values()))
        _, kind, label = primary
        self.seg_label.setText(" | ".join(parts) if len(parts) > 1 else (label or "—"))
        self.seg_label.setStyleSheet(f"color: {KIND_COLORS.get(kind, '#ECEFF1')};")
        if self.multi_traj.io_timeline is not None:
            io_state = self.multi_traj.io_timeline.state_at(t)
            for name, row in self.io_leds.items():
                row.set_on(io_state.get(name, False))
        if self.multi_traj.tower_timeline is not None:
            tower_state = self.multi_traj.tower_timeline.state_at(t)
            for name, widget in self.tower_widgets.items():
                widget.set_state(tower_state.get(name))
        self.plotter.render()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize a Task Definition Language file")
    parser.add_argument("task", nargs="?", help="Path to a .yaml/.yml/.json task")
    args = parser.parse_args()
    qt = QApplication(sys.argv)
    qt.setApplicationName("tdl-viewer")
    path = Path(args.task) if args.task else None
    win = Viewer(path)
    win.show()
    code = qt.exec()
    if getattr(win, "plotter", None) is not None:
        try:
            if not getattr(win.plotter, "_closed", True):
                win.plotter.close()
        except Exception:
            pass
        win.plotter = None
    sys.exit(code)


if __name__ == "__main__":
    main()
