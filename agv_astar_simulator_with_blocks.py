#!/usr/bin/env python3
"""
Pure AGV A* Grid Simulation

Purpose:
  - Simulation only.
  - No camera.
  - No AprilTag detection.
  - No calibration.
  - No ESP32 / serial.
  - Rows and columns can be changed from the UI.
  - A* path, heading changes, and turn commands can be checked visually.

Physical heading convention kept from AGV project:
  - Same row, tag number +1  => WEST
  - Same row, tag number -1  => EAST
  - Next row, tag number +COLS => NORTH
  - Previous row, tag number -COLS => SOUTH

This means tag numbers increase leftward across a row and upward by rows,
matching the physical AGV convention used in the main project.
"""

import sys
import time
import heapq
from dataclasses import dataclass

from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QPainter, QPen, QBrush
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


NORTH = 0
EAST = 1
SOUTH = 2
WEST = 3

HEADING_LABELS = {
    NORTH: "NORTH",
    EAST: "EAST",
    SOUTH: "SOUTH",
    WEST: "WEST",
}


def turn_delta_deg(current_heading, desired_heading):
    """
    Same turn convention as the AGV code:
      NORTH -> WEST = +90
      NORTH -> EAST = -90
    """
    if current_heading is None or desired_heading is None:
        return 0.0

    delta_steps = (desired_heading - current_heading) % 4

    if delta_steps == 0:
        return 0.0
    if delta_steps == 1:
        return -90.0
    if delta_steps == 2:
        return 180.0
    if delta_steps == 3:
        return 90.0

    return 0.0


def turn_action_between_headings(in_heading, out_heading):
    if in_heading is None or out_heading is None:
        return "UNKNOWN"

    delta_steps = (out_heading - in_heading) % 4

    if delta_steps == 0:
        return "STRAIGHT"
    if delta_steps == 1:
        return "RIGHT"
    if delta_steps == 3:
        return "LEFT"
    if delta_steps == 2:
        return "UTURN"

    return "UNKNOWN"


@dataclass
class GridModel:
    rows: int = 3
    cols: int = 5

    def max_tag(self):
        return self.rows * self.cols

    def tag_to_rc(self, tag_id: int):
        tag0 = int(tag_id) - 1
        return tag0 // self.cols, tag0 % self.cols

    def rc_to_tag(self, row: int, col: int):
        return row * self.cols + col + 1

    def in_bounds(self, row: int, col: int):
        return 0 <= row < self.rows and 0 <= col < self.cols

    def neighbors(self, tag_id: int, blocked=None):
        if blocked is None:
            blocked = set()

        r, c = self.tag_to_rc(tag_id)
        out = []

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if not self.in_bounds(nr, nc):
                continue

            nb = self.rc_to_tag(nr, nc)
            if nb not in blocked:
                out.append(nb)

        return out

    def manhattan(self, a: int, b: int):
        ar, ac = self.tag_to_rc(a)
        br, bc = self.tag_to_rc(b)
        return abs(ar - br) + abs(ac - bc)

    def astar_path(self, start: int, goal: int, blocked=None):
        if blocked is None:
            blocked = set()

        start = int(start)
        goal = int(goal)

        if start < 1 or start > self.max_tag():
            return []
        if goal < 1 or goal > self.max_tag():
            return []
        if start in blocked or goal in blocked:
            return []

        open_heap = []
        heapq.heappush(open_heap, (0, start))

        came_from = {}
        g_score = {start: 0}

        while open_heap:
            _, current = heapq.heappop(open_heap)

            if current == goal:
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path

            for nb in self.neighbors(current, blocked=blocked):
                tentative = g_score[current] + 1
                if tentative < g_score.get(nb, 10**9):
                    came_from[nb] = current
                    g_score[nb] = tentative
                    f = tentative + self.manhattan(nb, goal)
                    heapq.heappush(open_heap, (f, nb))

        return []

    def heading_between_tags(self, a: int, b: int):
        """
        Physical AGV heading convention.

        same row, +1      => WEST
        same row, -1      => EAST
        next row, +COLS   => NORTH
        previous row      => SOUTH
        """
        ar, ac = self.tag_to_rc(a)
        br, bc = self.tag_to_rc(b)

        if br == ar + 1 and bc == ac:
            return NORTH
        if br == ar - 1 and bc == ac:
            return SOUTH
        if br == ar and bc == ac + 1:
            return WEST
        if br == ar and bc == ac - 1:
            return EAST

        return None


class SimGridWidget(QWidget):
    tag_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.model = GridModel(3, 5)
        self.current_tag = 1
        self.goal_tag = None
        self.expected_tag = None
        self.path = []
        self.visited = []
        self.blocked = set()
        self.select_mode = "goal"

        self.setMinimumSize(520, 360)

    def set_model(self, model: GridModel):
        self.model = model
        self.current_tag = min(max(1, self.current_tag), self.model.max_tag())
        if self.goal_tag is not None:
            self.goal_tag = min(max(1, self.goal_tag), self.model.max_tag())
        self.blocked = {t for t in self.blocked if 1 <= t <= self.model.max_tag()}
        self.path = []
        self.visited = []
        self.expected_tag = None
        self.update()

    def set_state(self, current_tag=None, goal_tag=None, expected_tag=None, path=None, visited=None, blocked=None):
        if current_tag is not None:
            self.current_tag = current_tag
        self.goal_tag = goal_tag
        self.expected_tag = expected_tag
        self.path = path or []
        self.visited = visited or []
        if blocked is not None:
            self.blocked = set(blocked)
        self.update()

    def mousePressEvent(self, event):
        margin = 30
        w = max(1, self.width() - 2 * margin)
        h = max(1, self.height() - 2 * margin)
        cell_w = w / self.model.cols
        cell_h = h / self.model.rows

        col = int((event.x() - margin) / cell_w)
        row = int((event.y() - margin) / cell_h)

        if not self.model.in_bounds(row, col):
            return

        tag = self.model.rc_to_tag(row, col)
        self.tag_clicked.emit(tag)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.fillRect(self.rect(), QColor(55, 58, 60))

        margin = 30
        w = max(1, self.width() - 2 * margin)
        h = max(1, self.height() - 2 * margin)
        cell_w = w / self.model.cols
        cell_h = h / self.model.rows

        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor(235, 235, 235)))
        painter.drawText(
            20,
            20,
            f"Simulation Grid: {self.model.rows} rows × {self.model.cols} cols | Tags 1-{self.model.max_tag()}"
        )

        path_set = set(self.path)
        visited_set = set(self.visited)

        for r in range(self.model.rows):
            for c in range(self.model.cols):
                tag = self.model.rc_to_tag(r, c)

                x = margin + c * cell_w
                y = margin + r * cell_h

                fill = QColor(245, 245, 245)
                border = QColor(85, 85, 85)

                if tag in self.blocked:
                    fill = QColor(75, 75, 75)
                    border = QColor(20, 20, 20)
                elif tag in visited_set:
                    fill = QColor(195, 235, 205)
                elif tag in path_set:
                    fill = QColor(210, 230, 255)

                if tag == self.goal_tag:
                    fill = QColor(255, 225, 135)
                if tag == self.expected_tag:
                    fill = QColor(255, 195, 105)
                if tag == self.current_tag:
                    fill = QColor(120, 235, 150)
                    border = QColor(10, 130, 55)

                painter.setPen(QPen(border, 2))
                painter.setBrush(QBrush(fill))
                painter.drawRect(int(x), int(y), int(cell_w), int(cell_h))

                font = QFont()
                font.setPointSize(max(8, min(18, int(min(cell_w, cell_h) / 3.2))))
                font.setBold(True)
                painter.setFont(font)

                if tag in self.blocked:
                    painter.setPen(QPen(QColor(220, 220, 220)))
                    label = "X"
                else:
                    painter.setPen(QPen(QColor(45, 45, 45)))
                    label = str(tag)

                painter.drawText(
                    int(x),
                    int(y),
                    int(cell_w),
                    int(cell_h),
                    Qt.AlignCenter,
                    label
                )

        if len(self.path) >= 2:
            painter.setPen(QPen(QColor(40, 120, 220), 5))

            for i in range(len(self.path) - 1):
                a = self.path[i]
                b = self.path[i + 1]
                ar, ac = self.model.tag_to_rc(a)
                br, bc = self.model.tag_to_rc(b)

                ax = margin + ac * cell_w + cell_w / 2
                ay = margin + ar * cell_h + cell_h / 2
                bx = margin + bc * cell_w + cell_w / 2
                by = margin + br * cell_h + cell_h / 2

                painter.drawLine(int(ax), int(ay), int(bx), int(by))


class AGVGridSimulator(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Pure AGV A* Grid Simulator - No Calibration / No Camera / No ESP32")
        self.resize(1100, 720)

        self.model = GridModel(3, 5)

        self.current_tag = 1
        self.goal_tag = 15
        self.current_heading = NORTH
        self.path = []
        self.active_path = []
        self.path_index = 0
        self.expected_next_tag = None
        self.visited = [self.current_tag]
        self.blocked = set()
        self.running = False

        self.sim_timer = QTimer(self)
        self.sim_timer.timeout.connect(self.simulation_tick)

        self.build_ui()
        self.rebuild_destination_combo()
        self.compute_path()
        self.update_ui_state()

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)

        main = QHBoxLayout(root)

        left = QVBoxLayout()
        right = QVBoxLayout()

        main.addLayout(left, 4)
        main.addLayout(right, 2)

        grid_group = QGroupBox("Grid")
        grid_layout = QVBoxLayout(grid_group)

        self.grid = SimGridWidget()
        self.grid.tag_clicked.connect(self.on_grid_clicked)
        grid_layout.addWidget(self.grid)

        left.addWidget(grid_group, 1)

        setup_group = QGroupBox("Grid Setup")
        setup = QGridLayout(setup_group)

        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(1, 50)
        self.rows_spin.setValue(self.model.rows)

        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(1, 50)
        self.cols_spin.setValue(self.model.cols)

        self.apply_grid_btn = QPushButton("Apply Rows / Cols")
        self.apply_grid_btn.clicked.connect(self.apply_grid_size)

        self.start_tag_spin = QSpinBox()
        self.start_tag_spin.setRange(1, self.model.max_tag())
        self.start_tag_spin.setValue(self.current_tag)

        self.goal_combo = QComboBox()

        self.set_start_btn = QPushButton("Set Start")
        self.set_start_btn.clicked.connect(self.set_start_from_spin)

        self.compute_btn = QPushButton("Compute A*")
        self.compute_btn.clicked.connect(self.compute_path)

        setup.addWidget(QLabel("Rows:"), 0, 0)
        setup.addWidget(self.rows_spin, 0, 1)
        setup.addWidget(QLabel("Cols:"), 0, 2)
        setup.addWidget(self.cols_spin, 0, 3)
        setup.addWidget(self.apply_grid_btn, 0, 4)

        setup.addWidget(QLabel("Start Tag:"), 1, 0)
        setup.addWidget(self.start_tag_spin, 1, 1)
        setup.addWidget(self.set_start_btn, 1, 2)

        setup.addWidget(QLabel("Goal Tag:"), 2, 0)
        setup.addWidget(self.goal_combo, 2, 1, 1, 2)
        setup.addWidget(self.compute_btn, 2, 3, 1, 2)

        left.addWidget(setup_group)

        click_group = QGroupBox("Click / Block Mode")
        click_layout = QVBoxLayout(click_group)

        self.click_goal_cb = QCheckBox("Click cell sets goal")
        self.click_goal_cb.setChecked(True)

        self.click_start_cb = QCheckBox("Click cell sets start")

        self.block_mode_btn = QPushButton("Create / Remove Blocks")
        self.block_mode_btn.setCheckable(True)
        self.block_mode_btn.setToolTip("Turn this ON, then click grid cells to toggle blocked cells.")

        self.clear_blocks_btn = QPushButton("Clear All Blocks")
        self.demo_blocks_btn = QPushButton("Add Example Blocks")

        self.click_goal_cb.clicked.connect(lambda: self.set_click_mode("goal"))
        self.click_start_cb.clicked.connect(lambda: self.set_click_mode("start"))
        self.block_mode_btn.clicked.connect(self.toggle_block_mode)
        self.clear_blocks_btn.clicked.connect(self.clear_blocks)
        self.demo_blocks_btn.clicked.connect(self.add_example_blocks)

        click_layout.addWidget(self.click_goal_cb)
        click_layout.addWidget(self.click_start_cb)
        click_layout.addWidget(self.block_mode_btn)
        click_layout.addWidget(self.clear_blocks_btn)
        click_layout.addWidget(self.demo_blocks_btn)
        click_layout.addWidget(QLabel("Block mode: click a cell once to block it, click again to remove it."))

        left.addWidget(click_group)

        sim_group = QGroupBox("Simulation Controls")
        sim = QGridLayout(sim_group)

        self.current_tag_edit = QLineEdit()
        self.current_tag_edit.setReadOnly(True)

        self.heading_edit = QLineEdit()
        self.heading_edit.setReadOnly(True)

        self.expected_edit = QLineEdit()
        self.expected_edit.setReadOnly(True)

        self.turn_edit = QLineEdit()
        self.turn_edit.setReadOnly(True)

        self.path_edit = QTextEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setMinimumHeight(120)

        self.speed_spin = QSpinBox()
        self.speed_spin.setRange(100, 5000)
        self.speed_spin.setValue(700)
        self.speed_spin.setSuffix(" ms/step")

        self.start_btn = QPushButton("Start Simulation")
        self.start_btn.clicked.connect(self.start_simulation)

        self.step_btn = QPushButton("Step Once")
        self.step_btn.clicked.connect(self.step_once)

        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.pause_simulation)

        self.reset_btn = QPushButton("Reset to Start")
        self.reset_btn.clicked.connect(self.reset_to_start)

        sim.addWidget(QLabel("Current:"), 0, 0)
        sim.addWidget(self.current_tag_edit, 0, 1)
        sim.addWidget(QLabel("Heading:"), 1, 0)
        sim.addWidget(self.heading_edit, 1, 1)
        sim.addWidget(QLabel("Expected:"), 2, 0)
        sim.addWidget(self.expected_edit, 2, 1)
        sim.addWidget(QLabel("Last turn:"), 3, 0)
        sim.addWidget(self.turn_edit, 3, 1)
        sim.addWidget(QLabel("Speed:"), 4, 0)
        sim.addWidget(self.speed_spin, 4, 1)

        sim.addWidget(self.start_btn, 5, 0)
        sim.addWidget(self.step_btn, 5, 1)
        sim.addWidget(self.pause_btn, 6, 0)
        sim.addWidget(self.reset_btn, 6, 1)

        sim.addWidget(QLabel("Path:"), 7, 0, 1, 2)
        sim.addWidget(self.path_edit, 8, 0, 1, 2)

        right.addWidget(sim_group)

        log_group = QGroupBox("Simulation Log")
        log_layout = QVBoxLayout(log_group)

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(self.log.clear)

        log_layout.addWidget(self.clear_log_btn)
        log_layout.addWidget(self.log)

        right.addWidget(log_group, 1)

    def append_log(self, text):
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {text}")
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def set_click_mode(self, mode):
        self.click_goal_cb.setChecked(mode == "goal")
        self.click_start_cb.setChecked(mode == "start")
        self.block_mode_btn.setChecked(mode == "block")
        self.grid.select_mode = mode

    def toggle_block_mode(self, checked):
        if checked:
            self.set_click_mode("block")
            self.append_log("Block edit mode ON. Click cells to create/remove blocks.")
        else:
            self.set_click_mode("goal")
            self.append_log("Block edit mode OFF. Click cells now set the goal.")

    def clear_blocks(self):
        if not self.blocked:
            self.append_log("No blocks to clear.")
            return

        self.blocked.clear()
        self.visited = [self.current_tag]
        self.append_log("All blocks cleared.")
        self.compute_path()

    def add_example_blocks(self):
        """
        Add a simple obstacle pattern so A* must route around blocked cells.
        This is only for simulator testing; it does not change the tag numbering.
        """
        if self.model.rows < 3 or self.model.cols < 3:
            QMessageBox.information(
                self,
                "Grid too small",
                "Use at least 3 rows and 3 columns for example blocks."
            )
            return

        new_blocks = set()

        # Vertical wall near the middle, with one gap left open.
        wall_col = self.model.cols // 2
        gap_row = self.model.rows - 1

        for r in range(self.model.rows):
            if r == gap_row:
                continue
            tag = self.model.rc_to_tag(r, wall_col)
            if tag != self.current_tag and tag != self.goal_tag:
                new_blocks.add(tag)

        # Add a short horizontal piece to make the path visibly bend more.
        bend_row = max(0, self.model.rows // 2)
        for c in range(1, self.model.cols - 1):
            if c == wall_col:
                continue
            if c % 2 == 0:
                tag = self.model.rc_to_tag(bend_row, c)
                if tag != self.current_tag and tag != self.goal_tag:
                    new_blocks.add(tag)

        self.blocked.update(new_blocks)
        self.blocked.discard(self.current_tag)
        self.blocked.discard(self.goal_tag)
        self.visited = [self.current_tag]
        self.append_log(f"Example blocks added: {sorted(new_blocks)}")
        self.compute_path()

    def rebuild_destination_combo(self):
        self.goal_combo.blockSignals(True)
        self.goal_combo.clear()

        for tag in range(1, self.model.max_tag() + 1):
            self.goal_combo.addItem(f"Tag {tag}", tag)

        self.goal_tag = min(max(1, self.goal_tag), self.model.max_tag())
        self.goal_combo.setCurrentIndex(self.goal_tag - 1)
        self.goal_combo.blockSignals(False)

        try:
            self.goal_combo.currentIndexChanged.disconnect()
        except Exception:
            pass
        self.goal_combo.currentIndexChanged.connect(self.on_goal_combo_changed)

        self.start_tag_spin.setRange(1, self.model.max_tag())
        self.start_tag_spin.setValue(self.current_tag)

    def apply_grid_size(self):
        rows = int(self.rows_spin.value())
        cols = int(self.cols_spin.value())

        self.model = GridModel(rows, cols)

        self.current_tag = min(max(1, self.current_tag), self.model.max_tag())
        self.goal_tag = min(max(1, self.goal_tag), self.model.max_tag())
        self.blocked = {tag for tag in self.blocked if 1 <= tag <= self.model.max_tag()}
        self.blocked.discard(self.current_tag)
        self.blocked.discard(self.goal_tag)

        self.path = []
        self.active_path = []
        self.path_index = 0
        self.expected_next_tag = None
        self.visited = [self.current_tag]
        self.running = False
        self.sim_timer.stop()

        self.grid.set_model(self.model)
        self.rebuild_destination_combo()
        self.compute_path()

        self.append_log(f"Grid changed to {rows} rows × {cols} cols. Tags 1-{self.model.max_tag()}.")
        self.update_ui_state()

    def on_goal_combo_changed(self):
        tag = self.goal_combo.currentData()
        if tag is None:
            return
        self.goal_tag = int(tag)
        self.compute_path()

    def set_start_from_spin(self):
        self.current_tag = int(self.start_tag_spin.value())
        self.blocked.discard(self.current_tag)
        self.visited = [self.current_tag]
        self.active_path = []
        self.path_index = 0
        self.expected_next_tag = None
        self.running = False
        self.sim_timer.stop()
        self.compute_path()

    def on_grid_clicked(self, tag):
        mode = self.grid.select_mode

        if mode == "start":
            if tag in self.blocked:
                self.blocked.remove(tag)
            self.current_tag = tag
            self.start_tag_spin.setValue(tag)
            self.visited = [self.current_tag]
            self.append_log(f"Start set to Tag {tag}")
            self.compute_path()
            return

        if mode == "block":
            if tag == self.current_tag or tag == self.goal_tag:
                self.append_log(f"Tag {tag} cannot be blocked because it is start/goal.")
                return

            self.running = False
            self.sim_timer.stop()
            self.visited = [self.current_tag]
            self.active_path = []
            self.path_index = 0
            self.expected_next_tag = None

            if tag in self.blocked:
                self.blocked.remove(tag)
                self.append_log(f"Removed block at Tag {tag}")
            else:
                self.blocked.add(tag)
                self.append_log(f"Created block at Tag {tag}")

            self.compute_path()
            return

        self.goal_tag = tag
        self.goal_combo.setCurrentIndex(tag - 1)
        self.append_log(f"Goal set to Tag {tag}")
        self.compute_path()

    def compute_path(self):
        self.goal_tag = int(self.goal_combo.currentData() or self.goal_tag)
        self.blocked.discard(self.current_tag)
        self.blocked.discard(self.goal_tag)

        self.path = self.model.astar_path(self.current_tag, self.goal_tag, blocked=self.blocked)

        if not self.path:
            self.path_edit.setText(
                f"No path found from Tag {self.current_tag} to Tag {self.goal_tag}."
            )
            self.append_log(f"No path found: {self.current_tag} → {self.goal_tag}")
        else:
            headings = []
            turns = []

            sim_heading = self.current_heading

            for i in range(len(self.path) - 1):
                a = self.path[i]
                b = self.path[i + 1]
                h = self.model.heading_between_tags(a, b)
                td = turn_delta_deg(sim_heading, h)
                action = turn_action_between_headings(sim_heading, h)
                headings.append(f"{a}->{b}: {HEADING_LABELS.get(h, '---')}")
                turns.append(f"{a}->{b}: {action}, TURN_REL {td:.1f}")
                sim_heading = h

            block_text = ", ".join(str(x) for x in sorted(self.blocked)) if self.blocked else "None"

            self.path_edit.setText(
                "A* path:\n"
                + " → ".join(str(x) for x in self.path)
                + f"\n\nSteps: {len(self.path) - 1}"
                + f"\n\nBlocked tags: {block_text}"
                + "\n\nHeadings:\n"
                + "\n".join(headings)
                + "\n\nTurns:\n"
                + "\n".join(turns)
            )
            self.append_log(f"A* path computed: {' → '.join(str(x) for x in self.path)}")

        self.active_path = list(self.path)
        self.path_index = 0
        self.expected_next_tag = self.active_path[1] if len(self.active_path) > 1 else None
        self.update_ui_state()

    def start_simulation(self):
        if not self.path or len(self.path) < 2:
            self.compute_path()

        if not self.path or len(self.path) < 2:
            QMessageBox.information(self, "No movement", "No valid path to simulate.")
            return

        self.active_path = list(self.path)
        self.path_index = 0
        self.expected_next_tag = self.active_path[1]
        self.running = True
        self.visited = [self.current_tag]

        self.sim_timer.start(int(self.speed_spin.value()))
        self.append_log("Simulation started")
        self.update_ui_state()

    def pause_simulation(self):
        self.running = False
        self.sim_timer.stop()
        self.append_log("Simulation paused")
        self.update_ui_state()

    def reset_to_start(self):
        self.running = False
        self.sim_timer.stop()

        if self.path:
            self.current_tag = self.path[0]
        else:
            self.current_tag = int(self.start_tag_spin.value())

        self.visited = [self.current_tag]
        self.path_index = 0
        self.expected_next_tag = self.path[1] if len(self.path) > 1 else None
        self.turn_edit.setText("---")

        self.append_log("Simulation reset to start")
        self.update_ui_state()

    def step_once(self):
        if not self.path or len(self.path) < 2:
            self.compute_path()

        if not self.active_path:
            self.active_path = list(self.path)
            self.path_index = 0

        self.simulation_tick()

    def simulation_tick(self):
        if not self.active_path or len(self.active_path) < 2:
            self.running = False
            self.sim_timer.stop()
            return

        if self.path_index >= len(self.active_path) - 1:
            self.running = False
            self.sim_timer.stop()
            self.expected_next_tag = None
            self.append_log(f"Simulation complete at Tag {self.current_tag}")
            self.update_ui_state()
            return

        from_tag = self.active_path[self.path_index]
        to_tag = self.active_path[self.path_index + 1]

        desired_heading = self.model.heading_between_tags(from_tag, to_tag)
        td = turn_delta_deg(self.current_heading, desired_heading)
        action = turn_action_between_headings(self.current_heading, desired_heading)

        if abs(td) > 1.0:
            turn_text = f"{action}: TURN_REL {td:.1f}"
        else:
            turn_text = "STRAIGHT: TURN_REL 0.0"

        self.current_tag = to_tag
        self.current_heading = desired_heading
        self.path_index += 1

        if self.current_tag not in self.visited:
            self.visited.append(self.current_tag)

        if self.path_index < len(self.active_path) - 1:
            self.expected_next_tag = self.active_path[self.path_index + 1]
        else:
            self.expected_next_tag = None

        self.turn_edit.setText(turn_text)

        self.append_log(
            f"Move {from_tag} → {to_tag}: heading={HEADING_LABELS.get(desired_heading, '---')} | {turn_text}"
        )

        if self.path_index >= len(self.active_path) - 1:
            self.running = False
            self.sim_timer.stop()
            self.append_log(f"Reached goal Tag {self.current_tag}")

        self.update_ui_state()

    def update_ui_state(self):
        self.current_tag_edit.setText(str(self.current_tag))
        self.heading_edit.setText(HEADING_LABELS.get(self.current_heading, "---"))
        self.expected_edit.setText(str(self.expected_next_tag) if self.expected_next_tag else "---")

        self.grid.set_state(
            current_tag=self.current_tag,
            goal_tag=self.goal_tag,
            expected_tag=self.expected_next_tag,
            path=self.path,
            visited=self.visited,
            blocked=self.blocked,
        )


def main():
    app = QApplication(sys.argv)
    win = AGVGridSimulator()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()