#!/usr/bin/env python3
"""
Qt AGV A* Navigation - 5x3 AprilTag cluster map with Picamera2 feedback.

Design:
  Tag 0      = docking / calibration tag only
  Tags 1-15  = 5x3 grid map used by A*
  Camera     = Raspberry Pi Camera v2 through Picamera2/libcamera
  ESP32      = FireBeetle motor controller using serial commands:
               IMU RECAL, LOCK_HEADING_GO, TURN_REL, STOP,
               SET_BASE, SET_IMU_MAX, STATUS

This file is intentionally self-contained.
"""

import sys
import time
import math
import heapq
from dataclasses import dataclass

import cv2
import numpy as np

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

try:
    import serial
except Exception:
    serial = None


# =====================================================
# MAP CONFIG
# =====================================================

DOCK_TAG = 0
GRID_START_TAG = 1

ROWS = 3
COLS = 5

# Cell distance between adjacent grid tags. Used only for display and timed fallback.
CELL_DISTANCE_M = 0.50

# Real-mode timed fallback for one grid cell if encoder distance is not available.
# Tune this on your robot.
CELL_TRAVEL_SEC = 1.20

# Docking tag 0 to grid tag 1 move time.
DOCK_TO_TAG1_TRAVEL_SEC = 1.20
DOCK_TO_TAG1_TIMEOUT_SEC = 8.0

# Direction labels for display and turn computation.
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


def tag_to_rc(tag_id: int):
    """Convert grid tag 1-15 to zero-based row/col."""
    tag0 = int(tag_id) - 1
    return tag0 // COLS, tag0 % COLS


def rc_to_tag(row: int, col: int):
    """Convert zero-based row/col to grid tag 1-15."""
    return row * COLS + col + 1


def grid_neighbors(tag_id: int):
    r, c = tag_to_rc(tag_id)
    result = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            result.append(rc_to_tag(nr, nc))
    return result


def manhattan(a: int, b: int):
    ar, ac = tag_to_rc(a)
    br, bc = tag_to_rc(b)
    return abs(ar - br) + abs(ac - bc)


def astar_path(start: int, goal: int, blocked=None):
    """Simple A* for tags 1-15."""
    if blocked is None:
        blocked = set()

    start = int(start)
    goal = int(goal)

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

        for nb in grid_neighbors(current):
            if nb in blocked:
                continue

            tentative = g_score[current] + 1

            if tentative < g_score.get(nb, 10**9):
                came_from[nb] = current
                g_score[nb] = tentative
                f_score = tentative + manhattan(nb, goal)
                heapq.heappush(open_heap, (f_score, nb))

    return []


def heading_between_tags(a: int, b: int):
    ar, ac = tag_to_rc(a)
    br, bc = tag_to_rc(b)

    if br == ar - 1 and bc == ac:
        return NORTH
    if br == ar + 1 and bc == ac:
        return SOUTH
    if br == ar and bc == ac + 1:
        return EAST
    if br == ar and bc == ac - 1:
        return WEST

    return None


def turn_delta_deg(current_heading, desired_heading):
    """Return shortest relative turn in degrees. Positive means right/clockwise if ESP32 sign is correct."""
    if current_heading is None or desired_heading is None:
        return 0.0

    # N=0,E=1,S=2,W=3. +1 step = right turn 90 deg.
    delta_steps = (desired_heading - current_heading) % 4
    if delta_steps == 0:
        return 0.0
    if delta_steps == 1:
        return 90.0
    if delta_steps == 2:
        return 180.0
    if delta_steps == 3:
        return -90.0
    return 0.0


# =====================================================
# APRILTAG / CAMERA CONFIG
# =====================================================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

FX = 615.0
FY = 615.0
CX = FRAME_WIDTH / 2.0
CY = FRAME_HEIGHT / 2.0
CAMERA_PARAMS = (FX, FY, CX, CY)

# Central grid tags may be normal larger tags. Helper clusters are optional.
TAG_SIZE_M = 0.010
CLUSTER_SPACING_M = 0.015

# Helper grid:
#
#   508   501   502
#   507   CEN   503
#   506   505   504
#
HELPER_GRID_OFFSET = {
    501: (0, -1),
    502: (1, -1),
    503: (1, 0),
    504: (1, 1),
    505: (0, 1),
    506: (-1, 1),
    507: (-1, 0),
    508: (-1, -1),
}
HELPER_IDS = set(HELPER_GRID_OFFSET.keys())


def normalize_angle(angle):
    return ((angle + 180.0) % 360.0) - 180.0


def compute_yaw_deg_from_corners(corners, center):
    top_mid_x = (corners[0][0] + corners[1][0]) / 2.0
    top_mid_y = (corners[0][1] + corners[1][1]) / 2.0

    dx = top_mid_x - center[0]
    dy = center[1] - top_mid_y

    return math.degrees(math.atan2(dx, dy))


@dataclass
class TagDetection:
    tag_id: int
    center: tuple
    corners: np.ndarray
    yaw_deg: float
    x_m: float


# =====================================================
# CAMERA THREAD: uses the same working Picamera2 style
# from your previous AGV code.
# =====================================================

class AprilTagCameraThread(QThread):
    frame_ready = pyqtSignal(object)
    detections_ready = pyqtSignal(object)
    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = False
        self.picam2 = None
        self.detector = None

    def stop(self):
        self.running = False
        self.wait(2500)

    def run(self):
        self.running = True

        try:
            from picamera2 import Picamera2
            from pupil_apriltags import Detector
        except Exception as e:
            self.status.emit(f"Import failed: {e}")
            return

        try:
            self.picam2 = Picamera2()

            config = self.picam2.create_preview_configuration(
                main={
                    "size": (FRAME_WIDTH, FRAME_HEIGHT),
                    "format": "RGB888",
                }
            )

            self.picam2.configure(config)

            self.picam2.set_controls({
                "AeEnable": False,
                "AwbEnable": False,
                "ExposureTime": 5000,
                "AnalogueGain": 1.0,
            })

            self.picam2.start()

            self.detector = Detector(
                families="tag36h11",
                nthreads=4,
                quad_decimate=2.0,
                refine_edges=1,
            )

            self.status.emit("Camera running: Picamera2/libcamera")

        except Exception as e:
            self.status.emit(f"Picamera2 start failed: {e}")
            self.cleanup()
            return

        while self.running:
            try:
                frame_rgb = self.picam2.capture_array()
                if frame_rgb is None:
                    self.status.emit("Camera frame is None")
                    self.msleep(50)
                    continue

                gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

                raw_detections = self.detector.detect(
                    gray,
                    estimate_tag_pose=True,
                    camera_params=CAMERA_PARAMS,
                    tag_size=TAG_SIZE_M,
                )

                detections = []
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                for tag in raw_detections:
                    tag_id = int(tag.tag_id)
                    corners = tag.corners.astype(int)
                    center = (int(tag.center[0]), int(tag.center[1]))

                    yaw = compute_yaw_deg_from_corners(tag.corners, tag.center)
                    x_m = 0.0
                    if getattr(tag, "pose_t", None) is not None:
                        x_m = float(tag.pose_t[0][0])

                    detections.append(
                        TagDetection(
                            tag_id=tag_id,
                            center=center,
                            corners=corners,
                            yaw_deg=float(yaw),
                            x_m=float(x_m),
                        )
                    )

                    for i in range(4):
                        p1 = tuple(corners[i])
                        p2 = tuple(corners[(i + 1) % 4])
                        cv2.line(frame_bgr, p1, p2, (0, 255, 0), 2)

                    cv2.circle(frame_bgr, center, 5, (0, 0, 255), -1)
                    cv2.putText(
                        frame_bgr,
                        f"ID:{tag_id}",
                        (center[0] + 10, center[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame_bgr,
                        f"Yaw:{yaw:.1f}",
                        (center[0] + 10, center[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame_bgr,
                        f"xM:{x_m:.3f}",
                        (center[0] + 10, center[1] + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 0, 255),
                        2,
                    )

                # Draw center cross.
                cv2.line(frame_bgr, (int(CX) - 30, int(CY)), (int(CX) + 30, int(CY)), (255, 255, 255), 2)
                cv2.line(frame_bgr, (int(CX), int(CY) - 30), (int(CX), int(CY) + 30), (255, 255, 255), 2)

                ids = [d.tag_id for d in detections]
                if ids:
                    self.status.emit("Detected: " + ", ".join(str(x) for x in ids))
                else:
                    self.status.emit("No tag detected")

                self.detections_ready.emit(detections)
                self.frame_ready.emit(frame_bgr)

            except Exception as e:
                self.status.emit(f"Camera loop error: {e}")
                self.msleep(100)

            self.msleep(30)

        self.cleanup()
        self.status.emit("Camera stopped")

    def cleanup(self):
        try:
            if self.picam2 is not None:
                self.picam2.stop()
        except Exception:
            pass
        self.picam2 = None


# =====================================================
# GRID DISPLAY
# =====================================================

class TagGridWidget(QWidget):
    tag_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.current_tag = GRID_START_TAG
        self.goal_tag = None
        self.path = []
        self.expected_tag = None
        self.blocked = set()

        self.setMinimumSize(600, 360)

    def set_state(self, current_tag=None, goal_tag=None, path=None, expected_tag=None):
        if current_tag is not None:
            self.current_tag = current_tag
        self.goal_tag = goal_tag
        self.path = path or []
        self.expected_tag = expected_tag
        self.update()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return

        margin = 20
        w = max(1, self.width() - 2 * margin)
        h = max(1, self.height() - 2 * margin)
        cell_w = w / COLS
        cell_h = h / ROWS

        col = int((event.x() - margin) / cell_w)
        row = int((event.y() - margin) / cell_h)

        if 0 <= row < ROWS and 0 <= col < COLS:
            self.tag_clicked.emit(rc_to_tag(row, col))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        margin = 20
        w = self.width() - 2 * margin
        h = self.height() - 2 * margin
        cell_w = w / COLS
        cell_h = h / ROWS

        painter.fillRect(self.rect(), QColor(60, 64, 66))

        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QPen(QColor(230, 230, 230)))
        painter.drawText(20, 15, "5 x 3 Tag Grid: Tags 1-15   |   Tag 0 = Docking only")

        path_edges = set()
        for i in range(len(self.path) - 1):
            path_edges.add((self.path[i], self.path[i + 1]))
            path_edges.add((self.path[i + 1], self.path[i]))

        for r in range(ROWS):
            for c in range(COLS):
                tag = rc_to_tag(r, c)

                x = margin + c * cell_w
                y = margin + r * cell_h

                fill = QColor(245, 245, 245)
                border = QColor(80, 80, 80)

                if tag in self.path:
                    fill = QColor(210, 230, 255)

                if tag == self.goal_tag:
                    fill = QColor(255, 230, 150)

                if tag == self.expected_tag:
                    fill = QColor(255, 200, 120)

                if tag == self.current_tag:
                    fill = QColor(180, 245, 190)
                    border = QColor(20, 150, 70)

                painter.setPen(QPen(border, 2))
                painter.setBrush(fill)
                painter.drawRect(int(x), int(y), int(cell_w), int(cell_h))

                font = QFont()
                font.setPointSize(18)
                font.setBold(True)
                painter.setFont(font)
                painter.setPen(QPen(QColor(80, 80, 80)))
                painter.drawText(
                    int(x),
                    int(y),
                    int(cell_w),
                    int(cell_h),
                    Qt.AlignCenter,
                    str(tag),
                )

        # Draw path line.
        if len(self.path) >= 2:
            painter.setPen(QPen(QColor(40, 120, 220), 5))
            for i in range(len(self.path) - 1):
                a = self.path[i]
                b = self.path[i + 1]
                ar, ac = tag_to_rc(a)
                br, bc = tag_to_rc(b)
                ax = margin + ac * cell_w + cell_w / 2
                ay = margin + ar * cell_h + cell_h / 2
                bx = margin + bc * cell_w + cell_w / 2
                by = margin + br * cell_h + cell_h / 2
                painter.drawLine(int(ax), int(ay), int(bx), int(by))


# =====================================================
# MAIN APP
# =====================================================

class AGVQtApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("AGV Qt A-Star Closed Loop - Tag Cluster")
        self.resize(1450, 900)

        # Calibration flags.
        self.calibration_started = False
        self.dock_tag_confirmed = False
        self.waiting_for_manual_alignment = False
        self.calibration_done = False
        self.dock_move_active = False

        # Mission state.
        self.current_tag = GRID_START_TAG
        self.goal_tag = None
        self.path = []
        self.active_path = []
        self.path_index = 0
        self.expected_next_tag = None
        self.current_heading = SOUTH
        self.mission_running = False

        # Camera state.
        self.latest_detections = []
        self.latest_ids = []

        # Serial state.
        self.ser = None

        self.camera_thread = None

        self.build_ui()
        self.update_ui_state()

        QTimer.singleShot(300, self.start_camera_feedback)

    # -------------------------
    # UI
    # -------------------------

    def build_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        left = QVBoxLayout()
        right = QVBoxLayout()
        main.addLayout(left, 3)
        main.addLayout(right, 2)

        self.grid = TagGridWidget()
        self.grid.tag_clicked.connect(self.on_grid_tag_clicked)
        grid_group = QGroupBox("5 x 3 Tag Grid")
        grid_layout = QVBoxLayout(grid_group)
        grid_layout.addWidget(self.grid)
        left.addWidget(grid_group, 3)

        camera_group = QGroupBox("Live Camera Feedback")
        camera_layout = QVBoxLayout(camera_group)
        self.camera_label = QLabel("Camera not started")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 360)
        self.camera_label.setStyleSheet("background:#111;color:white;border:1px solid #555;")
        self.camera_status = QLabel("Camera status: idle")
        self.camera_status.setStyleSheet("font-weight:bold;")
        camera_layout.addWidget(self.camera_label)
        camera_layout.addWidget(self.camera_status)
        left.addWidget(camera_group, 2)

        # Calibration group.
        calib_group = QGroupBox("Docking Calibration")
        calib = QGridLayout(calib_group)

        self.calib_state = QLineEdit()
        self.calib_state.setReadOnly(True)
        self.calib_state.setText("WAITING FOR TAG 0")

        self.start_calib_btn = QPushButton("Start Calibration")
        self.start_calib_btn.clicked.connect(self.start_calibration)

        self.reset_calib_btn = QPushButton("Reset Calibration")
        self.reset_calib_btn.clicked.connect(self.reset_calibration)

        calib.addWidget(QLabel("State:"), 0, 0)
        calib.addWidget(self.calib_state, 0, 1, 1, 2)
        calib.addWidget(self.start_calib_btn, 1, 0)
        calib.addWidget(self.reset_calib_btn, 1, 1)
        calib.addWidget(QLabel("After Tag 0 is detected and robot is manually aligned, press S."), 2, 0, 1, 3)

        right.addWidget(calib_group)

        # Mission group.
        mission_group = QGroupBox("Mission Control")
        mission = QGridLayout(mission_group)

        self.current_tag_edit = QLineEdit()
        self.current_tag_edit.setReadOnly(True)

        self.goal_tag_edit = QLineEdit()
        self.goal_tag_edit.setReadOnly(True)

        self.heading_edit = QLineEdit()
        self.heading_edit.setReadOnly(True)

        self.expected_next_edit = QLineEdit()
        self.expected_next_edit.setReadOnly(True)

        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)

        self.simulation_checkbox = QCheckBox("Simulation mode")
        self.simulation_checkbox.setChecked(True)

        self.closed_loop_checkbox = QCheckBox("Closed-loop camera feedback")
        self.closed_loop_checkbox.setChecked(True)

        self.connect_btn = QPushButton("Connect ESP32")
        self.connect_btn.clicked.connect(self.connect_esp32)

        self.status_btn = QPushButton("ESP32 Status")
        self.status_btn.clicked.connect(lambda: self.send_esp32("STATUS"))

        self.start_btn = QPushButton("Start Mission")
        self.start_btn.clicked.connect(self.start_mission)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_mission)

        mission.addWidget(QLabel("Current Tag:"), 0, 0)
        mission.addWidget(self.current_tag_edit, 0, 1, 1, 2)
        mission.addWidget(QLabel("Goal Tag:"), 1, 0)
        mission.addWidget(self.goal_tag_edit, 1, 1, 1, 2)
        mission.addWidget(QLabel("Heading:"), 2, 0)
        mission.addWidget(self.heading_edit, 2, 1, 1, 2)
        mission.addWidget(QLabel("Expected Next:"), 3, 0)
        mission.addWidget(self.expected_next_edit, 3, 1, 1, 2)
        mission.addWidget(QLabel("Path:"), 4, 0)
        mission.addWidget(self.path_edit, 4, 1, 1, 2)
        mission.addWidget(self.simulation_checkbox, 5, 0)
        mission.addWidget(self.closed_loop_checkbox, 5, 1)
        mission.addWidget(self.connect_btn, 6, 0)
        mission.addWidget(self.status_btn, 6, 1)
        mission.addWidget(self.start_btn, 7, 0)
        mission.addWidget(self.stop_btn, 7, 1)

        right.addWidget(mission_group)

        # Destination group.
        dest_group = QGroupBox("Destination / A-Star Path")
        dest = QVBoxLayout(dest_group)

        self.destination_combo = QComboBox()
        for tag in range(1, ROWS * COLS + 1):
            self.destination_combo.addItem(f"Tag {tag}", tag)

        self.compute_btn = QPushButton("Compute A-Star Path")
        self.compute_btn.clicked.connect(self.compute_path_from_combo)

        self.path_text = QTextEdit()
        self.path_text.setReadOnly(True)
        self.path_text.setMinimumHeight(110)

        dest.addWidget(QLabel("Select destination:"))
        dest.addWidget(self.destination_combo)
        dest.addWidget(self.compute_btn)
        dest.addWidget(self.path_text)

        right.addWidget(dest_group)

        # Log.
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(180)
        log_layout.addWidget(self.log)
        right.addWidget(log_group, 1)

    def append_log(self, text):
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {text}")

    def update_ui_state(self):
        self.current_tag_edit.setText(str(self.current_tag) if self.current_tag is not None else "---")
        self.goal_tag_edit.setText(str(self.goal_tag) if self.goal_tag is not None else "---")
        self.heading_edit.setText(HEADING_LABELS.get(self.current_heading, "---"))
        self.expected_next_edit.setText(str(self.expected_next_tag) if self.expected_next_tag is not None else "---")
        self.path_edit.setText(" → ".join(str(x) for x in self.path) if self.path else "---")

        if self.calibration_done:
            self.calib_state.setText("DONE - GRID START TAG 1")
        elif self.dock_move_active:
            self.calib_state.setText("MOVING 0 -> 1, WAITING FOR TAG 1")
        elif self.waiting_for_manual_alignment:
            self.calib_state.setText("TAG 0 DETECTED - PRESS S")
        elif self.calibration_started:
            self.calib_state.setText("WAITING FOR TAG 0")
        else:
            self.calib_state.setText("NOT STARTED")

        self.grid.set_state(
            current_tag=self.current_tag,
            goal_tag=self.goal_tag,
            path=self.path,
            expected_tag=self.expected_next_tag,
        )

    # -------------------------
    # Camera
    # -------------------------

    def start_camera_feedback(self):
        if self.camera_thread is not None and self.camera_thread.isRunning():
            return

        self.camera_thread = AprilTagCameraThread(self)
        self.camera_thread.frame_ready.connect(self.update_camera_frame)
        self.camera_thread.detections_ready.connect(self.on_detections_ready)
        self.camera_thread.status.connect(self.on_camera_status)
        self.camera_thread.start()
        self.append_log("Camera feedback started")

    def stop_camera_feedback(self):
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread = None

    def update_camera_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        self.camera_label.setPixmap(
            pix.scaled(
                self.camera_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def on_camera_status(self, text):
        self.camera_status.setText(f"Camera status: {text}")

    def on_detections_ready(self, detections):
        self.latest_detections = detections
        self.latest_ids = [d.tag_id for d in detections]

        # Closed-loop docking movement: after IMU calibration and LOCK_HEADING_GO,
        # do not declare calibration done until the camera actually sees Tag 1.
        if self.dock_move_active:
            if GRID_START_TAG in self.latest_ids:
                self.append_log("Dock-to-grid feedback OK: Tag 1 detected")
                self.finish_dock_to_tag1()
            return

        # Calibration feedback.
        if self.calibration_started and not self.calibration_done:
            if DOCK_TAG in self.latest_ids:
                if not self.dock_tag_confirmed:
                    self.append_log("Docking Tag 0 detected. Manually align robot, then press S.")
                self.dock_tag_confirmed = True
                self.waiting_for_manual_alignment = True
                self.update_ui_state()

        # Closed-loop mission feedback.
        if (
            self.mission_running
            and self.closed_loop_checkbox.isChecked()
            and self.expected_next_tag is not None
        ):
            if self.expected_next_tag in self.latest_ids:
                self.append_log(f"Closed-loop feedback OK: Tag {self.expected_next_tag} detected")
                self.advance_to_detected_tag(self.expected_next_tag)
            elif len(self.latest_ids) > 0:
                # Do not spam too hard.
                pass

    # -------------------------
    # Serial
    # -------------------------

    def connect_esp32(self):
        if self.simulation_checkbox.isChecked():
            self.append_log("Simulation mode ON: ESP32 connection not required")
            return True

        if serial is None:
            QMessageBox.critical(self, "Serial missing", "pyserial is not installed. Run: pip install pyserial")
            return False

        if self.ser is not None and self.ser.is_open:
            self.append_log("ESP32 already connected")
            return True

        try:
            self.ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=0.2)
            time.sleep(2)
            self.append_log("ESP32 connected on /dev/ttyUSB0")
            return True
        except Exception as e:
            QMessageBox.critical(self, "ESP32 connection failed", str(e))
            self.append_log(f"ESP32 connection failed: {e}")
            return False

    def send_esp32(self, cmd):
        if self.simulation_checkbox.isChecked():
            self.append_log(f"SIM ESP32 <= {cmd}")
            return

        if not self.connect_esp32():
            return

        try:
            self.ser.write((cmd + "\n").encode())
            self.append_log(f"ESP32 <= {cmd}")
            QTimer.singleShot(100, self.read_esp32_available)
        except Exception as e:
            self.append_log(f"Serial send failed: {e}")

    def read_esp32_available(self):
        if self.ser is None:
            return
        try:
            while self.ser.in_waiting > 0:
                line = self.ser.readline().decode(errors="ignore").strip()
                if line:
                    self.append_log(f"ESP32: {line}")
        except Exception as e:
            self.append_log(f"Serial read failed: {e}")

    def wait_for_esp32_text(self, expected_text, timeout_sec=10.0):
        if self.simulation_checkbox.isChecked():
            return True

        if self.ser is None:
            return False

        start = time.time()
        while time.time() - start < timeout_sec:
            QApplication.processEvents()
            try:
                line = self.ser.readline().decode(errors="ignore").strip()
            except Exception:
                line = ""

            if line:
                self.append_log(f"ESP32: {line}")

            if expected_text in line:
                return True

            time.sleep(0.03)

        return False

    # -------------------------
    # Calibration
    # -------------------------

    def start_calibration(self):
        self.calibration_started = True
        self.dock_tag_confirmed = False
        self.waiting_for_manual_alignment = False
        self.calibration_done = False
        self.current_tag = None
        self.expected_next_tag = None
        self.path = []
        self.active_path = []
        self.mission_running = False
        self.append_log("Calibration started. Place robot at docking Tag 0.")
        self.append_log("Waiting for camera detection of Tag 0...")
        self.update_ui_state()

    def reset_calibration(self):
        self.calibration_started = False
        self.dock_tag_confirmed = False
        self.waiting_for_manual_alignment = False
        self.calibration_done = False
        self.dock_move_active = False
        self.current_tag = GRID_START_TAG
        self.goal_tag = None
        self.path = []
        self.active_path = []
        self.path_index = 0
        self.expected_next_tag = None
        self.current_heading = SOUTH
        self.mission_running = False
        self.send_esp32("STOP")
        self.append_log("Calibration reset")
        self.update_ui_state()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S:
            self.confirm_manual_alignment()
            return
        super().keyPressEvent(event)

    def confirm_manual_alignment(self):
        if not self.calibration_started:
            self.append_log("S ignored: calibration not started")
            return

        if not self.dock_tag_confirmed:
            self.append_log("S ignored: docking Tag 0 not detected yet")
            return

        if not self.waiting_for_manual_alignment:
            self.append_log("S ignored: not waiting for alignment")
            return

        self.append_log("Manual alignment confirmed by S key")

        if self.simulation_checkbox.isChecked():
            self.append_log("Simulation calibration only: no ESP32 motion will be sent.")
            self.append_log("For the real robot to move 0 -> 1, uncheck Simulation mode before pressing S.")
            self.finish_dock_to_tag1()
            return

        if not self.connect_esp32():
            return

        self.send_esp32("STOP")
        time.sleep(0.2)

        try:
            self.ser.reset_input_buffer()
        except Exception:
            pass

        self.send_esp32("SET_BASE 6500")
        self.send_esp32("SET_IMU_MAX 350")
        self.send_esp32("IMU RECAL")

        ok = self.wait_for_esp32_text("OK IMU RECAL", timeout_sec=10.0)

        if not ok:
            QMessageBox.warning(self, "Calibration failed", "ESP32 did not confirm OK IMU RECAL")
            self.append_log("Calibration failed: no OK IMU RECAL")
            return

        self.append_log("IMU calibrated. Starting real dock move 0 -> 1.")
        self.append_log("Calibration will complete only after camera detects Tag 1.")

        self.dock_move_active = True
        self.current_tag = DOCK_TAG
        self.expected_next_tag = GRID_START_TAG
        self.update_ui_state()

        self.send_esp32("LOCK_HEADING_GO")
        QTimer.singleShot(int(DOCK_TO_TAG1_TIMEOUT_SEC * 1000), self.check_dock_to_tag1_timeout)

    def check_dock_to_tag1_timeout(self):
        if not self.dock_move_active:
            return
        self.append_log("Dock-to-Tag-1 timeout: Tag 1 was not detected. Stopping robot.")
        self.send_esp32("STOP")
        QMessageBox.warning(
            self,
            "Dock move timeout",
            "Robot did not detect Tag 1 after leaving docking Tag 0. Check serial motion, direction, and camera view."
        )
        self.dock_move_active = False
        self.expected_next_tag = None
        self.update_ui_state()

    def finish_dock_to_tag1(self):
        self.send_esp32("STOP")

        self.dock_move_active = False
        self.current_tag = GRID_START_TAG
        self.current_heading = SOUTH
        self.expected_next_tag = None
        self.calibration_done = True
        self.calibration_started = False
        self.waiting_for_manual_alignment = False
        self.dock_tag_confirmed = True

        self.append_log("Calibration done. Robot is now at grid Tag 1.")
        self.append_log("Select destination tag and compute A-Star path.")
        self.update_ui_state()

    # -------------------------
    # Path planning
    # -------------------------

    def on_grid_tag_clicked(self, tag_id):
        self.goal_tag = int(tag_id)
        self.destination_combo.setCurrentIndex(self.goal_tag - 1)
        self.compute_path()
        self.update_ui_state()

    def compute_path_from_combo(self):
        self.goal_tag = int(self.destination_combo.currentData())
        self.compute_path()

    def compute_path(self):
        if not self.calibration_done:
            self.append_log("Cannot compute path: calibration not done")
            QMessageBox.information(self, "Calibration required", "Start calibration at Tag 0 first, then press S after alignment.")
            return

        if self.current_tag is None:
            self.append_log("Cannot compute path: current tag unknown")
            return

        if self.goal_tag is None:
            self.append_log("Select destination first")
            return

        self.path = astar_path(self.current_tag, self.goal_tag)
        if not self.path:
            self.path_text.setText("No path found")
            self.append_log(f"No path found from {self.current_tag} to {self.goal_tag}")
            return

        self.path_text.setText(
            f"A* path:\n{' → '.join(str(x) for x in self.path)}\n\n"
            f"Cells: {len(self.path) - 1}\n"
            f"Distance estimate: {(len(self.path) - 1) * CELL_DISTANCE_M:.2f} m"
        )
        self.append_log(f"A* path computed: {' → '.join(str(x) for x in self.path)}")
        self.update_ui_state()

    # -------------------------
    # Mission
    # -------------------------

    def start_mission(self):
        if not self.calibration_done:
            QMessageBox.warning(self, "Calibration required", "Calibrate at docking Tag 0 first.")
            return

        if not self.path:
            self.compute_path()

        if not self.path or len(self.path) < 2:
            self.append_log("Mission not started: no movement required")
            return

        if not self.simulation_checkbox.isChecked():
            if not self.connect_esp32():
                return
            self.send_esp32("SET_BASE 6500")
            self.send_esp32("SET_IMU_MAX 350")

        self.active_path = list(self.path)
        self.path_index = 1
        self.expected_next_tag = self.active_path[self.path_index]
        self.mission_running = True

        self.append_log(f"Mission started. Expected next tag: {self.expected_next_tag}")
        self.update_ui_state()
        self.command_next_segment()

    def stop_mission(self):
        self.mission_running = False
        self.expected_next_tag = None
        self.send_esp32("STOP")
        self.append_log("Mission stopped")
        self.update_ui_state()

    def command_next_segment(self):
        if not self.mission_running:
            return

        if self.path_index >= len(self.active_path):
            self.finish_mission()
            return

        from_tag = self.current_tag
        to_tag = self.active_path[self.path_index]

        desired_heading = heading_between_tags(from_tag, to_tag)
        turn_deg = turn_delta_deg(self.current_heading, desired_heading)

        self.append_log(
            f"Segment {from_tag} → {to_tag}, heading {HEADING_LABELS.get(self.current_heading)} "
            f"to {HEADING_LABELS.get(desired_heading)}, turn {turn_deg:.1f} deg"
        )

        self.current_heading = desired_heading if desired_heading is not None else self.current_heading

        if self.simulation_checkbox.isChecked():
            # In closed-loop simulation, do not auto-move. Wait for camera tag.
            if not self.closed_loop_checkbox.isChecked():
                QTimer.singleShot(int(CELL_TRAVEL_SEC * 1000), lambda: self.advance_to_detected_tag(to_tag))
            return

        # Real robot command sequence.
        # If turn is needed, ask ESP32 to turn. Then start IMU straight.
        if abs(turn_deg) > 1.0:
            self.send_esp32(f"TURN_REL {turn_deg:.1f}")
            QTimer.singleShot(2200, self.send_lock_heading_go)
        else:
            self.send_lock_heading_go()

    def send_lock_heading_go(self):
        if self.mission_running:
            self.send_esp32("LOCK_HEADING_GO")

    def advance_to_detected_tag(self, tag_id):
        if not self.mission_running:
            return

        if self.expected_next_tag != tag_id:
            return

        # Stop real robot as soon as expected central tag is seen.
        self.send_esp32("STOP")

        self.current_tag = tag_id
        self.path_index += 1

        self.append_log(f"Arrived at Tag {tag_id}")

        if self.path_index >= len(self.active_path):
            self.finish_mission()
            return

        self.expected_next_tag = self.active_path[self.path_index]
        self.update_ui_state()

        # Small delay before next segment.
        QTimer.singleShot(400, self.command_next_segment)

    def finish_mission(self):
        self.mission_running = False
        self.expected_next_tag = None
        self.send_esp32("STOP")
        self.append_log("Mission complete")
        QMessageBox.information(self, "Mission complete", "AGV reached the destination tag.")
        self.update_ui_state()

    # -------------------------
    # Cleanup
    # -------------------------

    def closeEvent(self, event):
        try:
            self.mission_running = False
            self.send_esp32("STOP")
        except Exception:
            pass

        try:
            self.stop_camera_feedback()
        except Exception as e:
            print(f"Camera cleanup error: {e}")

        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

        event.accept()


def main():
    app = QApplication(sys.argv)
    win = AGVQtApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
