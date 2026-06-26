#!/usr/bin/env python3
"""
Qt AGV A* Navigation with old cluster vision behavior.

Key behavior:
  - Tag 0 is docking/calibration only.
  - Tags 1-15 are the 5x3 A* grid, each adjacent tag distance is 50 cm.
  - Calibration:
      Place robot at Tag 0, click Start Calibration, wait for Tag 0, align manually, press S.
      Qt sends IMU RECAL, then moves from Tag 0 to Tag 1 using the same old cluster camera logic.
      Calibration becomes DONE only when Tag 1 is detected/reached.
  - Mission:
      User selects destination.
      A* creates a path from current tag to destination.
      Every segment uses old behavior:
        camera cluster correction -> VEL left right
        if cluster lost -> LOCK_HEADING_GO
        when next landmark central tag is seen -> STOP/update state
      Turns between grid segments are done using ESP32 TURN_REL.
  - ESP32 protocol expected:
      VEL left_pps right_pps
      LOCK_HEADING_GO
      TURN_REL deg
      IMU RECAL
      STOP
      STATUS
      SET_BASE value
      SET_IMU_MAX value
      SET_IMU_KP value
      SET_RAMP accel decel
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
    QDoubleSpinBox,
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

try:
    import serial
except Exception:
    serial = None


# =====================================================
# MAP / ROUTING
# =====================================================

DOCK_TAG = 0
GRID_START_TAG = 1

ROWS = 3
COLS = 5

CELL_DISTANCE_M = 0.50

# Fixed ESP32 tuning defaults.
# These are kept as normal Python values so they cannot be deleted by Qt layout cleanup.
DEFAULT_BASE_PPS = 6500
DEFAULT_IMU_MAX_CORR = 350
DEFAULT_IMU_KP = 70.0
DEFAULT_ACCEL_PPS_PER_SEC = 7000
DEFAULT_DECEL_PPS_PER_SEC = 9000

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
    tag0 = int(tag_id) - 1
    return tag0 // COLS, tag0 % COLS


def rc_to_tag(row: int, col: int):
    return row * COLS + col + 1


def grid_neighbors(tag_id: int):
    r, c = tag_to_rc(tag_id)
    out = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < ROWS and 0 <= nc < COLS:
            out.append(rc_to_tag(nr, nc))
    return out


def manhattan(a: int, b: int):
    ar, ac = tag_to_rc(a)
    br, bc = tag_to_rc(b)
    return abs(ar - br) + abs(ac - bc)


def astar_path(start: int, goal: int, blocked=None):
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
                f = tentative + manhattan(nb, goal)
                heapq.heappush(open_heap, (f, nb))

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
    """
    Return TURN_REL angle for the ESP32.

    Physical convention used for this AGV/grid:
      1 -> 2 is +90 degrees.

    Therefore the sign is intentionally opposite of the
    previous screen-coordinate convention.
    """
    if current_heading is None or desired_heading is None:
        return 0.0

    delta_steps = (desired_heading - current_heading) % 4

    if delta_steps == 0:
        return 0.0

    # Desired heading is one clockwise grid step from current heading.
    # For this robot/IMU setup that must be sent as -90.
    if delta_steps == 1:
        return -90.0

    if delta_steps == 2:
        return 180.0

    # Desired heading is one counter-clockwise grid step from current heading.
    # Example: current SOUTH -> desired EAST for 1 -> 2 after docking.
    # This must be +90 for the physical AGV.
    if delta_steps == 3:
        return 90.0

    return 0.0


# =====================================================
# OLD VISION / CLUSTER PARAMETERS
# =====================================================

MAX_PPS = 12000

VISION_BASE_PPS = 6500
VISION_BASE_PPS_SLOW = 5000
VISION_MIN_PPS = 3000
VISION_MAX_PPS = 7500

LOCAL_NUDGE_PPS = 500
TURN_TAG_FB_PPS = 550

FB_SIGN = 1

TAG_SIZE_M = 0.010
CLUSTER_SPACING_M = 0.015

EXPECTED_TAG_YAW_DEG = 0.0

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
CROSS_HELPERS = {501, 503, 505, 507}
CORNER_HELPERS = {502, 504, 506, 508}

CLUSTER_LOST_FRAMES_REQUIRED = 5
TARGET_CENTRAL_SEEN_FRAMES_REQUIRED = 2

LOCAL_NUDGE_CENTER_Y_OK_PX = 30
LOCAL_NUDGE_GOOD_FRAMES_REQUIRED = 3
LOCAL_NUDGE_TIMEOUT_SEC = 5.0

# Accept local helper tags as arrival evidence; central tag is not required.
LOCAL_HELPER_SEEN_FRAMES_REQUIRED = 1


KP_YAW_PPS_PER_DEG = 18
KP_X_PPS_PER_M = 16000

KP_YAW_STRONG_PPS_PER_DEG = 28
KP_X_STRONG_PPS_PER_M = 45000

X_SIGN = -1.0

YAW_DEADBAND_DEG = 0.30
X_DEADBAND_M = 0.0005

X_MEDIUM_ERROR_M = 0.008
X_LARGE_ERROR_M = 0.018

YAW_MEDIUM_ERROR_DEG = 3.0
YAW_LARGE_ERROR_DEG = 7.0

MAX_VISION_CORRECTION_PPS = 220
MAX_VISION_CORRECTION_STRONG_PPS = 500

CORRECTION_FILTER_ALPHA = 0.30
CORRECTION_FILTER_ALPHA_STRONG = 0.50

KP_TURN_TAG_YAW_PPS_PER_DEG = 20
MAX_TURN_TAG_YAW_CORRECTION_PPS = 150

TURN_TAG_YAW_OK_DEG = 3.0
TURN_TAG_CENTER_Y_OK_PX = 25
TURN_TAG_GOOD_FRAMES_REQUIRED = 3


# =====================================================
# CAMERA PARAMETERS
# =====================================================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

FX = 615.0
FY = 615.0
CX = FRAME_WIDTH / 2.0
CY = FRAME_HEIGHT / 2.0

CAMERA_PARAMS = (FX, FY, CX, CY)


# =====================================================
# TAG MATH
# =====================================================

def normalize_angle(angle):
    return ((angle + 180.0) % 360.0) - 180.0


@dataclass
class TagDetection:
    tag_id: int
    center: np.ndarray
    corners: np.ndarray
    pose_t: object


def compute_yaw_deg(tag):
    corners = tag.corners
    cx = tag.center[0]
    cy = tag.center[1]

    top_mid_x = (corners[0][0] + corners[1][0]) / 2.0
    top_mid_y = (corners[0][1] + corners[1][1]) / 2.0

    dx = top_mid_x - cx
    dy = cy - top_mid_y

    return math.degrees(math.atan2(dx, dy))


def compute_lateral_x_m(tag):
    if tag.pose_t is None:
        return 0.0
    return float(tag.pose_t[0][0])


def estimate_tag_side_px(tag):
    corners = tag.corners
    side_lengths = []

    for i in range(4):
        p1 = corners[i]
        p2 = corners[(i + 1) % 4]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        side_lengths.append(math.sqrt(dx * dx + dy * dy))

    return float(sum(side_lengths) / len(side_lengths))


def visible_ids(detections):
    return {int(tag.tag_id) for tag in detections}


def find_tag(detections, tag_id):
    for tag in detections:
        if int(tag.tag_id) == int(tag_id):
            return tag
    return None


def any_helper_visible(detections):
    ids = visible_ids(detections)
    return len(ids.intersection(HELPER_IDS)) > 0


def any_cluster_visible_now(detections, central_tag_id):
    if find_tag(detections, central_tag_id) is not None:
        return True
    return any_helper_visible(detections)


def choose_best_cluster_tag(detections, central_tag_id):
    central = find_tag(detections, central_tag_id)
    if central is not None:
        return central

    best_tag = None
    best_dist = 999999999.0

    for tag in detections:
        if tag.tag_id in HELPER_IDS:
            dx = tag.center[0] - CX
            dy = tag.center[1] - CY
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best_dist = dist
                best_tag = tag

    return best_tag


def choose_correction_tag(detections, central_tag_id):
    central = find_tag(detections, central_tag_id)
    if central is not None:
        return central

    best_tag = None
    best_dist = 999999999.0

    for tag in detections:
        if tag.tag_id in CROSS_HELPERS:
            dx = tag.center[0] - CX
            dy = tag.center[1] - CY
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best_dist = dist
                best_tag = tag

    if best_tag is not None:
        return best_tag

    return choose_best_cluster_tag(detections, central_tag_id)


def detect_side_pair(detections, target_landmark):
    ids = visible_ids(detections)

    if target_landmark in ids:
        return None

    groups = [
        (503, {502, 504}),
        (507, {506, 508}),
        (501, {508, 502}),
        (505, {506, 504}),
    ]

    for center_helper, corners in groups:
        if center_helper in ids and len(ids.intersection(corners)) > 0:
            return center_helper

    return None



def detect_local_arrival_helper(detections, target_landmark):
    """
    Restored cluster behavior for local landmark arrival.

    The robot should not require the central landmark tag to stop/update.
    If the target central tag is not visible, accept helper evidence:
      1) side-center + adjacent corner pair, preferred
      2) a single cross helper 501/503/505/507
      3) any helper if nothing else is available

    This lets the robot stop/nudge/update when it sees local cluster tags.
    """
    if find_tag(detections, target_landmark) is not None:
        return None

    pair_helper = detect_side_pair(detections, target_landmark)
    if pair_helper is not None:
        return pair_helper

    ids = visible_ids(detections)

    for helper_id in [501, 503, 505, 507]:
        if helper_id in ids:
            return helper_id

    for helper_id in [502, 504, 506, 508]:
        if helper_id in ids:
            return helper_id

    return None


def get_helper_grid_offset(tag_id, central_tag_id):
    if int(tag_id) == int(central_tag_id):
        return 0, 0
    if int(tag_id) in HELPER_GRID_OFFSET:
        return HELPER_GRID_OFFSET[int(tag_id)]
    return None


def get_landmark_pose_from_cluster_tag(tag, central_tag_id):
    grid_offset = get_helper_grid_offset(tag.tag_id, central_tag_id)
    if grid_offset is None:
        return None

    helper_x_grid, helper_y_grid = grid_offset

    raw_yaw = compute_yaw_deg(tag)
    yaw_error = normalize_angle(raw_yaw - EXPECTED_TAG_YAW_DEG)

    helper_x_m = compute_lateral_x_m(tag)
    center_x_m = helper_x_m - (helper_x_grid * CLUSTER_SPACING_M)

    tag_side_px = estimate_tag_side_px(tag)
    spacing_px = tag_side_px * (CLUSTER_SPACING_M / TAG_SIZE_M)
    center_py = tag.center[1] - (helper_y_grid * spacing_px)
    center_y_error_px = float(center_py - CY)

    return (
        raw_yaw,
        yaw_error,
        center_x_m,
        center_y_error_px,
        tag.tag_id,
        helper_x_grid,
        helper_y_grid,
    )


def get_visible_tag_pose(tag):
    raw_yaw = compute_yaw_deg(tag)
    yaw_error = normalize_angle(raw_yaw - EXPECTED_TAG_YAW_DEG)
    visible_x_error_px = float(tag.center[0] - CX)
    visible_y_error_px = float(tag.center[1] - CY)
    return raw_yaw, yaw_error, visible_x_error_px, visible_y_error_px, tag.tag_id


def adaptive_error_level(yaw_error, center_x_m):
    abs_x = abs(center_x_m)
    abs_yaw = abs(yaw_error)

    if abs_x >= X_LARGE_ERROR_M or abs_yaw >= YAW_LARGE_ERROR_DEG:
        return "LARGE"

    if abs_x >= X_MEDIUM_ERROR_M or abs_yaw >= YAW_MEDIUM_ERROR_DEG:
        return "MEDIUM"

    return "SMALL"


# =====================================================
# CAMERA THREAD
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
                gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

                raw_detections = self.detector.detect(
                    gray,
                    estimate_tag_pose=True,
                    camera_params=CAMERA_PARAMS,
                    tag_size=TAG_SIZE_M,
                )

                detections = []
                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

                for det in raw_detections:
                    tag = TagDetection(
                        tag_id=int(det.tag_id),
                        center=np.array(det.center, dtype=float),
                        corners=np.array(det.corners, dtype=float),
                        pose_t=det.pose_t,
                    )
                    detections.append(tag)

                    corners_i = tag.corners.astype(int)
                    center_i = tuple(tag.center.astype(int))

                    for i in range(4):
                        p1 = tuple(corners_i[i])
                        p2 = tuple(corners_i[(i + 1) % 4])
                        cv2.line(frame_bgr, p1, p2, (0, 255, 0), 2)

                    cv2.circle(frame_bgr, center_i, 5, (0, 0, 255), -1)

                    raw_yaw = compute_yaw_deg(tag)
                    yaw_error = normalize_angle(raw_yaw - EXPECTED_TAG_YAW_DEG)
                    x_m = compute_lateral_x_m(tag)

                    cv2.putText(
                        frame_bgr,
                        f"ID:{tag.tag_id}",
                        (center_i[0] + 10, center_i[1]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame_bgr,
                        f"Yaw:{yaw_error:.1f}",
                        (center_i[0] + 10, center_i[1] + 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )

                    cv2.putText(
                        frame_bgr,
                        f"xM:{x_m:.3f}",
                        (center_i[0] + 10, center_i[1] + 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 0, 255),
                        2,
                    )

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
# GRID WIDGET
# =====================================================

class TagGridWidget(QWidget):
    tag_clicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_tag = GRID_START_TAG
        self.goal_tag = None
        self.path = []
        self.expected_tag = None
        self.setMinimumSize(420, 210)

    def set_state(self, current_tag=None, goal_tag=None, path=None, expected_tag=None):
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

        painter.setPen(QPen(QColor(230, 230, 230)))
        title_font = QFont()
        title_font.setPointSize(10)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.drawText(20, 15, "5 x 3 Tag Grid: Tags 1-15 | Tag 0 = docking only | distance = 50 cm")

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
                painter.drawText(int(x), int(y), int(cell_w), int(cell_h), Qt.AlignCenter, str(tag))

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

        self.setWindowTitle("AGV Qt A-Star Closed Loop - Old Cluster Vision Behavior")
        # Raspberry Pi 4 display friendly size.
        screen = QApplication.primaryScreen()
        if screen is not None:
            geo = screen.availableGeometry()
            self.resize(int(geo.width() * 0.98), int(geo.height() * 0.96))
        else:
            self.resize(1024, 700)

        # Calibration flags.
        self.calibration_started = False
        self.dock_tag_confirmed = False
        self.waiting_for_manual_alignment = False
        self.calibration_done = False
        self.calibrating_move_to_tag1 = False

        # Mission state.
        self.current_tag = None
        self.goal_tag = None
        self.path = []
        self.active_path = []
        self.path_index = 0
        self.expected_next_tag = None
        self.current_heading = SOUTH
        self.mission_running = False

        # Old behavior segment state.
        self.route_state = "IDLE"
        self.segment_phase = "START_CLUSTER"
        self.cluster_lost_count = 0
        self.target_central_seen_count = 0
        self.travel_from_landmark = None
        self.travel_to_landmark = None

        self.local_arrival_landmark = None
        self.local_arrival_helper_id = None
        self.local_arrival_good_count = 0
        self.local_arrival_start_time = 0.0

        self.filtered_correction = 0.0

        self.turning_waiting = False
        self.pending_after_turn_segment = False
        self.turn_start_time = 0.0
        self.turn_timeout_sec = 20.0

        # Camera state.
        self.latest_detections = []
        self.latest_ids = []

        # Serial state.
        self.ser = None
        self.waiting_for_serial_response = False
        self.drive_left_pps = 0
        self.drive_right_pps = 0
        self.last_drive_mode = "STOP"
        self.last_vel_send_time = 0.0
        self.VEL_SEND_INTERVAL_SEC = 0.07

        self.camera_thread = None

        self.build_ui()
        self.update_ui_state()

        self.control_timer = QTimer(self)
        self.control_timer.timeout.connect(self.control_tick)
        self.control_timer.start(60)

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
        main.addLayout(left, 4)
        main.addLayout(right, 1)

        grid_group = QGroupBox("5 x 3 Tag Grid")
        grid_layout = QVBoxLayout(grid_group)
        self.grid = TagGridWidget()
        self.grid.tag_clicked.connect(self.on_grid_tag_clicked)
        grid_layout.addWidget(self.grid)
        left.addWidget(grid_group, 3)

        camera_group = QGroupBox("Live Camera Feedback")
        camera_layout = QVBoxLayout(camera_group)
        self.camera_label = QLabel("Camera not started")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(520, 390)
        self.camera_label.setStyleSheet("background:#111;color:white;border:1px solid #555;")
        self.camera_status = QLabel("Camera status: idle")
        self.camera_status.setStyleSheet("font-weight:bold;")
        camera_layout.addWidget(self.camera_label)
        camera_layout.addWidget(self.camera_status)
        left.addWidget(camera_group, 2)

        calib_group = QGroupBox("Docking Calibration")
        calib = QGridLayout(calib_group)

        self.calib_state = QLineEdit()
        self.calib_state.setReadOnly(True)

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
        self.route_state_edit = QLineEdit()
        self.route_state_edit.setReadOnly(True)

        self.simulation_checkbox = QCheckBox("Simulation mode")
        self.simulation_checkbox.setChecked(False)

        self.connect_btn = QPushButton("Connect ESP32")
        self.connect_btn.clicked.connect(self.connect_esp32)

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
        mission.addWidget(QLabel("Route State:"), 5, 0)
        mission.addWidget(self.route_state_edit, 5, 1, 1, 2)
        mission.addWidget(self.simulation_checkbox, 6, 0)
        mission.addWidget(self.connect_btn, 7, 0)
        mission.addWidget(self.start_btn, 7, 1)
        mission.addWidget(self.stop_btn, 8, 0, 1, 2)
        right.addWidget(mission_group)

        tuning_group = QGroupBox("PPS / ESP32 Tuning")
        tuning = QGridLayout(tuning_group)

        self.base_pps_spin = QSpinBox()
        self.base_pps_spin.setRange(0, 20000)
        self.base_pps_spin.setValue(6500)

        self.imu_max_spin = QSpinBox()
        self.imu_max_spin.setRange(0, 3000)
        self.imu_max_spin.setValue(350)

        self.imu_kp_spin = QDoubleSpinBox()
        self.imu_kp_spin.setRange(0, 300)
        self.imu_kp_spin.setDecimals(1)
        self.imu_kp_spin.setValue(70.0)

        self.accel_spin = QSpinBox()
        self.accel_spin.setRange(100, 50000)
        self.accel_spin.setValue(7000)

        self.decel_spin = QSpinBox()
        self.decel_spin.setRange(100, 50000)
        self.decel_spin.setValue(9000)

        self.apply_tuning_btn = QPushButton("Apply ESP32 Tuning")
        self.apply_tuning_btn.clicked.connect(self.apply_esp32_tuning)

        tuning.addWidget(QLabel("BASE_PPS"), 0, 0)
        tuning.addWidget(self.base_pps_spin, 0, 1)
        tuning.addWidget(QLabel("IMU_MAX_CORR"), 1, 0)
        tuning.addWidget(self.imu_max_spin, 1, 1)
        tuning.addWidget(QLabel("IMU_KP"), 2, 0)
        tuning.addWidget(self.imu_kp_spin, 2, 1)
        tuning.addWidget(QLabel("ACCEL"), 3, 0)
        tuning.addWidget(self.accel_spin, 3, 1)
        tuning.addWidget(QLabel("DECEL"), 4, 0)
        tuning.addWidget(self.decel_spin, 4, 1)
        tuning.addWidget(self.apply_tuning_btn, 5, 0, 1, 2)
        # Tuning panel removed from visible UI. Defaults are sent from Python constants.
        self.tuning_group = tuning_group
        self.tuning_group.setVisible(False)

        dest_group = QGroupBox("Destination / A-Star Path")
        dest = QVBoxLayout(dest_group)
        self.destination_combo = QComboBox()
        for tag in range(1, ROWS * COLS + 1):
            self.destination_combo.addItem(f"Tag {tag}", tag)

        self.compute_btn = QPushButton("Compute A-Star Path")
        self.compute_btn.clicked.connect(self.compute_path_from_combo)

        self.path_text = QTextEdit()
        self.path_text.setReadOnly(True)
        self.path_text.setMinimumHeight(45)

        dest.addWidget(QLabel("Select destination:"))
        dest.addWidget(self.destination_combo)
        dest.addWidget(self.compute_btn)
        dest.addWidget(self.path_text)
        right.addWidget(dest_group)

        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(220)
        log_layout.addWidget(self.log)
        right.addWidget(log_group, 1)

    def append_log(self, text):
        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {text}")
        self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def update_ui_state(self):
        self.current_tag_edit.setText(str(self.current_tag) if self.current_tag is not None else "---")
        self.goal_tag_edit.setText(str(self.goal_tag) if self.goal_tag is not None else "---")
        self.heading_edit.setText(HEADING_LABELS.get(self.current_heading, "---"))
        self.expected_next_edit.setText(str(self.expected_next_tag) if self.expected_next_tag is not None else "---")
        self.path_edit.setText(" → ".join(str(x) for x in self.path) if self.path else "---")
        self.route_state_edit.setText(f"{self.route_state} / {self.segment_phase}")

        if self.calibration_done:
            self.calib_state.setText("DONE - GRID START TAG 1")
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
        # Add state overlay.
        cv2.putText(
            frame_bgr,
            f"State:{self.route_state} Seg:{self.travel_from_landmark}->{self.travel_to_landmark}",
            (25, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame_bgr,
            f"Drive:{self.last_drive_mode} L:{self.drive_left_pps} R:{self.drive_right_pps}",
            (25, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pix = QPixmap.fromImage(qimg)
        self.camera_label.setPixmap(
            pix.scaled(self.camera_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

    def on_camera_status(self, text):
        self.camera_status.setText(f"Camera status: {text}")

    def on_detections_ready(self, detections):
        self.latest_detections = detections
        self.latest_ids = [d.tag_id for d in detections]

        if self.calibration_started and not self.calibration_done and not self.calibrating_move_to_tag1:
            if DOCK_TAG in self.latest_ids:
                if not self.dock_tag_confirmed:
                    self.append_log("Docking Tag 0 detected. Manually align robot, then press S.")
                self.dock_tag_confirmed = True
                self.waiting_for_manual_alignment = True
                self.update_ui_state()

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
            try:
                self.ser.flush()
            except Exception:
                pass
            self.append_log(f"ESP32 <= {cmd}")
        except Exception as e:
            self.append_log(f"Serial send failed: {e}")

    def read_esp32_available(self):
        # Do not let the periodic control timer steal expected replies
        # while wait_for_esp32_text() is waiting for OK IMU RECAL / OK TURN_DONE.
        if self.waiting_for_serial_response:
            return []

        if self.ser is None:
            return []

        lines = []
        try:
            while self.ser.in_waiting > 0:
                line = self.ser.readline().decode(errors="ignore").strip()
                if line:
                    lines.append(line)
                    self.append_log(f"ESP32: {line}")
        except Exception as e:
            self.append_log(f"Serial read failed: {e}")
        return lines

    def wait_for_esp32_text(self, expected_text, timeout_sec=20.0):
        """
        Wait for a specific ESP32 reply.

        Important:
        The normal 60 ms control timer also reads serial. During this wait,
        pause that reader so it cannot consume the expected OK line first.
        """
        if self.simulation_checkbox.isChecked():
            return True

        if self.ser is None:
            return False

        self.waiting_for_serial_response = True
        start = time.time()
        found = False

        try:
            while time.time() - start < timeout_sec:
                try:
                    line = self.ser.readline().decode(errors="ignore").strip()
                except Exception:
                    line = ""

                if line:
                    self.append_log(f"ESP32: {line}")

                    if expected_text in line:
                        found = True
                        break

                    if line.startswith("ERR"):
                        # Keep reading until timeout because ESP32 may print diagnostic
                        # lines before the final result, but make the error visible.
                        self.append_log(f"ESP32 error while waiting for {expected_text}: {line}")

                # Keep UI alive without allowing the periodic serial reader to steal lines.
                QApplication.processEvents()
                time.sleep(0.03)

        finally:
            self.waiting_for_serial_response = False

        return found

    def apply_esp32_tuning(self):
        """
        Send fixed ESP32 tuning values.

        Do not read hidden QSpinBox widgets here. In the compact UI those widgets
        may be removed/garbage-collected.
        """
        self.send_esp32(f"SET_BASE {DEFAULT_BASE_PPS}")
        time.sleep(0.05)
        self.send_esp32(f"SET_IMU_MAX {DEFAULT_IMU_MAX_CORR}")
        time.sleep(0.05)
        self.send_esp32(f"SET_IMU_KP {DEFAULT_IMU_KP:.1f}")
        time.sleep(0.05)
        self.send_esp32(f"SET_RAMP {DEFAULT_ACCEL_PPS_PER_SEC} {DEFAULT_DECEL_PPS_PER_SEC}")
        time.sleep(0.05)

    def send_velocity(self, left_pps, right_pps, force=False):
        now = time.time()

        left_pps = int(np.clip(left_pps, -MAX_PPS, MAX_PPS))
        right_pps = int(np.clip(right_pps, -MAX_PPS, MAX_PPS))

        if not force:
            if (
                left_pps == self.drive_left_pps
                and right_pps == self.drive_right_pps
                and now - self.last_vel_send_time < self.VEL_SEND_INTERVAL_SEC
            ):
                return

        self.drive_left_pps = left_pps
        self.drive_right_pps = right_pps
        self.last_vel_send_time = now
        self.last_drive_mode = "VISION"

        self.send_esp32(f"VEL {left_pps} {right_pps}")

    def stop_robot(self):
        self.drive_left_pps = 0
        self.drive_right_pps = 0
        self.last_drive_mode = "STOP"
        self.send_esp32("STOP")

    def lock_heading_go(self):
        if self.last_drive_mode == "IMU":
            return
        self.filtered_correction = 0.0
        self.send_esp32("LOCK_HEADING_GO")
        self.last_drive_mode = "IMU"
        self.append_log("LOCK_HEADING_GO sent")

    # -------------------------
    # Old correction methods
    # -------------------------

    def reset_correction_filter(self):
        self.filtered_correction = 0.0

    def travelling_velocity(self, raw_yaw, yaw_error, center_x_m):
        error_level = adaptive_error_level(yaw_error, center_x_m)

        if error_level == "LARGE":
            kp_yaw = KP_YAW_STRONG_PPS_PER_DEG
            kp_x = KP_X_STRONG_PPS_PER_M
            max_corr = MAX_VISION_CORRECTION_STRONG_PPS
            base_pps = VISION_BASE_PPS_SLOW
            alpha = CORRECTION_FILTER_ALPHA_STRONG
        elif error_level == "MEDIUM":
            kp_yaw = (KP_YAW_PPS_PER_DEG + KP_YAW_STRONG_PPS_PER_DEG) * 0.5
            kp_x = (KP_X_PPS_PER_M + KP_X_STRONG_PPS_PER_M) * 0.5
            max_corr = int((MAX_VISION_CORRECTION_PPS + MAX_VISION_CORRECTION_STRONG_PPS) * 0.5)
            base_pps = int((VISION_BASE_PPS + VISION_BASE_PPS_SLOW) * 0.5)
            alpha = 0.40
        else:
            kp_yaw = KP_YAW_PPS_PER_DEG
            kp_x = KP_X_PPS_PER_M
            max_corr = MAX_VISION_CORRECTION_PPS
            base_pps = VISION_BASE_PPS
            alpha = CORRECTION_FILTER_ALPHA

        yaw_for_control = yaw_error
        x_for_control = center_x_m

        if abs(yaw_for_control) < YAW_DEADBAND_DEG:
            yaw_for_control = 0.0
        if abs(x_for_control) < X_DEADBAND_M:
            x_for_control = 0.0

        yaw_corr = kp_yaw * yaw_for_control
        x_corr = kp_x * x_for_control * X_SIGN

        if abs(x_for_control) > X_DEADBAND_M:
            if yaw_corr * x_corr < 0:
                yaw_corr *= 0.25

        raw_correction = yaw_corr + x_corr
        raw_correction = float(np.clip(raw_correction, -max_corr, max_corr))

        self.filtered_correction = ((1.0 - alpha) * self.filtered_correction + alpha * raw_correction)

        correction = int(self.filtered_correction)

        left = base_pps - correction
        right = base_pps + correction

        left = int(np.clip(left, VISION_MIN_PPS, VISION_MAX_PPS))
        right = int(np.clip(right, VISION_MIN_PPS, VISION_MAX_PPS))

        return left, right, correction, yaw_corr, x_corr, error_level

    def visible_tag_center_velocity(self, raw_yaw, yaw_error, y_error_px, pps):
        yaw_for_control = yaw_error
        if abs(yaw_for_control) < YAW_DEADBAND_DEG:
            yaw_for_control = 0.0

        yaw_corr = -KP_TURN_TAG_YAW_PPS_PER_DEG * yaw_for_control
        yaw_corr = int(np.clip(yaw_corr, -MAX_TURN_TAG_YAW_CORRECTION_PPS, MAX_TURN_TAG_YAW_CORRECTION_PPS))

        if abs(y_error_px) <= TURN_TAG_CENTER_Y_OK_PX:
            fb = 0
        else:
            if y_error_px > 0:
                fb = TURN_TAG_FB_PPS * FB_SIGN
            else:
                fb = -TURN_TAG_FB_PPS * FB_SIGN

        left = fb - yaw_corr
        right = fb + yaw_corr

        if fb == 0:
            left = -yaw_corr
            right = yaw_corr

        left = int(np.clip(left, -MAX_PPS, MAX_PPS))
        right = int(np.clip(right, -MAX_PPS, MAX_PPS))

        return left, right, yaw_corr

    # -------------------------
    # Segment state machine
    # -------------------------

    def start_segment(self, from_tag, to_tag):
        self.travel_from_landmark = int(from_tag)
        self.travel_to_landmark = int(to_tag)
        self.segment_phase = "START_CLUSTER"
        self.cluster_lost_count = 0
        self.target_central_seen_count = 0
        self.reset_correction_filter()
        self.route_state = "MOVE"
        self.expected_next_tag = int(to_tag)
        self.append_log(f"Starting segment {from_tag} → {to_tag}")
        self.update_ui_state()

    def handle_landmark_arrival(self, landmark_id):
        self.stop_robot()
        self.reset_correction_filter()

        landmark_id = int(landmark_id)
        self.append_log(f"Reached landmark Tag {landmark_id}")

        if self.calibrating_move_to_tag1 and landmark_id == GRID_START_TAG:
            self.finish_calibration_at_tag1()
            return

        if not self.mission_running:
            self.route_state = "IDLE"
            self.update_ui_state()
            return

        if self.path_index < len(self.active_path) and landmark_id == self.active_path[self.path_index]:
            self.current_tag = landmark_id
            self.path_index += 1

        if self.path_index >= len(self.active_path):
            self.finish_mission()
            return

        next_tag = self.active_path[self.path_index]
        desired_heading = heading_between_tags(self.current_tag, next_tag)
        turn_deg = turn_delta_deg(self.current_heading, desired_heading)

        self.expected_next_tag = next_tag
        self.update_ui_state()

        if abs(turn_deg) > 1.0:
            self.route_state = "TURNING"
            self.turning_waiting = True
            self.pending_after_turn_segment = True
            self.turn_start_time = time.time()
            self.append_log(f"Turning before next segment: TURN_REL {turn_deg:.1f}")
            self.send_esp32(f"TURN_REL {turn_deg:.1f}")
            self.current_heading = desired_heading
        else:
            self.current_heading = desired_heading
            self.start_segment(self.current_tag, next_tag)
            self.lock_heading_go()

    def start_local_arrival(self, landmark_id, helper_id):
        self.stop_robot()
        self.local_arrival_landmark = int(landmark_id)
        self.local_arrival_helper_id = int(helper_id)
        self.local_arrival_good_count = 0
        self.local_arrival_start_time = time.time()
        self.route_state = "LOCAL_ARRIVAL"
        self.append_log(f"Local arrival at Tag {landmark_id}, nudging to helper {helper_id}")

    def choose_move_correction(self, detections):
        central_target = find_tag(detections, self.travel_to_landmark)

        if central_target is not None:
            self.target_central_seen_count += 1

            if self.target_central_seen_count >= TARGET_CENTRAL_SEEN_FRAMES_REQUIRED:
                self.handle_landmark_arrival(self.travel_to_landmark)
                return None, None, ""

            return central_target, self.travel_to_landmark, f"TAG{self.travel_to_landmark}"

        self.target_central_seen_count = 0

        if self.segment_phase == "START_CLUSTER":
            if any_cluster_visible_now(detections, self.travel_from_landmark):
                self.cluster_lost_count = 0
                tag = choose_best_cluster_tag(detections, self.travel_from_landmark)
                return tag, self.travel_from_landmark, f"TAG{self.travel_from_landmark}"

            self.cluster_lost_count += 1

            if self.cluster_lost_count >= CLUSTER_LOST_FRAMES_REQUIRED:
                self.segment_phase = "SEARCH_TARGET"
                self.append_log(
                    f"Fully left Tag {self.travel_from_landmark}. Helpers now belong to target {self.travel_to_landmark}."
                )

            return None, None, ""

        if self.segment_phase == "SEARCH_TARGET":
            helper_id = detect_local_arrival_helper(detections, self.travel_to_landmark)
            if helper_id is not None:
                self.append_log(
                    f"Local helper Tag {helper_id} seen for target {self.travel_to_landmark}. "
                    "Central tag is not required."
                )
                self.start_local_arrival(self.travel_to_landmark, helper_id)
                return None, None, ""

            tag = choose_best_cluster_tag(detections, self.travel_to_landmark)
            if tag is not None:
                return tag, self.travel_to_landmark, f"TAG{self.travel_to_landmark}"

            return None, None, ""

        return None, None, ""

    def control_tick(self):
        # Read ESP32 output.
        lines = self.read_esp32_available()

        if self.route_state == "TURNING" and self.turning_waiting:
            done = any("OK TURN_DONE" in line for line in lines)
            timeout = (time.time() - self.turn_start_time) > self.turn_timeout_sec

            if done:
                self.append_log("ESP32 reported OK TURN_DONE")
                self.turning_waiting = False
                self.route_state = "MOVE"
                if self.path_index < len(self.active_path):
                    next_tag = self.active_path[self.path_index]
                    self.start_segment(self.current_tag, next_tag)
                    self.lock_heading_go()
                self.update_ui_state()

            elif timeout:
                # Do NOT continue movement on timeout.
                # The previous version continued after timeout, which can make
                # a slow 90 degree turn look like only 45 degrees before travel starts.
                self.append_log("Turn timeout: stopping. ESP32 did not report OK TURN_DONE.")
                self.turning_waiting = False
                self.mission_running = False
                self.route_state = "TURN_TIMEOUT"
                self.expected_next_tag = None
                self.stop_robot()
                self.update_ui_state()

            return

        if self.route_state == "MOVE":
            self.process_move_state()

        elif self.route_state == "LOCAL_ARRIVAL":
            self.process_local_arrival_state()

        self.update_ui_state()

    def process_move_state(self):
        detections = self.latest_detections
        if not detections:
            self.lock_heading_go()
            return

        tag, landmark_id, label = self.choose_move_correction(detections)

        if tag is not None and self.route_state == "MOVE":
            pose = get_landmark_pose_from_cluster_tag(tag, landmark_id)

            if pose is not None:
                (
                    raw_yaw,
                    yaw_error,
                    center_x_m,
                    center_y_error_px,
                    seen_tag_id,
                    helper_x_grid,
                    helper_y_grid,
                ) = pose

                (
                    left,
                    right,
                    correction,
                    yaw_corr,
                    x_corr,
                    error_level,
                ) = self.travelling_velocity(raw_yaw, yaw_error, center_x_m)

                self.send_velocity(left, right)

                # Log at lower rate only.
                if time.time() - getattr(self, "_last_corr_log", 0.0) > 0.70:
                    self._last_corr_log = time.time()
                    self.append_log(
                        f"{label} CORR seg={self.travel_from_landmark}->{self.travel_to_landmark} "
                        f"phase={self.segment_phase} seen={seen_tag_id} grid=({helper_x_grid},{helper_y_grid}) "
                        f"yawErr={yaw_error:.2f} centerXM={center_x_m:.4f} level={error_level} "
                        f"corr={correction} L={left} R={right}"
                    )
        elif self.route_state == "MOVE":
            self.lock_heading_go()

    def process_local_arrival_state(self):
        detections = self.latest_detections

        central = find_tag(detections, self.local_arrival_landmark)
        if central is not None:
            self.append_log(f"Central Tag {self.local_arrival_landmark} became visible during local arrival.")
            self.handle_landmark_arrival(self.local_arrival_landmark)
            return

        helper = find_tag(detections, self.local_arrival_helper_id)
        if helper is None:
            helper = choose_best_cluster_tag(detections, self.local_arrival_landmark)

        if helper is None:
            self.append_log("Local arrival tag lost. Stopping as reached.")
            self.handle_landmark_arrival(self.local_arrival_landmark)
            return

        raw_yaw, yaw_error, visible_x_error_px, visible_y_error_px, seen_tag_id = get_visible_tag_pose(helper)

        if abs(visible_y_error_px) <= LOCAL_NUDGE_CENTER_Y_OK_PX:
            self.local_arrival_good_count += 1
            self.stop_robot()

            self.append_log(
                f"LOCAL_ARRIVAL_GOOD target={self.local_arrival_landmark} "
                f"helper={seen_tag_id} yErr={visible_y_error_px:.1f}px "
                f"good={self.local_arrival_good_count}/{LOCAL_NUDGE_GOOD_FRAMES_REQUIRED}"
            )

            if self.local_arrival_good_count >= LOCAL_NUDGE_GOOD_FRAMES_REQUIRED:
                self.handle_landmark_arrival(self.local_arrival_landmark)
                return
        else:
            self.local_arrival_good_count = 0
            left, right, yaw_corr = self.visible_tag_center_velocity(
                raw_yaw,
                yaw_error,
                visible_y_error_px,
                LOCAL_NUDGE_PPS,
            )
            self.send_velocity(left, right)

        if time.time() - self.local_arrival_start_time > LOCAL_NUDGE_TIMEOUT_SEC:
            self.append_log("Local arrival timeout. Stopping as reached.")
            self.handle_landmark_arrival(self.local_arrival_landmark)

    # -------------------------
    # Calibration
    # -------------------------

    def start_calibration(self):
        self.calibration_started = True
        self.dock_tag_confirmed = False
        self.waiting_for_manual_alignment = False
        self.calibration_done = False
        self.calibrating_move_to_tag1 = False

        self.current_tag = None
        self.expected_next_tag = None
        self.path = []
        self.active_path = []
        self.path_index = 0
        self.route_state = "WAIT_DOCK"
        self.mission_running = False

        self.append_log("Calibration started. Place robot at docking Tag 0.")
        self.append_log("Waiting for camera detection of Tag 0...")
        self.update_ui_state()

    def reset_calibration(self):
        self.stop_robot()
        self.calibration_started = False
        self.dock_tag_confirmed = False
        self.waiting_for_manual_alignment = False
        self.calibration_done = False
        self.calibrating_move_to_tag1 = False

        self.current_tag = None
        self.goal_tag = None
        self.path = []
        self.active_path = []
        self.path_index = 0
        self.expected_next_tag = None
        self.current_heading = SOUTH
        self.mission_running = False
        self.route_state = "IDLE"

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

        if not self.simulation_checkbox.isChecked():
            if not self.connect_esp32():
                return
            self.stop_robot()
            time.sleep(0.2)

            try:
                self.ser.reset_input_buffer()
            except Exception:
                pass

            self.apply_esp32_tuning()
            self.send_esp32("IMU RECAL")
            ok = self.wait_for_esp32_text("OK IMU RECAL", timeout_sec=20.0)
            if not ok:
                QMessageBox.warning(self, "Calibration failed", "ESP32 did not confirm OK IMU RECAL")
                self.append_log("Calibration failed: no OK IMU RECAL")
                return
            self.append_log("IMU calibrated.")

        self.calibrating_move_to_tag1 = True
        self.calibration_started = False
        self.waiting_for_manual_alignment = False
        self.current_tag = DOCK_TAG
        self.current_heading = SOUTH

        self.append_log("Moving from docking Tag 0 to grid Tag 1 using old camera cluster logic.")
        self.start_segment(DOCK_TAG, GRID_START_TAG)
        self.lock_heading_go()

    def finish_calibration_at_tag1(self):
        self.stop_robot()
        self.current_tag = GRID_START_TAG
        self.current_heading = SOUTH
        self.calibration_done = True
        self.calibrating_move_to_tag1 = False
        self.route_state = "IDLE"
        self.expected_next_tag = None
        self.append_log("Calibration done. Robot reached grid Tag 1.")
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

        if self.current_tag is None or self.current_tag == DOCK_TAG:
            self.append_log("Cannot compute path: current grid tag unknown")
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
            f"Distance: {(len(self.path) - 1) * CELL_DISTANCE_M:.2f} m\n"
            "Movement uses old PPS camera correction + IMU fallback."
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
            self.apply_esp32_tuning()

        self.active_path = list(self.path)
        self.path_index = 1
        self.expected_next_tag = self.active_path[self.path_index]
        self.mission_running = True

        next_tag = self.expected_next_tag
        desired_heading = heading_between_tags(self.current_tag, next_tag)
        turn_deg = turn_delta_deg(self.current_heading, desired_heading)

        self.append_log(f"Mission started. Expected next tag: {self.expected_next_tag}")

        if abs(turn_deg) > 1.0:
            self.route_state = "TURNING"
            self.turning_waiting = True
            self.turn_start_time = time.time()
            self.append_log(f"Initial turn: TURN_REL {turn_deg:.1f}")
            self.send_esp32(f"TURN_REL {turn_deg:.1f}")
            self.current_heading = desired_heading
        else:
            self.current_heading = desired_heading
            self.start_segment(self.current_tag, next_tag)
            self.lock_heading_go()

        self.update_ui_state()

    def stop_mission(self):
        self.mission_running = False
        self.expected_next_tag = None
        self.route_state = "IDLE"
        self.stop_robot()
        self.append_log("Mission stopped")
        self.update_ui_state()

    def finish_mission(self):
        self.mission_running = False
        self.expected_next_tag = None
        self.route_state = "DONE"
        self.stop_robot()
        self.append_log("Mission complete")
        QMessageBox.information(self, "Mission complete", "AGV reached the destination tag.")
        self.update_ui_state()

    # -------------------------
    # Cleanup
    # -------------------------

    def closeEvent(self, event):
        try:
            self.mission_running = False
            self.stop_robot()
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
