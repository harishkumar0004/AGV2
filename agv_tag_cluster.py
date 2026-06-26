#!/usr/bin/env python3
"""
Qt based AGV navigation for 5x3 AprilTag grid with docking Tag 0.

Map:
    Docking/calibration tag: 0

    Row 0:   1   2   3   4   5
    Row 1:   6   7   8   9   10
    Row 2:   11  12  13  14  15

Flow:
    1. Start app.
    2. Camera feedback starts automatically.
    3. Place robot at Tag 0.
    4. Click Start Calibration.
    5. When Tag 0 is detected, manually align robot.
    6. Press S.
    7. Robot/planner moves to Tag 1.
    8. Select destination tag 1..15.
    9. Click Start Mission.

Simulation closed loop:
    The simulated robot marker advances only when the camera detects the expected next tag.

Real hardware:
    Python sends serial commands to ESP32:
        IMU RECAL
        SET_BASE <value>
        SET_IMU_MAX <value>
        LOCK_HEADING_GO
        TURN_REL <deg>
        STOP
        STATUS

Install:
    pip install PyQt5 pyserial opencv-python pupil-apriltags networkx

Run:
    python3 app_qt_cluster_closed_loop.py
"""

import math
import sys
import time
from itertools import islice
from typing import Dict, List, Optional, Tuple

import cv2
import networkx as nx

try:
    import serial
except Exception:
    serial = None

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


# ============================================================
# CONFIGURATION
# ============================================================

DOCK_TAG = 0
GRID_START_TAG = 1
ROWS = 3
COLS = 5
NUM_PATHS = 5
CELL_SIZE = 95
CELL_DISTANCE_M = 0.5

SERIAL_PORT = "/dev/ttyUSB0"
SERIAL_BAUD = 115200

# Speed values sent to ESP32 during startup/calibration.
ESP32_BASE_PPS = 6500
ESP32_IMU_MAX_CORR = 350

# Timed movement fallback. Tune on real robot.
CELL_TRAVEL_SEC = 1.20
DOCK_TO_TAG1_TRAVEL_SEC = 1.20
TURN_SETTLE_SEC = 0.25

# Camera settings.
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_FPS = 15

# Heading convention used by UI.
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

PATH_COLORS = [
    QColor(52, 152, 219),
    QColor(46, 204, 113),
    QColor(155, 89, 182),
    QColor(241, 196, 15),
    QColor(230, 126, 34),
]


# ============================================================
# GRID / A-STAR HELPERS
# ============================================================

def node_id_to_grid_coord(node_id: int) -> Tuple[int, int]:
    """Convert 1-based tag ID 1..15 to 0-based row,col."""
    node_id_0 = node_id - 1
    row = node_id_0 // COLS
    col = node_id_0 % COLS
    return row, col


def grid_coord_to_node_id(row: int, col: int) -> int:
    """Convert 0-based row,col to 1-based tag ID 1..15."""
    return row * COLS + col + 1


def create_grid_graph(blocked_tags: Optional[set] = None) -> nx.Graph:
    """Create 5x3 grid graph using tag IDs 1..15 as nodes."""
    if blocked_tags is None:
        blocked_tags = set()

    G = nx.Graph()

    for tag_id in range(1, ROWS * COLS + 1):
        if tag_id not in blocked_tags:
            G.add_node(tag_id)

    for r in range(ROWS):
        for c in range(COLS):
            a = grid_coord_to_node_id(r, c)
            if a not in G:
                continue
            if c + 1 < COLS:
                b = grid_coord_to_node_id(r, c + 1)
                if b in G:
                    G.add_edge(a, b, weight=CELL_DISTANCE_M)
            if r + 1 < ROWS:
                b = grid_coord_to_node_id(r + 1, c)
                if b in G:
                    G.add_edge(a, b, weight=CELL_DISTANCE_M)

    return G


def manhattan_heuristic_tag(a: int, b: int) -> float:
    ar, ac = node_id_to_grid_coord(a)
    br, bc = node_id_to_grid_coord(b)
    return abs(ar - br) + abs(ac - bc)


def k_shortest_paths(G: nx.Graph, source: int, target: int, k: int = NUM_PATHS) -> List[List[int]]:
    """Return k shortest paths using NetworkX shortest_simple_paths."""
    if source not in G or target not in G:
        return []
    try:
        return list(islice(nx.shortest_simple_paths(G, source, target, weight="weight"), k))
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def astar_path(G: nx.Graph, source: int, target: int) -> List[int]:
    """Return one A-star path. Kept here for explicit A-star operation."""
    if source not in G or target not in G:
        return []
    try:
        return nx.astar_path(
            G,
            source,
            target,
            heuristic=manhattan_heuristic_tag,
            weight="weight",
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return []


def path_length_m(G: nx.Graph, path: List[int]) -> float:
    if len(path) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(path[:-1], path[1:]):
        if G.has_edge(a, b):
            total += float(G[a][b].get("weight", CELL_DISTANCE_M))
        else:
            return float("inf")
    return total


def heading_between_tags(a: int, b: int) -> Optional[int]:
    ar, ac = node_id_to_grid_coord(a)
    br, bc = node_id_to_grid_coord(b)
    dr = br - ar
    dc = bc - ac
    if dr == -1 and dc == 0:
        return NORTH
    if dr == 1 and dc == 0:
        return SOUTH
    if dr == 0 and dc == 1:
        return EAST
    if dr == 0 and dc == -1:
        return WEST
    return None


def turn_degrees(current_heading: Optional[int], target_heading: int) -> float:
    """
    Convert current heading to shortest relative turn.
    Positive/negative sign must match ESP32 TURN_REL behavior.
    If turning direction is reversed physically, change sign here or TURN_SIGN in ESP32.
    """
    if current_heading is None:
        return 0.0
    diff = (target_heading - current_heading) % 4
    if diff == 0:
        return 0.0
    if diff == 1:
        return 90.0
    if diff == 2:
        return 180.0
    if diff == 3:
        return -90.0
    return 0.0


# ============================================================
# APRILTAG CAMERA THREAD
# ============================================================

class AprilTagCameraThread(QThread):
    """
    Camera thread for Raspberry Pi Camera.

    It tries Picamera2 first because Raspberry Pi Camera v2 is normally served by
    libcamera, not by a normal OpenCV /dev/video device. If Picamera2 is not
    available, it falls back to OpenCV V4L2.
    """

    tag_detected = pyqtSignal(int)
    frame_ready = pyqtSignal(object)   # BGR frame for Qt display
    status = pyqtSignal(str)

    def __init__(self, camera_index: int = CAMERA_INDEX, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.running = False
        self.last_emit_time_by_tag: Dict[int, float] = {}
        self.detector = None
        self.picam2 = None
        self.cap = None
        self.backend = "none"

    def stop(self):
        self.running = False
        if self.isRunning():
            self.wait(2500)

    def _open_picamera2(self) -> bool:
        try:
            from picamera2 import Picamera2
        except Exception as e:
            self.status.emit(f"Picamera2 unavailable: {e}")
            return False

        try:
            self.picam2 = Picamera2()
            config = self.picam2.create_preview_configuration(
                main={
                    "size": (CAMERA_WIDTH, CAMERA_HEIGHT),
                    "format": "RGB888",
                },
                buffer_count=3,
            )
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(0.5)

            frame = self.picam2.capture_array()
            if frame is None:
                self.status.emit("Picamera2 started but frame is empty")
                self.picam2.stop()
                self.picam2.close()
                self.picam2 = None
                return False

            self.backend = "picamera2"
            self.status.emit("Camera running: Picamera2/libcamera")
            return True

        except Exception as e:
            self.status.emit(f"Picamera2 open failed: {e}")
            try:
                if self.picam2 is not None:
                    self.picam2.stop()
                    self.picam2.close()
            except Exception:
                pass
            self.picam2 = None
            return False

    def _open_opencv_v4l2(self) -> bool:
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)

            if not self.cap.isOpened():
                self.status.emit("OpenCV V4L2 open failed")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            ok, frame = self.cap.read()
            if not ok or frame is None:
                self.status.emit("OpenCV V4L2 opened but frame read failed")
                self.cap.release()
                self.cap = None
                return False

            self.backend = "opencv_v4l2"
            self.status.emit("Camera running: OpenCV V4L2")
            return True

        except Exception as e:
            self.status.emit(f"OpenCV V4L2 open failed: {e}")
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:
                pass
            self.cap = None
            return False

    def _open_camera(self) -> bool:
        # Raspberry Pi Camera v2 should use Picamera2/libcamera first.
        if self._open_picamera2():
            return True

        # USB cameras or legacy /dev/video camera nodes may work here.
        if self._open_opencv_v4l2():
            return True

        return False

    def _read_frame_bgr(self):
        if self.backend == "picamera2" and self.picam2 is not None:
            frame_rgb = self.picam2.capture_array()
            if frame_rgb is None:
                return None
            # Picamera2 config gives RGB888. Convert to BGR because drawing code uses OpenCV colors.
            return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        if self.backend == "opencv_v4l2" and self.cap is not None:
            ok, frame_bgr = self.cap.read()
            if not ok:
                return None
            return frame_bgr

        return None

    def _close_camera(self):
        try:
            if self.picam2 is not None:
                self.picam2.stop()
                self.picam2.close()
        except Exception:
            pass
        self.picam2 = None

        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.cap = None
        self.backend = "none"

    def run(self):
        self.running = True

        try:
            from pupil_apriltags import Detector
            self.detector = Detector(
                families="tag36h11",
                nthreads=2,
                quad_decimate=2.0,
                quad_sigma=0.0,
                refine_edges=1,
                decode_sharpening=0.25,
                debug=0,
            )
        except Exception as e:
            self.status.emit(f"AprilTag detector init failed: {e}")
            return

        if not self._open_camera():
            self.status.emit(
                "Camera unavailable. For Raspberry Pi Camera install/enable Picamera2, "
                "or close other apps using the camera."
            )
            return

        while self.running:
            frame = self._read_frame_bgr()
            if frame is None:
                self.status.emit("Camera frame read failed")
                self.msleep(100)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            try:
                detections = self.detector.detect(gray)
            except Exception as e:
                self.status.emit(f"Detection error: {e}")
                detections = []

            detected_ids = []
            now = time.time()

            for det in detections:
                tag_id = int(det.tag_id)
                detected_ids.append(tag_id)

                corners = det.corners.astype(int)
                for i in range(4):
                    p1 = tuple(corners[i])
                    p2 = tuple(corners[(i + 1) % 4])
                    cv2.line(frame, p1, p2, (0, 255, 0), 2)

                cx, cy = int(det.center[0]), int(det.center[1])
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                cv2.putText(
                    frame,
                    f"Tag {tag_id}",
                    (cx + 8, cy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                # Emit same tag at most 4 times per second.
                last_emit = self.last_emit_time_by_tag.get(tag_id, 0.0)
                if now - last_emit >= 0.25:
                    self.last_emit_time_by_tag[tag_id] = now
                    self.tag_detected.emit(tag_id)

            if detected_ids:
                self.status.emit(
                    f"{self.backend}: detected " + ", ".join(str(x) for x in sorted(set(detected_ids)))
                )
            else:
                self.status.emit(f"{self.backend}: no tag detected")

            self.frame_ready.emit(frame.copy())
            self.msleep(30)

        self._close_camera()
        self.status.emit("Camera stopped")


# ============================================================
# GRID VIEW
# ============================================================

class TagGridView(QGraphicsView):
    tagClicked = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.Antialiasing)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.cells = {}
        self.labels = {}
        self.path_items = []
        self.robot_item = None
        self.robot_heading_item = None
        self.robot_tag = None
        self.robot_heading = SOUTH
        self.start_tag = GRID_START_TAG
        self.goal_tag = None
        self.blocked_tags = set()
        self.active_path = []

        self._draw_base()

    def _cell_rect(self, tag_id: int):
        r, c = node_id_to_grid_coord(tag_id)
        return c * CELL_SIZE, r * CELL_SIZE, CELL_SIZE, CELL_SIZE

    def _cell_center(self, tag_id: int) -> Tuple[float, float]:
        x, y, w, h = self._cell_rect(tag_id)
        return x + w / 2.0, y + h / 2.0

    def _draw_base(self):
        self.scene.clear()
        self.cells.clear()
        self.labels.clear()
        self.path_items.clear()
        self.robot_item = None
        self.robot_heading_item = None

        self.scene.setSceneRect(0, 0, COLS * CELL_SIZE, ROWS * CELL_SIZE)

        font = QFont("Arial", 15)
        font.setBold(True)

        for tag_id in range(1, ROWS * COLS + 1):
            x, y, w, h = self._cell_rect(tag_id)
            rect = self.scene.addRect(x, y, w, h, QPen(QColor(35, 35, 35), 1), QColor(245, 245, 245))
            rect.setZValue(0)
            self.cells[tag_id] = rect

            label = QGraphicsTextItem(str(tag_id))
            label.setFont(font)
            label.setDefaultTextColor(QColor(80, 80, 80))
            self.scene.addItem(label)
            br = label.boundingRect()
            label.setPos(x + (w - br.width()) / 2, y + (h - br.height()) / 2)
            label.setZValue(2)
            self.labels[tag_id] = label

        self._ensure_robot_item()
        self.apply_styles()
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def _ensure_robot_item(self):
        if self.robot_item is None:
            radius = CELL_SIZE * 0.20
            self.robot_item = QGraphicsEllipseItem(0, 0, radius * 2, radius * 2)
            self.robot_item.setBrush(QColor(52, 152, 219))
            self.robot_item.setPen(QPen(QColor(20, 90, 160), 2))
            self.robot_item.setZValue(10)
            self.scene.addItem(self.robot_item)
            self.robot_item.setVisible(False)

        if self.robot_heading_item is None:
            self.robot_heading_item = QGraphicsPathItem()
            pen = QPen(QColor(20, 60, 120), 4)
            pen.setCapStyle(Qt.RoundCap)
            self.robot_heading_item.setPen(pen)
            self.robot_heading_item.setZValue(11)
            self.scene.addItem(self.robot_heading_item)
            self.robot_heading_item.setVisible(False)

    def apply_styles(self):
        for tag_id, rect in self.cells.items():
            rect.setPen(QPen(QColor(35, 35, 35), 1))
            rect.setBrush(QColor(245, 245, 245))

        for tag_id in self.blocked_tags:
            if tag_id in self.cells:
                self.cells[tag_id].setPen(QPen(QColor(180, 40, 40), 3))
                self.cells[tag_id].setBrush(QColor(235, 180, 180))

        if self.start_tag in self.cells:
            self.cells[self.start_tag].setPen(QPen(QColor(20, 140, 60), 3))
            self.cells[self.start_tag].setBrush(QColor(180, 235, 190))

        if self.goal_tag in self.cells:
            self.cells[self.goal_tag].setPen(QPen(QColor(180, 130, 20), 3))
            self.cells[self.goal_tag].setBrush(QColor(245, 225, 140))

    def set_start_goal_blocked(self, start_tag, goal_tag, blocked_tags):
        self.start_tag = start_tag
        self.goal_tag = goal_tag
        self.blocked_tags = set(blocked_tags)
        self.apply_styles()

    def clear_paths(self):
        for item in self.path_items:
            self.scene.removeItem(item)
        self.path_items.clear()

    def set_paths(self, paths: List[List[int]], selected_index: int = 0):
        self.clear_paths()
        for idx, path in enumerate(paths):
            if not path:
                continue
            color = PATH_COLORS[idx % len(PATH_COLORS)]
            alpha = 230 if idx == selected_index else 120
            pen_width = 5 if idx == selected_index else 3
            pen_color = QColor(color)
            pen_color.setAlpha(alpha)
            pen = QPen(pen_color, pen_width)
            pen.setCosmetic(True)

            painter_path = QPainterPath()
            for i, tag_id in enumerate(path):
                cx, cy = self._cell_center(tag_id)
                if i == 0:
                    painter_path.moveTo(cx, cy)
                else:
                    painter_path.lineTo(cx, cy)

            item = QGraphicsPathItem(painter_path)
            item.setPen(pen)
            item.setZValue(5)
            self.scene.addItem(item)
            self.path_items.append(item)

    def show_robot_at_tag(self, tag_id: Optional[int], heading: Optional[int] = None):
        self._ensure_robot_item()
        if tag_id is None:
            self.robot_item.setVisible(False)
            self.robot_heading_item.setVisible(False)
            return

        self.robot_tag = tag_id
        if heading is not None:
            self.robot_heading = heading

        cx, cy = self._cell_center(tag_id)
        radius = CELL_SIZE * 0.20
        self.robot_item.setRect(cx - radius, cy - radius, radius * 2, radius * 2)
        self.robot_item.setVisible(True)

        vec = {
            NORTH: (0, -1),
            EAST: (1, 0),
            SOUTH: (0, 1),
            WEST: (-1, 0),
        }.get(self.robot_heading)

        if vec is None:
            self.robot_heading_item.setVisible(False)
            return

        length = CELL_SIZE * 0.28
        path = QPainterPath()
        path.moveTo(cx, cy)
        path.lineTo(cx + vec[0] * length, cy + vec[1] * length)
        self.robot_heading_item.setPath(path)
        self.robot_heading_item.setVisible(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            pos = self.mapToScene(event.pos())
            col = int(pos.x() / CELL_SIZE)
            row = int(pos.y() / CELL_SIZE)
            if 0 <= row < ROWS and 0 <= col < COLS:
                self.tagClicked.emit(grid_coord_to_node_id(row, col))
                return
        super().mousePressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)


# ============================================================
# MAIN QT APP
# ============================================================

class AGVClusterQtApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AGV Qt A-Star Closed Loop - Tag Cluster")
        self.resize(1500, 900)

        # Planning state.
        self.blocked_tags = set()
        self.graph = create_grid_graph(self.blocked_tags)
        self.current_tag: Optional[int] = None
        self.goal_tag: Optional[int] = None
        self.paths: List[List[int]] = []
        self.selected_path: List[int] = []

        # Robot heading and path execution state.
        self.current_heading: Optional[int] = SOUTH
        self.active_path: List[int] = []
        self.current_path_index = 0
        self.expected_next_tag: Optional[int] = None
        self.mission_running = False
        self.waiting_for_feedback = False

        # Calibration flags.
        self.calibration_started = False
        self.dock_tag_confirmed = False
        self.waiting_for_manual_alignment = False
        self.calibration_done = False

        # Camera state.
        self.camera_thread: Optional[AprilTagCameraThread] = None
        self.latest_camera_tag: Optional[int] = None

        # Serial state.
        self.ser = None

        self._build_ui()
        self._connect_signals()
        self._reset_to_uncalibrated_state()

        QTimer.singleShot(500, self.start_camera_feedback)

    # --------------------------------------------------------
    # UI
    # --------------------------------------------------------

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main = QHBoxLayout(central)

        left = QVBoxLayout()
        right = QVBoxLayout()
        main.addLayout(left, 3)
        main.addLayout(right, 2)

        # Map group.
        map_group = QGroupBox("5 x 3 Tag Grid: Tags 1-15")
        map_layout = QVBoxLayout(map_group)
        self.grid_view = TagGridView(self)
        map_layout.addWidget(self.grid_view)
        left.addWidget(map_group, 3)

        # Camera group.
        camera_group = QGroupBox("Live Camera Feedback")
        camera_layout = QVBoxLayout(camera_group)
        self.camera_label = QLabel("Camera not started")
        self.camera_label.setAlignment(Qt.AlignCenter)
        self.camera_label.setMinimumSize(640, 360)
        self.camera_label.setStyleSheet("background:#111; color:white; border:1px solid #555;")
        self.camera_status_label = QLabel("Camera status: idle")
        self.camera_status_label.setStyleSheet("font-weight:bold;")
        camera_layout.addWidget(self.camera_label)
        camera_layout.addWidget(self.camera_status_label)
        left.addWidget(camera_group, 2)

        # Calibration group.
        calibration_group = QGroupBox("Docking Calibration")
        calibration_layout = QGridLayout(calibration_group)
        self.calibration_state = QLineEdit()
        self.calibration_state.setReadOnly(True)
        self.calibration_state.setText("NOT CALIBRATED")

        self.start_calibration_btn = QPushButton("Start Calibration")
        self.reset_calibration_btn = QPushButton("Reset Calibration")
        self.manual_s_label = QLabel("After Tag 0 is detected and robot is manually aligned, press S")
        self.manual_s_label.setWordWrap(True)

        calibration_layout.addWidget(QLabel("State:"), 0, 0)
        calibration_layout.addWidget(self.calibration_state, 0, 1, 1, 2)
        calibration_layout.addWidget(self.start_calibration_btn, 1, 0)
        calibration_layout.addWidget(self.reset_calibration_btn, 1, 1)
        calibration_layout.addWidget(self.manual_s_label, 2, 0, 1, 3)
        right.addWidget(calibration_group)

        # Mission group.
        mission_group = QGroupBox("Mission Control")
        mission_layout = QGridLayout(mission_group)

        self.current_tag_text = QLineEdit()
        self.current_tag_text.setReadOnly(True)
        self.goal_tag_text = QLineEdit()
        self.goal_tag_text.setReadOnly(True)
        self.heading_text = QLineEdit()
        self.heading_text.setReadOnly(True)
        self.expected_text = QLineEdit()
        self.expected_text.setReadOnly(True)
        self.path_text = QLineEdit()
        self.path_text.setReadOnly(True)

        self.simulation_checkbox = QCheckBox("Simulation mode")
        self.simulation_checkbox.setChecked(True)
        self.closed_loop_checkbox = QCheckBox("Closed-loop camera feedback")
        self.closed_loop_checkbox.setChecked(True)

        self.start_mission_btn = QPushButton("Start Mission")
        self.stop_mission_btn = QPushButton("Stop")
        self.status_btn = QPushButton("ESP32 Status")
        self.connect_btn = QPushButton("Connect ESP32")

        mission_layout.addWidget(QLabel("Current Tag:"), 0, 0)
        mission_layout.addWidget(self.current_tag_text, 0, 1, 1, 2)
        mission_layout.addWidget(QLabel("Goal Tag:"), 1, 0)
        mission_layout.addWidget(self.goal_tag_text, 1, 1, 1, 2)
        mission_layout.addWidget(QLabel("Heading:"), 2, 0)
        mission_layout.addWidget(self.heading_text, 2, 1, 1, 2)
        mission_layout.addWidget(QLabel("Expected Next:"), 3, 0)
        mission_layout.addWidget(self.expected_text, 3, 1, 1, 2)
        mission_layout.addWidget(QLabel("Path:"), 4, 0)
        mission_layout.addWidget(self.path_text, 4, 1, 1, 2)
        mission_layout.addWidget(self.simulation_checkbox, 5, 0)
        mission_layout.addWidget(self.closed_loop_checkbox, 5, 1, 1, 2)
        mission_layout.addWidget(self.connect_btn, 6, 0)
        mission_layout.addWidget(self.status_btn, 6, 1)
        mission_layout.addWidget(self.start_mission_btn, 7, 0)
        mission_layout.addWidget(self.stop_mission_btn, 7, 1)
        right.addWidget(mission_group)

        # Path selection group.
        path_group = QGroupBox("Destination / A-Star Path")
        path_layout = QVBoxLayout(path_group)
        self.destination_combo = QComboBox()
        for tag_id in range(1, ROWS * COLS + 1):
            self.destination_combo.addItem(f"Tag {tag_id}", tag_id)
        self.compute_btn = QPushButton("Compute A-Star Path")
        self.path_list_text = QTextEdit()
        self.path_list_text.setReadOnly(True)
        self.path_list_text.setMinimumHeight(120)
        path_layout.addWidget(QLabel("Select destination:"))
        path_layout.addWidget(self.destination_combo)
        path_layout.addWidget(self.compute_btn)
        path_layout.addWidget(self.path_list_text)
        right.addWidget(path_group)

        # Log group.
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMinimumHeight(250)
        log_layout.addWidget(self.log_text)
        right.addWidget(log_group, 1)

        self.setStyleSheet("""
            QMainWindow { background: #ecf0f1; }
            QGroupBox { font-weight: bold; border: 1px solid #95a5a6; border-radius: 5px; margin-top: 10px; padding-top: 10px; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QPushButton { padding: 8px; font-weight: bold; }
            QLineEdit { padding: 6px; background: white; }
            QTextEdit { background: #fbfbfb; }
        """)

    def _connect_signals(self):
        self.grid_view.tagClicked.connect(self.on_grid_tag_clicked)
        self.start_calibration_btn.clicked.connect(self.start_calibration)
        self.reset_calibration_btn.clicked.connect(self.reset_calibration)
        self.compute_btn.clicked.connect(self.compute_path_to_selected_destination)
        self.start_mission_btn.clicked.connect(self.start_mission)
        self.stop_mission_btn.clicked.connect(self.stop_mission)
        self.connect_btn.clicked.connect(self.connect_esp32)
        self.status_btn.clicked.connect(lambda: self.send_esp32("STATUS"))
        self.destination_combo.currentIndexChanged.connect(lambda _: self.compute_path_to_selected_destination(auto=True))

    # --------------------------------------------------------
    # Logging / UI state
    # --------------------------------------------------------

    def append_log(self, text: str):
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.append(f"[{timestamp}] {text}")

    def _update_status_fields(self):
        self.current_tag_text.setText("---" if self.current_tag is None else f"Tag {self.current_tag}")
        self.goal_tag_text.setText("---" if self.goal_tag is None else f"Tag {self.goal_tag}")
        self.heading_text.setText("---" if self.current_heading is None else HEADING_LABELS.get(self.current_heading, "---"))
        self.expected_text.setText("---" if self.expected_next_tag is None else f"Tag {self.expected_next_tag}")
        self.path_text.setText("---" if not self.selected_path else " → ".join(map(str, self.selected_path)))

        if self.calibration_done:
            state = "DONE"
        elif self.waiting_for_manual_alignment:
            state = "TAG 0 CONFIRMED - PRESS S AFTER ALIGNMENT"
        elif self.calibration_started:
            state = "WAITING FOR TAG 0"
        else:
            state = "NOT CALIBRATED"
        self.calibration_state.setText(state)

        self.grid_view.set_start_goal_blocked(
            self.current_tag if self.current_tag is not None else GRID_START_TAG,
            self.goal_tag,
            self.blocked_tags,
        )
        self.grid_view.show_robot_at_tag(self.current_tag, self.current_heading)

    def _reset_to_uncalibrated_state(self):
        self.current_tag = None
        self.goal_tag = None
        self.selected_path = []
        self.paths = []
        self.active_path = []
        self.current_path_index = 0
        self.expected_next_tag = None
        self.mission_running = False
        self.waiting_for_feedback = False
        self.current_heading = SOUTH
        self.calibration_started = False
        self.dock_tag_confirmed = False
        self.waiting_for_manual_alignment = False
        self.calibration_done = False
        self.grid_view.clear_paths()
        self.path_list_text.clear()
        self._update_status_fields()

    # --------------------------------------------------------
    # Camera feedback
    # --------------------------------------------------------

    def start_camera_feedback(self):
        if self.camera_thread is not None and self.camera_thread.isRunning():
            return
        self.camera_thread = AprilTagCameraThread(camera_index=CAMERA_INDEX)
        self.camera_thread.frame_ready.connect(self.update_camera_frame)
        self.camera_thread.tag_detected.connect(self.on_camera_tag_detected)
        self.camera_thread.status.connect(self.on_camera_status)
        self.camera_thread.start()
        self.append_log("Camera feedback started")

    def stop_camera_feedback(self):
        if self.camera_thread is not None:
            self.camera_thread.stop()
            self.camera_thread = None

    def update_camera_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)
        self.camera_label.setPixmap(
            pixmap.scaled(
                self.camera_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation,
            )
        )

    def on_camera_status(self, text: str):
        self.camera_status_label.setText(f"Camera status: {text}")

    def on_camera_tag_detected(self, tag_id: int):
        self.latest_camera_tag = tag_id

        # Calibration feedback.
        if self.calibration_started and not self.calibration_done:
            if tag_id == DOCK_TAG:
                if not self.dock_tag_confirmed:
                    self.append_log("Dock Tag 0 detected. Manually align robot and press S.")
                self.dock_tag_confirmed = True
                self.waiting_for_manual_alignment = True
                self._update_status_fields()
            return

        # Mission closed-loop feedback.
        if not self.mission_running:
            return

        if self.expected_next_tag is None:
            return

        if tag_id == self.expected_next_tag:
            self.append_log(f"Closed-loop feedback OK: detected expected Tag {tag_id}")
            self.advance_after_tag_feedback(tag_id)
        else:
            self.append_log(f"Feedback mismatch: detected Tag {tag_id}, expected Tag {self.expected_next_tag}")

    # --------------------------------------------------------
    # ESP32 serial
    # --------------------------------------------------------

    def connect_esp32(self) -> bool:
        if self.simulation_checkbox.isChecked():
            self.append_log("Simulation mode ON: ESP32 connection not required")
            return True

        if serial is None:
            QMessageBox.critical(self, "Serial Error", "pyserial is not installed")
            return False

        if self.ser is not None and self.ser.is_open:
            self.append_log("ESP32 already connected")
            return True

        try:
            self.ser = serial.Serial(SERIAL_PORT, SERIAL_BAUD, timeout=0.2)
            time.sleep(1.5)
            self.append_log(f"Connected to ESP32 on {SERIAL_PORT} @ {SERIAL_BAUD}")
            return True
        except Exception as e:
            QMessageBox.critical(self, "ESP32 Connection Failed", str(e))
            self.append_log(f"ESP32 connection failed: {e}")
            return False

    def send_esp32(self, cmd: str):
        if self.simulation_checkbox.isChecked():
            self.append_log(f"SIM ESP32 <- {cmd}")
            return

        if not self.connect_esp32():
            return

        try:
            self.ser.write((cmd.strip() + "\n").encode("utf-8"))
            self.ser.flush()
            self.append_log(f"ESP32 <- {cmd}")

            # Read short response window.
            deadline = time.time() + 0.35
            while time.time() < deadline:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if line:
                    self.append_log(f"ESP32 -> {line}")
        except Exception as e:
            self.append_log(f"Serial send failed: {e}")

    # --------------------------------------------------------
    # Calibration
    # --------------------------------------------------------

    def start_calibration(self):
        if self.mission_running:
            QMessageBox.warning(self, "Mission Running", "Stop the mission before calibration")
            return

        self.calibration_started = True
        self.dock_tag_confirmed = False
        self.waiting_for_manual_alignment = False
        self.calibration_done = False
        self.current_tag = None
        self.goal_tag = None
        self.selected_path = []
        self.paths = []
        self.active_path = []
        self.expected_next_tag = None
        self.grid_view.clear_paths()
        self.path_list_text.clear()
        self.append_log("Calibration started. Place robot at docking Tag 0.")
        self.append_log("Waiting for camera detection of Tag 0...")
        self._update_status_fields()

    def reset_calibration(self):
        self.send_esp32("STOP")
        self._reset_to_uncalibrated_state()
        self.append_log("Calibration reset. Robot must start again from Tag 0.")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S:
            self.confirm_manual_alignment()
            return
        super().keyPressEvent(event)

    def confirm_manual_alignment(self):
        if not self.calibration_started:
            self.append_log("Press Start Calibration first")
            return
        if not self.dock_tag_confirmed:
            self.append_log("Cannot confirm alignment yet. Tag 0 has not been detected.")
            return
        if not self.waiting_for_manual_alignment:
            self.append_log("Not waiting for manual alignment")
            return

        self.append_log("Manual alignment confirmed by S key")

        # ESP32 IMU recalibration should happen while robot is still aligned at Tag 0.
        self.send_esp32("STOP")
        self.send_esp32("SET_BASE %d" % ESP32_BASE_PPS)
        self.send_esp32("SET_IMU_MAX %d" % ESP32_IMU_MAX_CORR)
        self.send_esp32("IMU RECAL")

        if self.simulation_checkbox.isChecked():
            self.append_log("Simulation: setting robot to grid Tag 1")
            self.complete_calibration_at_tag_1()
            return

        # Timed move from Dock 0 to Grid Tag 1.
        self.append_log("Moving from Dock Tag 0 to Grid Tag 1")
        self.send_esp32("LOCK_HEADING_GO")
        QTimer.singleShot(int(DOCK_TO_TAG1_TRAVEL_SEC * 1000), self.finish_dock_to_tag_1_real)

    def finish_dock_to_tag_1_real(self):
        self.send_esp32("STOP")
        self.complete_calibration_at_tag_1()

    def complete_calibration_at_tag_1(self):
        self.calibration_started = False
        self.dock_tag_confirmed = True
        self.waiting_for_manual_alignment = False
        self.calibration_done = True
        self.current_tag = GRID_START_TAG
        self.current_heading = SOUTH
        self.goal_tag = None
        self.selected_path = []
        self.active_path = []
        self.expected_next_tag = None
        self.append_log("Calibration complete. Planner is now at Tag 1.")
        self.append_log("Select a destination tag and start mission.")
        self._update_status_fields()

    # --------------------------------------------------------
    # Path planning
    # --------------------------------------------------------

    def on_grid_tag_clicked(self, tag_id: int):
        if not self.calibration_done:
            self.append_log("Grid selection ignored. Complete docking calibration first.")
            return
        self.goal_tag = tag_id
        index = self.destination_combo.findData(tag_id)
        if index >= 0:
            self.destination_combo.setCurrentIndex(index)
        self.compute_path_to_selected_destination()

    def compute_path_to_selected_destination(self, auto: bool = False):
        if not self.calibration_done:
            if not auto:
                self.append_log("Complete calibration before computing path")
            return
        if self.current_tag is None:
            return

        self.goal_tag = int(self.destination_combo.currentData())
        if self.goal_tag == self.current_tag:
            self.selected_path = [self.current_tag]
            self.paths = [self.selected_path]
            self.path_list_text.setText("Already at selected tag")
            self.grid_view.set_paths(self.paths, 0)
            self._update_status_fields()
            return

        self.graph = create_grid_graph(self.blocked_tags)

        # Use A-star for primary path.
        primary = astar_path(self.graph, self.current_tag, self.goal_tag)

        # Use k shortest for alternatives display.
        alternatives = k_shortest_paths(self.graph, self.current_tag, self.goal_tag, NUM_PATHS)

        self.paths = []
        if primary:
            self.paths.append(primary)
        for path in alternatives:
            if path not in self.paths:
                self.paths.append(path)

        if not self.paths:
            self.selected_path = []
            self.path_list_text.setText(f"No path from Tag {self.current_tag} to Tag {self.goal_tag}")
            self.grid_view.clear_paths()
            self._update_status_fields()
            return

        self.selected_path = self.paths[0]
        lines = []
        for i, path in enumerate(self.paths):
            dist_mm = int(path_length_m(self.graph, path) * 1000)
            prefix = "A*" if i == 0 else f"Alt {i}"
            lines.append(f"{prefix}: {' → '.join(map(str, path))}    {dist_mm} mm")
        self.path_list_text.setText("\n".join(lines))
        self.grid_view.set_paths(self.paths, 0)
        self.append_log(f"Computed path: {' → '.join(map(str, self.selected_path))}")
        self._update_status_fields()

    # --------------------------------------------------------
    # Mission execution
    # --------------------------------------------------------

    def start_mission(self):
        if not self.calibration_done:
            QMessageBox.warning(self, "Calibration Required", "Complete docking calibration first")
            return

        if not self.selected_path:
            self.compute_path_to_selected_destination()

        if not self.selected_path or len(self.selected_path) < 2:
            QMessageBox.information(self, "No Movement", "Select a destination different from current tag")
            return

        if not self.simulation_checkbox.isChecked() and not self.connect_esp32():
            return

        self.mission_running = True
        self.active_path = list(self.selected_path)
        self.current_path_index = 1
        self.expected_next_tag = self.active_path[self.current_path_index]
        self.waiting_for_feedback = bool(self.closed_loop_checkbox.isChecked())

        self.append_log(f"Mission started: {' → '.join(map(str, self.active_path))}")
        self.append_log(f"Expected next tag: {self.expected_next_tag}")
        self._update_status_fields()

        self.command_next_segment()

    def stop_mission(self):
        self.mission_running = False
        self.waiting_for_feedback = False
        self.expected_next_tag = None
        self.send_esp32("STOP")
        self.append_log("Mission stopped")
        self._update_status_fields()

    def command_next_segment(self):
        if not self.mission_running:
            return
        if self.current_path_index >= len(self.active_path):
            self.finish_mission()
            return

        prev_tag = self.current_tag
        next_tag = self.active_path[self.current_path_index]

        if prev_tag is None:
            self.append_log("Cannot move: current tag unknown")
            self.stop_mission()
            return

        required_heading = heading_between_tags(prev_tag, next_tag)
        if required_heading is None:
            self.append_log(f"Invalid segment: Tag {prev_tag} to Tag {next_tag} is not adjacent")
            self.stop_mission()
            return

        turn = turn_degrees(self.current_heading, required_heading)
        self.append_log(
            f"Segment {prev_tag} → {next_tag}: required heading {HEADING_LABELS[required_heading]}, turn {turn:.0f} deg"
        )

        self.current_heading = required_heading
        self.grid_view.show_robot_at_tag(self.current_tag, self.current_heading)
        self._update_status_fields()

        if abs(turn) > 0.1:
            self.send_esp32(f"TURN_REL {turn:.1f}")
            QTimer.singleShot(int(TURN_SETTLE_SEC * 1000), self._start_straight_for_segment)
        else:
            self._start_straight_for_segment()

    def _start_straight_for_segment(self):
        if not self.mission_running:
            return

        next_tag = self.active_path[self.current_path_index]
        self.expected_next_tag = next_tag
        self._update_status_fields()

        self.send_esp32("LOCK_HEADING_GO")
        self.append_log(f"Driving toward Tag {next_tag}")

        if self.closed_loop_checkbox.isChecked():
            self.waiting_for_feedback = True
            self.append_log(f"Closed loop: waiting for camera to detect Tag {next_tag}")
        else:
            # Open loop fallback. Advance by timer.
            self.waiting_for_feedback = False
            QTimer.singleShot(int(CELL_TRAVEL_SEC * 1000), lambda: self.advance_after_tag_feedback(next_tag))

    def advance_after_tag_feedback(self, tag_id: int):
        if not self.mission_running:
            return
        if self.current_path_index >= len(self.active_path):
            return

        expected = self.active_path[self.current_path_index]
        if tag_id != expected:
            return

        self.send_esp32("STOP")
        self.current_tag = tag_id
        self.current_path_index += 1
        self.waiting_for_feedback = False
        self.append_log(f"Arrived at Tag {tag_id}")

        if self.current_path_index >= len(self.active_path):
            self.finish_mission()
            return

        self.expected_next_tag = self.active_path[self.current_path_index]
        self._update_status_fields()
        QTimer.singleShot(150, self.command_next_segment)

    def finish_mission(self):
        self.send_esp32("STOP")
        self.mission_running = False
        self.waiting_for_feedback = False
        self.expected_next_tag = None
        self.append_log("Mission complete")
        self._update_status_fields()
        QMessageBox.information(self, "Mission Complete", "AGV reached the destination tag")

    # --------------------------------------------------------
    # Clean close
    # --------------------------------------------------------

    def closeEvent(self, event):
        try:
            self.stop_camera_feedback()
        except Exception as e:
            print(f"Camera cleanup error: {e}")

        try:
            self.send_esp32("STOP")
        except Exception:
            pass

        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

        event.accept()


# ============================================================
# MAIN
# ============================================================

def main():
    app = QApplication(sys.argv)
    win = AGVClusterQtApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
