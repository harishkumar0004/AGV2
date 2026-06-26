"""
Qt based AGV navigation for 5x3 AprilTag cluster with closed-loop simulation.

Tag 0  = docking / IMU calibration only
Tags 1-15 = A-star grid map

Closed-loop simulation idea:
    The simulated robot marker does NOT advance only by timer.
    It advances only when the camera detects the expected next AprilTag.

Calibration flow:
    1. Place robot on docking tag 0.
    2. Click Start Calibration.
    3. Camera must detect tag 0.
    4. Manually align the robot.
    5. Press S.
    6. Python sends IMU RECAL in real mode, then moves from tag 0 to tag 1.
    7. Planner becomes active from tag 1.

Install:
    pip install PyQt5 pyserial opencv-python pupil-apriltags

Run:
    python3 qt_agv_cluster_closed_loop.py

ESP32 serial protocol used for real hardware:
    IMU RECAL
    SET_BASE value
    SET_IMU_MAX value
    TURN_REL deg
    LOCK_HEADING_GO
    STOP
    STATUS
"""

import sys
import time
import heapq
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    import serial
except ImportError:
    serial = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from pupil_apriltags import Detector
except ImportError:
    Detector = None

from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
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


# =========================================================
# MAP CONFIGURATION
# =========================================================

DOCK_TAG = 0
ROWS = 3
COLS = 5
CELL_DISTANCE_M = 0.5

# Used only for open-loop fallback simulation.
OPEN_LOOP_SIM_STEP_MS = 700

# Timed docking-to-grid-start move. Tune this for the physical distance from tag 0 to tag 1.
DOCK_TO_TAG1_TRAVEL_SEC = 1.20

SERIAL_PORT = "/dev/ttyUSB0"
BAUDRATE = 115200
CAMERA_INDEX = 0

BASE_PPS = 6500
IMU_MAX_CORR = 350

# Heading convention used by the planner.
NORTH = 0
EAST = 90
SOUTH = 180
WEST = -90

HEADING_NAMES = {
    NORTH: "NORTH",
    EAST: "EAST",
    SOUTH: "SOUTH",
    WEST: "WEST",
}


# =========================================================
# GRID / TAG HELPERS
# =========================================================

def tag_to_coord(tag_id: int) -> Tuple[int, int]:
    """Convert tag 1-15 to 0-based row,col."""
    if tag_id < 1 or tag_id > ROWS * COLS:
        raise ValueError(f"Grid tag must be 1-{ROWS * COLS}, got {tag_id}")
    zero = tag_id - 1
    return zero // COLS, zero % COLS


def coord_to_tag(row: int, col: int) -> int:
    """Convert 0-based row,col to tag 1-15."""
    return row * COLS + col + 1


def valid_coord(row: int, col: int) -> bool:
    return 0 <= row < ROWS and 0 <= col < COLS


def neighbors(tag_id: int, blocked: set) -> List[int]:
    row, col = tag_to_coord(tag_id)
    result = []
    for dr, dc in [(-1, 0), (0, 1), (1, 0), (0, -1)]:
        nr, nc = row + dr, col + dc
        if valid_coord(nr, nc):
            nt = coord_to_tag(nr, nc)
            if nt not in blocked:
                result.append(nt)
    return result


def manhattan(a: int, b: int) -> int:
    ar, ac = tag_to_coord(a)
    br, bc = tag_to_coord(b)
    return abs(ar - br) + abs(ac - bc)


def astar_path(start: int, goal: int, blocked: set) -> List[int]:
    """A-star route over tags 1-15. Tag 0 is not part of the graph."""
    if start in blocked or goal in blocked:
        return []

    open_heap = []
    heapq.heappush(open_heap, (0, start))

    came_from: Dict[int, Optional[int]] = {start: None}
    g_score: Dict[int, float] = {start: 0.0}

    while open_heap:
        _, current = heapq.heappop(open_heap)

        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = came_from[current]
            return list(reversed(path))

        for nxt in neighbors(current, blocked):
            tentative_g = g_score[current] + 1.0
            if nxt not in g_score or tentative_g < g_score[nxt]:
                came_from[nxt] = current
                g_score[nxt] = tentative_g
                f_score = tentative_g + manhattan(nxt, goal)
                heapq.heappush(open_heap, (f_score, nxt))

    return []


def heading_between(a: int, b: int) -> int:
    ar, ac = tag_to_coord(a)
    br, bc = tag_to_coord(b)
    dr = br - ar
    dc = bc - ac

    if dr == -1 and dc == 0:
        return NORTH
    if dr == 0 and dc == 1:
        return EAST
    if dr == 1 and dc == 0:
        return SOUTH
    if dr == 0 and dc == -1:
        return WEST

    raise ValueError(f"Tags {a}->{b} are not adjacent in the 5x3 grid")


def normalize_turn_deg(deg: int) -> int:
    while deg > 180:
        deg -= 360
    while deg <= -180:
        deg += 360
    return deg


def turn_needed(current_heading: int, target_heading: int) -> int:
    return normalize_turn_deg(target_heading - current_heading)


# =========================================================
# APRILTAG DETECTION THREAD
# =========================================================

class AprilTagDetectorThread(QThread):
    tag_detected = pyqtSignal(int)
    status = pyqtSignal(str)

    def __init__(self, camera_index: int = CAMERA_INDEX, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self._running = False
        self._last_emit_tag: Optional[int] = None
        self._last_emit_time = 0.0

    def stop(self):
        self._running = False

    def run(self):
        if cv2 is None:
            self.status.emit("OpenCV is not installed. Run: pip install opencv-python")
            return
        if Detector is None:
            self.status.emit("pupil-apriltags is not installed. Run: pip install pupil-apriltags")
            return

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            self.status.emit(f"Could not open camera index {self.camera_index}")
            return

        detector = Detector(
            families="tag36h11",
            nthreads=2,
            quad_decimate=1.5,
            quad_sigma=0.0,
            refine_edges=1,
            decode_sharpening=0.25,
            debug=0,
        )

        self._running = True
        self.status.emit("AprilTag detector running")

        while self._running:
            ok, frame = cap.read()
            if not ok:
                self.msleep(30)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray)

            if detections:
                # Choose the biggest detection, which is usually the closest / most stable tag.
                best = max(detections, key=lambda d: abs(cv2.contourArea(d.corners.astype("float32"))))
                tag_id = int(best.tag_id)

                now = time.time()
                # Avoid flooding the UI with the same tag hundreds of times per second.
                if tag_id != self._last_emit_tag or now - self._last_emit_time > 0.35:
                    self._last_emit_tag = tag_id
                    self._last_emit_time = now
                    self.tag_detected.emit(tag_id)

            self.msleep(20)

        cap.release()
        self.status.emit("AprilTag detector stopped")


# =========================================================
# SERIAL EXECUTOR THREAD FOR REAL HARDWARE
# =========================================================

@dataclass
class SerialConfig:
    port: str = SERIAL_PORT
    baudrate: int = BAUDRATE
    timeout: float = 0.1


class RobotExecutor(QThread):
    log = pyqtSignal(str)
    position_changed = pyqtSignal(int)
    heading_changed = pyqtSignal(int)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, path: List[int], parent=None):
        super().__init__(parent)
        self.path = path
        self._stop_requested = False
        self.ser = None
        self.current_heading = SOUTH
        self.config = SerialConfig()

    def request_stop(self):
        self._stop_requested = True
        try:
            self.send("STOP")
        except Exception:
            pass

    def open_serial(self):
        if serial is None:
            raise RuntimeError("pyserial is not installed. Run: pip install pyserial")
        self.ser = serial.Serial(self.config.port, self.config.baudrate, timeout=self.config.timeout)
        time.sleep(2.0)
        self.flush_input()

    def flush_input(self):
        if self.ser:
            self.ser.reset_input_buffer()

    def send(self, cmd: str):
        if self.ser is None:
            raise RuntimeError("Serial port is not open")
        self.log.emit(f">> {cmd}")
        self.ser.write((cmd + "\n").encode("utf-8"))
        self.ser.flush()

    def read_lines_for(self, seconds: float) -> List[str]:
        lines = []
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self._stop_requested:
                break
            line = self.ser.readline().decode(errors="ignore").strip()
            if line:
                lines.append(line)
                self.log.emit(line)
        return lines

    def wait_for_text(self, text: str, timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self._stop_requested:
                return False
            line = self.ser.readline().decode(errors="ignore").strip()
            if line:
                self.log.emit(line)
                if text in line:
                    return True
        return False

    def run(self):
        try:
            if len(self.path) < 1:
                raise RuntimeError("Empty path")

            self.open_serial()

            self.log.emit("Dock tag 0 is calibration-only. Keep AGV still at docking area.")
            self.send("STOP")
            self.read_lines_for(0.3)

            self.send(f"SET_BASE {BASE_PPS}")
            self.read_lines_for(0.4)

            self.send(f"SET_IMU_MAX {IMU_MAX_CORR}")
            self.read_lines_for(0.4)

            self.send("IMU RECAL")
            if not self.wait_for_text("OK IMU RECAL", 15.0):
                raise RuntimeError("IMU recalibration failed or timed out")

            self.current_heading = SOUTH
            self.heading_changed.emit(self.current_heading)
            self.position_changed.emit(self.path[0])

            # Real hardware still uses short timed cells here.
            # Closed-loop real stop can be added by sharing the detector state
            # and stopping when the expected tag is detected.
            for i in range(len(self.path) - 1):
                if self._stop_requested:
                    raise RuntimeError("Stopped by user")

                src = self.path[i]
                dst = self.path[i + 1]
                target_heading = heading_between(src, dst)
                deg = turn_needed(self.current_heading, target_heading)

                self.log.emit(f"Segment {src} -> {dst}")

                if deg != 0:
                    self.send(f"TURN_REL {deg}")
                    if not self.wait_for_text("OK TURN_DONE", 12.0):
                        raise RuntimeError(f"Turn {deg} deg timed out")
                    self.current_heading = target_heading
                    self.heading_changed.emit(self.current_heading)

                self.send("LOCK_HEADING_GO")
                if not self.wait_for_text("OK LOCK_HEADING_GO", 3.0):
                    raise RuntimeError("LOCK_HEADING_GO failed")

                self.read_lines_for(1.20)
                self.send("STOP")
                self.read_lines_for(0.5)

                self.position_changed.emit(dst)

            self.send("STOP")
            self.finished_ok.emit()

        except Exception as e:
            try:
                self.send("STOP")
            except Exception:
                pass
            self.failed.emit(str(e))
        finally:
            try:
                if self.ser:
                    self.ser.close()
            except Exception:
                pass


# =========================================================
# QT UI
# =========================================================

class CalibrationExecutor(QThread):
    log = pyqtSignal(str)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._stop_requested = False
        self.ser = None
        self.config = SerialConfig()

    def request_stop(self):
        self._stop_requested = True
        try:
            self.send("STOP")
        except Exception:
            pass

    def open_serial(self):
        if serial is None:
            raise RuntimeError("pyserial is not installed. Run: pip install pyserial")
        self.ser = serial.Serial(self.config.port, self.config.baudrate, timeout=self.config.timeout)
        time.sleep(2.0)
        self.ser.reset_input_buffer()

    def send(self, cmd: str):
        if self.ser is None:
            raise RuntimeError("Serial port is not open")
        self.log.emit(f">> {cmd}")
        self.ser.write((cmd + "\n").encode("utf-8"))
        self.ser.flush()

    def read_lines_for(self, seconds: float):
        deadline = time.time() + seconds
        while time.time() < deadline:
            if self._stop_requested:
                return
            line = self.ser.readline().decode(errors="ignore").strip()
            if line:
                self.log.emit(line)

    def wait_for_text(self, text: str, timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self._stop_requested:
                return False
            line = self.ser.readline().decode(errors="ignore").strip()
            if line:
                self.log.emit(line)
                if text in line:
                    return True
        return False

    def run(self):
        try:
            self.open_serial()
            self.send("STOP")
            self.read_lines_for(0.3)
            self.send(f"SET_BASE {BASE_PPS}")
            self.read_lines_for(0.3)
            self.send(f"SET_IMU_MAX {IMU_MAX_CORR}")
            self.read_lines_for(0.3)

            self.log.emit("Starting IMU calibration at docking tag 0. Keep robot still.")
            self.send("IMU RECAL")
            if not self.wait_for_text("OK IMU RECAL", 15.0):
                raise RuntimeError("IMU recalibration failed or timed out")

            if self._stop_requested:
                raise RuntimeError("Calibration stopped by user")

            self.log.emit("Moving from docking tag 0 to grid tag 1.")
            self.send("LOCK_HEADING_GO")
            if not self.wait_for_text("OK LOCK_HEADING_GO", 3.0):
                raise RuntimeError("LOCK_HEADING_GO failed during dock-to-tag-1 move")
            self.read_lines_for(DOCK_TO_TAG1_TRAVEL_SEC)
            self.send("STOP")
            self.read_lines_for(0.5)

            self.finished_ok.emit()

        except Exception as e:
            try:
                self.send("STOP")
            except Exception:
                pass
            self.failed.emit(str(e))
        finally:
            try:
                if self.ser:
                    self.ser.close()
            except Exception:
                pass


class AGVQtApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AGV Qt A-star Navigation - Closed Loop AprilTag Simulation")
        self.resize(1180, 760)

        self.current_tag: Optional[int] = None
        self.goal_tag: Optional[int] = None
        self.path: List[int] = []
        self.blocked = set()
        self.executor: Optional[RobotExecutor] = None
        self.calibration_executor: Optional[CalibrationExecutor] = None
        self.simulation_mode = True
        self.feedback_simulation = True
        self.current_heading = SOUTH

        self.detector_thread: Optional[AprilTagDetectorThread] = None
        self.latest_detected_tag: Optional[int] = None

        self.calibration_started = False
        self.dock_tag_confirmed = False
        self.waiting_for_manual_alignment = False
        self.calibration_done = False

        self.sim_running = False
        self.sim_index = 0
        self.sim_expected_tag: Optional[int] = None
        self.open_loop_timer = QTimer(self)
        self.open_loop_timer.timeout.connect(self.open_loop_advance)

        self.buttons: Dict[int, QPushButton] = {}
        self.init_ui()
        self.update_map()

    def init_ui(self):
        root = QWidget()
        self.setCentralWidget(root)
        main = QHBoxLayout(root)

        map_group = QGroupBox("5x3 AprilTag Grid  |  Tag 0 = Docking/Calibration only")
        map_layout = QGridLayout(map_group)
        map_layout.setSpacing(12)

        for r in range(ROWS):
            for c in range(COLS):
                tag = coord_to_tag(r, c)
                btn = QPushButton(str(tag))
                btn.setMinimumSize(110, 90)
                btn.setFont(QFont("Arial", 18, QFont.Bold))
                btn.clicked.connect(lambda checked=False, t=tag: self.on_tag_clicked(t))
                self.buttons[tag] = btn
                map_layout.addWidget(btn, r, c)

        side = QVBoxLayout()

        self.current_label = QLineEdit()
        self.current_label.setReadOnly(True)
        self.goal_label = QLineEdit()
        self.goal_label.setReadOnly(True)
        self.heading_label = QLineEdit()
        self.heading_label.setReadOnly(True)
        self.detected_label = QLineEdit()
        self.detected_label.setReadOnly(True)
        self.expected_label = QLineEdit()
        self.expected_label.setReadOnly(True)
        self.path_label = QTextEdit()
        self.path_label.setReadOnly(True)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        self.calibration_label = QLineEdit()
        self.calibration_label.setReadOnly(True)

        side.addWidget(QLabel("Calibration state"))
        side.addWidget(self.calibration_label)

        cal_row = QHBoxLayout()
        self.start_cal_btn = QPushButton("Start Calibration")
        self.start_cal_btn.clicked.connect(self.start_calibration)
        self.reset_cal_btn = QPushButton("Reset Calibration")
        self.reset_cal_btn.clicked.connect(self.reset_calibration)
        cal_row.addWidget(self.start_cal_btn)
        cal_row.addWidget(self.reset_cal_btn)
        side.addLayout(cal_row)

        side.addWidget(QLabel("Current simulated/planner tag"))
        side.addWidget(self.current_label)
        side.addWidget(QLabel("Goal tag"))
        side.addWidget(self.goal_label)
        side.addWidget(QLabel("Heading"))
        side.addWidget(self.heading_label)
        side.addWidget(QLabel("Latest detected AprilTag"))
        side.addWidget(self.detected_label)
        side.addWidget(QLabel("Expected next tag"))
        side.addWidget(self.expected_label)
        side.addWidget(QLabel("Computed A-star path"))
        side.addWidget(self.path_label)

        btn_row1 = QHBoxLayout()
        self.plan_btn = QPushButton("Plan")
        self.plan_btn.clicked.connect(self.plan_path)
        self.start_btn = QPushButton("Start")
        self.start_btn.clicked.connect(self.start_navigation)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_navigation)
        btn_row1.addWidget(self.plan_btn)
        btn_row1.addWidget(self.start_btn)
        btn_row1.addWidget(self.stop_btn)
        side.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        self.reset_btn = QPushButton("Reset to Tag 1")
        self.reset_btn.clicked.connect(self.reset_to_tag_1)
        self.clear_blocks_btn = QPushButton("Clear Blocks")
        self.clear_blocks_btn.clicked.connect(self.clear_blocks)
        btn_row2.addWidget(self.reset_btn)
        btn_row2.addWidget(self.clear_blocks_btn)
        side.addLayout(btn_row2)

        btn_row3 = QHBoxLayout()
        self.detector_btn = QPushButton("Start Detector")
        self.detector_btn.clicked.connect(self.toggle_detector)
        btn_row3.addWidget(self.detector_btn)
        side.addLayout(btn_row3)

        self.sim_check = QCheckBox("Simulation mode")
        self.sim_check.setChecked(True)
        self.sim_check.stateChanged.connect(self.on_sim_changed)
        side.addWidget(self.sim_check)

        self.feedback_check = QCheckBox("Closed-loop simulation using AprilTag detection")
        self.feedback_check.setChecked(True)
        self.feedback_check.stateChanged.connect(self.on_feedback_changed)
        side.addWidget(self.feedback_check)

        side.addWidget(QLabel("Log"))
        side.addWidget(self.log_box, 1)

        main.addWidget(map_group, 2)
        main.addLayout(side, 1)

    def append_log(self, text: str):
        self.log_box.append(text)

    def on_sim_changed(self, state):
        self.simulation_mode = state == Qt.Checked
        self.append_log(f"Simulation mode: {'ON' if self.simulation_mode else 'OFF'}")

    def on_feedback_changed(self, state):
        self.feedback_simulation = state == Qt.Checked
        self.append_log(f"Closed-loop simulation feedback: {'ON' if self.feedback_simulation else 'OFF'}")

    def toggle_detector(self):
        if self.detector_thread and self.detector_thread.isRunning():
            self.detector_thread.stop()
            self.detector_thread.wait(1500)
            self.detector_thread = None
            self.detector_btn.setText("Start Detector")
            return
        self.start_detector()

    def start_detector(self):
        if self.detector_thread and self.detector_thread.isRunning():
            return True
        self.detector_thread = AprilTagDetectorThread(CAMERA_INDEX, self)
        self.detector_thread.tag_detected.connect(self.on_apriltag_detected)
        self.detector_thread.status.connect(self.append_log)
        self.detector_thread.start()
        self.detector_btn.setText("Stop Detector")
        return True

    def on_apriltag_detected(self, tag_id: int):
        self.latest_detected_tag = tag_id
        self.detected_label.setText(f"Tag {tag_id}")

        # Docking tag is not part of the A-star grid.
        if tag_id == DOCK_TAG:
            if self.calibration_started and not self.calibration_done:
                if not self.dock_tag_confirmed:
                    self.append_log("Docking tag 0 detected. Manually align robot, then press S.")
                self.dock_tag_confirmed = True
                self.waiting_for_manual_alignment = True
                self.update_map()
            else:
                self.append_log("Detected docking tag 0. Calibration/start reference only.")
            return

        if not self.sim_running or not self.feedback_simulation:
            return

        if self.sim_expected_tag is None:
            return

        if tag_id == self.sim_expected_tag:
            self.append_log(f"Feedback OK: expected Tag {self.sim_expected_tag} detected")
            self.advance_sim_to_detected_tag(tag_id)
        else:
            self.append_log(f"Feedback mismatch: detected Tag {tag_id}, expected Tag {self.sim_expected_tag}")

    def on_tag_clicked(self, tag: int):
        if not self.calibration_done or self.current_tag is None:
            QMessageBox.warning(self, "Calibration Required", "Start calibration at Tag 0 first. After Tag 0 is detected, manually align and press S to move to Tag 1.")
            return

        # Left click chooses the goal. Shift-click toggles blocked node.
        if QApplication.keyboardModifiers() & Qt.ShiftModifier:
            if self.current_tag is not None and tag != self.current_tag:
                if tag in self.blocked:
                    self.blocked.remove(tag)
                else:
                    self.blocked.add(tag)
            self.plan_path()
        else:
            self.goal_tag = tag
            self.plan_path()
        self.update_map()

    def reset_to_tag_1(self):
        if not self.calibration_done:
            QMessageBox.warning(self, "Calibration Required", "Calibrate from docking tag 0 first.")
            return
        self.current_tag = 1
        self.goal_tag = None
        self.path = []
        self.current_heading = SOUTH
        self.sim_running = False
        self.sim_expected_tag = None
        self.open_loop_timer.stop()
        self.update_map()
        self.append_log("Reset planner position to Tag 1. Dock tag 0 remains calibration-only.")

    def start_calibration(self):
        self.stop_navigation()
        self.goal_tag = None
        self.path = []
        self.current_tag = None
        self.calibration_started = True
        self.dock_tag_confirmed = False
        self.waiting_for_manual_alignment = False
        self.calibration_done = False
        self.start_detector()
        self.append_log("Calibration started. Place robot on docking Tag 0 and keep it visible to the camera.")
        self.update_map()

    def reset_calibration(self):
        self.stop_navigation()
        if self.calibration_executor:
            self.calibration_executor.request_stop()
            self.calibration_executor.wait(1500)
            self.calibration_executor = None
        self.current_tag = None
        self.goal_tag = None
        self.path = []
        self.current_heading = SOUTH
        self.calibration_started = False
        self.dock_tag_confirmed = False
        self.waiting_for_manual_alignment = False
        self.calibration_done = False
        self.sim_expected_tag = None
        self.update_map()
        self.append_log("Calibration reset. Start again from docking Tag 0.")

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S:
            self.confirm_alignment_and_move_to_tag_1()
            return
        super().keyPressEvent(event)

    def confirm_alignment_and_move_to_tag_1(self):
        if not self.calibration_started:
            QMessageBox.warning(self, "Calibration Not Started", "Click Start Calibration first.")
            return
        if not self.dock_tag_confirmed:
            QMessageBox.warning(self, "Dock Tag Not Detected", "Tag 0 must be detected before pressing S.")
            return
        if not self.waiting_for_manual_alignment:
            return

        self.waiting_for_manual_alignment = False
        self.append_log("Manual alignment confirmed with S key.")

        if self.simulation_mode:
            self.current_tag = 1
            self.current_heading = SOUTH
            self.calibration_done = True
            self.calibration_started = False
            self.append_log("Simulation calibration complete. Robot moved from Tag 0 to Tag 1. Select next node on the grid.")
            self.update_map()
            return

        self.calibration_executor = CalibrationExecutor(self)
        self.calibration_executor.log.connect(self.append_log)
        self.calibration_executor.finished_ok.connect(self.on_calibration_finished_ok)
        self.calibration_executor.failed.connect(self.on_calibration_failed)
        self.calibration_executor.start()
        self.append_log("Real calibration/move started. ESP32 will recalibrate IMU and move to Tag 1.")

    def on_calibration_finished_ok(self):
        self.current_tag = 1
        self.current_heading = SOUTH
        self.calibration_done = True
        self.calibration_started = False
        self.append_log("Calibration complete. Robot is now at Tag 1. Select the next node on the Qt grid.")
        self.update_map()

    def on_calibration_failed(self, msg: str):
        self.waiting_for_manual_alignment = True
        self.append_log("CALIBRATION ERROR: " + msg)
        QMessageBox.critical(self, "Calibration Failed", msg)
        self.update_map()

    def clear_blocks(self):
        self.blocked.clear()
        self.plan_path()
        self.update_map()

    def plan_path(self):
        if self.current_tag is None:
            self.path = []
            self.update_map()
            return
        if self.goal_tag is None:
            self.path = []
            self.update_map()
            return
        self.path = astar_path(self.current_tag, self.goal_tag, self.blocked)
        if not self.path:
            self.append_log(f"No path from {self.current_tag} to {self.goal_tag}")
        else:
            self.append_log("Path: " + " -> ".join(map(str, self.path)))
        self.update_map()

    def update_map(self):
        for tag, btn in self.buttons.items():
            style = "background-color: #ecf0f1; color: #2c3e50; border: 2px solid #7f8c8d;"

            if tag in self.blocked:
                style = "background-color: #e74c3c; color: white; border: 2px solid #922b21;"
            if tag in self.path:
                style = "background-color: #85c1e9; color: black; border: 3px solid #2471a3;"
            if tag == self.goal_tag:
                style = "background-color: #f7dc6f; color: black; border: 3px solid #b7950b;"
            if tag == self.current_tag:
                style = "background-color: #58d68d; color: black; border: 4px solid #1e8449;"
            if self.sim_expected_tag == tag:
                style = "background-color: #f5b041; color: black; border: 4px solid #af601a;"

            btn.setStyleSheet(style)

        if self.calibration_done:
            cal_text = "DONE - planner active from Tag 1"
        elif self.waiting_for_manual_alignment:
            cal_text = "TAG 0 DETECTED - manually align, then press S"
        elif self.calibration_started:
            cal_text = "WAITING FOR DOCKING TAG 0"
        else:
            cal_text = "NOT DONE"
        self.calibration_label.setText(cal_text)

        self.current_label.setText("Dock / ---" if self.current_tag is None else f"Tag {self.current_tag}")
        self.goal_label.setText("---" if self.goal_tag is None else f"Tag {self.goal_tag}")
        self.heading_label.setText(HEADING_NAMES.get(self.current_heading, "---"))
        self.expected_label.setText("---" if self.sim_expected_tag is None else f"Tag {self.sim_expected_tag}")
        self.path_label.setText(" -> ".join(map(str, self.path)) if self.path else "No path")

    def start_navigation(self):
        if not self.calibration_done or self.current_tag is None:
            QMessageBox.warning(self, "Calibration Required", "Complete docking calibration first: Start Calibration → detect Tag 0 → manually align → press S.")
            return
        if not self.path:
            QMessageBox.warning(self, "No Path", "Select a goal tag first.")
            return

        if self.simulation_mode:
            self.start_simulation()
            return

        self.executor = RobotExecutor(self.path)
        self.executor.log.connect(self.append_log)
        self.executor.position_changed.connect(self.on_position_changed)
        self.executor.heading_changed.connect(self.on_heading_changed)
        self.executor.finished_ok.connect(self.on_finished_ok)
        self.executor.failed.connect(self.on_failed)
        self.executor.start()
        self.append_log("Real navigation started.")

    def stop_navigation(self):
        self.sim_running = False
        self.sim_expected_tag = None
        self.open_loop_timer.stop()
        if self.executor:
            self.executor.request_stop()
        self.update_map()
        self.append_log("STOP requested.")

    def start_simulation(self):
        if len(self.path) < 2:
            self.append_log("Simulation not needed. Already at goal.")
            return

        self.sim_running = True
        self.sim_index = 0
        self.sim_expected_tag = self.path[1]

        first_heading = heading_between(self.path[0], self.path[1])
        self.current_heading = first_heading

        if self.feedback_simulation:
            self.start_detector()
            self.append_log(
                "Closed-loop simulation started. "
                f"Move/show the camera to expected Tag {self.sim_expected_tag}."
            )
        else:
            self.append_log("Open-loop simulation started. Timer will advance the marker.")
            self.open_loop_timer.start(OPEN_LOOP_SIM_STEP_MS)

        self.update_map()

    def open_loop_advance(self):
        if not self.sim_running or not self.path:
            self.open_loop_timer.stop()
            return
        if self.sim_expected_tag is None:
            self.open_loop_timer.stop()
            return
        self.advance_sim_to_detected_tag(self.sim_expected_tag)

    def advance_sim_to_detected_tag(self, tag_id: int):
        if not self.sim_running:
            return

        self.current_tag = tag_id
        self.sim_index += 1

        if self.sim_index >= len(self.path) - 1:
            self.sim_running = False
            self.sim_expected_tag = None
            self.open_loop_timer.stop()
            self.append_log("Simulation complete. Goal reached by tag feedback.")
            self.update_map()
            return

        src = self.path[self.sim_index]
        dst = self.path[self.sim_index + 1]
        self.current_heading = heading_between(src, dst)
        self.sim_expected_tag = dst
        self.append_log(f"Next expected tag: {self.sim_expected_tag}")
        self.update_map()

    def on_position_changed(self, tag: int):
        self.current_tag = tag
        self.update_map()

    def on_heading_changed(self, heading: int):
        self.current_heading = heading
        self.update_map()

    def on_finished_ok(self):
        self.append_log("Mission complete.")
        QMessageBox.information(self, "Done", "AGV mission complete.")

    def on_failed(self, msg: str):
        self.append_log("ERROR: " + msg)
        QMessageBox.critical(self, "Navigation Failed", msg)

    def closeEvent(self, event):
        self.open_loop_timer.stop()
        if self.detector_thread:
            self.detector_thread.stop()
            self.detector_thread.wait(1500)
        if self.executor:
            self.executor.request_stop()
            self.executor.wait(1500)
        if self.calibration_executor:
            self.calibration_executor.request_stop()
            self.calibration_executor.wait(1500)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    win = AGVQtApp()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
