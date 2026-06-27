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
DEFAULT_BASE_PPS = 4500
DEFAULT_IMU_MAX_CORR = 350
DEFAULT_IMU_KP = 70.0
DEFAULT_ACCEL_PPS_PER_SEC = 4000
DEFAULT_DECEL_PPS_PER_SEC = 5500

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
    """
    Physical grid heading.

    In the real AGV layout:
      0 -> 1 is NORTH
      1 -> 2 is WEST because Tag 2 is physically left of Tag 1

    Therefore tag numbers increase to the LEFT and upward:
      same row, +1  => WEST
      same row, -1  => EAST
      next row, +5  => NORTH
      previous row, -5 => SOUTH
    """
    ar, ac = tag_to_rc(a)
    br, bc = tag_to_rc(b)

    if br == ar + 1 and bc == ac:
        return NORTH
    if br == ar - 1 and bc == ac:
        return SOUTH
    if br == ar and bc == ac + 1:
        return WEST
    if br == ar and bc == ac - 1:
        return EAST
    return None


def turn_delta_deg(current_heading, desired_heading):
    """
    Return TURN_REL angle for the ESP32.

    Physical convention used for this AGV/grid:
      0 -> 1 is NORTH
      1 -> 2 is WEST because Tag 2 is physically left of Tag 1.
      From NORTH to WEST, the robot uses +90 degrees.

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
    # Example: current NORTH -> desired WEST for 1 -> 2 after docking.
    # This must be +90 for the physical AGV.
    if delta_steps == 3:
        return 90.0

    return 0.0



def expected_entry_helper_for_heading(heading):
    if heading is None:
        return None
    return ENTRY_CENTER_BY_HEADING.get(heading)


def exit_helper_for_heading(heading):
    if heading is None:
        return None
    return EXIT_CENTER_BY_HEADING.get(heading)


def helper_group_name(center_helper):
    group = helper_group_ids_for_center(int(center_helper))
    if not group:
        return str(center_helper)
    return "/".join(str(x) for x in sorted(group))


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


# =====================================================
# OLD VISION / CLUSTER PARAMETERS
# =====================================================

MAX_PPS = 12000

VISION_BASE_PPS = 4500
VISION_BASE_PPS_SLOW = 3200
VISION_MIN_PPS = 1800
VISION_MAX_PPS = 6500

LOCAL_NUDGE_PPS = 500
TURN_TAG_FB_PPS = 550

FB_SIGN = 1

TAG_SIZE_M = 0.010
CLUSTER_SPACING_M = 0.015

EXPECTED_TAG_YAW_DEG = 0.0

# =====================================================
# UNIQUE LOCAL HELPER TAGS
# =====================================================
#
# Central landmark tags:
#   Docking: 0
#   Grid:    1..15
#
# Local helper IDs are unique per central landmark.
#
# Grid helpers:
#   helper_id = 100 + ((central_tag - 1) * 8) + local_position
#
# Docking helpers:
#   helper_id = 260 + local_position
#   Dock helpers are 261..268
#
# Physical local-position layout around every central tag:
#
#       pos8   pos1   pos2
#       pos7  CENTER  pos3
#       pos6   pos5   pos4
#
# This matches the printable sheet:
#   Dock 0: 261..268
#   Tag 1: 101..108
#   Tag 2: 109..116
#   ...
#   Tag 15: 213..220

GRID_MIN_TAG = 1
GRID_MAX_TAG = ROWS * COLS

DOCK_HELPER_BASE = 260
GRID_HELPER_BASE = 100
HELPERS_PER_LANDMARK = 8

LOCAL_HELPER_GRID_OFFSET = {
    1: (0, -1),
    2: (1, -1),
    3: (1, 0),
    4: (1, 1),
    5: (0, 1),
    6: (-1, 1),
    7: (-1, 0),
    8: (-1, -1),
}

LOCAL_CROSS_POS = {1, 3, 5, 7}
LOCAL_CORNER_POS = {2, 4, 6, 8}


def local_helper_id(central_tag_id, local_position):
    central_tag_id = int(central_tag_id)
    local_position = int(local_position)

    if central_tag_id == DOCK_TAG:
        return DOCK_HELPER_BASE + local_position

    if GRID_MIN_TAG <= central_tag_id <= GRID_MAX_TAG:
        return GRID_HELPER_BASE + ((central_tag_id - 1) * HELPERS_PER_LANDMARK) + local_position

    return None


def helper_owner_tag_id(tag_id):
    tag_id = int(tag_id)

    if DOCK_HELPER_BASE + 1 <= tag_id <= DOCK_HELPER_BASE + HELPERS_PER_LANDMARK:
        return DOCK_TAG

    first_grid_helper = GRID_HELPER_BASE + 1
    last_grid_helper = GRID_HELPER_BASE + (GRID_MAX_TAG * HELPERS_PER_LANDMARK)

    if first_grid_helper <= tag_id <= last_grid_helper:
        return ((tag_id - first_grid_helper) // HELPERS_PER_LANDMARK) + 1

    return None


def helper_local_position(tag_id):
    tag_id = int(tag_id)

    if DOCK_HELPER_BASE + 1 <= tag_id <= DOCK_HELPER_BASE + HELPERS_PER_LANDMARK:
        return tag_id - DOCK_HELPER_BASE

    first_grid_helper = GRID_HELPER_BASE + 1
    last_grid_helper = GRID_HELPER_BASE + (GRID_MAX_TAG * HELPERS_PER_LANDMARK)

    if first_grid_helper <= tag_id <= last_grid_helper:
        return ((tag_id - first_grid_helper) % HELPERS_PER_LANDMARK) + 1

    return None


def is_helper_tag_id(tag_id):
    return helper_owner_tag_id(tag_id) is not None


def helper_ids_for_landmark(central_tag_id):
    return {
        local_helper_id(central_tag_id, pos)
        for pos in range(1, HELPERS_PER_LANDMARK + 1)
    }


HELPER_IDS = set()
for _central_tag in range(GRID_MIN_TAG, GRID_MAX_TAG + 1):
    HELPER_IDS.update(helper_ids_for_landmark(_central_tag))
HELPER_IDS.update(helper_ids_for_landmark(DOCK_TAG))

CROSS_HELPERS = {
    tid for tid in HELPER_IDS
    if helper_local_position(tid) in LOCAL_CROSS_POS
}

CORNER_HELPERS = {
    tid for tid in HELPER_IDS
    if helper_local_position(tid) in LOCAL_CORNER_POS
}

# Direction-based helper local positions.
ENTRY_HELPER_POS_BY_HEADING = {
    WEST: {2, 3, 4},
    EAST: {6, 7, 8},
    NORTH: {4, 5, 6},
    SOUTH: {8, 1, 2},
}

EXIT_HELPER_POS_BY_HEADING = {
    WEST: {6, 7, 8},
    EAST: {2, 3, 4},
    NORTH: {8, 1, 2},
    SOUTH: {4, 5, 6},
}

ENTRY_CENTER_POS_BY_HEADING = {
    WEST: 3,
    EAST: 7,
    NORTH: 5,
    SOUTH: 1,
}

EXIT_CENTER_POS_BY_HEADING = {
    WEST: 7,
    EAST: 3,
    NORTH: 1,
    SOUTH: 5,
}

VALID_ENTRY_CENTER_POS_BY_HEADING = {
    WEST: {1, 3, 5},
    EAST: {1, 5, 7},
    NORTH: {3, 5, 7},
    SOUTH: {1, 3, 7},
}

ENTRY_SEQUENCE_POS_BY_HEADING = {
    WEST: {2: 1, 3: "CENTRAL", 4: 5},
    EAST: {6: 5, 7: "CENTRAL", 8: 1},
    NORTH: {4: 3, 5: "CENTRAL", 6: 7},
    SOUTH: {8: 7, 1: "CENTRAL", 2: 3},
}

CORNER_TO_NEXT_CENTER_POS_BY_HEADING = {
    WEST: {2: {1}, 4: {5}},
    EAST: {6: {5}, 8: {1}},
    NORTH: {4: {3}, 6: {7}},
    SOUTH: {8: {7}, 2: {3}},
}


def entry_helper_ids_for_heading(central_tag_id, heading):
    return {
        local_helper_id(central_tag_id, pos)
        for pos in ENTRY_HELPER_POS_BY_HEADING.get(heading, set())
    }


def exit_helper_ids_for_heading(central_tag_id, heading):
    return {
        local_helper_id(central_tag_id, pos)
        for pos in EXIT_HELPER_POS_BY_HEADING.get(heading, set())
    }


def valid_entry_center_ids_for_heading(central_tag_id, heading):
    return {
        local_helper_id(central_tag_id, pos)
        for pos in VALID_ENTRY_CENTER_POS_BY_HEADING.get(heading, set())
    }


def entry_center_id_for_heading(central_tag_id, heading):
    pos = ENTRY_CENTER_POS_BY_HEADING.get(heading)
    if pos is None:
        return None
    return local_helper_id(central_tag_id, pos)


def exit_center_id_for_heading(central_tag_id, heading):
    pos = EXIT_CENTER_POS_BY_HEADING.get(heading)
    if pos is None:
        return None
    return local_helper_id(central_tag_id, pos)


def helper_group_ids_for_center(center_helper_id):
    center_helper_id = int(center_helper_id)
    owner = helper_owner_tag_id(center_helper_id)
    pos = helper_local_position(center_helper_id)

    if owner is None or pos is None:
        return set()

    if pos == 3:
        group_pos = {2, 3, 4}
    elif pos == 7:
        group_pos = {6, 7, 8}
    elif pos == 1:
        group_pos = {8, 1, 2}
    elif pos == 5:
        group_pos = {4, 5, 6}
    else:
        group_pos = {pos}

    return {local_helper_id(owner, p) for p in group_pos}


# Compatibility names kept for comments/old references.
ENTRY_CENTER_BY_HEADING = ENTRY_CENTER_POS_BY_HEADING
EXIT_CENTER_BY_HEADING = EXIT_CENTER_POS_BY_HEADING
ENTRY_HELPER_IDS_BY_HEADING = ENTRY_HELPER_POS_BY_HEADING
EXIT_HELPER_IDS_BY_HEADING = EXIT_HELPER_POS_BY_HEADING
ENTRY_CENTER_BY_HEADING_STRICT = ENTRY_CENTER_POS_BY_HEADING
ENTRY_SEQUENCE_BY_HEADING = ENTRY_SEQUENCE_POS_BY_HEADING
CORNER_TO_NEXT_CENTER_BY_HEADING = CORNER_TO_NEXT_CENTER_POS_BY_HEADING
VALID_ENTRY_CENTERS_BY_HEADING = VALID_ENTRY_CENTER_POS_BY_HEADING
HELPER_GROUP_BY_CENTER = {}

CLUSTER_LOST_FRAMES_REQUIRED = 5
TARGET_CENTRAL_SEEN_FRAMES_REQUIRED = 2

# Local helper arrival confirmation. Cross helpers are important.
TARGET_HELPER_SEEN_FRAMES_REQUIRED = 1

# Geometry-only local helper arrival gate.
# No time-based arrival and no single-helper arrival.
# A side-pair helper is accepted only when the estimated landmark center
# is reasonably close to the camera center line.
LOCAL_PAIR_CENTER_Y_OK_PX = 9999
LOCAL_SINGLE_CENTER_Y_OK_PX = 90
# Minimum time after segment start before helpers are allowed to confirm target arrival.
# Prevents immediately accepting old-cluster helpers at segment start.
# Previous central tag should be absent briefly before accepting helper-only target arrival.

LOCAL_NUDGE_CENTER_Y_OK_PX = 30
LOCAL_NUDGE_GOOD_FRAMES_REQUIRED = 3
LOCAL_NUDGE_TIMEOUT_SEC = 5.0

# Accept local helper tags as arrival evidence where enabled; EAST mirrors WEST and uses them as correction-only.
LOCAL_HELPER_SEEN_FRAMES_REQUIRED = 1

# Local-center reached rule from ENTRY_SEQUENCE_BY_HEADING.
#
# Integer VALUES in the table are local center helpers and may confirm reached
# after their entry helper created them as candidates.
#
# A helper that maps to "CENTRAL" is correction evidence only. It guides toward
# the actual central landmark tag, but it does not confirm reached by itself.
#
# Example EAST for target 12:
#   506 -> 505       # 505 can confirm reached after 506
#   507 -> CENTRAL   # 507 cannot confirm reached by itself
#   508 -> 501       # 501 can confirm reached after 508
ACCEPT_EXPECTED_LOCAL_CENTER_AS_REACHED = True

# Extra gate for expected local-center reached.
# Example EAST 508 -> 501:
# 501 is valid, but it must be close to the camera center before Tag 12 is
# considered reached/pass-through. This fixes early acceptance such as:
#   helper=501 centerY=316.6px
EXPECTED_LOCAL_CENTER_REACHED_Y_OK_PX = 90


# Stronger correction tuning.
# Previous logs showed repeated LARGE corrections saturating at about ±500 pps
# while the robot still drifted. This version increases vision authority and
# applies it faster, especially when x/yaw errors are large.
KP_YAW_PPS_PER_DEG = 24
KP_X_PPS_PER_M = 24000

KP_YAW_STRONG_PPS_PER_DEG = 42
KP_X_STRONG_PPS_PER_M = 85000

# Direction-specific X correction sign.
# This is the only X-sign source used by travel correction.
X_SIGN_BY_HEADING = {
    NORTH: -1.0,
    EAST: 1.0,
    SOUTH: -1.0,
    WEST: -1.0,
}

# IMPORTANT:
# During grid travel the ESP32/IMU owns heading hold.
# Camera correction should mainly correct lateral X offset.
#
# The logs showed AprilTag yaw around +/-90 deg on side/helper views,
# causing huge wrong steering:
#   TAG2 CORR seen=502 yawErr=89 -> strong turn
# Then another helper produced the opposite correction.
#
# Therefore travel correction uses lateral X only. Yaw is still logged, but not
# used for wheel correction while moving between grid tags.
USE_YAW_IN_TRAVEL_CORRECTION = False


YAW_DEADBAND_DEG = 0.30
X_DEADBAND_M = 0.0005

X_MEDIUM_ERROR_M = 0.005
X_LARGE_ERROR_M = 0.012

YAW_MEDIUM_ERROR_DEG = 2.0
YAW_LARGE_ERROR_DEG = 5.0

MAX_VISION_CORRECTION_PPS = 300
MAX_VISION_CORRECTION_STRONG_PPS = 700

CORRECTION_FILTER_ALPHA = 0.35
CORRECTION_FILTER_ALPHA_STRONG = 0.62

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


def visible_helper_ids(detections):
    return {int(tag.tag_id) for tag in detections if is_helper_tag_id(int(tag.tag_id))}


def visible_helper_ids_for_landmark(detections, central_tag_id):
    central_tag_id = int(central_tag_id)
    return {
        int(tag.tag_id)
        for tag in detections
        if helper_owner_tag_id(int(tag.tag_id)) == central_tag_id
    }


def any_helper_visible(detections):
    return bool(visible_helper_ids(detections))


def any_cluster_visible_now(detections, central_tag_id):
    central_tag_id = int(central_tag_id)

    if find_tag(detections, central_tag_id) is not None:
        return True

    return bool(visible_helper_ids_for_landmark(detections, central_tag_id))


def choose_best_cluster_tag(detections, central_tag_id):
    central_tag_id = int(central_tag_id)
    central = find_tag(detections, central_tag_id)
    if central is not None:
        return central

    best_tag = None
    best_dist = 999999999.0

    for tag in detections:
        tid = int(tag.tag_id)
        if helper_owner_tag_id(tid) != central_tag_id:
            continue

        dx = tag.center[0] - CX
        dy = tag.center[1] - CY
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_dist = dist
            best_tag = tag

    return best_tag


def choose_correction_tag(detections, central_tag_id):
    central_tag_id = int(central_tag_id)
    central = find_tag(detections, central_tag_id)
    if central is not None:
        return central

    best_tag = None
    best_dist = 999999999.0

    for tag in detections:
        tid = int(tag.tag_id)
        if helper_owner_tag_id(tid) != central_tag_id:
            continue
        if helper_local_position(tid) not in LOCAL_CROSS_POS:
            continue

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
    """
    Valid side-pair local helper evidence for the TARGET landmark only.
    """
    target_landmark = int(target_landmark)
    ids = visible_ids(detections)

    if target_landmark in ids:
        return None

    target_helpers = visible_helper_ids_for_landmark(detections, target_landmark)
    pos_to_id = {
        helper_local_position(tid): tid
        for tid in target_helpers
    }

    groups = [
        (3, {2, 4}),
        (7, {6, 8}),
        (1, {8, 2}),
        (5, {6, 4}),
    ]

    for center_pos, corner_positions in groups:
        center_id = pos_to_id.get(center_pos)
        if center_id is None:
            continue

        if any(pos in pos_to_id for pos in corner_positions):
            return center_id

    return None


def detect_single_cross_helper(detections, target_landmark):
    """
    Single local cross helper for the TARGET landmark only.
    """
    target_landmark = int(target_landmark)
    ids = visible_ids(detections)

    if target_landmark in ids:
        return None

    best_tag = None
    best_dist = 999999999.0

    for tag in detections:
        tid = int(tag.tag_id)
        if helper_owner_tag_id(tid) != target_landmark:
            continue
        if helper_local_position(tid) not in LOCAL_CROSS_POS:
            continue

        dx = tag.center[0] - CX
        dy = tag.center[1] - CY
        dist = dx * dx + dy * dy
        if dist < best_dist:
            best_dist = dist
            best_tag = tag

    if best_tag is None:
        return None

    return int(best_tag.tag_id)


def detect_local_arrival_helper(detections, target_landmark, allow_single_cross=False):
    """
    Local helper detector for the target-owned helper ID scheme.
    """
    pair_helper = detect_side_pair(detections, target_landmark)
    if pair_helper is not None:
        return pair_helper, "PAIR"

    if allow_single_cross:
        single_helper = detect_single_cross_helper(detections, target_landmark)
        if single_helper is not None:
            return single_helper, "SINGLE"

    return None, None


def get_helper_grid_offset(tag_id, central_tag_id):
    tag_id = int(tag_id)
    central_tag_id = int(central_tag_id)

    if tag_id == central_tag_id:
        return 0, 0

    if helper_owner_tag_id(tag_id) != central_tag_id:
        return None

    pos = helper_local_position(tag_id)
    if pos is None:
        return None

    return LOCAL_HELPER_GRID_OFFSET.get(pos)


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

    if USE_YAW_IN_TRAVEL_CORRECTION:
        abs_yaw = abs(yaw_error)
    else:
        abs_yaw = 0.0

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

        self.setWindowTitle("AGV Qt A-Star Closed Loop - Unique Local Helper Tags")
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
        self.current_heading = NORTH
        self.mission_running = False

        # Old behavior segment state.
        self.route_state = "IDLE"
        self.segment_phase = "START_CLUSTER"
        self.cluster_lost_count = 0
        self.old_central_lost_count = 0
        self.target_helper_seen_count = 0
        self.target_central_seen_count = 0
        self.segment_start_time = 0.0
        self.travel_from_landmark = None
        self.travel_to_landmark = None

        self.local_arrival_landmark = None
        self.local_arrival_helper_id = None
        self.local_arrival_good_count = 0
        self.local_arrival_start_time = 0.0

        # Helpers visible at the instant a new segment starts are treated as
        # previous-landmark local tags. They must disappear once before the same
        # helper ID can be accepted as the next target. This prevents:
        #   1 -> 2 reached by helper 503
        #   immediately starting 2 -> 3
        #   same visible 503 falsely becoming Tag 3.
        self.previous_helper_block_ids = set()

        # For distinguishing previous local tag vs arrived local tag.
        # If two consecutive segments are in the same heading, the expected
        # arriving side-center helper should normally remain the same.
        # If the heading changes, helper arrival in START_CLUSTER is blocked
        # until there is a clean "no helper visible" transition.
        self.last_arrival_helper_id = None
        self.last_arrival_heading = None
        self.segment_heading = None
        self.expected_straight_helper_id = None
        self.all_helpers_cleared_once = False
        self.expected_entry_helper_id = None
        self.expected_exit_helper_id = None
        self.segment_next_action = "UNKNOWN"
        self.corner_single_arrival_candidates = set()
        self.skip_next_segment_lock = False

        self.filtered_correction = 0.0

        self.turning_waiting = False
        self.pending_after_turn_segment = False
        self.turn_start_time = 0.0
        # No Python-side turn timeout.
        # Wait until ESP32 reports OK TURN_DONE, even for slow 180 degree turns.
        self.turn_timeout_sec = None

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

        self.log_autoscroll_checkbox = QCheckBox("Auto-scroll log")
        self.log_autoscroll_checkbox.setChecked(True)
        self.pause_log_checkbox = QCheckBox("Pause log")
        self.clear_log_btn = QPushButton("Clear Log")
        self.clear_log_btn.clicked.connect(self.log.clear)

        log_controls = QHBoxLayout()
        log_controls.addWidget(self.log_autoscroll_checkbox)
        log_controls.addWidget(self.pause_log_checkbox)
        log_controls.addWidget(self.clear_log_btn)

        log_layout.addLayout(log_controls)
        log_layout.addWidget(self.log)
        right.addWidget(log_group, 1)

    def append_log(self, text):
        if hasattr(self, "pause_log_checkbox") and self.pause_log_checkbox.isChecked():
            return

        ts = time.strftime("%H:%M:%S")
        self.log.append(f"[{ts}] {text}")

        if hasattr(self, "log_autoscroll_checkbox") and self.log_autoscroll_checkbox.isChecked():
            self.log.verticalScrollBar().setValue(self.log.verticalScrollBar().maximum())

    def update_ui_state(self):
        self.current_tag_edit.setText(str(self.current_tag) if self.current_tag is not None else "---")
        self.goal_tag_edit.setText(str(self.goal_tag) if self.goal_tag is not None else "---")
        self.heading_edit.setText(HEADING_LABELS.get(self.current_heading, "---"))
        self.expected_next_edit.setText(str(self.expected_next_tag) if self.expected_next_tag is not None else "---")
        self.path_edit.setText(" → ".join(str(x) for x in self.path) if self.path else "---")
        self.route_state_edit.setText(f"{self.route_state} / {self.segment_phase} / lost={self.cluster_lost_count} h={self.target_helper_seen_count}")

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
            f"State:{self.route_state} Seg:{self.travel_from_landmark}->{self.travel_to_landmark} phase:{self.segment_phase} lost:{self.cluster_lost_count}",
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
        if getattr(self, "skip_next_segment_lock", False):
            self.skip_next_segment_lock = False

            # Straight pass-through case:
            # Do not immediately clear the previous visual VEL command.
            #
            # Reason:
            #   On a straight chain such as 15->14->13->12->11, the robot is
            #   still physically leaving the previous cluster when the next
            #   segment starts. If we send LOCK_HEADING_GO immediately here,
            #   the active visual steering is removed and the robot may continue
            #   on IMU hold while only the exit-side helper is visible.
            #
            # This does not affect turn waypoints because those use STOP/TURN_REL.
            if self.last_drive_mode == "VISION":
                self.append_log(
                    "Skipped LOCK_HEADING_GO after straight pass-through; keeping visual correction carry-over."
                )
                return

            if self.last_drive_mode == "IMU":
                self.append_log("Skipped LOCK_HEADING_GO after straight pass-through; already in IMU heading hold.")
                return

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
            alpha = 0.45
        else:
            kp_yaw = KP_YAW_PPS_PER_DEG
            kp_x = KP_X_PPS_PER_M
            max_corr = MAX_VISION_CORRECTION_PPS
            base_pps = VISION_BASE_PPS
            alpha = CORRECTION_FILTER_ALPHA

        yaw_for_control = yaw_error
        x_for_control = center_x_m

        if not USE_YAW_IN_TRAVEL_CORRECTION:
            yaw_for_control = 0.0

        if abs(yaw_for_control) < YAW_DEADBAND_DEG:
            yaw_for_control = 0.0
        if abs(x_for_control) < X_DEADBAND_M:
            x_for_control = 0.0

        yaw_corr = kp_yaw * yaw_for_control
        direction_x_sign = X_SIGN_BY_HEADING.get(getattr(self, "segment_heading", None), -1.0)
        x_corr = kp_x * x_for_control * direction_x_sign

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
        self.target_helper_seen_count = 0
        self.target_central_seen_count = 0

        self.segment_heading = heading_between_tags(
            self.travel_from_landmark,
            self.travel_to_landmark
        )

        self.expected_entry_helper_id = entry_center_id_for_heading(
            self.travel_to_landmark,
            self.segment_heading
        )
        self.expected_exit_helper_id = exit_center_id_for_heading(
            self.travel_to_landmark,
            self.segment_heading
        )

        # Determine the next action at the target landmark: straight/right/left/stop/u-turn.
        if self.mission_running and self.path_index < len(self.active_path):
            # The target of this segment is active_path[path_index].
            next_index_after_target = self.path_index + 1
            if next_index_after_target < len(self.active_path):
                outgoing_heading = heading_between_tags(
                    self.travel_to_landmark,
                    self.active_path[next_index_after_target]
                )
                self.segment_next_action = turn_action_between_headings(
                    self.segment_heading,
                    outgoing_heading
                )
            else:
                self.segment_next_action = "STOP"
        else:
            self.segment_next_action = "STOP"

        self.all_helpers_cleared_once = False
        self.corner_single_arrival_candidates = set()

        # Previous-local vs arrived-local protection:
        # Any helper IDs visible at the exact start of the new segment usually
        # belong to the previous landmark. Do not accept those IDs as arrival for
        # the next landmark until they disappear once.
        self.previous_helper_block_ids = visible_ids(self.latest_detections).intersection(HELPER_IDS)

        # Straight pass-through correction fix:
        # When two consecutive segments have the same heading, the strict entry
        # center helper is expected to be the first useful target-side helper.
        #
        # Example EAST:
        #   14->13 starts while 507 is already visible.
        #   507 is the EAST entry-center helper, so blocking it makes the robot
        #   ignore the correct early correction and then wait until 507 appears
        #   again later.
        #
        # Therefore do not previous-block the strict entry center when continuing
        # straight in the same heading. This is generic:
        #   WEST  keeps 503 usable
        #   EAST  keeps 507 usable
        #   NORTH keeps 505 usable
        #   SOUTH keeps 501 usable
        if (
            getattr(self, "last_arrival_heading", None) is not None
            and self.segment_heading == self.last_arrival_heading
            and self.segment_next_action in ("STRAIGHT", "STOP")
        ):
            strict_entry_center = entry_center_id_for_heading(
                self.travel_to_landmark,
                self.segment_heading
            )
            if strict_entry_center is not None and int(strict_entry_center) in self.previous_helper_block_ids:
                self.previous_helper_block_ids.discard(int(strict_entry_center))
                self.append_log(
                    f"Straight continuation: not blocking entry-center helper {int(strict_entry_center)} "
                    f"for heading={HEADING_LABELS.get(self.segment_heading, '---')}."
                )

        if self.previous_helper_block_ids:
            self.append_log(
                "Blocking previous local helpers at segment start: "
                + ", ".join(str(x) for x in sorted(self.previous_helper_block_ids))
            )

        self.append_log(
            f"Direction rule {from_tag}->{to_tag}: heading={HEADING_LABELS.get(self.segment_heading, '---')} "
            f"validEntryCenters={sorted(self.valid_entry_centers_for_heading())} "
            f"exitHelper={self.expected_exit_helper_id} "
            f"nextAction={self.segment_next_action}"
        )

        self.reset_correction_filter()
        self.route_state = "MOVE"
        self.expected_next_tag = int(to_tag)
        self.append_log(f"Starting segment {from_tag} → {to_tag}")
        self.update_ui_state()

    def arrival_requires_stop_or_turn(self, landmark_id):
        """
        Return True if this reached landmark must physically stop.

        Stop is required for:
          - docking/calibration completion
          - final destination
          - left/right/U-turn before the next segment

        Stop is NOT required for a straight pass-through waypoint.
        """
        landmark_id = int(landmark_id)

        if self.calibrating_move_to_tag1:
            return True

        if not self.mission_running:
            return True

        if not (self.path_index < len(self.active_path) and landmark_id == self.active_path[self.path_index]):
            return True

        next_index = self.path_index + 1

        if next_index >= len(self.active_path):
            return True

        next_tag = self.active_path[next_index]
        desired_heading = heading_between_tags(landmark_id, next_tag)
        turn_deg = turn_delta_deg(self.current_heading, desired_heading)

        return abs(turn_deg) > 1.0

    def handle_landmark_arrival(self, landmark_id):
        landmark_id = int(landmark_id)

        must_stop = self.arrival_requires_stop_or_turn(landmark_id)

        if must_stop:
            self.stop_robot()

        self.reset_correction_filter()

        if must_stop:
            self.append_log(f"Reached landmark Tag {landmark_id}")
        else:
            self.append_log(f"Pass-through landmark Tag {landmark_id}; continuing without STOP")

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
            # Turn waypoint: stop already sent above, then rotate.
            self.route_state = "TURNING"
            self.turning_waiting = True
            self.pending_after_turn_segment = True
            self.turn_start_time = time.time()
            self.append_log(f"Turning before next segment: TURN_REL {turn_deg:.1f} -- waiting for ESP32 OK TURN_DONE")
            self.send_esp32(f"TURN_REL {turn_deg:.1f}")
            self.current_heading = desired_heading
        else:
            # Straight pass-through: do not stop and do not re-lock heading here.
            # The next control tick will continue with vision correction if a tag
            # is visible, otherwise ESP32 IMU heading hold remains active.
            self.current_heading = desired_heading
            self.skip_next_segment_lock = True
            self.start_segment(self.current_tag, next_tag)

            if must_stop:
                # This path is mostly for non-mission/manual cases. For normal
                # straight pass-through, must_stop is False and no pause is created.
                self.lock_heading_go()

    def start_local_arrival(self, landmark_id, helper_id):
        """
        Local helper arrival updates the mission state.

        It does NOT blindly send STOP anymore.
        STOP is sent only if this landmark is a turn point, final target, or
        calibration target. Straight pass-through waypoints continue smoothly.
        """
        self.remember_local_arrival_helper(helper_id)
        self.corner_single_arrival_candidates = set()
        self.local_arrival_landmark = int(landmark_id)
        self.local_arrival_helper_id = int(helper_id)
        self.local_arrival_good_count = 0
        self.local_arrival_start_time = time.time()

        if self.arrival_requires_stop_or_turn(landmark_id):
            self.append_log(
                f"Local helper accepted as reached: target={landmark_id} helper={helper_id}. "
                "STOP/turn handling required."
            )
        else:
            self.append_log(
                f"Local helper accepted as pass-through: target={landmark_id} helper={helper_id}. "
                "No STOP."
            )

        self.handle_landmark_arrival(landmark_id)

    def local_helper_arrival_ready(self, detections, helper_id, evidence_type):
        """
        Helper confirmation.

        PAIR:
          accepted immediately when helper-arrival is allowed.
          This matches the standalone local-arrival decision better. The previous
          centerY gate blocked Tag 3 even though the side-pair was visible.

        SINGLE:
          accepted only in SEARCH_TARGET for normal mission and only when the
          estimated cluster center is reasonably near the camera center line.
        """
        helper = find_tag(detections, helper_id)
        if helper is None:
            return False, None

        pose = get_landmark_pose_from_cluster_tag(helper, self.travel_to_landmark)
        center_y_error_px = None

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

        if evidence_type == "PAIR":
            return True, center_y_error_px

        if center_y_error_px is None:
            return False, None

        ready = abs(center_y_error_px) <= LOCAL_SINGLE_CENTER_Y_OK_PX
        return ready, center_y_error_px

    def update_previous_helper_blocks(self, detections):
        """
        Remove helper IDs from previous_helper_block_ids only after that exact
        helper disappears from camera view.

        Arrival is decided from side-pair CENTER helper, not from a sequence of
        corner/center/corner helpers.
        """
        ids = visible_ids(detections)
        helper_ids_now = ids.intersection(HELPER_IDS)

        if not helper_ids_now and not self.all_helpers_cleared_once:
            self.all_helpers_cleared_once = True
            self.append_log("All helpers cleared once.")

        if not self.previous_helper_block_ids:
            return

        still_visible = self.previous_helper_block_ids.intersection(ids)
        cleared = self.previous_helper_block_ids.difference(still_visible)

        if cleared:
            self.append_log(
                "Previous helper cleared after disappearing: "
                + ", ".join(str(x) for x in sorted(cleared))
            )

        self.previous_helper_block_ids = still_visible

    def helper_is_previous_blocked(self, helper_id):
        return int(helper_id) in self.previous_helper_block_ids

    def helper_is_exit_side(self, helper_id):
        return (
            self.expected_exit_helper_id is not None
            and int(helper_id) == int(self.expected_exit_helper_id)
        )

    def valid_entry_centers_for_heading(self):
        if self.segment_heading is None or self.travel_to_landmark is None:
            return set()
        return valid_entry_center_ids_for_heading(
            self.travel_to_landmark,
            self.segment_heading
        )


    def update_corner_single_arrival_candidates(self, detections):
        """
        Track direction-aware corner -> next-side-center sequence for the
        TARGET landmark only.

        With unique local IDs, old landmark helpers are ignored because their
        owner tag is not self.travel_to_landmark.
        """
        if self.segment_heading is None or self.travel_to_landmark is None:
            return

        ids = visible_ids(detections)
        new_candidates = set()

        entry_corners = CORNER_TO_NEXT_CENTER_POS_BY_HEADING.get(self.segment_heading, {})
        valid_centers = self.valid_entry_centers_for_heading()

        for corner_pos, next_center_positions in entry_corners.items():
            corner_id = local_helper_id(self.travel_to_landmark, corner_pos)
            if corner_id not in ids:
                continue

            if self.helper_is_previous_blocked(corner_id):
                continue

            for center_pos in next_center_positions:
                center_id = local_helper_id(self.travel_to_landmark, center_pos)
                if center_id in valid_centers and not self.helper_is_exit_side(center_id):
                    new_candidates.add(center_id)

        if new_candidates:
            before = set(self.corner_single_arrival_candidates)
            self.corner_single_arrival_candidates.update(new_candidates)
            added = self.corner_single_arrival_candidates.difference(before)
            if added and time.time() - getattr(self, "_last_corner_candidate_log", 0.0) > 0.50:
                self._last_corner_candidate_log = time.time()
                self.append_log(
                    "Entry-corner sequence: allow next local center(s) "
                    + ", ".join(str(x) for x in sorted(added))
                    + f" for target {self.travel_to_landmark}"
                )


    def single_helper_allowed_by_corner_sequence(self, helper_id):
        helper_id = int(helper_id)

        if helper_owner_tag_id(helper_id) != int(self.travel_to_landmark):
            return False, f"single_{helper_id}_belongs_to_tag_{helper_owner_tag_id(helper_id)}_not_target_{self.travel_to_landmark}"

        if helper_id not in self.corner_single_arrival_candidates:
            return False, f"single_{helper_id}_has_no_corner_evidence"

        if self.helper_is_previous_blocked(helper_id):
            return False, f"single_{helper_id}_still_previous_blocked"

        if self.helper_is_exit_side(helper_id):
            return False, f"single_{helper_id}_is_exit_side"

        if helper_id not in self.valid_entry_centers_for_heading():
            return False, f"single_{helper_id}_not_valid_for_heading"

        return True, f"corner_then_single_{helper_id}"

    def pair_matches_corner_sequence(self, helper_id):
        """
        If a target-owned entry-corner sequence has already been observed, then
        the next accepted center must match that sequence.

        Direct strict entry-center remains valid.
        """
        helper_id = int(helper_id)

        if not self.corner_single_arrival_candidates:
            return True, "no_corner_sequence_active"

        if helper_id in self.corner_single_arrival_candidates:
            return True, f"pair_matches_corner_sequence_{helper_id}"

        strict_entry_center = entry_center_id_for_heading(
            self.travel_to_landmark,
            self.segment_heading
        )
        if strict_entry_center is not None and helper_id == int(strict_entry_center):
            return True, f"pair_matches_strict_entry_center_{helper_id}"

        return False, (
            "pair_does_not_match_corner_sequence_"
            + ",".join(str(x) for x in sorted(self.corner_single_arrival_candidates))
        )


    def start_pair_helper_allowed(self, helper_id):
        """
        Direction-based false-positive eliminator.

        detect_local_arrival_helper() returns only the middle/center helper of a
        valid side-pair:
            502+503 or 503+504 -> 503
            508+501 or 501+502 -> 501
            504+505 or 505+506 -> 505
            506+507 or 507+508 -> 507

        For the current movement heading:
          - reject the known exit-side center
          - accept the other side-center pairs as possible target entry
          - still block a helper if the exact same helper is visible from the
            previous segment start and has not disappeared once
        """
        helper_id = int(helper_id)

        if helper_owner_tag_id(helper_id) != int(self.travel_to_landmark):
            return False, f"helper_{helper_id}_belongs_to_tag_{helper_owner_tag_id(helper_id)}_not_target_{self.travel_to_landmark}"

        if self.helper_is_previous_blocked(helper_id):
            return False, f"helper_{helper_id}_still_previous_blocked"

        corner_ok, corner_reason = self.pair_matches_corner_sequence(helper_id)
        if not corner_ok:
            return False, corner_reason

        valid_centers = self.valid_entry_centers_for_heading()

        if helper_id not in valid_centers:
            if self.helper_is_exit_side(helper_id):
                return False, f"exit_side_{helper_id}_for_heading_{HEADING_LABELS.get(self.segment_heading, '---')}"
            return False, f"helper_{helper_id}_not_valid_for_heading_{HEADING_LABELS.get(self.segment_heading, '---')}"

        return True, f"valid_entry_center_{helper_id}"

    def entry_side_helper_visible(self, detections):
        """
        Return a visible helper that is valid TARGET-side correction evidence.

        With unique local helper IDs, only helpers owned by travel_to_landmark
        are allowed here. Old/current landmark helpers are ignored by ID.
        """
        if self.segment_heading is None or self.travel_to_landmark is None:
            return None

        entry_helpers = entry_helper_ids_for_heading(
            self.travel_to_landmark,
            self.segment_heading
        )
        sequence_centers = set(getattr(self, "corner_single_arrival_candidates", set()))
        allowed_helpers = entry_helpers.union(sequence_centers)

        best_tag = None
        best_dist = 999999999.0

        for tag in detections:
            tid = int(tag.tag_id)

            if helper_owner_tag_id(tid) != int(self.travel_to_landmark):
                continue

            if tid not in allowed_helpers:
                continue

            if self.helper_is_previous_blocked(tid):
                continue

            if self.helper_is_exit_side(tid):
                continue

            dx = tag.center[0] - CX
            dy = tag.center[1] - CY
            dist = dx * dx + dy * dy

            if dist < best_dist:
                best_dist = dist
                best_tag = tag

        return best_tag


    def entry_side_arrival_helper_visible(self, detections):
        """
        Helpers never directly confirm arrival in this build.

        Entry helpers and sequence helpers are correction-only evidence.
        The target is reached only when the central target tag is visible.
        """
        return None

    def helper_sequence_note(self, helper_id):
        """
        Explain the role of a target-owned helper using local positions.
        """
        helper_id = int(helper_id)
        owner = helper_owner_tag_id(helper_id)
        pos = helper_local_position(helper_id)

        if owner != int(self.travel_to_landmark) or pos is None:
            return f"helper_{helper_id}_belongs_to_tag_{owner}_not_target_{self.travel_to_landmark}"

        seq = ENTRY_SEQUENCE_POS_BY_HEADING.get(self.segment_heading, {})

        # Case 1: helper is an entry helper key.
        nxt = seq.get(pos, None)
        if nxt == "CENTRAL":
            return f"entry_helper_{helper_id}_pos{pos}_expect_central_tag_{self.travel_to_landmark}"
        if nxt is not None:
            expected_id = local_helper_id(self.travel_to_landmark, int(nxt))
            return f"entry_helper_{helper_id}_pos{pos}_expect_helper_{expected_id}_pos{nxt}"

        # Case 2: helper is an expected value from a previous entry helper.
        creators = []
        for entry_pos, expected_pos in seq.items():
            if isinstance(expected_pos, int) and int(expected_pos) == pos:
                creators.append(int(entry_pos))

        if helper_id in set(getattr(self, "corner_single_arrival_candidates", set())):
            if creators:
                creator_ids = [
                    local_helper_id(self.travel_to_landmark, p)
                    for p in sorted(creators)
                ]
                return (
                    f"expected_helper_{helper_id}_pos{pos}_valid_after_entry_"
                    + ",".join(str(x) for x in creator_ids)
                    + f"_toward_central_tag_{self.travel_to_landmark}"
                )
            return f"expected_helper_{helper_id}_pos{pos}_valid_candidate_toward_central_tag_{self.travel_to_landmark}"

        if creators:
            creator_ids = [
                local_helper_id(self.travel_to_landmark, p)
                for p in sorted(creators)
            ]
            return (
                f"expected_helper_{helper_id}_pos{pos}_but_waiting_for_entry_"
                + ",".join(str(x) for x in creator_ids)
            )

        return "not_in_entry_sequence_for_this_heading"


    def helper_expects_central_now(self, helper_id):
        helper_id = int(helper_id)
        if helper_owner_tag_id(helper_id) != int(self.travel_to_landmark):
            return False

        pos = helper_local_position(helper_id)
        seq = ENTRY_SEQUENCE_POS_BY_HEADING.get(self.segment_heading, {})
        return seq.get(pos, None) == "CENTRAL"


    def helper_is_expected_value_for_heading(self, helper_id):
        helper_id = int(helper_id)
        if helper_owner_tag_id(helper_id) != int(self.travel_to_landmark):
            return False

        pos = helper_local_position(helper_id)
        seq = ENTRY_SEQUENCE_POS_BY_HEADING.get(self.segment_heading, {})
        for expected in seq.values():
            if isinstance(expected, int) and int(expected) == pos:
                return True
        return False


    def helper_is_valid_sequence_center_now(self, helper_id):
        helper_id = int(helper_id)

        # Direct center-side helper for this heading is always allowed.
        # Example EAST: 507 -> CENTRAL.
        if self.helper_expects_central_now(helper_id):
            return True

        # Expected helper values are valid only after the corresponding entry
        # helper has created them as a candidate.
        # Example EAST: 508 -> 501, so 501 is valid after 508 was seen.
        if helper_id in set(getattr(self, "corner_single_arrival_candidates", set())):
            return True

        return False


    def helper_can_confirm_reached_now(self, helper_id, center_y_error_px=None):
        """
        Reached/pass-through decision.

        Generic rule for every heading:
          - A helper that maps to "CENTRAL" is correction-only.
            Example EAST: 507 -> CENTRAL. 507 guides toward the target tag,
            but does not confirm reached by itself.
          - An integer VALUE helper from ENTRY_SEQUENCE_BY_HEADING can confirm
            reached only after its entry corner created it as a candidate.
            Example EAST: 508 -> 501. 501 can confirm reached only after 508.
          - Even then, it must be near the camera center line.

        This keeps the table as the single source of direction logic and does
        not use separate EAST parameters.
        """
        helper_id = int(helper_id)

        if not ACCEPT_EXPECTED_LOCAL_CENTER_AS_REACHED:
            return False

        # Only integer VALUE helpers that were created by a valid entry corner
        # may confirm reached. Direct "CENTRAL" helpers such as EAST 507 or
        # WEST 503 remain correction-only unless the actual central tag appears.
        if helper_id not in set(getattr(self, "corner_single_arrival_candidates", set())):
            return False

        if center_y_error_px is None:
            return False

        gate = globals().get("EXPECTED_LOCAL_CENTER_REACHED_Y_OK_PX", LOCAL_PAIR_CENTER_Y_OK_PX)
        return abs(center_y_error_px) <= gate

    def remember_local_arrival_helper(self, helper_id):
        self.last_arrival_helper_id = int(helper_id)
        self.last_arrival_heading = self.segment_heading


    def choose_move_correction(self, detections):
        """
        Improved previous-local vs arrived-local logic.

        Core idea:
          - A valid side-pair is strong arrival evidence.
          - A single cross helper is useful, but ambiguous.

        0 -> 1 calibration:
          START_CLUSTER:
            helpers 501-508 are OLD Tag 0, never reached.
          SEARCH_TARGET:
            central target or valid side-pair can stop.
            single helper is not accepted.

        Normal A* mission:
          START_CLUSTER:
            exact old central visible -> keep old correction.
            valid side-pair -> target reached.
            single 501/503/505/507 -> correction only, not reached.
          SEARCH_TARGET:
            valid side-pair -> target reached.
            single 501/503/505/507 -> target reached only if centered.

        Removed:
          - time-based helper arrival
          - frame-count helper arrival
          - pair centerY gate
          - single-helper early arrival
        """
        self.update_previous_helper_blocks(detections)
        self.update_corner_single_arrival_candidates(detections)

        central_target = find_tag(detections, self.travel_to_landmark)

        if central_target is not None:
            self.target_central_seen_count += 1
            self.target_helper_seen_count = 0

            if self.target_central_seen_count >= TARGET_CENTRAL_SEEN_FRAMES_REQUIRED:
                self.last_arrival_helper_id = None
                self.last_arrival_heading = self.segment_heading
                self.handle_landmark_arrival(self.travel_to_landmark)
                return None, None, ""

            return central_target, self.travel_to_landmark, f"TAG{self.travel_to_landmark}"

        self.target_central_seen_count = 0
        old_central = find_tag(detections, self.travel_from_landmark)

        # ------------------------------------------------------------
        # START_CLUSTER
        # ------------------------------------------------------------
        if self.segment_phase == "START_CLUSTER":

            # Docking/calibration 0 -> 1 protection:
            # no helper arrival while Tag 0 cluster is still present.
            if self.calibrating_move_to_tag1:
                if any_cluster_visible_now(detections, self.travel_from_landmark):
                    self.cluster_lost_count = 0
                    tag = choose_best_cluster_tag(detections, self.travel_from_landmark)
                    return tag, self.travel_from_landmark, f"TAG{self.travel_from_landmark}"

                self.cluster_lost_count += 1

                if self.cluster_lost_count >= CLUSTER_LOST_FRAMES_REQUIRED:
                    self.segment_phase = "SEARCH_TARGET"
                    self.append_log(
                        f"Fully left docking Tag {self.travel_from_landmark}. "
                        f"Now target {self.travel_to_landmark} can use central tag or side-pair helpers."
                    )

                return None, None, ""

            # Normal mission: if exact old central is visible, it is definitely
            # still the previous landmark. Do not accept helpers yet.
            if old_central is not None:
                self.cluster_lost_count = 0
                return old_central, self.travel_from_landmark, f"TAG{self.travel_from_landmark}"

            # Helper arrival disabled:
            # helpers are correction / sequence evidence only.
            # The target is reached only when the central target tag is visible.
            # Valid side-pair is strong enough to mark target reached.
            helper_id, evidence_type = detect_local_arrival_helper(
                detections,
                self.travel_to_landmark,
                allow_single_cross=False
            )

            if helper_id is not None and evidence_type == "PAIR":
                allowed, reason = self.start_pair_helper_allowed(helper_id)

                if not allowed:
                    if time.time() - getattr(self, "_last_pair_wait_log", 0.0) > 0.80:
                        self._last_pair_wait_log = time.time()
                        self.append_log(
                            f"PAIR REJECTED BY DIRECTION: target={self.travel_to_landmark} "
                            f"helper={helper_id} group={helper_group_name(helper_id)} "
                            f"validEntryCenters={sorted(self.valid_entry_centers_for_heading())} "
                            f"exitSide={self.expected_exit_helper_id} "
                            f"reason={reason}. Correction only."
                        )

                    # Do not steer from rejected helper evidence.
                    # In the 15->11 / 11->4 tests, rejected previous/exit helpers
                    # created strong visual corrections in the wrong direction.
                    # Safer behavior: keep IMU heading until a valid target-side
                    # helper/central tag is seen.
                    return None, None, ""

                ready, center_y_error_px = self.local_helper_arrival_ready(
                    detections,
                    helper_id,
                    evidence_type
                )

                if not self.helper_is_valid_sequence_center_now(helper_id):
                    self.append_log(
                        f"HELPER PAIR REJECTED BY SEQUENCE: target={self.travel_to_landmark} "
                        f"helper={helper_id} group={helper_group_name(helper_id)} "
                        f"validEntryCenters={sorted(self.valid_entry_centers_for_heading())} "
                        f"reason={reason} seq={self.helper_sequence_note(helper_id)} "
                        f"candidates={sorted(self.corner_single_arrival_candidates)}. IMU hold."
                    )
                    return None, None, ""

                if ready and self.helper_can_confirm_reached_now(helper_id, center_y_error_px):
                    self.append_log(
                        f"EXPECTED LOCAL CENTER REACHED: target={self.travel_to_landmark} "
                        f"helper={helper_id} group={helper_group_name(helper_id)} "
                        f"seq={self.helper_sequence_note(helper_id)} "
                        f"centerY={center_y_error_px if center_y_error_px is not None else 999:.1f}px. "
                        "Applying landmark reached/pass-through."
                    )
                    self.start_local_arrival(self.travel_to_landmark, helper_id)
                    return None, None, ""

                if helper_id in set(getattr(self, "corner_single_arrival_candidates", set())) and center_y_error_px is not None:
                    if abs(center_y_error_px) > EXPECTED_LOCAL_CENTER_REACHED_Y_OK_PX:
                        self.append_log(
                            f"EXPECTED LOCAL CENTER NOT READY: target={self.travel_to_landmark} "
                            f"helper={helper_id} seq={self.helper_sequence_note(helper_id)} "
                            f"centerY={center_y_error_px:.1f}px "
                            f"need<={EXPECTED_LOCAL_CENTER_REACHED_Y_OK_PX}px. Correction continues."
                        )

                self.append_log(
                    f"HELPER PAIR CORRECTION ONLY: target={self.travel_to_landmark} "
                    f"helper={helper_id} group={helper_group_name(helper_id)} "
                    f"validEntryCenters={sorted(self.valid_entry_centers_for_heading())} "
                    f"reason={reason} seq={self.helper_sequence_note(helper_id)} "
                    f"centerY={center_y_error_px if center_y_error_px is not None else 999:.1f}px. "
                    "Waiting for central tag."
                )
                tag = find_tag(detections, helper_id)
                if tag is not None:
                    return tag, self.travel_to_landmark, f"TAG{self.travel_to_landmark}"
                return None, None, ""

            # Single cross helper while START_CLUSTER is only correction.
            single_helper = detect_single_cross_helper(detections, self.travel_to_landmark)

            if single_helper is not None:
                allowed_single, single_reason = self.single_helper_allowed_by_corner_sequence(single_helper)

                if allowed_single:
                    ready, center_y_error_px = self.local_helper_arrival_ready(
                        detections,
                        single_helper,
                        "SINGLE"
                    )

                    if ready and self.helper_can_confirm_reached_now(single_helper, center_y_error_px):
                        self.append_log(
                            f"EXPECTED SINGLE LOCAL CENTER REACHED: target={self.travel_to_landmark} "
                            f"helper={single_helper} reason={single_reason} "
                            f"seq={self.helper_sequence_note(single_helper)} "
                            f"centerY={center_y_error_px if center_y_error_px is not None else 999:.1f}px. "
                            "Applying landmark reached/pass-through."
                        )
                        self.start_local_arrival(self.travel_to_landmark, single_helper)
                        return None, None, ""

                    self.append_log(
                        f"SINGLE HELPER CORRECTION ONLY AFTER CORNER: target={self.travel_to_landmark} "
                        f"helper={single_helper} reason={single_reason} "
                        f"seq={self.helper_sequence_note(single_helper)} "
                        f"centerY={center_y_error_px if center_y_error_px is not None else 999:.1f}px. "
                        "Waiting for central tag."
                    )
                    tag = find_tag(detections, single_helper)
                    if tag is not None:
                        return tag, self.travel_to_landmark, f"TAG{self.travel_to_landmark}"
                    return None, None, ""

                if time.time() - getattr(self, "_last_single_log", 0.0) > 0.90:
                    self._last_single_log = time.time()
                    self.append_log(
                        f"SINGLE HELPER CORRECTION ONLY: target={self.travel_to_landmark} "
                        f"helper={single_helper} exitSide={self.expected_exit_helper_id} "
                        f"cornerCandidates={sorted(self.corner_single_arrival_candidates)} "
                        f"reason={single_reason}; not reached."
                    )

                # Do not steer from ambiguous single helper evidence.
                # Wait for a valid pair, valid corner->single sequence, or central tag.
                return None, None, ""

            # Direction-entry helper correction.
            # If old central is gone and a helper from the target entry side is
            # visible, steer as TARGET correction, not old correction.
            entry_tag = self.entry_side_helper_visible(detections)
            if entry_tag is not None:
                if time.time() - getattr(self, "_last_entry_helper_corr_log", 0.0) > 0.80:
                    self._last_entry_helper_corr_log = time.time()
                    self.append_log(
                        f"ENTRY SIDE TARGET CORRECTION: seg={self.travel_from_landmark}->{self.travel_to_landmark} "
                        f"heading={HEADING_LABELS.get(self.segment_heading, '---')} "
                        f"seen={int(entry_tag.tag_id)} "
                        f"seq={self.helper_sequence_note(int(entry_tag.tag_id))} "
                        f"entrySide={sorted(ENTRY_HELPER_IDS_BY_HEADING.get(self.segment_heading, set()))}"
                    )
                return entry_tag, self.travel_to_landmark, f"TAG{self.travel_to_landmark}"

            # Minimal leak fix:
            # If old central is not visible, do NOT use arbitrary helper tags as
            # old-cluster correction. This was the 13->12 drift case:
            #   508 was valid EAST entry evidence,
            #   501 was rejected,
            #   but fallback still used 501 as old helper correction and sent
            #   a strong wrong VEL.
            #
            # Old central correction is already handled above by old_central.
            # Non-valid helper-only evidence should fall back to IMU hold.
            if any_cluster_visible_now(detections, self.travel_from_landmark):
                self.cluster_lost_count = 0
                return None, None, ""

            self.cluster_lost_count += 1

            if self.cluster_lost_count >= CLUSTER_LOST_FRAMES_REQUIRED:
                self.segment_phase = "SEARCH_TARGET"
                self.append_log(
                    f"Fully left Tag {self.travel_from_landmark}. "
                    f"Now target {self.travel_to_landmark} can use central, side-pair, or centered single cross helper."
                )

            return None, None, ""



        # ------------------------------------------------------------
        # SEARCH_TARGET
        # ------------------------------------------------------------
        if self.segment_phase == "SEARCH_TARGET":

            allow_single = not self.calibrating_move_to_tag1

            # Helper arrival disabled in SEARCH_TARGET as well.
            helper_id, evidence_type = detect_local_arrival_helper(
                detections,
                self.travel_to_landmark,
                allow_single_cross=allow_single
            )

            if helper_id is not None:
                allowed, reason = self.start_pair_helper_allowed(helper_id)

                if not allowed:
                    if time.time() - getattr(self, "_last_pair_wait_log", 0.0) > 0.80:
                        self._last_pair_wait_log = time.time()
                        self.append_log(
                            f"SEARCH HELPER REJECTED BY DIRECTION: target={self.travel_to_landmark} "
                            f"helper={helper_id} exitSide={self.expected_exit_helper_id} reason={reason}. "
                            "Correction only."
                        )

                    return None, None, ""

                ready, center_y_error_px = self.local_helper_arrival_ready(
                    detections,
                    helper_id,
                    evidence_type
                )

                if time.time() - getattr(self, "_last_pair_log", 0.0) > 0.60:
                    self._last_pair_log = time.time()
                    self.append_log(
                        f"SEARCH {evidence_type} target={self.travel_to_landmark} helper={helper_id} "
                        f"centerY={center_y_error_px if center_y_error_px is not None else 999:.1f}px "
                        f"ready={ready} singleAllowed={allow_single}"
                    )

                if ready:
                    # Helper is only correction / sequence evidence.
                    # Do not mark target reached until central target tag is visible.
                    if not self.helper_is_valid_sequence_center_now(helper_id):
                        self.append_log(
                            f"SEARCH HELPER REJECTED BY SEQUENCE: target={self.travel_to_landmark} "
                            f"helper={helper_id} evidence={evidence_type} "
                            f"seq={self.helper_sequence_note(helper_id)} "
                            f"candidates={sorted(self.corner_single_arrival_candidates)}. IMU hold."
                        )
                        return None, None, ""

                    if self.helper_can_confirm_reached_now(helper_id, center_y_error_px):
                        self.append_log(
                            f"SEARCH EXPECTED LOCAL CENTER REACHED: target={self.travel_to_landmark} "
                            f"helper={helper_id} evidence={evidence_type} "
                            f"seq={self.helper_sequence_note(helper_id)} "
                            f"centerY={center_y_error_px if center_y_error_px is not None else 999:.1f}px. "
                            "Applying landmark reached/pass-through."
                        )
                        self.start_local_arrival(self.travel_to_landmark, helper_id)
                        return None, None, ""

                    tag = find_tag(detections, helper_id)
                    if tag is not None:
                        return tag, self.travel_to_landmark, f"TAG{self.travel_to_landmark}"
                    return None, None, ""

                tag = find_tag(detections, helper_id)
                if tag is not None:
                    return tag, self.travel_to_landmark, f"TAG{self.travel_to_landmark}"

            tag = choose_best_cluster_tag(detections, self.travel_to_landmark)

            if tag is not None:
                if int(tag.tag_id) in HELPER_IDS:
                    entry_tag = self.entry_side_helper_visible(detections)
                    if entry_tag is None:
                        return None, None, ""
                    return entry_tag, self.travel_to_landmark, f"TAG{self.travel_to_landmark}"

                return tag, self.travel_to_landmark, f"TAG{self.travel_to_landmark}"

            return None, None, ""

        return None, None, ""


    def control_tick(self):
        # Read ESP32 output.
        lines = self.read_esp32_available()

        if self.route_state == "TURNING" and self.turning_waiting:
            done = any("OK TURN_DONE" in line for line in lines)

            if done:
                self.append_log("ESP32 reported OK TURN_DONE")
                self.turning_waiting = False
                self.route_state = "MOVE"
                if self.path_index < len(self.active_path):
                    next_tag = self.active_path[self.path_index]
                    self.start_segment(self.current_tag, next_tag)
                    self.lock_heading_go()
                self.update_ui_state()

            # No Python-side timeout. ESP32 IMU owns turn completion.
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
                if time.time() - getattr(self, "_last_corr_log", 0.0) > 1.20:
                    self._last_corr_log = time.time()
                    self.append_log(
                        f"{label} CORR seg={self.travel_from_landmark}->{self.travel_to_landmark} "
                        f"phase={self.segment_phase} seen={seen_tag_id} grid=({helper_x_grid},{helper_y_grid}) "
                        f"yawErr={yaw_error:.2f} centerXM={center_x_m:.4f} level={error_level} "
                        f"xSign={X_SIGN_BY_HEADING.get(getattr(self, 'segment_heading', None), -1.0):.1f} "
                        f"corr={correction} L={left} R={right}"
                    )
        elif self.route_state == "MOVE":

            helper_ids_now = visible_helper_ids(detections)

            if self.last_drive_mode == "VISION" and helper_ids_now:
                # A helper is visible, but choose_move_correction() returned no
                # valid correction target. That means the helper is previous
                # blocked, exit-side, sequence-rejected, or otherwise ambiguous.
                #
                # Do NOT keep the last visual VEL here.
                # Keeping the previous VEL was the drift source:
                #   pass-through applies a strong correction near the old tag,
                #   rejected helper appears,
                #   old correction keeps turning the robot away from the lane.
                #
                # Correct behavior:
                #   reject the helper for correction/arrival,
                #   clear the old visual command,
                #   let ESP32 hold the current IMU heading until a valid target
                #   entry helper or central tag appears.
                self.skip_next_segment_lock = False
                self.filtered_correction = 0.0

                if self.last_drive_mode != "IMU":
                    self.send_esp32("LOCK_HEADING_GO")
                    self.last_drive_mode = "IMU"

                if time.time() - getattr(self, "_last_rejected_helper_imu_log", 0.0) > 0.80:
                    self._last_rejected_helper_imu_log = time.time()
                    self.append_log(
                        "Rejected/ambiguous helper visible; using IMU heading hold instead of carrying old visual VEL."
                    )
                return

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
            self.append_log("Local arrival helper lost. Treating landmark as reached, same as old cluster code.")
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
        self.current_heading = NORTH
        self.mission_running = False
        self.route_state = "IDLE"
        self.previous_helper_block_ids = set()

        # For distinguishing previous local tag vs arrived local tag.
        # If two consecutive segments are in the same heading, the expected
        # arriving side-center helper should normally remain the same.
        # If the heading changes, helper arrival in START_CLUSTER is blocked
        # until there is a clean "no helper visible" transition.
        self.last_arrival_helper_id = None
        self.last_arrival_heading = None
        self.segment_heading = None
        self.expected_straight_helper_id = None
        self.all_helpers_cleared_once = False
        self.expected_entry_helper_id = None
        self.expected_exit_helper_id = None
        self.segment_next_action = "UNKNOWN"

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
        self.current_heading = NORTH

        self.append_log("Moving from docking Tag 0 to grid Tag 1. Rule: no single-helper arrival during docking; helpers remain OLD until full loss.")
        self.start_segment(DOCK_TAG, GRID_START_TAG)
        self.lock_heading_go()

    def finish_calibration_at_tag1(self):
        self.stop_robot()
        self.current_tag = GRID_START_TAG
        self.current_heading = NORTH
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

        self.append_log("BUILD: unique_local_helpers_15_with_dock active")

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
            self.append_log(f"Initial turn: TURN_REL {turn_deg:.1f} -- waiting for ESP32 OK TURN_DONE")
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
        self.previous_helper_block_ids = set()

        # For distinguishing previous local tag vs arrived local tag.
        # If two consecutive segments are in the same heading, the expected
        # arriving side-center helper should normally remain the same.
        # If the heading changes, helper arrival in START_CLUSTER is blocked
        # until there is a clean "no helper visible" transition.
        self.last_arrival_helper_id = None
        self.last_arrival_heading = None
        self.segment_heading = None
        self.expected_straight_helper_id = None
        self.all_helpers_cleared_once = False
        self.expected_entry_helper_id = None
        self.expected_exit_helper_id = None
        self.segment_next_action = "UNKNOWN"
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