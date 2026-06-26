#!/usr/bin/env python3

import math
import time

import cv2
import numpy as np
import serial

from picamera2 import Picamera2
from pupil_apriltags import Detector


# =====================================================
# ROUTE
# =====================================================

START_TAG = 11
TAG_7 = 7
TAG_6 = 6
TAG_5 = 5
TAG_9 = 9

ROUTE = [START_TAG, TAG_7, TAG_6, TAG_5, TAG_9]

TURN_DEG_BY_LANDMARK = {
    TAG_6: 90.0,
    TAG_5: -90.0,
}

TURN_LANDMARKS = {TAG_6, TAG_5}
FINAL_LANDMARK = TAG_9
PASS_THROUGH_LANDMARKS = {TAG_7}

route_state = "WAIT_START"

travel_from_landmark = None
travel_to_landmark = None

turn_landmark_id = None


# =====================================================
# TRAVEL PHASE
# =====================================================

segment_phase = "START_CLUSTER"
cluster_lost_count = 0
target_central_seen_count = 0

CLUSTER_LOST_FRAMES_REQUIRED = 5
TARGET_CENTRAL_SEEN_FRAMES_REQUIRED = 2


# =====================================================
# LOCAL ARRIVAL NUDGE
# =====================================================

local_arrival_landmark = None
local_arrival_helper_id = None
local_arrival_good_count = 0
local_arrival_start_time = 0.0

LOCAL_NUDGE_PPS = 500
LOCAL_NUDGE_CENTER_Y_OK_PX = 30
LOCAL_NUDGE_GOOD_FRAMES_REQUIRED = 3
LOCAL_NUDGE_TIMEOUT_SEC = 5.0


# =====================================================
# DRIVE PARAMETERS
# =====================================================

MAX_PPS = 10000

VISION_BASE_PPS = 6500 #3500
VISION_BASE_PPS_SLOW = 3500 #2600
VISION_MIN_PPS = 2500 #1600
VISION_MAX_PPS = 6500 #5200

# If manual correction or local nudge moves wrong direction, flip this to -1
FB_SIGN = 1


# =====================================================
# APRILTAG / CLUSTER PARAMETERS
# =====================================================

TAG_SIZE_M = 0.010
CLUSTER_SPACING_M = 0.015

EXPECTED_TAG_YAW_DEG = 0.0

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
CROSS_HELPERS = {501, 503, 505, 507}
CORNER_HELPERS = {502, 504, 506, 508}


# =====================================================
# TRAVEL CORRECTION PARAMETERS
# =====================================================

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

filtered_correction = 0.0


# =====================================================
# TURN TAG CORRECTION PARAMETERS
# =====================================================

KP_TURN_TAG_YAW_PPS_PER_DEG = 20
MAX_TURN_TAG_YAW_CORRECTION_PPS = 150

TURN_TAG_FB_PPS = 550
TURN_TAG_YAW_OK_DEG = 3.0
TURN_TAG_CENTER_Y_OK_PX = 25
TURN_TAG_GOOD_FRAMES_REQUIRED = 3

turn_tag_good_count = 0


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
# SERIAL STATE
# =====================================================

drive_left_pps = 0
drive_right_pps = 0

last_drive_mode = "STOP"
last_vel_send_time = 0.0
VEL_SEND_INTERVAL_SEC = 0.07


# =====================================================
# SERIAL SETUP
# =====================================================

ser = serial.Serial(
    "/dev/ttyUSB0",
    115200,
    timeout=0.2
)

time.sleep(2)


# =====================================================
# CAMERA SETUP
# =====================================================

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (FRAME_WIDTH, FRAME_HEIGHT),
        "format": "RGB888"
    }
)

picam2.configure(config)

picam2.set_controls({
    "AeEnable": False,
    "AwbEnable": False,
    "ExposureTime": 5000,
    "AnalogueGain": 1.0
})

picam2.start()


# =====================================================
# APRILTAG DETECTOR
# =====================================================

detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=2.0,
    refine_edges=1
)


# =====================================================
# SERIAL HELPERS
# =====================================================

def send_command(cmd):
    ser.write((cmd + "\n").encode())


def send_velocity(left_pps, right_pps, force=False):
    global drive_left_pps
    global drive_right_pps
    global last_vel_send_time
    global last_drive_mode

    now = time.time()

    left_pps = int(np.clip(left_pps, -MAX_PPS, MAX_PPS))
    right_pps = int(np.clip(right_pps, -MAX_PPS, MAX_PPS))

    if not force:
        if (
            left_pps == drive_left_pps
            and right_pps == drive_right_pps
            and now - last_vel_send_time < VEL_SEND_INTERVAL_SEC
        ):
            return

    drive_left_pps = left_pps
    drive_right_pps = right_pps
    last_vel_send_time = now
    last_drive_mode = "VISION"

    send_command(f"VEL {left_pps} {right_pps}")


def stop_robot():
    global drive_left_pps
    global drive_right_pps
    global last_drive_mode

    drive_left_pps = 0
    drive_right_pps = 0
    last_drive_mode = "STOP"

    send_command("STOP")


def lock_heading_go():
    global last_drive_mode
    global filtered_correction

    if last_drive_mode == "IMU":
        return

    filtered_correction = 0.0

    send_command("LOCK_HEADING_GO")
    last_drive_mode = "IMU"

    print("RPI: LOCK_HEADING_GO sent")


def read_esp32_lines():
    lines = []

    while ser.in_waiting > 0:
        line = ser.readline().decode(errors="ignore").strip()

        if line:
            print(f"ESP32: {line}")
            lines.append(line)

    return lines


def wait_for_esp32_text(expected_text, timeout_sec=10.0):
    start = time.time()

    while time.time() - start < timeout_sec:
        line = ser.readline().decode(errors="ignore").strip()

        if line:
            print(f"ESP32: {line}")

        if expected_text in line:
            return True

    return False


# =====================================================
# BASIC TAG MATH
# =====================================================

def normalize_angle(angle):
    return ((angle + 180.0) % 360.0) - 180.0


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


# =====================================================
# TAG SELECTION
# =====================================================

def visible_ids(detections):
    ids = set()

    for tag in detections:
        ids.add(tag.tag_id)

    return ids


def find_tag(detections, tag_id):
    for tag in detections:
        if tag.tag_id == tag_id:
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
    """
    This function assumes the current state already knows
    which landmark the helper tags belong to.
    """

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
    """
    Used after pressing c.
    Prefer central tag, then cross helpers, then any helper.
    """

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
    """
    Called only after the old cluster has completely disappeared.

    Tag 7 is pass-through, so side-pair local arrival is disabled for Tag 7.

    Single 501/503/505/507 is not enough.
    Side-center + adjacent corner is enough.

    502 + 503 or 503 + 504 -> 503
    506 + 507 or 507 + 508 -> 507
    508 + 501 or 501 + 502 -> 501
    506 + 505 or 505 + 504 -> 505
    """

    if target_landmark in PASS_THROUGH_LANDMARKS:
        return None

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


# =====================================================
# LANDMARK POSE FROM CLUSTER
# =====================================================

def get_helper_grid_offset(tag_id, central_tag_id):
    if tag_id == central_tag_id:
        return 0, 0

    if tag_id in HELPER_GRID_OFFSET:
        return HELPER_GRID_OFFSET[tag_id]

    return None


def get_landmark_pose_from_cluster_tag(tag, central_tag_id):
    grid_offset = get_helper_grid_offset(tag.tag_id, central_tag_id)

    if grid_offset is None:
        return None

    helper_x_grid, helper_y_grid = grid_offset

    raw_yaw = compute_yaw_deg(tag)

    yaw_error = normalize_angle(
        raw_yaw - EXPECTED_TAG_YAW_DEG
    )

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
        helper_y_grid
    )


def get_visible_tag_pose(tag):
    raw_yaw = compute_yaw_deg(tag)

    yaw_error = normalize_angle(
        raw_yaw - EXPECTED_TAG_YAW_DEG
    )

    visible_x_error_px = float(tag.center[0] - CX)
    visible_y_error_px = float(tag.center[1] - CY)

    return (
        raw_yaw,
        yaw_error,
        visible_x_error_px,
        visible_y_error_px,
        tag.tag_id
    )


# =====================================================
# TRAVEL CORRECTION
# =====================================================

def reset_correction_filter():
    global filtered_correction
    filtered_correction = 0.0


def adaptive_error_level(yaw_error, center_x_m):
    abs_x = abs(center_x_m)
    abs_yaw = abs(yaw_error)

    if abs_x >= X_LARGE_ERROR_M or abs_yaw >= YAW_LARGE_ERROR_DEG:
        return "LARGE"

    if abs_x >= X_MEDIUM_ERROR_M or abs_yaw >= YAW_MEDIUM_ERROR_DEG:
        return "MEDIUM"

    return "SMALL"


def travelling_velocity(raw_yaw, yaw_error, center_x_m):
    global filtered_correction

    error_level = adaptive_error_level(
        yaw_error,
        center_x_m
    )

    if error_level == "LARGE":
        kp_yaw = KP_YAW_STRONG_PPS_PER_DEG
        kp_x = KP_X_STRONG_PPS_PER_M
        max_corr = MAX_VISION_CORRECTION_STRONG_PPS
        base_pps = VISION_BASE_PPS_SLOW
        alpha = CORRECTION_FILTER_ALPHA_STRONG

    elif error_level == "MEDIUM":
        kp_yaw = (KP_YAW_PPS_PER_DEG + KP_YAW_STRONG_PPS_PER_DEG) * 0.5
        kp_x = (KP_X_PPS_PER_M + KP_X_STRONG_PPS_PER_M) * 0.5
        max_corr = int(
            (MAX_VISION_CORRECTION_PPS + MAX_VISION_CORRECTION_STRONG_PPS)
            * 0.5
        )
        base_pps = int(
            (VISION_BASE_PPS + VISION_BASE_PPS_SLOW)
            * 0.5
        )
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

    raw_correction = float(np.clip(
        raw_correction,
        -max_corr,
        max_corr
    ))

    filtered_correction = (
        (1.0 - alpha) * filtered_correction
        + alpha * raw_correction
    )

    correction = int(filtered_correction)

    left = base_pps - correction
    right = base_pps + correction

    left = int(np.clip(left, VISION_MIN_PPS, VISION_MAX_PPS))
    right = int(np.clip(right, VISION_MIN_PPS, VISION_MAX_PPS))

    return left, right, correction, yaw_corr, x_corr, error_level


# =====================================================
# TURN / LOCAL CORRECTION
# =====================================================

def visible_tag_center_velocity(raw_yaw, yaw_error, y_error_px, pps):
    yaw_for_control = yaw_error

    if abs(yaw_for_control) < YAW_DEADBAND_DEG:
        yaw_for_control = 0.0

    yaw_corr = -KP_TURN_TAG_YAW_PPS_PER_DEG * yaw_for_control

    yaw_corr = int(np.clip(
        yaw_corr,
        -MAX_TURN_TAG_YAW_CORRECTION_PPS,
        MAX_TURN_TAG_YAW_CORRECTION_PPS
    ))

    if abs(y_error_px) <= TURN_TAG_CENTER_Y_OK_PX:
        fb = 0
    else:
        if y_error_px > 0:
            fb = pps * FB_SIGN
        else:
            fb = -pps * FB_SIGN

    left = fb - yaw_corr
    right = fb + yaw_corr

    if fb == 0:
        left = -yaw_corr
        right = yaw_corr

    left = int(np.clip(left, -MAX_PPS, MAX_PPS))
    right = int(np.clip(right, -MAX_PPS, MAX_PPS))

    return left, right, yaw_corr


# =====================================================
# SEGMENT / ARRIVAL
# =====================================================

def start_segment(from_tag, to_tag):
    global route_state
    global travel_from_landmark
    global travel_to_landmark
    global segment_phase
    global cluster_lost_count
    global target_central_seen_count

    travel_from_landmark = from_tag
    travel_to_landmark = to_tag

    segment_phase = "START_CLUSTER"
    cluster_lost_count = 0
    target_central_seen_count = 0

    reset_correction_filter()

    route_state = "MOVE"

    print(f"RPI: Starting segment {from_tag} -> {to_tag}")


def next_route_after(tag_id):
    for i in range(len(ROUTE) - 1):
        if ROUTE[i] == tag_id:
            return ROUTE[i + 1]

    return None


def handle_landmark_arrival(landmark_id):
    """
    Turn and final landmarks stop.
    Pass-through landmarks are handled separately before this function is called.
    """

    global route_state
    global turn_landmark_id

    stop_robot()
    reset_correction_filter()

    if landmark_id in TURN_LANDMARKS:

        turn_landmark_id = landmark_id
        route_state = "WAIT_TURN_TAG_CORRECT_COMMAND"

        print(f"RPI: Reached turn landmark {landmark_id}.")
        print("RPI: Press 'c' to correct.")
        print("RPI: Press 't' after correction to turn.")

        return

    if landmark_id == FINAL_LANDMARK:

        route_state = "DONE"
        print("RPI: Final landmark reached. Robot stopped.")


def start_local_arrival(landmark_id, helper_id):
    global route_state
    global local_arrival_landmark
    global local_arrival_helper_id
    global local_arrival_good_count
    global local_arrival_start_time

    stop_robot()

    local_arrival_landmark = landmark_id
    local_arrival_helper_id = helper_id
    local_arrival_good_count = 0
    local_arrival_start_time = time.time()

    route_state = "LOCAL_ARRIVAL"

    print(
        f"RPI: Local arrival at landmark {landmark_id}. "
        f"Nudging to helper {helper_id}."
    )


def pass_through_to_next(current_tag, visible_tag):
    """
    Switch segment without stopping.
    Return the visible pass-through tag so the same frame still sends VISION velocity.
    This prevents the visible jerk at Tag 7.
    """

    next_tag = next_route_after(current_tag)

    if next_tag is None:
        return None, None, ""

    print(
        f"RPI: Pass-through landmark {current_tag}. "
        f"Switching to {current_tag}->{next_tag} without stopping."
    )

    start_segment(
        current_tag,
        next_tag
    )

    return visible_tag, current_tag, f"TAG{current_tag}"


# =====================================================
# MOVE STATE LOGIC
# =====================================================

def choose_move_correction(detections):
    global segment_phase
    global cluster_lost_count
    global target_central_seen_count

    central_target = find_tag(
        detections,
        travel_to_landmark
    )

    if central_target is not None:
        target_central_seen_count += 1

        if target_central_seen_count >= TARGET_CENTRAL_SEEN_FRAMES_REQUIRED:

            if travel_to_landmark in PASS_THROUGH_LANDMARKS:
                return pass_through_to_next(
                    travel_to_landmark,
                    central_target
                )

            handle_landmark_arrival(
                travel_to_landmark
            )

            return None, None, ""

        return central_target, travel_to_landmark, f"TAG{travel_to_landmark}"

    target_central_seen_count = 0

    if segment_phase == "START_CLUSTER":

        if any_cluster_visible_now(detections, travel_from_landmark):
            cluster_lost_count = 0

            tag = choose_best_cluster_tag(
                detections,
                travel_from_landmark
            )

            return tag, travel_from_landmark, f"TAG{travel_from_landmark}"

        cluster_lost_count += 1

        if cluster_lost_count >= CLUSTER_LOST_FRAMES_REQUIRED:
            segment_phase = "SEARCH_TARGET"
            print(
                f"RPI: Fully left landmark {travel_from_landmark}. "
                f"Now helpers can belong to {travel_to_landmark}."
            )

        return None, None, ""

    if segment_phase == "SEARCH_TARGET":

        helper_id = detect_side_pair(
            detections,
            travel_to_landmark
        )

        if helper_id is not None:
            start_local_arrival(
                travel_to_landmark,
                helper_id
            )

            return None, None, ""

        tag = choose_best_cluster_tag(
            detections,
            travel_to_landmark
        )

        if tag is not None:
            return tag, travel_to_landmark, f"TAG{travel_to_landmark}"

        return None, None, ""

    return None, None, ""


# =====================================================
# DISPLAY
# =====================================================

def draw_tags(frame, detections):
    ids = []

    for tag in detections:

        ids.append(tag.tag_id)

        corners = tag.corners.astype(int)

        for i in range(4):
            p1 = tuple(corners[i])
            p2 = tuple(corners[(i + 1) % 4])
            cv2.line(frame, p1, p2, (0, 255, 0), 2)

        center = tuple(tag.center.astype(int))
        cv2.circle(frame, center, 5, (0, 0, 255), -1)

        raw_yaw = compute_yaw_deg(tag)
        yaw_error = normalize_angle(
            raw_yaw - EXPECTED_TAG_YAW_DEG
        )

        x_m = compute_lateral_x_m(tag)

        cv2.putText(
            frame,
            f"ID:{tag.tag_id}",
            (center[0] + 10, center[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Yaw:{yaw_error:.1f}",
            (center[0] + 10, center[1] + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"xM:{x_m:.3f}",
            (center[0] + 10, center[1] + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 255),
            2
        )

    return ids


# =====================================================
# STARTUP PRINT
# =====================================================

print("Waiting for docking Tag 11...")
print("Press 's' when Tag 11 is visible and robot is aligned.")
print("")
print("Current logic:")
print("  Helpers are old landmark helpers until all helpers disappear.")
print("  After full cluster loss, helpers can belong to target landmark.")
print("  Tag 7 is pass-through with no stop.")
print("  Tag 6 and Tag 5 are turn landmarks.")
print("  Tag 9 is final.")
print("")
print("Keys:")
print("  s = start")
print("  c = correct turn landmark")
print("  t = turn")
print("  q = quit")


# =====================================================
# MAIN LOOP
# =====================================================

try:
    while True:

        frame = picam2.capture_array()

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2GRAY
        )

        detections = detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=CAMERA_PARAMS,
            tag_size=TAG_SIZE_M
        )

        ids = draw_tags(frame, detections)

        # =================================================
        # STATE MACHINE
        # =================================================

        if route_state == "WAIT_START":

            pass

        elif route_state == "MOVE":

            tag, landmark_id, label = choose_move_correction(detections)

            if tag is not None and route_state == "MOVE":

                pose = get_landmark_pose_from_cluster_tag(
                    tag,
                    landmark_id
                )

                if pose is not None:

                    (
                        raw_yaw,
                        yaw_error,
                        center_x_m,
                        center_y_error_px,
                        seen_tag_id,
                        helper_x_grid,
                        helper_y_grid
                    ) = pose

                    (
                        left,
                        right,
                        correction,
                        yaw_corr,
                        x_corr,
                        error_level
                    ) = travelling_velocity(
                        raw_yaw,
                        yaw_error,
                        center_x_m
                    )

                    send_velocity(left, right)

                    print(
                        f"{label} CORR "
                        f"seg={travel_from_landmark}->{travel_to_landmark} "
                        f"phase={segment_phase} "
                        f"seen={seen_tag_id} "
                        f"grid=({helper_x_grid},{helper_y_grid}) "
                        f"yawErr={yaw_error:.2f} "
                        f"centerXM={center_x_m:.4f} "
                        f"centerYerr={center_y_error_px:.1f}px "
                        f"level={error_level} "
                        f"corr={correction} "
                        f"L={left} R={right}"
                    )

            elif route_state == "MOVE":

                lock_heading_go()

        elif route_state == "LOCAL_ARRIVAL":

            central = find_tag(
                detections,
                local_arrival_landmark
            )

            if central is not None:

                print(
                    f"RPI: Central tag {local_arrival_landmark} became visible during local arrival."
                )

                handle_landmark_arrival(
                    local_arrival_landmark
                )

            else:

                helper = find_tag(
                    detections,
                    local_arrival_helper_id
                )

                if helper is None:
                    helper = choose_best_cluster_tag(
                        detections,
                        local_arrival_landmark
                    )

                if helper is None:

                    print("RPI: Local arrival tag lost. Stopping as reached.")
                    handle_landmark_arrival(
                        local_arrival_landmark
                    )

                else:

                    (
                        raw_yaw,
                        yaw_error,
                        visible_x_error_px,
                        visible_y_error_px,
                        seen_tag_id
                    ) = get_visible_tag_pose(helper)

                    if abs(visible_y_error_px) <= LOCAL_NUDGE_CENTER_Y_OK_PX:

                        local_arrival_good_count += 1
                        stop_robot()

                        print(
                            f"LOCAL_ARRIVAL_GOOD "
                            f"landmark={local_arrival_landmark} "
                            f"helper={seen_tag_id} "
                            f"yErr={visible_y_error_px:.1f}px "
                            f"good={local_arrival_good_count}/{LOCAL_NUDGE_GOOD_FRAMES_REQUIRED}"
                        )

                        if local_arrival_good_count >= LOCAL_NUDGE_GOOD_FRAMES_REQUIRED:
                            handle_landmark_arrival(
                                local_arrival_landmark
                            )

                    else:

                        local_arrival_good_count = 0

                        left, right, yaw_corr = visible_tag_center_velocity(
                            raw_yaw,
                            yaw_error,
                            visible_y_error_px,
                            LOCAL_NUDGE_PPS
                        )

                        send_velocity(left, right)

                        print(
                            f"LOCAL_ARRIVAL "
                            f"landmark={local_arrival_landmark} "
                            f"helper={seen_tag_id} "
                            f"targetHelper={local_arrival_helper_id} "
                            f"yErr={visible_y_error_px:.1f}px "
                            f"L={left} R={right}"
                        )

                if time.time() - local_arrival_start_time > LOCAL_NUDGE_TIMEOUT_SEC:

                    print("RPI: Local arrival timeout. Stopping as reached.")
                    handle_landmark_arrival(
                        local_arrival_landmark
                    )

        elif route_state == "WAIT_TURN_TAG_CORRECT_COMMAND":

            stop_robot()

        elif route_state == "TURN_TAG_CORRECTING":

            tag = choose_correction_tag(
                detections,
                turn_landmark_id
            )

            if tag is None:

                stop_robot()
                route_state = "WAIT_TURN_TAG_CORRECT_COMMAND"

                print("RPI: Correction tag lost. Press c again.")

            else:

                (
                    raw_yaw,
                    yaw_error,
                    visible_x_error_px,
                    visible_y_error_px,
                    seen_tag_id
                ) = get_visible_tag_pose(tag)

                good = (
                    abs(yaw_error) <= TURN_TAG_YAW_OK_DEG
                    and abs(visible_y_error_px) <= TURN_TAG_CENTER_Y_OK_PX
                )

                if good:

                    turn_tag_good_count += 1
                    stop_robot()

                    print(
                        f"TURN_CORR_GOOD "
                        f"landmark={turn_landmark_id} "
                        f"seen={seen_tag_id} "
                        f"good={turn_tag_good_count}/{TURN_TAG_GOOD_FRAMES_REQUIRED} "
                        f"yawErr={yaw_error:.2f} "
                        f"yErr={visible_y_error_px:.1f}px"
                    )

                    if turn_tag_good_count >= TURN_TAG_GOOD_FRAMES_REQUIRED:
                        route_state = "WAIT_TURN_COMMAND"
                        print("RPI: Correction complete. Press t to turn.")

                else:

                    turn_tag_good_count = 0

                    left, right, yaw_corr = visible_tag_center_velocity(
                        raw_yaw,
                        yaw_error,
                        visible_y_error_px,
                        TURN_TAG_FB_PPS
                    )

                    send_velocity(left, right)

                    print(
                        f"TURN_CORR "
                        f"landmark={turn_landmark_id} "
                        f"seen={seen_tag_id} "
                        f"yawErr={yaw_error:.2f} "
                        f"yErr={visible_y_error_px:.1f}px "
                        f"L={left} R={right}"
                    )

        elif route_state == "WAIT_TURN_COMMAND":

            stop_robot()

        elif route_state == "TURNING":

            lines = read_esp32_lines()

            for line in lines:
                if "OK TURN_DONE" in line:

                    print(f"RPI: Turn at Tag {turn_landmark_id} complete.")

                    next_tag = next_route_after(
                        turn_landmark_id
                    )

                    if next_tag is not None:
                        start_segment(
                            turn_landmark_id,
                            next_tag
                        )

                        lock_heading_go()

                    else:
                        route_state = "DONE"

        elif route_state == "DONE":

            stop_robot()

        # =================================================
        # DISPLAY
        # =================================================

        cv2.line(
            frame,
            (int(CX) - 30, int(CY)),
            (int(CX) + 30, int(CY)),
            (255, 255, 255),
            2
        )

        cv2.line(
            frame,
            (int(CX), int(CY) - 30),
            (int(CX), int(CY) + 30),
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"State: {route_state}",
            (40, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Seg: {travel_from_landmark}->{travel_to_landmark} phase:{segment_phase}",
            (40, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Lost:{cluster_lost_count} Seen:{target_central_seen_count}",
            (40, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"TurnTag:{turn_landmark_id} Local:{local_arrival_landmark}->{local_arrival_helper_id}",
            (40, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Drive:{last_drive_mode} L:{drive_left_pps} R:{drive_right_pps}",
            (40, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Visible:{ids}",
            (40, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "AGV Minimal Route Logic - Smooth Pass Through",
            frame
        )

        if route_state != "TURNING":
            read_esp32_lines()

        key = cv2.waitKey(1) & 0xFF

        # =================================================
        # KEY COMMANDS
        # =================================================

        if key == ord('s'):

            if route_state == "WAIT_START":

                tag11 = choose_best_cluster_tag(
                    detections,
                    START_TAG
                )

                if tag11 is not None:

                    print("RPI: Docking landmark 11 visible.")
                    print("RPI: Requesting IMU recalibration.")
                    print("RPI: Keep robot still.")

                    stop_robot()
                    time.sleep(0.3)

                    ser.reset_input_buffer()
                    send_command("IMU RECAL")

                    ok = wait_for_esp32_text(
                        "OK IMU RECAL",
                        timeout_sec=10.0
                    )

                    if ok:

                        print("RPI: IMU calibrated.")

                        start_segment(
                            START_TAG,
                            TAG_7
                        )

                        lock_heading_go()

                    else:

                        print("RPI: Start failed. ESP32 did not confirm IMU RECAL.")

                else:

                    print("RPI: Start ignored. Tag 11 cluster not visible.")

        elif key == ord('c'):

            if route_state == "WAIT_TURN_TAG_CORRECT_COMMAND":

                if turn_landmark_id is None:

                    print("RPI: No turn landmark selected.")

                else:

                    tag = choose_correction_tag(
                        detections,
                        turn_landmark_id
                    )

                    if tag is None:

                        print("RPI: Cannot correct. No tag visible.")

                    else:

                        turn_tag_good_count = 0
                        route_state = "TURN_TAG_CORRECTING"

                        print(f"RPI: Starting correction for Tag {turn_landmark_id}.")

            else:

                print("RPI: c ignored. Not waiting for correction.")

        elif key == ord('t'):

            if route_state == "WAIT_TURN_COMMAND":

                if turn_landmark_id is None:

                    print("RPI: No turn landmark selected.")

                else:

                    turn_deg = TURN_DEG_BY_LANDMARK.get(
                        turn_landmark_id,
                        90.0
                    )

                    print(f"RPI: Sending TURN_REL {turn_deg:.1f}")

                    ser.reset_input_buffer()
                    send_command(f"TURN_REL {turn_deg:.1f}")

                    route_state = "TURNING"

            else:

                print("RPI: t ignored. Correct current landmark first.")

        elif key == ord('q'):

            break

        time.sleep(0.03)

except KeyboardInterrupt:
    pass

finally:

    stop_robot()

    picam2.stop()
    cv2.destroyAllWindows()
    ser.close()

    print("Robot stopped")
