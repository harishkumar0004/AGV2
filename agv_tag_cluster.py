#!/usr/bin/env python3

import math
import time

import cv2
import numpy as np
import serial

from picamera2 import Picamera2
from pupil_apriltags import Detector


# =====================================================
# ROUTE LANDMARKS
# =====================================================

START_TAG = 11
TAG_7 = 7
TAG_6 = 6
TAG_5 = 5
TAG_9 = 9

route_state = "WAIT_START"

turn_landmark_id = None
next_move_state_after_turn = None


# =====================================================
# TRAVEL SEGMENT TRACKING
# =====================================================

travel_from_landmark = None
travel_to_landmark = None

left_start_cluster = False
start_cluster_lost_count = 0
target_cluster_seen_count = 0
target_helper_reached_count = 0

travel_segment_start_time = 0.0

START_CLUSTER_LOST_FRAMES_REQUIRED = 4
TARGET_CLUSTER_SEEN_FRAMES_REQUIRED = 2

# Important:
# helper 501/503/505/507/502/504/506/508 visible alone is NOT enough.
# The estimated central landmark must be close.
TARGET_CENTER_Y_REACHED_PX = 35
TARGET_CENTER_X_REACHED_M = 0.018
TARGET_HELPER_REACHED_FRAMES_REQUIRED = 3

MIN_TRAVEL_TIME_BEFORE_TARGET_SEC = 1.0


# =====================================================
# TURN SETTINGS
# =====================================================

TURN_DEG_BY_LANDMARK = {
    TAG_6: 90.0,
    TAG_5: -90.0,
}


# =====================================================
# DRIVE PARAMETERS
# =====================================================

MAX_PPS = 10000

VISION_BASE_PPS = 3500
VISION_BASE_PPS_SLOW = 2600

VISION_MIN_PPS = 1600
VISION_MAX_PPS = 5200

TURN_TAG_FB_PPS = 550

# If pressing c moves front/back wrong way, change this to -1
FB_SIGN = 1


# =====================================================
# APRILTAG / CLUSTER PARAMETERS
# =====================================================

TAG_SIZE_M = 0.010

# tag size = 10 mm, gap = 5 mm, center-to-center = 15 mm
CLUSTER_SPACING_M = 0.015

EXPECTED_TAG_YAW_DEG = 0.0

# Helper tag grid:
#
#   508   501   502
#   507   CEN   503
#   506   505   504
#
# x_grid: right positive
# y_grid: image-down positive
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

CORRECTION_HELPER_PRIORITY = [501, 503, 505, 507]


# =====================================================
# GLOBAL ADAPTIVE CORRECTION PARAMETERS
# =====================================================

KP_YAW_PPS_PER_DEG = 18
KP_X_PPS_PER_M = 16000

KP_YAW_STRONG_PPS_PER_DEG = 28
KP_X_STRONG_PPS_PER_M = 45000

# If xM correction moves away from zero during travel, change to +1.0
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
# TURN LANDMARK CORRECTION PARAMETERS
# =====================================================

KP_TURN_TAG_YAW_PPS_PER_DEG = 20
MAX_TURN_TAG_YAW_CORRECTION_PPS = 150

TURN_TAG_YAW_OK_DEG = 3.0
TURN_TAG_CENTER_Y_OK_PX = 25
TURN_TAG_GOOD_FRAMES_REQUIRED = 3

turn_tag_good_count = 0

POST_TURN_VERIFY_TIMEOUT_SEC = 6.0
post_turn_start_time = 0.0


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
    global last_drive_mode
    global drive_left_pps
    global drive_right_pps

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
# BASIC APRILTAG MATH
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


def get_helper_grid_offset(tag_id, central_tag_id):
    if tag_id == central_tag_id:
        return 0, 0

    if tag_id in HELPER_GRID_OFFSET:
        return HELPER_GRID_OFFSET[tag_id]

    return None


# =====================================================
# CLUSTER LANDMARK POSE
# =====================================================

def is_cluster_tag(tag_id, central_tag_id):
    if tag_id == central_tag_id:
        return True

    return tag_id in HELPER_GRID_OFFSET


def choose_best_cluster_tag(detections, central_tag_id):
    """
    Used for travelling.
    Prefer central landmark tag if visible.
    Otherwise choose helper closest to image center.
    """

    candidate_tags = []

    for tag in detections:
        if is_cluster_tag(tag.tag_id, central_tag_id):
            candidate_tags.append(tag)

    if not candidate_tags:
        return None

    for tag in candidate_tags:
        if tag.tag_id == central_tag_id:
            return tag

    best_tag = None
    best_dist = 999999999.0

    for tag in candidate_tags:
        dx = tag.center[0] - CX
        dy = tag.center[1] - CY
        dist = dx * dx + dy * dy

        if dist < best_dist:
            best_dist = dist
            best_tag = tag

    return best_tag


def choose_best_correction_tag(detections, central_tag_id):
    """
    Used after pressing c.
    Priority:
    1. central landmark tag
    2. cross helpers 501, 503, 505, 507
    3. any helper closest to image center
    """

    for tag in detections:
        if tag.tag_id == central_tag_id:
            return tag

    best_tag = None
    best_dist = 999999999.0

    for tag in detections:
        if tag.tag_id in CORRECTION_HELPER_PRIORITY:
            dx = tag.center[0] - CX
            dy = tag.center[1] - CY
            dist = dx * dx + dy * dy

            if dist < best_dist:
                best_dist = dist
                best_tag = tag

    if best_tag is not None:
        return best_tag

    return choose_best_cluster_tag(detections, central_tag_id)


def central_tag_visible(detections, central_tag_id):
    for tag in detections:
        if tag.tag_id == central_tag_id:
            return True

    return False


def any_cluster_visible(detections, central_tag_id):
    return choose_best_cluster_tag(detections, central_tag_id) is not None


def get_landmark_pose_from_cluster_tag(tag, central_tag_id):
    """
    Converts visible central/helper tag pose into estimated central landmark pose.
    Used during travelling.
    """

    grid_offset = get_helper_grid_offset(
        tag.tag_id,
        central_tag_id
    )

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

    center_px = tag.center[0] - (helper_x_grid * spacing_px)
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


def get_visible_tag_center_pose(tag):
    """
    Used only after pressing c.
    Centers the selected visible tag itself.
    """

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
# TRAVEL SEGMENT MONITOR
# =====================================================

def start_travel_segment(from_landmark, to_landmark, new_state):
    global route_state
    global travel_from_landmark
    global travel_to_landmark
    global left_start_cluster
    global start_cluster_lost_count
    global target_cluster_seen_count
    global target_helper_reached_count
    global travel_segment_start_time

    travel_from_landmark = from_landmark
    travel_to_landmark = to_landmark

    left_start_cluster = False
    start_cluster_lost_count = 0
    target_cluster_seen_count = 0
    target_helper_reached_count = 0
    travel_segment_start_time = time.time()

    reset_correction_filter()

    route_state = new_state

    print(
        f"RPI: Starting travel segment "
        f"{from_landmark} -> {to_landmark}"
    )


def monitor_travel_progress(detections):
    """
    Returns:
        USE_START   = still near start cluster, correct using start landmark
        USE_TARGET  = target cluster visible, but not yet centered/reached
        ARRIVED     = target landmark confirmed
        NO_TAG      = no useful tag

    Helper visible alone is not enough for arrival.
    The estimated central target landmark must be close.
    """

    global left_start_cluster
    global start_cluster_lost_count
    global target_cluster_seen_count
    global target_helper_reached_count

    if travel_from_landmark is None or travel_to_landmark is None:
        return "NO_TAG"

    elapsed = time.time() - travel_segment_start_time

    # -------------------------------------------------
    # Central target tag is reliable.
    # -------------------------------------------------

    if elapsed >= MIN_TRAVEL_TIME_BEFORE_TARGET_SEC:
        if central_tag_visible(detections, travel_to_landmark):
            target_cluster_seen_count += 1

            if target_cluster_seen_count >= TARGET_CLUSTER_SEEN_FRAMES_REQUIRED:
                return "ARRIVED"

            return "USE_TARGET"

    # -------------------------------------------------
    # Before leaving start cluster:
    # helpers belong to start landmark.
    # -------------------------------------------------

    if not left_start_cluster:

        if central_tag_visible(detections, travel_from_landmark):
            start_cluster_lost_count = 0
            target_cluster_seen_count = 0
            target_helper_reached_count = 0
            return "USE_START"

        start_cluster_lost_count += 1

        if start_cluster_lost_count < START_CLUSTER_LOST_FRAMES_REQUIRED:
            if any_cluster_visible(detections, travel_from_landmark):
                return "USE_START"

        if (
            start_cluster_lost_count >= START_CLUSTER_LOST_FRAMES_REQUIRED
            and elapsed >= MIN_TRAVEL_TIME_BEFORE_TARGET_SEC
        ):
            left_start_cluster = True
            target_cluster_seen_count = 0
            target_helper_reached_count = 0

            print(
                f"RPI: Left landmark {travel_from_landmark} central tag. "
                f"Now searching for landmark {travel_to_landmark} cluster."
            )

        return "NO_TAG"

    # -------------------------------------------------
    # After leaving start:
    # helpers are interpreted as target helpers.
    # But do not stop just because 501/503/505/507 is visible.
    # -------------------------------------------------

    target_tag = choose_best_cluster_tag(
        detections,
        travel_to_landmark
    )

    if target_tag is None:
        target_cluster_seen_count = 0
        target_helper_reached_count = 0
        return "NO_TAG"

    if target_tag.tag_id == travel_to_landmark:
        target_cluster_seen_count += 1

        if target_cluster_seen_count >= TARGET_CLUSTER_SEEN_FRAMES_REQUIRED:
            return "ARRIVED"

        return "USE_TARGET"

    pose = get_landmark_pose_from_cluster_tag(
        target_tag,
        travel_to_landmark
    )

    if pose is None:
        return "USE_TARGET"

    (
        raw_yaw,
        yaw_error,
        center_x_m,
        center_y_error_px,
        seen_tag_id,
        helper_x_grid,
        helper_y_grid
    ) = pose

    helper_center_good = (
        abs(center_y_error_px) <= TARGET_CENTER_Y_REACHED_PX
        and abs(center_x_m) <= TARGET_CENTER_X_REACHED_M
    )

    if helper_center_good:
        target_helper_reached_count += 1
    else:
        target_helper_reached_count = 0

    print(
        f"TARGET_HELPER_CHECK "
        f"target={travel_to_landmark} "
        f"seen={seen_tag_id} "
        f"grid=({helper_x_grid},{helper_y_grid}) "
        f"centerXM={center_x_m:.4f} "
        f"centerYerr={center_y_error_px:.1f}px "
        f"good={target_helper_reached_count}/{TARGET_HELPER_REACHED_FRAMES_REQUIRED}"
    )

    if target_helper_reached_count >= TARGET_HELPER_REACHED_FRAMES_REQUIRED:
        return "ARRIVED"

    return "USE_TARGET"


# =====================================================
# ADAPTIVE TRAVELLING CORRECTION
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


def travelling_correction_velocity_from_landmark(
    raw_yaw,
    yaw_error,
    center_x_m
):
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
# TURN LANDMARK CORRECTION
# =====================================================

def turn_tag_heading_center_velocity_from_visible_tag(
    raw_yaw,
    yaw_error,
    visible_y_error_px
):
    yaw_for_control = yaw_error

    if abs(yaw_for_control) < YAW_DEADBAND_DEG:
        yaw_for_control = 0.0

    yaw_corr = -KP_TURN_TAG_YAW_PPS_PER_DEG * yaw_for_control

    yaw_corr = int(np.clip(
        yaw_corr,
        -MAX_TURN_TAG_YAW_CORRECTION_PPS,
        MAX_TURN_TAG_YAW_CORRECTION_PPS
    ))

    if abs(visible_y_error_px) <= TURN_TAG_CENTER_Y_OK_PX:
        fb = 0
    else:
        if visible_y_error_px > 0:
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


def turn_tag_good(yaw_error, visible_y_error_px):
    return (
        abs(yaw_error) <= TURN_TAG_YAW_OK_DEG
        and abs(visible_y_error_px) <= TURN_TAG_CENTER_Y_OK_PX
    )


# =====================================================
# ROUTE HELPERS
# =====================================================

def start_turn_wait(landmark_id, next_state_after_turn):
    global route_state
    global turn_landmark_id
    global next_move_state_after_turn
    global turn_tag_good_count

    stop_robot()
    reset_correction_filter()

    turn_landmark_id = landmark_id
    next_move_state_after_turn = next_state_after_turn
    turn_tag_good_count = 0

    route_state = "WAIT_TURN_TAG_CORRECT_COMMAND"

    print(f"RPI: Landmark {landmark_id} cluster reached. Robot stopped.")
    print(f"RPI: Press 'c' to correct Tag {landmark_id} cluster.")
    print(f"RPI: Press 't' after correction to turn {TURN_DEG_BY_LANDMARK[landmark_id]:.1f} degrees.")


def choose_travel_landmark(detections):
    global route_state

    progress = monitor_travel_progress(detections)

    if route_state == "MOVE_TO_7":

        if progress == "ARRIVED":
            print("RPI: Landmark 7 cluster reached.")
            start_travel_segment(TAG_7, TAG_6, "MOVE_TO_6")
            return choose_best_cluster_tag(detections, TAG_7), TAG_7, "TAG7"

        if progress == "USE_START":
            return choose_best_cluster_tag(detections, START_TAG), START_TAG, "TAG11"

        if progress == "USE_TARGET":
            return choose_best_cluster_tag(detections, TAG_7), TAG_7, "TAG7"

        return None, None, ""

    if route_state == "MOVE_TO_6":

        if progress == "ARRIVED":
            start_turn_wait(TAG_6, "MOVE_TO_5")
            return None, None, ""

        if progress == "USE_START":
            return choose_best_cluster_tag(detections, TAG_7), TAG_7, "TAG7"

        if progress == "USE_TARGET":
            return choose_best_cluster_tag(detections, TAG_6), TAG_6, "TAG6"

        return None, None, ""

    if route_state == "MOVE_TO_5":

        if progress == "ARRIVED":
            start_turn_wait(TAG_5, "MOVE_TO_9")
            return None, None, ""

        if progress == "USE_START":
            return choose_best_cluster_tag(detections, TAG_6), TAG_6, "TAG6"

        if progress == "USE_TARGET":
            return choose_best_cluster_tag(detections, TAG_5), TAG_5, "TAG5"

        return None, None, ""

    if route_state == "MOVE_TO_9":

        if progress == "ARRIVED":
            stop_robot()
            route_state = "DONE"

            print("RPI: Landmark 9 cluster reached.")
            print("RPI: Final destination reached. Robot stopped.")

            return None, None, ""

        if progress == "USE_START":
            return choose_best_cluster_tag(detections, TAG_5), TAG_5, "TAG5"

        if progress == "USE_TARGET":
            return choose_best_cluster_tag(detections, TAG_9), TAG_9, "TAG9"

        return None, None, ""

    return None, None, ""


# =====================================================
# DISPLAY DRAWING
# =====================================================

def draw_tags(frame, detections):
    visible_ids = []

    for tag in detections:

        visible_ids.append(tag.tag_id)

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

    return visible_ids


# =====================================================
# STARTUP
# =====================================================

print("Waiting for docking Tag 11...")
print("Press 's' when Tag 11 is visible and robot is manually aligned.")
print("")
print("Route:")
print("  11 -> 7 -> 6")
print("  At 6: stop, press c, then press t")
print("  6 -> 5")
print("  At 5: stop, press c, then press t")
print("  5 -> 9 and stop")
print("")
print("Helper arrival rule:")
print("  central target tag visible -> reached")
print("  helper visible only -> estimate central target position")
print("  stop only when estimated central target is near image center")
print("")
print("Keys:")
print("  s = start route")
print("  c = correct current turn landmark")
print("  t = turn current turn landmark")
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

        visible_ids = draw_tags(frame, detections)

        # =================================================
        # STATE MACHINE
        # =================================================

        if route_state == "WAIT_START":

            pass

        elif route_state in ("MOVE_TO_7", "MOVE_TO_6", "MOVE_TO_5", "MOVE_TO_9"):

            tag, landmark_id, label = choose_travel_landmark(detections)

            if tag is not None:

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
                    ) = travelling_correction_velocity_from_landmark(
                        raw_yaw,
                        yaw_error,
                        center_x_m
                    )

                    send_velocity(left, right)

                    print(
                        f"{label} CLUSTER_CORR "
                        f"travel={travel_from_landmark}->{travel_to_landmark} "
                        f"leftStart={left_start_cluster} "
                        f"seen={seen_tag_id} "
                        f"grid=({helper_x_grid},{helper_y_grid}) "
                        f"rawYaw={raw_yaw:.2f} "
                        f"yawErr={yaw_error:.2f} "
                        f"centerXM={center_x_m:.4f} "
                        f"centerYerr={center_y_error_px:.1f}px "
                        f"yawCorr={yaw_corr:.1f} "
                        f"xCorr={x_corr:.1f} "
                        f"level={error_level} "
                        f"corr={correction} "
                        f"L={left} "
                        f"R={right}"
                    )

            else:

                lock_heading_go()

        elif route_state == "WAIT_TURN_TAG_CORRECT_COMMAND":

            stop_robot()

        elif route_state == "TURN_TAG_CORRECTING":

            if turn_landmark_id is None:

                stop_robot()
                route_state = "DONE"

            else:

                correction_tag = choose_best_correction_tag(
                    detections,
                    turn_landmark_id
                )

                if correction_tag is None:

                    stop_robot()
                    route_state = "WAIT_TURN_TAG_CORRECT_COMMAND"

                    print(f"RPI: Tag {turn_landmark_id} correction tag lost.")
                    print("RPI: Robot stopped. Press 'c' again when tag is visible.")

                else:

                    (
                        raw_yaw,
                        yaw_error,
                        visible_x_error_px,
                        visible_y_error_px,
                        seen_tag_id
                    ) = get_visible_tag_center_pose(correction_tag)

                    if turn_tag_good(yaw_error, visible_y_error_px):

                        turn_tag_good_count += 1

                        stop_robot()

                        print(
                            f"TAG{turn_landmark_id} GOOD "
                            f"{turn_tag_good_count}/{TURN_TAG_GOOD_FRAMES_REQUIRED} "
                            f"seen={seen_tag_id} "
                            f"rawYaw={raw_yaw:.2f} "
                            f"yawErr={yaw_error:.2f} "
                            f"visibleXerr={visible_x_error_px:.1f}px "
                            f"visibleYerr={visible_y_error_px:.1f}px"
                        )

                        if turn_tag_good_count >= TURN_TAG_GOOD_FRAMES_REQUIRED:

                            stop_robot()
                            route_state = "WAIT_TURN_COMMAND"

                            print(f"RPI: Tag {turn_landmark_id} correction complete.")
                            print("RPI: Press 't' to turn.")

                    else:

                        turn_tag_good_count = 0

                        left, right, yaw_corr = turn_tag_heading_center_velocity_from_visible_tag(
                            raw_yaw,
                            yaw_error,
                            visible_y_error_px
                        )

                        send_velocity(left, right)

                        print(
                            f"TAG{turn_landmark_id} ALIGN_VISIBLE "
                            f"seen={seen_tag_id} "
                            f"rawYaw={raw_yaw:.2f} "
                            f"yawErr={yaw_error:.2f} "
                            f"visibleXerr={visible_x_error_px:.1f}px "
                            f"visibleYerr={visible_y_error_px:.1f}px "
                            f"yawCorr={yaw_corr} "
                            f"L={left} "
                            f"R={right}"
                        )

        elif route_state == "WAIT_TURN_COMMAND":

            stop_robot()

        elif route_state == "TURNING":

            lines = read_esp32_lines()

            for line in lines:
                if "OK TURN_DONE" in line:

                    print(f"RPI: Turn at Tag {turn_landmark_id} complete.")
                    print("RPI: Verifying current cluster if visible.")

                    post_turn_start_time = time.time()
                    route_state = "POST_TURN_VERIFY"

        elif route_state == "POST_TURN_VERIFY":

            tag_cluster = choose_best_cluster_tag(
                detections,
                turn_landmark_id
            )

            if tag_cluster is not None:

                pose = get_landmark_pose_from_cluster_tag(
                    tag_cluster,
                    turn_landmark_id
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

                    print(
                        f"POST_TURN VERIFY "
                        f"landmark={turn_landmark_id} "
                        f"seen={seen_tag_id} "
                        f"grid=({helper_x_grid},{helper_y_grid}) "
                        f"rawYaw={raw_yaw:.2f} "
                        f"yawErr={yaw_error:.2f} "
                        f"centerXM={center_x_m:.4f} "
                        f"centerYerr={center_y_error_px:.1f}px"
                    )

                    if turn_landmark_id == TAG_6:
                        start_travel_segment(
                            TAG_6,
                            TAG_5,
                            "MOVE_TO_5"
                        )

                    elif turn_landmark_id == TAG_5:
                        start_travel_segment(
                            TAG_5,
                            TAG_9,
                            "MOVE_TO_9"
                        )

                    else:
                        route_state = next_move_state_after_turn
                        reset_correction_filter()

                    lock_heading_go()

            elif time.time() - post_turn_start_time > POST_TURN_VERIFY_TIMEOUT_SEC:

                print("RPI: Post-turn verify timeout. Continuing with IMU heading.")

                if turn_landmark_id == TAG_6:
                    start_travel_segment(
                        TAG_6,
                        TAG_5,
                        "MOVE_TO_5"
                    )

                elif turn_landmark_id == TAG_5:
                    start_travel_segment(
                        TAG_5,
                        TAG_9,
                        "MOVE_TO_9"
                    )

                else:
                    route_state = next_move_state_after_turn
                    reset_correction_filter()

                lock_heading_go()

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
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Travel: {travel_from_landmark}->{travel_to_landmark} left:{left_start_cluster}",
            (40, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"TurnTag: {turn_landmark_id}",
            (40, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Drive: {last_drive_mode} L:{drive_left_pps} R:{drive_right_pps}",
            (40, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Visible: {visible_ids}",
            (40, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Lost:{start_cluster_lost_count} Seen:{target_cluster_seen_count} HelperGood:{target_helper_reached_count}",
            (40, 250),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "AGV Cluster Route With Helper Center Arrival",
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
                    print("RPI: Requesting ESP32 IMU recalibration.")
                    print("RPI: Keep AGV completely still.")

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
                        print("RPI: Starting travel 11 -> 7.")

                        start_travel_segment(
                            START_TAG,
                            TAG_7,
                            "MOVE_TO_7"
                        )

                        lock_heading_go()

                    else:

                        print("RPI: Start failed. ESP32 did not confirm IMU RECAL.")

                else:

                    print("RPI: Start ignored. Landmark 11 cluster not visible.")

        elif key == ord('c'):

            if route_state == "WAIT_TURN_TAG_CORRECT_COMMAND":

                if turn_landmark_id is None:

                    print("RPI: No turn landmark selected.")

                else:

                    correction_tag = choose_best_correction_tag(
                        detections,
                        turn_landmark_id
                    )

                    if correction_tag is None:

                        print(f"RPI: Cannot correct. Tag {turn_landmark_id} cluster is not visible.")

                    else:

                        turn_tag_good_count = 0

                        print(f"RPI: Starting Tag {turn_landmark_id} visible-tag correction.")
                        print("RPI: Priority: central tag, then 501/503/505/507, then any helper.")
                        print("RPI: Using yaw + selected visible tag image Y only. No xM steering.")

                        route_state = "TURN_TAG_CORRECTING"

            else:

                print("RPI: 'c' ignored. Robot is not waiting for correction.")

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

                print("RPI: Turn ignored. Correct current turn tag first, then press 't'.")

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
