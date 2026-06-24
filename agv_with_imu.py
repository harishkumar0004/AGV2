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

route_state = "WAIT_START"


# =====================================================
# DRIVE PARAMETERS
# =====================================================

MAX_PPS = 10000

# Live AprilTag correction speed while travelling
VISION_BASE_PPS = 4200
VISION_MIN_PPS = 2500
VISION_MAX_PPS = 6000

# Final tag 6 alignment speed
FINAL_BASE_PPS = 1000
FINAL_MIN_PPS = 600
FINAL_MAX_PPS = 1800

# Front/back centering speed at tag 6
CENTER_FB_PPS = 850

# If front/back moves wrong way, change this to -1
FB_SIGN = 1


# =====================================================
# APRILTAG CORRECTION PARAMETERS
# =====================================================

EXPECTED_TAG_YAW_DEG = 0.0

# Before tag 6, use yaw + xM
KP_YAW_PPS_PER_DEG = 18
KP_X_PPS_PER_M = 16000

# At tag 6, use yaw only
KP_TAG6_YAW_PPS_PER_DEG = 20

# If xM correction moves away from zero before tag 6, change this to +1.0
X_SIGN = -1.0

YAW_DEADBAND_DEG = 0.30
X_DEADBAND_M = 0.002

MAX_VISION_CORRECTION_PPS = 260
MAX_TAG6_YAW_CORRECTION_PPS = 180

# Smooth steering
CORRECTION_FILTER_ALPHA = 0.35
filtered_correction = 0.0


# =====================================================
# TAG 6 FINAL CONDITIONS
# =====================================================

TAG6_YAW_OK_DEG = 3.0

# Only use image center Y for front/back centering.
# Do not use xM.
TAG6_CENTER_Y_OK_PX = 25

TAG6_GOOD_FRAMES_REQUIRED = 3
tag6_good_count = 0

TURN_DEG = 90.0


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
TAG_SIZE_M = 0.020


# =====================================================
# SERIAL STATE
# =====================================================

drive_left_pps = 0
drive_right_pps = 0

last_drive_mode = "STOP"
last_vel_send_time = 0.0
VEL_SEND_INTERVAL_SEC = 0.08


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
# APRILTAG MATH
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


def get_tag_pose(tag):
    raw_yaw = compute_yaw_deg(tag)
    yaw_error = normalize_angle(raw_yaw - EXPECTED_TAG_YAW_DEG)
    x_m = compute_lateral_x_m(tag)

    return raw_yaw, yaw_error, x_m


def tag_y_error_px(tag):
    return float(tag.center[1] - (FRAME_HEIGHT / 2.0))


def tag6_good(yaw_error, y_error_px):
    return (
        abs(yaw_error) <= TAG6_YAW_OK_DEG
        and abs(y_error_px) <= TAG6_CENTER_Y_OK_PX
    )


# =====================================================
# CORRECTION BEFORE TAG 6: YAW + XM
# =====================================================

def travelling_correction_velocity(tag):
    global filtered_correction

    raw_yaw, yaw_error, x_m = get_tag_pose(tag)

    yaw_for_control = yaw_error
    x_for_control = x_m

    if abs(yaw_for_control) < YAW_DEADBAND_DEG:
        yaw_for_control = 0.0

    if abs(x_for_control) < X_DEADBAND_M:
        x_for_control = 0.0

    yaw_corr = KP_YAW_PPS_PER_DEG * yaw_for_control
    x_corr = KP_X_PPS_PER_M * x_for_control * X_SIGN

    if abs(x_for_control) > X_DEADBAND_M:
        if yaw_corr * x_corr < 0:
            yaw_corr *= 0.25

    raw_correction = yaw_corr + x_corr

    raw_correction = float(np.clip(
        raw_correction,
        -MAX_VISION_CORRECTION_PPS,
        MAX_VISION_CORRECTION_PPS
    ))

    filtered_correction = (
        (1.0 - CORRECTION_FILTER_ALPHA) * filtered_correction
        + CORRECTION_FILTER_ALPHA * raw_correction
    )

    correction = int(filtered_correction)

    left = VISION_BASE_PPS - correction
    right = VISION_BASE_PPS + correction

    left = int(np.clip(left, VISION_MIN_PPS, VISION_MAX_PPS))
    right = int(np.clip(right, VISION_MIN_PPS, VISION_MAX_PPS))

    return (
        left,
        right,
        correction,
        raw_yaw,
        yaw_error,
        x_m,
        yaw_corr,
        x_corr
    )


# =====================================================
# TAG 6: YAW ONLY + IMAGE CENTER Y
# =====================================================

def tag6_heading_center_velocity(tag):
    raw_yaw, yaw_error, x_m = get_tag_pose(tag)
    y_err = tag_y_error_px(tag)

    yaw_for_control = yaw_error

    if abs(yaw_for_control) < YAW_DEADBAND_DEG:
        yaw_for_control = 0.0

    yaw_corr = -KP_TAG6_YAW_PPS_PER_DEG * yaw_for_control

    yaw_corr = int(np.clip(
        yaw_corr,
        -MAX_TAG6_YAW_CORRECTION_PPS,
        MAX_TAG6_YAW_CORRECTION_PPS
    ))

    # Front/back movement to center tag vertically in image.
    # No xM correction is used here.
    if abs(y_err) <= TAG6_CENTER_Y_OK_PX:
        fb = 0
    else:
        if y_err > 0:
            fb = CENTER_FB_PPS * FB_SIGN
        else:
            fb = -CENTER_FB_PPS * FB_SIGN

    # Heading correction is differential.
    # Front/back centering is common speed.
    left = fb - yaw_corr
    right = fb + yaw_corr

    # If front/back is centered, still allow pure yaw correction.
    if fb == 0:
        left = -yaw_corr
        right = yaw_corr

    left = int(np.clip(left, -FINAL_MAX_PPS, FINAL_MAX_PPS))
    right = int(np.clip(right, -FINAL_MAX_PPS, FINAL_MAX_PPS))

    return (
        left,
        right,
        yaw_corr,
        raw_yaw,
        yaw_error,
        x_m,
        y_err
    )


def reset_correction_filter():
    global filtered_correction
    filtered_correction = 0.0


# =====================================================
# TAG SELECTION
# =====================================================

def find_tag(detections, tag_id):
    for tag in detections:
        if tag.tag_id == tag_id:
            return tag

    return None


def choose_travel_tag(detections):
    global route_state

    tag11 = find_tag(detections, START_TAG)
    tag7 = find_tag(detections, TAG_7)
    tag6 = find_tag(detections, TAG_6)

    if route_state == "MOVE_TO_7":

        if tag7 is not None:
            route_state = "MOVE_TO_6"
            reset_correction_filter()
            print("RPI: Tag 7 detected. Now moving toward Tag 6.")
            return tag7, "TAG7"

        if tag11 is not None:
            return tag11, "TAG11"

        return None, ""

    if route_state == "MOVE_TO_6":

        if tag6 is not None:
            route_state = "TAG6_ALIGN"
            reset_correction_filter()
            print("RPI: Tag 6 detected. Align heading and image center before turn.")
            return None, ""

        if tag7 is not None:
            return tag7, "TAG7"

        return None, ""

    return None, ""


# =====================================================
# STARTUP
# =====================================================

print("Waiting for docking Tag 11...")
print("Press 's' when Tag 11 is visible and robot is manually aligned.")
print("Route:")
print("  11->7 uses live AprilTag + IMU")
print("  7->6 uses live AprilTag + IMU")
print("  At Tag 6: heading only + image center Y, then TURN_REL 90")


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

            raw_yaw, yaw_error, x_m = get_tag_pose(tag)
            y_err = tag_y_error_px(tag)

            cv2.putText(
                frame,
                f"ID:{tag.tag_id}",
                (center[0] + 10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"YawErr:{yaw_error:.1f}",
                (center[0] + 10, center[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"xM:{x_m:.3f}",
                (center[0] + 10, center[1] + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2
            )

            cv2.putText(
                frame,
                f"yErr:{y_err:.0f}px",
                (center[0] + 10, center[1] + 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 180, 255),
                2
            )

        # =================================================
        # STATE MACHINE
        # =================================================

        if route_state == "WAIT_START":

            pass

        elif route_state in ("MOVE_TO_7", "MOVE_TO_6"):

            tag, label = choose_travel_tag(detections)

            if tag is not None:

                (
                    left,
                    right,
                    correction,
                    raw_yaw,
                    yaw_error,
                    x_m,
                    yaw_corr,
                    x_corr
                ) = travelling_correction_velocity(tag)

                send_velocity(left, right)

                print(
                    f"{label} LIVE_CORR "
                    f"rawYaw={raw_yaw:.2f} "
                    f"yawErr={yaw_error:.2f} "
                    f"xM={x_m:.4f} "
                    f"yawCorr={yaw_corr:.1f} "
                    f"xCorr={x_corr:.1f} "
                    f"corr={correction} "
                    f"L={left} "
                    f"R={right}"
                )

            else:

                lock_heading_go()

        elif route_state == "TAG6_ALIGN":

            tag6 = find_tag(detections, TAG_6)

            if tag6 is None:

                print("RPI: Tag 6 lost during final alignment. Stop.")
                stop_robot()
                route_state = "DONE"

            else:

                (
                    left,
                    right,
                    yaw_corr,
                    raw_yaw,
                    yaw_error,
                    x_m,
                    y_err
                ) = tag6_heading_center_velocity(tag6)

                if tag6_good(yaw_error, y_err):

                    tag6_good_count += 1

                    stop_robot()

                    print(
                        f"TAG6 GOOD "
                        f"{tag6_good_count}/{TAG6_GOOD_FRAMES_REQUIRED} "
                        f"rawYaw={raw_yaw:.2f} "
                        f"yawErr={yaw_error:.2f} "
                        f"xM={x_m:.4f} "
                        f"yErr={y_err:.1f}"
                    )

                    if tag6_good_count >= TAG6_GOOD_FRAMES_REQUIRED:

                        print("RPI: Tag 6 heading and image center OK.")
                        print(f"RPI: Sending TURN_REL {TURN_DEG:.1f}")

                        send_command(f"TURN_REL {TURN_DEG:.1f}")

                        route_state = "TAG6_TURN"

                else:

                    tag6_good_count = 0

                    send_velocity(left, right)

                    print(
                        f"TAG6 ALIGN "
                        f"rawYaw={raw_yaw:.2f} "
                        f"yawErr={yaw_error:.2f} "
                        f"xM={x_m:.4f} "
                        f"yErr={y_err:.1f} "
                        f"yawCorr={yaw_corr} "
                        f"L={left} "
                        f"R={right}"
                    )

        elif route_state == "TAG6_TURN":

            lines = read_esp32_lines()

            for line in lines:
                if "OK TURN_DONE" in line:
                    print("RPI: 90 degree turn complete.")
                    route_state = "DONE"

        elif route_state == "DONE":

            pass

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
            f"Drive: {last_drive_mode} L:{drive_left_pps} R:{drive_right_pps}",
            (40, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Visible: {visible_ids}",
            (40, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Tag6 good: {tag6_good_count}",
            (40, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "AGV Tag6 Heading Center Turn",
            frame
        )

        if route_state != "TAG6_TURN":
            read_esp32_lines()

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):

            if route_state == "WAIT_START":

                tag11 = find_tag(detections, START_TAG)

                if tag11 is not None:

                    print("RPI: Docking Tag 11 visible.")
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
                        print("RPI: Locking heading and starting movement.")

                        reset_correction_filter()

                        route_state = "MOVE_TO_7"

                        lock_heading_go()

                    else:

                        print("RPI: Start failed. ESP32 did not confirm IMU RECAL.")

                else:

                    print("RPI: Start ignored. Tag 11 not visible.")

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
