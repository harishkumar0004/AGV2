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

seen_tag7 = False


# =====================================================
# DRIVE PARAMETERS
# =====================================================

MAX_PPS = 10000

# Live AprilTag correction speed while travelling
VISION_BASE_PPS = 4200
VISION_MIN_PPS = 2500
VISION_MAX_PPS = 6000

# Final tag speed
FINAL_BASE_PPS = 1200
FINAL_MIN_PPS = 700
FINAL_MAX_PPS = 2200


# =====================================================
# APRILTAG CORRECTION PARAMETERS
# =====================================================

EXPECTED_TAG_YAW_DEG = 0.0

KP_YAW_PPS_PER_DEG = 18
KP_X_PPS_PER_M = 16000

# If xM correction moves away from zero, change this to +1.0
X_SIGN = -1.0

YAW_DEADBAND_DEG = 0.30
X_DEADBAND_M = 0.002

MAX_VISION_CORRECTION_PPS = 260
MAX_FINAL_CORRECTION_PPS = 160

# Low-pass filter for smooth steering
CORRECTION_FILTER_ALPHA = 0.35

filtered_correction = 0.0


# =====================================================
# FINAL TAG STOP TOLERANCE
# =====================================================

TAG6_X_OK_M = 0.010
TAG6_YAW_OK_DEG = 3.0
TAG6_GOOD_FRAMES_REQUIRED = 3

tag6_good_count = 0


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
    while ser.in_waiting > 0:
        line = ser.readline().decode(errors="ignore").strip()

        if line:
            print(f"ESP32: {line}")


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


def tag6_pose_good(yaw_error, x_m):
    return (
        abs(yaw_error) <= TAG6_YAW_OK_DEG
        and abs(x_m) <= TAG6_X_OK_M
    )


def correction_velocity(tag, final=False):
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

    # If yaw and xM fight each other, reduce yaw effect.
    if abs(x_for_control) > X_DEADBAND_M:
        if yaw_corr * x_corr < 0:
            yaw_corr *= 0.25

    raw_correction = yaw_corr + x_corr

    if final:
        max_corr = MAX_FINAL_CORRECTION_PPS
        base_pps = FINAL_BASE_PPS
        min_pps = FINAL_MIN_PPS
        max_pps = FINAL_MAX_PPS
    else:
        max_corr = MAX_VISION_CORRECTION_PPS
        base_pps = VISION_BASE_PPS
        min_pps = VISION_MIN_PPS
        max_pps = VISION_MAX_PPS

    raw_correction = float(np.clip(
        raw_correction,
        -max_corr,
        max_corr
    ))

    filtered_correction = (
        (1.0 - CORRECTION_FILTER_ALPHA) * filtered_correction
        + CORRECTION_FILTER_ALPHA * raw_correction
    )

    correction = int(filtered_correction)

    left = base_pps - correction
    right = base_pps + correction

    left = int(np.clip(left, min_pps, max_pps))
    right = int(np.clip(right, min_pps, max_pps))

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


def choose_vision_tag(detections):
    global route_state
    global seen_tag7

    tag11 = find_tag(detections, START_TAG)
    tag7 = find_tag(detections, TAG_7)
    tag6 = find_tag(detections, TAG_6)

    if route_state == "MOVE_TO_7":

        if tag7 is not None:
            seen_tag7 = True
            route_state = "MOVE_TO_6"
            print("RPI: Tag 7 detected. Now target is Tag 6.")
            return tag7, "TAG7", False

        if tag11 is not None:
            return tag11, "TAG11", False

        return None, "", False

    if route_state == "MOVE_TO_6":

        if tag6 is not None:
            route_state = "FINAL_TAG6"
            print("RPI: Tag 6 detected. Final approach.")
            return tag6, "TAG6", True

        if tag7 is not None:
            return tag7, "TAG7", False

        return None, "", False

    if route_state == "FINAL_TAG6":

        if tag6 is not None:
            return tag6, "TAG6", True

        return None, "", True

    return None, "", False


# =====================================================
# STARTUP
# =====================================================

print("Waiting for docking Tag 11...")
print("Press 's' when Tag 11 is visible and robot is manually aligned.")
print("Flow:")
print("  s -> IMU RECAL -> LOCK_HEADING_GO")
print("  visible tag -> live AprilTag yaw + xM correction")
print("  no tag -> ESP32 IMU heading hold")
print("  Tag 6 -> slow final correction and stop")


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

        # =================================================
        # DRAW TAGS
        # =================================================

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

        # =================================================
        # STATE MACHINE
        # =================================================

        if route_state == "WAIT_START":

            pass

        elif route_state in ("MOVE_TO_7", "MOVE_TO_6", "FINAL_TAG6"):

            vision_tag, label, final = choose_vision_tag(detections)

            if vision_tag is not None:

                (
                    left,
                    right,
                    correction,
                    raw_yaw,
                    yaw_error,
                    x_m,
                    yaw_corr,
                    x_corr
                ) = correction_velocity(
                    vision_tag,
                    final=final
                )

                if final:

                    if tag6_pose_good(yaw_error, x_m):

                        tag6_good_count += 1

                        send_velocity(
                            0,
                            0,
                            force=True
                        )

                        print(
                            f"TAG6 GOOD "
                            f"{tag6_good_count}/{TAG6_GOOD_FRAMES_REQUIRED} "
                            f"rawYaw={raw_yaw:.2f} "
                            f"yawErr={yaw_error:.2f} "
                            f"xM={x_m:.4f}"
                        )

                        if tag6_good_count >= TAG6_GOOD_FRAMES_REQUIRED:

                            print("RPI: Tag 6 aligned. Route complete.")
                            stop_robot()
                            route_state = "DONE"

                    else:

                        tag6_good_count = 0

                        send_velocity(
                            left,
                            right
                        )

                        print(
                            f"{label} FINAL_CORR "
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

                    tag6_good_count = 0

                    send_velocity(
                        left,
                        right
                    )

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

                tag6_good_count = 0

                if route_state == "FINAL_TAG6":
                    print("RPI: Tag 6 lost during final approach. Stop.")
                    stop_robot()
                    route_state = "DONE"
                else:
                    lock_heading_go()

        elif route_state == "DONE":

            pass

        # =================================================
        # DISPLAY
        # =================================================

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
            "AGV Live AprilTag + IMU Route 11-7-6",
            frame
        )

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
