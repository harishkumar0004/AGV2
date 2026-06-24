#!/usr/bin/env python3

import math
import time

import cv2
import numpy as np
import serial

from picamera2 import Picamera2
from pupil_apriltags import Detector


# =====================================================
# DRIVE PARAMETERS
# =====================================================

BASE_PPS = 4000
MAX_PPS = 10000

# Normal IMU movement speed is controlled by ESP32 BASE_PPS.
# These values are only used for AprilTag correction from Raspberry Pi.
CORR_BASE_PPS = 2200
CORR_MIN_PPS = 1200
CORR_MAX_PPS = 4000


# =====================================================
# APRILTAG CORRECTION PARAMETERS
# =====================================================

EXPECTED_TAG_YAW_DEG = 0.0

KP_YAW_PPS_PER_DEG = 18
KP_X_PPS_PER_M = 16000

X_SIGN = -1.0

YAW_DEADBAND_DEG = 0.30
X_DEADBAND_M = 0.002

MAX_CORRECTION_PPS = 220

TAG_X_OK_M = 0.008
TAG_YAW_OK_DEG = 3.0

TAG_CORRECTION_TIMEOUT_SEC = 4.0
TAG_GOOD_FRAMES_REQUIRED = 5


# =====================================================
# DOCKING ALIGNMENT PARAMETERS
# =====================================================

DOCK_YAW_OK_DEG = 1.0
DOCK_TURN_PPS = 800
DOCK_ALIGN_TIMEOUT_SEC = 4.0


# =====================================================
# ROUTE PARAMETERS
# =====================================================

START_TAG = 11
TAG_SEQUENCE = [11, 7, 6]

current_index = 0
current_target = TAG_SEQUENCE[current_index]

route_state = "WAIT_START"

tag_correction_start_time = 0.0
tag_good_count = 0
dock_align_start_time = 0.0


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
# MOTION STATE
# =====================================================

drive_left_pps = 0
drive_right_pps = 0


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


def send_velocity(left_pps, right_pps):
    global drive_left_pps
    global drive_right_pps

    left_pps = int(np.clip(left_pps, -MAX_PPS, MAX_PPS))
    right_pps = int(np.clip(right_pps, -MAX_PPS, MAX_PPS))

    drive_left_pps = left_pps
    drive_right_pps = right_pps

    send_command(f"VEL {left_pps} {right_pps}")


def stop_robot():
    send_velocity(0, 0)


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


def lock_heading_go():
    send_command("LOCK_HEADING_GO")
    print("RPI: LOCK_HEADING_GO sent")


# =====================================================
# MATH / APRILTAG HELPERS
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

    yaw_rad = math.atan2(dx, dy)

    return math.degrees(yaw_rad)


def compute_lateral_x_m(tag):
    if tag.pose_t is None:
        return 0.0

    return float(tag.pose_t[0][0])


def yaw_error_from_tag(raw_yaw_deg):
    return normalize_angle(raw_yaw_deg - EXPECTED_TAG_YAW_DEG)


def get_tag_pose_error(tag):
    raw_yaw_deg = compute_yaw_deg(tag)
    yaw_error_deg = yaw_error_from_tag(raw_yaw_deg)
    x_m = compute_lateral_x_m(tag)

    return raw_yaw_deg, yaw_error_deg, x_m


def tag_pose_good(yaw_error_deg, x_m):
    return (
        abs(yaw_error_deg) <= TAG_YAW_OK_DEG
        and abs(x_m) <= TAG_X_OK_M
    )


def pose_error_to_pps(yaw_error_deg, x_error_m):

    if abs(yaw_error_deg) < YAW_DEADBAND_DEG:
        yaw_error_deg = 0.0

    if abs(x_error_m) < X_DEADBAND_M:
        x_error_m = 0.0

    yaw_correction = KP_YAW_PPS_PER_DEG * yaw_error_deg
    x_correction = KP_X_PPS_PER_M * x_error_m * X_SIGN

    # If yaw and xM corrections fight, prioritize xM.
    if abs(x_error_m) > X_DEADBAND_M:
        if yaw_correction * x_correction < 0:
            yaw_correction *= 0.25

    correction = yaw_correction + x_correction

    correction = int(np.clip(
        correction,
        -MAX_CORRECTION_PPS,
        MAX_CORRECTION_PPS
    ))

    left_pps = CORR_BASE_PPS - correction
    right_pps = CORR_BASE_PPS + correction

    left_pps = int(np.clip(left_pps, CORR_MIN_PPS, CORR_MAX_PPS))
    right_pps = int(np.clip(right_pps, CORR_MIN_PPS, CORR_MAX_PPS))

    return left_pps, right_pps, correction, yaw_correction, x_correction


def dock_align_command(yaw_error_deg):
    if abs(yaw_error_deg) <= DOCK_YAW_OK_DEG:
        return 0, 0

    # If dock alignment turns wrong way, swap these two returns.
    if yaw_error_deg > 0:
        return DOCK_TURN_PPS, -DOCK_TURN_PPS
    else:
        return -DOCK_TURN_PPS, DOCK_TURN_PPS


def advance_target():
    global current_index
    global current_target

    current_index += 1

    if current_index < len(TAG_SEQUENCE):
        current_target = TAG_SEQUENCE[current_index]
    else:
        current_target = TAG_SEQUENCE[-1]

    print(f"RPI: Target advanced to Tag {current_target}")


def start_tag_correction(label):
    global tag_correction_start_time
    global tag_good_count

    tag_correction_start_time = time.time()
    tag_good_count = 0

    print(f"RPI: Starting continuous correction for {label}")


def run_tag_correction(tag, label):
    global tag_good_count

    raw_yaw_deg, yaw_error_deg, x_m = get_tag_pose_error(tag)

    if tag_pose_good(yaw_error_deg, x_m):
        tag_good_count += 1
        stop_robot()

        print(
            f"{label} GOOD "
            f"count={tag_good_count}/{TAG_GOOD_FRAMES_REQUIRED} "
            f"rawYaw={raw_yaw_deg:.2f} "
            f"yawErr={yaw_error_deg:.2f} "
            f"xM={x_m:.4f}"
        )

        if tag_good_count >= TAG_GOOD_FRAMES_REQUIRED:
            return True

        return False

    tag_good_count = 0

    left_pps, right_pps, correction, yaw_corr, x_corr = pose_error_to_pps(
        yaw_error_deg,
        x_m
    )

    send_velocity(left_pps, right_pps)

    print(
        f"{label} CORR "
        f"rawYaw={raw_yaw_deg:.2f} "
        f"yawErr={yaw_error_deg:.2f} "
        f"xM={x_m:.4f} "
        f"yawCorr={yaw_corr:.1f} "
        f"xCorr={x_corr:.1f} "
        f"corr={correction} "
        f"L={left_pps} "
        f"R={right_pps}"
    )

    return False


# =====================================================
# STARTUP MESSAGE
# =====================================================

print("Waiting for docking Tag 11...")
print("Press 's' when Tag 11 is visible.")
print(f"Expected raw tag yaw = {EXPECTED_TAG_YAW_DEG:.1f} deg")
print("This version uses continuous AprilTag correction, no correction tries.")
print("This test stops after Tag 6 is corrected. No 90 degree turn.")


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

        target_found = False
        target_tag = None

        for tag in detections:

            corners = tag.corners.astype(int)

            for i in range(4):
                p1 = tuple(corners[i])
                p2 = tuple(corners[(i + 1) % 4])
                cv2.line(frame, p1, p2, (0, 255, 0), 2)

            center = tuple(tag.center.astype(int))
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            raw_yaw_deg = compute_yaw_deg(tag)
            yaw_error_deg = yaw_error_from_tag(raw_yaw_deg)
            x_m = compute_lateral_x_m(tag)

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
                f"RawYaw:{raw_yaw_deg:.1f}",
                (center[0] + 10, center[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"YawErr:{yaw_error_deg:.1f}",
                (center[0] + 10, center[1] + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"xM:{x_m:.3f}",
                (center[0] + 10, center[1] + 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2
            )

            if tag.tag_id == current_target:
                target_found = True
                target_tag = tag

        # =================================================
        # STATE MACHINE
        # =================================================

        if route_state == "WAIT_START":

            stop_robot()

        elif route_state == "DOCK_ALIGN_TAG11":

            if target_found and current_target == START_TAG:

                raw_yaw_deg, yaw_error_deg, x_m = get_tag_pose_error(target_tag)

                if abs(yaw_error_deg) <= DOCK_YAW_OK_DEG:

                    stop_robot()
                    time.sleep(0.20)

                    print(
                        f"RPI: Dock aligned "
                        f"rawYaw={raw_yaw_deg:.2f} "
                        f"yawErr={yaw_error_deg:.2f} "
                        f"xM={x_m:.4f}"
                    )

                    advance_target()
                    lock_heading_go()

                    print("RPI: Moving from Tag 11 to Tag 7 using IMU heading.")
                    route_state = "MOVE_11_TO_7"

                elif time.time() - dock_align_start_time > DOCK_ALIGN_TIMEOUT_SEC:

                    stop_robot()
                    time.sleep(0.20)

                    print(
                        f"RPI: Dock align timeout. "
                        f"rawYaw={raw_yaw_deg:.2f} "
                        f"yawErr={yaw_error_deg:.2f}. "
                        f"Starting anyway."
                    )

                    advance_target()
                    lock_heading_go()

                    print("RPI: Moving from Tag 11 to Tag 7 using IMU heading.")
                    route_state = "MOVE_11_TO_7"

                else:

                    left_pps, right_pps = dock_align_command(yaw_error_deg)

                    send_velocity(left_pps, right_pps)

                    print(
                        f"DOCK ALIGN "
                        f"rawYaw={raw_yaw_deg:.2f} "
                        f"yawErr={yaw_error_deg:.2f} "
                        f"xM={x_m:.4f} "
                        f"L={left_pps} "
                        f"R={right_pps}"
                    )

            else:

                stop_robot()
                print("RPI: Tag 11 lost during dock alignment. Stop.")
                route_state = "DONE"

        elif route_state == "MOVE_11_TO_7":

            if target_found and current_target == 7:

                print("RPI: Tag 7 reached. Starting continuous AprilTag correction.")

                stop_robot()
                time.sleep(0.15)

                start_tag_correction("TAG7")

                route_state = "TAG7_CORRECTION"

        elif route_state == "TAG7_CORRECTION":

            if not target_found or current_target != 7:

                stop_robot()
                print("RPI: Tag 7 lost during correction. Stop.")
                route_state = "DONE"

            elif time.time() - tag_correction_start_time > TAG_CORRECTION_TIMEOUT_SEC:

                stop_robot()
                print("RPI: Tag 7 correction timeout. Stop.")
                route_state = "DONE"

            else:

                done = run_tag_correction(target_tag, "TAG7")

                if done:

                    stop_robot()
                    time.sleep(0.20)

                    print("RPI: Tag 7 corrected. Locking heading and going to Tag 6.")

                    lock_heading_go()
                    advance_target()

                    print("RPI: Moving from Tag 7 to Tag 6 using IMU heading.")
                    route_state = "MOVE_7_TO_6"

        elif route_state == "MOVE_7_TO_6":

            if target_found and current_target == 6:

                print("RPI: Tag 6 reached. Starting continuous AprilTag correction.")

                stop_robot()
                time.sleep(0.15)

                start_tag_correction("TAG6")

                route_state = "TAG6_CORRECTION"

        elif route_state == "TAG6_CORRECTION":

            if not target_found or current_target != 6:

                stop_robot()
                print("RPI: Tag 6 lost during correction. Stop.")
                route_state = "DONE"

            elif time.time() - tag_correction_start_time > TAG_CORRECTION_TIMEOUT_SEC:

                stop_robot()
                print("RPI: Tag 6 correction timeout. Stop.")
                route_state = "DONE"

            else:

                done = run_tag_correction(target_tag, "TAG6")

                if done:

                    stop_robot()

                    print("RPI: Tag 6 corrected. Route test complete.")
                    print("RPI: Ready for next step: slow 90 degree turn test.")

                    route_state = "DONE"

        elif route_state == "DONE":

            stop_robot()

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
            f"Target: {current_target}",
            (40, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Drive L:{drive_left_pps} R:{drive_right_pps}",
            (40, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Good frames:{tag_good_count}",
            (40, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "AGV Route 11-7-6 Continuous Correction",
            frame
        )

        read_esp32_lines()

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):

            if route_state == "WAIT_START":

                if target_found and current_target == START_TAG:

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
                        print("RPI: Aligning robot with docking Tag 11 yaw.")

                        dock_align_start_time = time.time()
                        route_state = "DOCK_ALIGN_TAG11"

                    else:

                        print("RPI: Start failed. ESP32 did not confirm IMU RECAL.")

                else:

                    print("RPI: Start ignored. Tag 11 not visible.")

        elif key == ord('q'):
            break

        time.sleep(0.05)

except KeyboardInterrupt:
    pass

finally:
    stop_robot()

    picam2.stop()

    cv2.destroyAllWindows()

    ser.close()

    print("Robot stopped")
