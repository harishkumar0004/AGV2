#!/usr/bin/env python3

import math
import time

import cv2
import numpy as np
import serial

from picamera2 import Picamera2
from pupil_apriltags import Detector


# =====================================================
# ROBOT PARAMETERS
# =====================================================

WHEEL_BASE = 0.324
WHEEL_DIAMETER = 0.117
PULSES_PER_REV = 20000

WHEEL_CIRCUMFERENCE = math.pi * WHEEL_DIAMETER
PULSES_PER_METER = PULSES_PER_REV / WHEEL_CIRCUMFERENCE


# =====================================================
# CONTROLLER PARAMETERS
# =====================================================

BASE_PPS = 5500
MAX_PPS = 12000

MIN_DRIVE_PPS = 3000
MAX_DRIVE_PPS = 8000

# Yaw correction gain
KP_YAW_PPS_PER_DEG = 25

# Lateral correction gain
KP_X_PPS_PER_M = 30000

# Your tested working lateral sign
X_SIGN = -1.0

# Deadbands
YAW_DEADBAND_DEG = 0.20
X_DEADBAND_M = 0.002

# Maximum combined correction
MAX_CORRECTION_PPS = 400

# Apply tag correction only for this time.
# After this, ESP32 locks IMU heading and holds straight.
CORRECTION_DURATION_SEC = 0.4


# =====================================================
# TAG PARAMETERS
# =====================================================

TAG_SEQUENCE = [11, 7, 6, 5, 4, 3, 2, 1, 0]

current_index = 0
current_target = TAG_SEQUENCE[current_index]

captured_tags = set()

mission_started = False

TAG_SIZE_M = 0.020   # 20 mm tag

# =====================================================
# SERIAL
# =====================================================

ser = serial.Serial(
    "/dev/ttyUSB0",
    115200,
    timeout=0.2
)

time.sleep(2)


# =====================================================
# CAMERA
# =====================================================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Approximate camera values.
# Later replace these with real calibration values.
FX = 615.0
FY = 615.0
CX = FRAME_WIDTH / 2.0
CY = FRAME_HEIGHT / 2.0

CAMERA_PARAMS = (FX, FY, CX, CY)

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
    "ExposureTime": 5000,
    "AnalogueGain": 1.5
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
    left_pps = int(np.clip(left_pps, -MAX_PPS, MAX_PPS))
    right_pps = int(np.clip(right_pps, -MAX_PPS, MAX_PPS))

    cmd = f"VEL {left_pps} {right_pps}"
    send_command(cmd)


def stop_robot():
    send_command("STOP")


def wait_for_esp32_text(expected_text, timeout_sec=10.0):
    start_time = time.time()

    while time.time() - start_time < timeout_sec:
        line = ser.readline().decode(errors="ignore").strip()

        if line:
            print(f"ESP32: {line}")

        if expected_text in line:
            return True

    return False

# =====================================================
# APRILTAG HELPERS
# =====================================================

def normalize_angle(angle):
    return ((angle + 180)  % 360.0) - 180


def compute_yaw_deg(tag):
    corners = tag.corners

    cx = tag.center[0]
    cy = tag.center[1]

    top_mid_x = (corners[0][0] + corners[1][0]) / 2.0
    top_mid_y = (corners[0][1] + corners[1][1]) / 2.0

    dx = top_mid_x - cx
    dy = cy - top_mid_y

    yaw_deg = math.degrees(math.atan2(dx, dy))

    yaw_deg = yaw_deg - 180.0

    yaw_deg = ((yaw_deg + 180.0) % 360.0) - 180.0

    return yaw_deg


def compute_lateral_x_m(tag):
    if tag.pose_t is None:
        return 0.0

    return float(tag.pose_t[0][0])


def pose_error_to_pps(yaw_error_deg, x_error_m):

    # ---------------- deadband ----------------

    if abs(yaw_error_deg) < YAW_DEADBAND_DEG:
        yaw_error_deg = 0.0

    if abs(x_error_m) < X_DEADBAND_M:
        x_error_m = 0.0

    # ---------------- correction ----------------

    yaw_correction = KP_YAW_PPS_PER_DEG * yaw_error_deg

    x_correction = KP_X_PPS_PER_M * x_error_m * X_SIGN

    # If lateral correction and yaw correction fight,
    # reduce yaw effect and trust xM more.
    if abs(x_error_m) > X_DEADBAND_M:
        if yaw_correction * x_correction < 0:
            yaw_correction = yaw_correction * 0.20

    correction = x_correction + yaw_correction

    correction = int(np.clip(
        correction,
        -MAX_CORRECTION_PPS,
        MAX_CORRECTION_PPS
    ))

    left_pps = BASE_PPS + correction
    right_pps = BASE_PPS - correction

    left_pps = int(np.clip(left_pps, MIN_DRIVE_PPS, MAX_DRIVE_PPS))
    right_pps = int(np.clip(right_pps, MIN_DRIVE_PPS, MAX_DRIVE_PPS))

    return left_pps, right_pps, correction, yaw_correction, x_correction


def advance_target():
    global current_index
    global current_target

    current_index += 1

    if current_index < len(TAG_SEQUENCE):
        current_target = TAG_SEQUENCE[current_index]
    else:
        current_target = TAG_SEQUENCE[-1]


# =====================================================
# INITIAL STATE
# =====================================================

# Last calculated correction command
frozen_left_pps = BASE_PPS
frozen_right_pps = BASE_PPS

# Actual correction command sent during correction_active
drive_left_pps = 0
drive_right_pps = 0

# Short correction timing
correction_active = False
correction_end_time = 0.0

print("Waiting for Tag 11...")


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

        for tag in detections:

            # -------------------------------------
            # Draw all detected tags
            # -------------------------------------

            corners = tag.corners.astype(int)

            for i in range(4):
                p1 = tuple(corners[i])
                p2 = tuple(corners[(i + 1) % 4])
                cv2.line(frame, p1, p2, (0, 255, 0), 2)

            center = tuple(tag.center.astype(int))

            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            cv2.putText(
                frame,
                f"ID:{tag.tag_id}",
                (center[0] + 10, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            # -------------------------------------
            # Only process current target tag
            # -------------------------------------

            if tag.tag_id != current_target:
                continue

            target_found = True

            yaw_deg = compute_yaw_deg(tag)
            tag_x_m = compute_lateral_x_m(tag)

            cv2.putText(
                frame,
                f"Yaw:{yaw_deg:.1f}",
                (center[0] + 10, center[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"xM:{tag_x_m:.3f}",
                (center[0] + 10, center[1] + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2
            )

            # -------------------------------------
            # WAIT FOR START ON TAG 0
            # -------------------------------------

            if current_target == 11 and not mission_started:
                cv2.putText(
                    frame,
                    "Press 's' to calibrate IMU and start",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            # -------------------------------------
            # TAG 0 / 1 / 2 / 3 / 4 = TAG CORRECTION
            # Tag is main source here, not IMU.
            # -------------------------------------

            if (
                mission_started and
                current_target in [11, 7, 6, 5, 4, 3, 2, 1, 0] and
                current_target not in captured_tags
            ):

                yaw_error = normalize_angle(yaw_deg)

                x_error_m = tag_x_m

                (
                    frozen_left_pps,
                    frozen_right_pps,
                    correction,
                    yaw_correction,
                    x_correction
                ) = pose_error_to_pps(
                    yaw_error,
                    x_error_m
                )

                # Apply AprilTag correction only for short time.
                drive_left_pps = frozen_left_pps
                drive_right_pps = frozen_right_pps

                correction_active = True
                correction_end_time = time.time() + CORRECTION_DURATION_SEC

                captured_tags.add(current_target)

                print(
                    f"Tag {current_target} "
                    f"yaw={yaw_deg:.2f} "
                    f"yawErr={yaw_error:.2f} "
                    f"xM={tag_x_m:.4f} "
                    f"xErr={x_error_m:.4f} "
                    f"yawCorr={yaw_correction:.1f} "
                    f"xCorr={x_correction:.1f} "
                    f"corr={correction} "
                    f"L={drive_left_pps} "
                    f"R={drive_right_pps} "
                    f"duration={CORRECTION_DURATION_SEC:.2f}s"
                )

                advance_target()

            # -------------------------------------
            # TAG 5 = STOP
            # -------------------------------------

            elif (
                mission_started and
                current_target == 6 and
                current_target not in captured_tags
            ):

                print("Final tag reached")

                stop_robot()

                raise KeyboardInterrupt

        # -----------------------------------------
        # DRIVE SECTION
        # -----------------------------------------

        if mission_started:

            # If tag correction time finished,
            # tell ESP32 to lock current IMU heading
            # and continue straight using IMU.
            if correction_active and time.time() >= correction_end_time:

                correction_active = False

                send_command("LOCK_HEADING_GO")

                print("Correction finished -> LOCK_HEADING_GO")

            # During tag correction only, send VEL L R.
            # Between tags, send nothing.
            # ESP32 handles IMU straight hold.
            if correction_active:
                send_velocity(
                    drive_left_pps,
                    drive_right_pps
                )

        else:
            # Before mission start, keep ESP32 stopped.
            # This may print repeated STOP replies on ESP32.
            stop_robot()

        # -----------------------------------------
        # Display status
        # -----------------------------------------

        cv2.putText(
            frame,
            f"Target: {current_target}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Drive L:{drive_left_pps} R:{drive_right_pps}",
            (50, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        correction_text = "TAG_CORR:ON" if correction_active else "TAG_CORR:OFF"

        cv2.putText(
            frame,
            correction_text,
            (50, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255) if correction_active else (180, 180, 180),
            2
        )

        imu_text = "IMU:ESP32 between tags"

        cv2.putText(
            frame,
            imu_text,
            (50, 220),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "AprilTag Navigation",
            frame
        )

        key = cv2.waitKey(1) & 0xFF


        # -----------------------------------------
        # START KEY
        # -----------------------------------------

        if key == ord('s'):

            if target_found and current_target == 11:

                print("Tag 11 visible.")
                print("Requesting ESP32 IMU recalibration...")
                print("Keep AGV completely still.")

                stop_robot()
                time.sleep(0.2)

                # Clear old ESP32 prints before waiting for IMU RECAL response.
                ser.reset_input_buffer()

                send_command("IMU RECAL")

                ok = wait_for_esp32_text(
                    "OK IMU RECAL",
                    timeout_sec=10.0
                )

                if ok:
                    mission_started = True
                    print("Mission started after IMU calibration")
                else:
                    mission_started = False
                    print("Start failed: ESP32 did not confirm IMU calibration")

            else:
                print("Start ignored: Tag 11 not visible")

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
