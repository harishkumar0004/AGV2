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

FORWARD_SPEED = 0.10          # m/s

KP_YAW = 1.2

MAX_OMEGA = 0.4               # rad/s
MAX_PPS = 12000

YAW_DEADBAND = 2.0            # degrees

# =====================================================
# TAG PARAMETERS
# =====================================================

TAG_SEQUENCE = [0, 1, 2, 3]

current_index = 0
current_target = TAG_SEQUENCE[current_index]

captured_tags = set()

mission_started = False
reference_yaw = None

BASE_PPS = 5500


# =====================================================
# SERIAL
# =====================================================

ser = serial.Serial(
    "/dev/ttyUSB0",
    115200,
    timeout=1
)

time.sleep(2)


# =====================================================
# CAMERA
# =====================================================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

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
# HELPERS
# =====================================================

def send_velocity(left_pps, right_pps):

    left_pps = int(np.clip(left_pps, -MAX_PPS, MAX_PPS))
    right_pps = int(np.clip(right_pps, -MAX_PPS, MAX_PPS))

    cmd = f"VEL {left_pps} {right_pps}\n"

    ser.write(cmd.encode())


def stop_robot():

    send_velocity(0, 0)


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


def normalize_angle(angle):

    return ((angle + 180.0) % 360.0) - 180.0


def yaw_error_to_pps(yaw_error_deg):

    if abs(yaw_error_deg) < 0.20:
        yaw_error_deg = 0.0

    kp = 100

    correction = int(kp * yaw_error_deg)

    left_pps = BASE_PPS - correction
    right_pps = BASE_PPS + correction

    left_pps = int(np.clip(left_pps, 3000, 8000))
    right_pps = int(np.clip(right_pps, 3000, 8000))

    return left_pps, right_pps


# =====================================================
# INITIAL STATE
# =====================================================

frozen_left_pps = 0
frozen_right_pps = 0

print("Waiting for Tag 6...")

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

        detections = detector.detect(gray)

        target_found = False

        for tag in detections:

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

            if tag.tag_id != current_target:
                continue

            target_found = True

            yaw_deg = compute_yaw_deg(tag)

            cv2.putText(
                frame,
                f"Yaw:{yaw_deg:.1f}",
                (center[0] + 10, center[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )


            # -------------------------------------
            # WAIT FOR START
            # -------------------------------------

            if current_target == 0 and not mission_started:

                cv2.putText(
                    frame,
                    "Press 's' to start",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2
                )

            # -------------------------------------
            # TAG 6 = REFERENCE
            # -------------------------------------

            if (
                mission_started and
                current_target == 0 and
                current_target not in captured_tags
            ):

                reference_yaw = yaw_deg

                frozen_left_pps = BASE_PPS
                frozen_right_pps = BASE_PPS

                captured_tags.add(0)

                print(
                    f"Reference Tag 0 "
                    f"yaw={reference_yaw:.2f} "
                    f"L={frozen_left_pps} "
                    f"R={frozen_right_pps}"
                )

                current_index += 1
                current_target = TAG_SEQUENCE[current_index]

            # -------------------------------------
            # TAG 7 / TAG 8 CORRECTION
            # -------------------------------------

            elif (
                mission_started and
                current_target in [1, 2] and
                current_target not in captured_tags
            ):

                yaw_error = normalize_angle(
                    yaw_deg - reference_yaw
                )

                frozen_left_pps, frozen_right_pps = yaw_error_to_pps(yaw_error)

                captured_tags.add(current_target)

                print(
                    f"Tag {current_target} "
                    f"yaw={yaw_deg:.2f} "
                    f"error={yaw_error:.2f} "
                    f"L={frozen_left_pps} "
                    f"R={frozen_right_pps}"
                )

                current_index += 1
                current_target = TAG_SEQUENCE[current_index]

            # -------------------------------------
            # TAG 9 STOP
            # -------------------------------------

            elif (
                mission_started and
                current_target == 3 and
                current_target not in captured_tags
            ):

                print("Final tag reached")

                stop_robot()

                raise KeyboardInterrupt

        # -----------------------------------------
        # DRIVE USING FROZEN COMMANDS
        # -----------------------------------------

        if mission_started:

            send_velocity(
                frozen_left_pps,
                frozen_right_pps
            )

        else:

            stop_robot()

        cv2.putText(
            frame,
            f"Target: {current_target}",
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2
        )

        cv2.imshow(
            "AprilTag Navigation",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):

            if target_found and current_target == 0:

                mission_started = True

                print("Mission started")

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
