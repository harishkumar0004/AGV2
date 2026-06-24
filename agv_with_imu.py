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

WHEEL_DIAMETER = 0.117
PULSES_PER_REV = 20000

WHEEL_CIRCUMFERENCE = math.pi * WHEEL_DIAMETER
PULSES_PER_METER = PULSES_PER_REV / WHEEL_CIRCUMFERENCE


# =====================================================
# DRIVE PARAMETERS
# =====================================================

BASE_PPS = 5500
MAX_PPS = 12000

MIN_DRIVE_PPS = 3000
MAX_DRIVE_PPS = 8000


# =====================================================
# APRILTAG CORRECTION PARAMETERS
# =====================================================

KP_YAW_PPS_PER_DEG = 25
KP_X_PPS_PER_M = 30000

X_SIGN = -1.0

YAW_DEADBAND_DEG = 0.20
X_DEADBAND_M = 0.002

MAX_CORRECTION_PPS = 400
CORRECTION_DURATION_SEC = 0.5


# =====================================================
# TAG FLIP / HEADING PARAMETERS
# =====================================================

# Tags are physically rotated 180 degrees.
# Therefore, when robot is correctly aligned, raw tag yaw is around 180 or -180.
TAG_YAW_OFFSET_DEG = 180.0

# At tag 6, ESP32 will rotate robot by this much using IMU.
# If turn direction is wrong, change to -90.0.
TURN_REL_DEG = 90.0

# After robot turns +90, expected raw tag yaw changes by +90 from flipped yaw.
EXPECTED_AFTER_TURN_YAW_DEG = None

VERIFY_YAW_OK_DEG = 3.0


# =====================================================
# ROUTE PARAMETERS
# =====================================================

START_TAG = 11
TAG_SEQUENCE = [11, 7, 6]

current_index = 0
current_target = TAG_SEQUENCE[current_index]

route_state = "WAIT_START"
mission_started = False

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

correction_active = False
correction_end_time = 0.0

drive_left_pps = 0
drive_right_pps = 0


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
    left_pps = int(np.clip(left_pps, -MAX_PPS, MAX_PPS))
    right_pps = int(np.clip(right_pps, -MAX_PPS, MAX_PPS))

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
    """
    Raw visual yaw of the AprilTag.

    This function does NOT correct for 180 degree tag flip.
    If the tag is physically flipped, this may return 179.5 or -179.5
    when the robot is actually aligned correctly.
    """
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


def yaw_error_from_tag(raw_yaw_deg, expected_tag_yaw_deg):
    """
    Converts raw tag yaw into useful yaw error.

    Example for flipped tags:
        raw_yaw_deg = 179.5
        expected_tag_yaw_deg = 180.0
        yaw_error = -0.5

        raw_yaw_deg = -179.5
        expected_tag_yaw_deg = 180.0
        yaw_error = 0.5
    """
    return normalize_angle(raw_yaw_deg - expected_tag_yaw_deg)


def pose_error_to_pps(yaw_error_deg, x_error_m):

    if abs(yaw_error_deg) < YAW_DEADBAND_DEG:
        yaw_error_deg = 0.0

    if abs(x_error_m) < X_DEADBAND_M:
        x_error_m = 0.0

    yaw_correction = KP_YAW_PPS_PER_DEG * yaw_error_deg
    x_correction = KP_X_PPS_PER_M * x_error_m * X_SIGN

    if abs(x_error_m) > X_DEADBAND_M:
        if yaw_correction * x_correction < 0:
            yaw_correction *= 0.20

    correction = yaw_correction + x_correction

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

    print(f"RPI: Target advanced to Tag {current_target}")


def start_apriltag_correction(tag, label, expected_tag_yaw_deg):
    global correction_active
    global correction_end_time
    global drive_left_pps
    global drive_right_pps

    raw_yaw_deg = compute_yaw_deg(tag)
    x_m = compute_lateral_x_m(tag)

    yaw_error = yaw_error_from_tag(
        raw_yaw_deg,
        expected_tag_yaw_deg
    )

    x_error_m = x_m

    (
        drive_left_pps,
        drive_right_pps,
        correction,
        yaw_correction,
        x_correction
    ) = pose_error_to_pps(
        yaw_error,
        x_error_m
    )

    correction_active = True
    correction_end_time = time.time() + CORRECTION_DURATION_SEC

    send_velocity(
        drive_left_pps,
        drive_right_pps
    )

    print(
        f"{label} CORR "
        f"rawYaw={raw_yaw_deg:.2f} "
        f"expectedYaw={expected_tag_yaw_deg:.2f} "
        f"yawErr={yaw_error:.2f} "
        f"xM={x_m:.4f} "
        f"xErr={x_error_m:.4f} "
        f"yawCorr={yaw_correction:.1f} "
        f"xCorr={x_correction:.1f} "
        f"corr={correction} "
        f"L={drive_left_pps} "
        f"R={drive_right_pps} "
        f"duration={CORRECTION_DURATION_SEC:.2f}s"
    )


def verify_tag6_after_turn(tag):
    raw_yaw_deg = compute_yaw_deg(tag)

    yaw_error = yaw_error_from_tag(
        raw_yaw_deg,
        EXPECTED_AFTER_TURN_YAW_DEG
    )

    print(
        f"TAG6 VERIFY "
        f"rawYaw={raw_yaw_deg:.2f} "
        f"expectedYaw={EXPECTED_AFTER_TURN_YAW_DEG:.2f} "
        f"yawErr={yaw_error:.2f}"
    )

    return abs(yaw_error) <= VERIFY_YAW_OK_DEG


# =====================================================
# INITIALIZE EXPECTED AFTER TURN YAW
# =====================================================

EXPECTED_AFTER_TURN_YAW_DEG = normalize_angle(
    TAG_YAW_OFFSET_DEG + TURN_REL_DEG
)

print("Waiting for docking Tag 11...")
print("Press 's' when Tag 11 is visible.")
print(f"Straight expected raw tag yaw = {TAG_YAW_OFFSET_DEG:.1f} deg")
print(f"After turn expected raw tag yaw = {EXPECTED_AFTER_TURN_YAW_DEG:.1f} deg")


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


        # ------------------------------------------------
        # Draw tags and find current target
        # ------------------------------------------------
        for tag in detections:

            corners = tag.corners.astype(int)

            for i in range(4):
                p1 = tuple(corners[i])
                p2 = tuple(corners[(i + 1) % 4])
                cv2.line(frame, p1, p2, (0, 255, 0), 2)

            center = tuple(tag.center.astype(int))

            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            raw_yaw_deg = compute_yaw_deg(tag)
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
                f"xM:{x_m:.3f}",
                (center[0] + 10, center[1] + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 255),
                2
            )

            if tag.tag_id == current_target:
                target_found = True
                target_tag = tag

        # ------------------------------------------------
        # STATE MACHINE
        # ------------------------------------------------

        if route_state == "WAIT_START":

            stop_robot()
            drive_left_pps = 0
            drive_right_pps = 0

        elif route_state == "MOVE_11_TO_7":

            # ESP32 is holding heading using IMU.
            # Raspberry Pi only waits for Tag 7.

            if target_found and current_target == 7:

                print("RPI: Tag 7 reached. Starting AprilTag correction.")

                stop_robot()
                time.sleep(0.15)

                start_apriltag_correction(
                    target_tag,
                    "TAG7",
                    TAG_YAW_OFFSET_DEG
                )

                route_state = "TAG7_CORRECTION"

        elif route_state == "TAG7_CORRECTION":

            if correction_active and time.time() >= correction_end_time:

                correction_active = False

                stop_robot()
                time.sleep(0.15)

                lock_heading_go()

                advance_target()

                print("RPI: Moving from Tag 7 to Tag 6 using IMU heading.")
                route_state = "MOVE_7_TO_6"

            elif correction_active:

                send_velocity(
                    drive_left_pps,
                    drive_right_pps
                )

        elif route_state == "MOVE_7_TO_6":

            # ESP32 is holding heading using IMU.
            # Raspberry Pi only waits for Tag 6.

            if target_found and current_target == 6:

                print("RPI: Tag 6 reached. Starting AprilTag correction before turn.")

                stop_robot()
                time.sleep(0.15)

                start_apriltag_correction(
                    target_tag,
                    "TAG6",
                    TAG_YAW_OFFSET_DEG
                )

                route_state = "TAG6_CORRECTION"

        elif route_state == "TAG6_CORRECTION":

            if correction_active and time.time() >= correction_end_time:

                correction_active = False

                stop_robot()
                time.sleep(0.3)

                print(f"RPI: Requesting IMU turn {TURN_REL_DEG:.1f} deg at Tag 6.")
                send_command(f"TURN_REL {TURN_REL_DEG:.1f}")

                route_state = "WAIT_TURN_90_DONE"

            elif correction_active:

                send_velocity(
                    drive_left_pps,
                    drive_right_pps
                )

        elif route_state == "WAIT_TURN_90_DONE":

            lines = read_esp32_lines()

            for line in lines:
                if "OK TURN_DONE" in line:
                    print("RPI: ESP32 reports 90 degree turn done.")
                    route_state = "VERIFY_TAG6_HEADING"

        elif route_state == "VERIFY_TAG6_HEADING":

            if target_found and current_target == 6:

                ok = verify_tag6_after_turn(target_tag)

                if ok:
                    print("RPI: Tag 6 heading verified. Test complete.")
                    stop_robot()
                    route_state = "DONE"
                else:
                    print("RPI: Tag 6 heading verification failed. Stop for inspection.")
                    stop_robot()
                    route_state = "DONE"

            else:
                stop_robot()
                print("RPI: Waiting to see Tag 6 for heading verification.")

        elif route_state == "DONE":

            stop_robot()
            drive_left_pps = 0
            drive_right_pps = 0

        # ------------------------------------------------
        # DISPLAY
        # ------------------------------------------------

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

        cv2.imshow(
            "AGV Route Test 11-7-6",
            frame
        )

        if route_state != "WAIT_TURN_90_DONE":
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

                        advance_target()

                        lock_heading_go()

                        print("RPI: Moving from Tag 11 to Tag 7 using IMU heading.")
                        route_state = "MOVE_11_TO_7"

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
