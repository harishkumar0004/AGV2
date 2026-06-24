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

KP_YAW_PPS_PER_DEG = 25
KP_X_PPS_PER_M = 30000

X_SIGN = -1.0

YAW_DEADBAND_DEG = 0.20
X_DEADBAND_M = 0.002

MAX_CORRECTION_PPS = 400
CORRECTION_DURATION_SEC = 0.5


# =====================================================
# TAG-BASED ALIGNMENT PARAMETERS
# =====================================================

ALIGN_X_OK_M = 0.004          # 6 mm xM tolerance
ALIGN_YAW_OK_DEG = 2.0        # yaw tolerance

ALIGN_ROTATE_PPS = 1500       # slow rotation speed
ALIGN_MOVE_PPS = 1800         # slow forward/backward correction speed

ALIGN_STAGE_TIMEOUT_SEC = 8.0

# Tune these only if direction is wrong
ROTATE_SIGN = 1
XM_MOVE_SIGN = 1


# =====================================================
# TAG PARAMETERS
# =====================================================

TAG_SEQUENCE = [0, 1, 2, 3, 4, 5]

current_index = 0
current_target = TAG_SEQUENCE[current_index]

captured_tags = set()
mission_started = False

TAG_SIZE_M = 0.020


# =====================================================
# ALIGNMENT STATE
# =====================================================

align_active = False
align_stage = "IDLE"

align_start_yaw = 0.0
align_target_yaw = 0.0
align_stage_start_time = 0.0

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
# CAMERA
# =====================================================

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

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
    "AwbEnable": False,
    "ExposureTime": 10000,
    "AnalogueGain": 2.0
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
    while ser.in_waiting > 0:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print(f"ESP32: {line}")


# =====================================================
# MATH / POSE HELPERS
# =====================================================

def normalize_angle(angle):
    return ((angle + 180.0) % 360.0) - 180.0


def angle_error_deg(target_deg, current_deg):
    return normalize_angle(target_deg - current_deg)


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


# =====================================================
# NORMAL COURSE CORRECTION
# =====================================================

def pose_error_to_pps(yaw_error_deg, x_error_m):

    if abs(yaw_error_deg) < YAW_DEADBAND_DEG:
        yaw_error_deg = 0.0

    if abs(x_error_m) < X_DEADBAND_M:
        x_error_m = 0.0

    yaw_correction = KP_YAW_PPS_PER_DEG * yaw_error_deg
    x_correction = KP_X_PPS_PER_M * x_error_m * X_SIGN

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




# =====================================================
# ALIGNMENT HELPERS
# =====================================================

def desired_heading_for_next_move(current_tag, next_tag):
    # Current prototype: always depart with AprilTag yaw = 0 degrees.
    # Future matrix version: return 0, 90, 180, or -90 based on grid map.
    return 0.0


def rotate_command_from_error(yaw_error):
    if yaw_error > 0:
        left_pps = -ALIGN_ROTATE_PPS * ROTATE_SIGN
        right_pps = ALIGN_ROTATE_PPS * ROTATE_SIGN
    else:
        left_pps = ALIGN_ROTATE_PPS * ROTATE_SIGN
        right_pps = -ALIGN_ROTATE_PPS * ROTATE_SIGN

    return left_pps, right_pps


def xm_move_command(x_error_m):
    if x_error_m > 0:
        move_pps = -ALIGN_MOVE_PPS * XM_MOVE_SIGN
    else:
        move_pps = ALIGN_MOVE_PPS * XM_MOVE_SIGN

    return move_pps, move_pps


def advance_target():
    global current_index
    global current_target

    current_index += 1

    if current_index < len(TAG_SEQUENCE):
        current_target = TAG_SEQUENCE[current_index]
    else:
        current_target = TAG_SEQUENCE[-1]


def start_alignment(tag_id):
    global align_active
    global align_stage
    global align_stage_start_time

    align_active = True
    align_stage = "CHECK_X"
    align_stage_start_time = time.time()

    stop_robot()

    print(f"ALIGN START at Tag {tag_id}")


def finish_alignment(yaw_error, x_error_m):
    global align_active
    global align_stage
    global drive_left_pps
    global drive_right_pps
    global current_target

    stop_robot()
    time.sleep(0.15)

    captured_tags.add(current_target)

    print(
        f"ALIGN DONE Tag {current_target} "
        f"yawErr={yaw_error:.2f} "
        f"xErr={x_error_m:.4f}"
    )

    advance_target()

    align_active = False
    align_stage = "IDLE"

    drive_left_pps = BASE_PPS
    drive_right_pps = BASE_PPS

    print(f"Target advanced to Tag {current_target}")


def fail_alignment(reason, yaw_error, x_error_m):
    global mission_started
    global align_active
    global align_stage
    global drive_left_pps
    global drive_right_pps

    stop_robot()

    drive_left_pps = 0
    drive_right_pps = 0

    mission_started = False
    align_active = False
    align_stage = "IDLE"

    print(
        f"ALIGN FAILED: {reason} "
        f"yawErr={yaw_error:.2f} "
        f"xErr={x_error_m:.4f} "
        f"manual check needed"
    )



def update_alignment(tag):
    global align_stage
    global align_start_yaw
    global align_target_yaw
    global align_stage_start_time
    global drive_left_pps
    global drive_right_pps

    yaw_deg = compute_yaw_deg(tag)
    x_error_m = compute_lateral_x_m(tag)

    now = time.time()

    if now - align_stage_start_time > ALIGN_STAGE_TIMEOUT_SEC:
        fail_alignment(
            f"timeout in {align_stage}",
            yaw_deg,
            x_error_m
        )
        return

    # -------------------------------------------------
    # STAGE 1: Check xM
    # -------------------------------------------------

    if align_stage == "CHECK_X":

        if abs(x_error_m) <= ALIGN_X_OK_M:
            align_stage = "YAW_ALIGN"
            align_stage_start_time = now

            print(
                f"ALIGN CHECK_X OK "
                f"xErr={x_error_m:.4f} "
                f"-> YAW_ALIGN"
            )

            return

        align_start_yaw = yaw_deg
        align_target_yaw = normalize_angle(align_start_yaw + 90.0)

        align_stage = "ROTATE_90"
        align_stage_start_time = now

        print(
            f"ALIGN CHECK_X xErr={x_error_m:.4f} "
            f"too high -> ROTATE_90 "
            f"from={align_start_yaw:.2f} "
            f"to={align_target_yaw:.2f}"
        )

        return

    # -------------------------------------------------
    # STAGE 2: Rotate 90 degrees using AprilTag yaw
    # -------------------------------------------------

    if align_stage == "ROTATE_90":

        err = angle_error_deg(align_target_yaw, yaw_deg)

        if abs(err) <= ALIGN_YAW_OK_DEG:
            stop_robot()
            time.sleep(0.15)

            drive_left_pps = 0
            drive_right_pps = 0

            align_stage = "XM_MOVE"
            align_stage_start_time = now

            print(
                f"ALIGN ROTATE_90 DONE "
                f"yaw={yaw_deg:.2f} "
                f"target={align_target_yaw:.2f} "
                f"-> XM_MOVE"
            )

            return

        drive_left_pps, drive_right_pps = rotate_command_from_error(err)

        send_velocity(drive_left_pps, drive_right_pps)

        print(
            f"ALIGN ROTATE_90 "
            f"yaw={yaw_deg:.2f} "
            f"target={align_target_yaw:.2f} "
            f"err={err:.2f} "
            f"L={drive_left_pps} "
            f"R={drive_right_pps}"
        )

        return


    # -------------------------------------------------
    # STAGE 3: Move forward/backward until xM is small
    # -------------------------------------------------

    if align_stage == "XM_MOVE":

        if abs(x_error_m) <= ALIGN_X_OK_M:
            stop_robot()
            time.sleep(0.15)

            drive_left_pps = 0
            drive_right_pps = 0

            align_target_yaw = align_start_yaw

            align_stage = "ROTATE_BACK"
            align_stage_start_time = now

            print(
                f"ALIGN XM_MOVE DONE "
                f"xErr={x_error_m:.4f} "
                f"-> ROTATE_BACK "
                f"target={align_target_yaw:.2f}"
            )

            return

        drive_left_pps, drive_right_pps = xm_move_command(x_error_m)

        send_velocity(drive_left_pps, drive_right_pps)

        print(
            f"ALIGN XM_MOVE "
            f"xErr={x_error_m:.4f} "
            f"L={drive_left_pps} "
            f"R={drive_right_pps}"
        )

        return

    # -------------------------------------------------
    # STAGE 4: Rotate back using AprilTag yaw
    # -------------------------------------------------

    if align_stage == "ROTATE_BACK":

        err = angle_error_deg(align_target_yaw, yaw_deg)

        if abs(err) <= ALIGN_YAW_OK_DEG:
            stop_robot()
            time.sleep(0.15)

            drive_left_pps = 0
            drive_right_pps = 0

            align_stage = "YAW_ALIGN"
            align_stage_start_time = now

            print(
                f"ALIGN ROTATE_BACK DONE "
                f"yaw={yaw_deg:.2f} "
                f"-> YAW_ALIGN"
            )

            return

        drive_left_pps, drive_right_pps = rotate_command_from_error(err)

        send_velocity(drive_left_pps, drive_right_pps)

        print(
            f"ALIGN ROTATE_BACK "
            f"yaw={yaw_deg:.2f} "
            f"target={align_target_yaw:.2f} "
            f"err={err:.2f} "
            f"L={drive_left_pps} "
            f"R={drive_right_pps}"
        )

        return

    # -------------------------------------------------
    # STAGE 5: Final yaw alignment to desired heading
    # -------------------------------------------------

    if align_stage == "YAW_ALIGN":

        if current_index + 1 < len(TAG_SEQUENCE):
            next_tag = TAG_SEQUENCE[current_index + 1]
        else:
            next_tag = current_target

        desired_heading = desired_heading_for_next_move(
            current_target,
            next_tag
        )

        yaw_error = normalize_angle(yaw_deg - desired_heading)

        if abs(yaw_error) <= ALIGN_YAW_OK_DEG:
            finish_alignment(
                yaw_error,
                x_error_m
            )
            return

        drive_left_pps, drive_right_pps = rotate_command_from_error(
            -yaw_error
        )

        send_velocity(drive_left_pps, drive_right_pps)

        print(
            f"ALIGN YAW_ALIGN "
            f"yaw={yaw_deg:.2f} "
            f"desired={desired_heading:.2f} "
            f"yawErr={yaw_error:.2f} "
            f"L={drive_left_pps} "
            f"R={drive_right_pps}"
        )

        return


# =====================================================
# INITIAL STATE
# =====================================================

print("Waiting for Tag 0...")


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

        # ---------------------------------------------
        # Draw all tags and find current target
        # ---------------------------------------------

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

            yaw_deg = compute_yaw_deg(tag)
            x_m = compute_lateral_x_m(tag)

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



        # ---------------------------------------------
        # Start message
        # ---------------------------------------------

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

        # ---------------------------------------------
        # Robot logic
        # ---------------------------------------------

        if mission_started:

            if align_active:

                if target_found and target_tag is not None:
                    update_alignment(target_tag)
                else:
                    stop_robot()
                    drive_left_pps = 0
                    drive_right_pps = 0
                    print("ALIGN PAUSED: current tag not visible")

            else:

                # Final tag reached
                if (
                    target_found and
                    target_tag is not None and
                    current_target == TAG_SEQUENCE[-1]
                ):
                    print("Final tag reached")
                    stop_robot()
                    raise KeyboardInterrupt

                # Reached target tag: stop and align before departure
                elif (
                    target_found and
                    target_tag is not None and
                    current_target not in captured_tags
                ):
                    start_alignment(current_target)

                # Between tags: move straight
                else:
                    drive_left_pps = BASE_PPS
                    drive_right_pps = BASE_PPS

                    send_velocity(
                        drive_left_pps,
                        drive_right_pps
                    )

        else:
            stop_robot()
            drive_left_pps = 0
            drive_right_pps = 0


        # ---------------------------------------------
        # Display status
        # ---------------------------------------------

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

        cv2.putText(
            frame,
            f"Align:{align_stage}",
            (50, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255) if align_active else (180, 180, 180),
            2
        )

        cv2.imshow(
            "AprilTag Navigation",
            frame
        )

        read_esp32_lines()

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):

            if target_found and current_target == 0:
                mission_started = True
                print("Mission started")
                start_alignment(current_target)
            else:
                print("Start ignored: Tag 0 not visible")

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
