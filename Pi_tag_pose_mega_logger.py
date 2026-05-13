#!/usr/bin/env python3
"""
Raspberry Pi AprilTag + Mega telemetry logger.

Purpose:
  1. Send FORWARD / STOP to the Arduino Mega.
  2. Read AprilTag lateral offset and yaw error from the Pi camera.
  3. Read Mega gyro/motor telemetry lines such as:
       TEL:H=...,E=...,C=...,L=...,R=...
  4. Save both camera and Mega values into one CSV.

This file is for logging only. It does not send tag correction to the Mega.
"""

import argparse
import csv
import math
import select
import subprocess
import sys
import threading
import time
from datetime import datetime

import apriltag
import cv2
import numpy as np
import serial


DEFAULT_TAG_SIZE_CM = 3.0
DEFAULT_CAMERA_INDEX = 0
DEFAULT_SERIAL_PORT = "/dev/ttyUSB0"
DEFAULT_BAUD = 115200
DEFAULT_FRAME_WIDTH = 640
DEFAULT_FRAME_HEIGHT = 480
DEFAULT_TARGET_FPS = 30
DEFAULT_CAMERA_FOURCC = "MJPG"
DEFAULT_CAMERA_WARMUP_SEC = 3.0
DEFAULT_EXPOSURE = 40
DEFAULT_GAIN = 64
DEFAULT_POWER_LINE_FREQUENCY = 1


FIELDNAMES = [
    "time_s",
    "frame_index",
    "tag_detected",
    "tag_count",
    "tag_id",
    "lateral_offset_cm",
    "lateral_offset_px",
    "lateral_offset_norm",
    "yaw_error_deg",
    "raw_yaw_image_deg",
    "center_x_px",
    "center_y_px",
    "tag_width_px",
    "mega_time_s",
    "mega_age_s",
    "motion_state",
    "last_command",
    "mega_event",
    "mega_heading_error_deg",
    "mega_control_error_deg",
    "mega_correction_rpm",
    "mega_left_rpm",
    "mega_right_rpm",
    "mega_last_line",
]


def make_initial_mega_state():
    return {
        "time_s": None,
        "last_line": "",
        "event": "",
        "motion_state": "STOPPED",
        "last_command": "",
        "H": None,
        "E": None,
        "C": None,
        "L": None,
        "R": None,
    }


def fmt(value, digits=3):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def parse_float(text):
    try:
        return float(text)
    except ValueError:
        return None


def parse_telemetry(line):
    if not line.startswith("TEL:"):
        return {}

    values = {}
    for item in line[4:].split(","):
        if "=" not in item:
            continue

        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key == "Rpm":
            key = "R"

        values[key] = parse_float(value)

    return values


def update_mega_state_from_line(state, line, start_time):
    state["time_s"] = time.time() - start_time
    state["last_line"] = line

    if line.startswith("TEL:"):
        telemetry = parse_telemetry(line)
        for key in ("H", "E", "C", "L", "R"):
            if key in telemetry:
                state[key] = telemetry[key]
        return

    state["event"] = line

    if line == "ACK:FORWARD":
        state["motion_state"] = "MOVING"
    elif line == "ACK:STOP":
        state["motion_state"] = "STOPPED"
    elif line == "EVT:READY":
        if state["motion_state"] == "":
            state["motion_state"] = "READY"


def serial_reader(ser, stop_event, state, state_lock, start_time):
    while not stop_event.is_set():
        try:
            line = ser.readline().decode("utf-8", errors="replace").strip()
        except serial.SerialException as exc:
            line = f"SERIAL_READ_ERROR:{exc}"
            stop_event.set()

        if not line:
            continue

        with state_lock:
            update_mega_state_from_line(state, line, start_time)

        print("[MEGA]", line)


def open_serial(port, baud):
    if not port:
        return None

    ser = serial.Serial(port, baud, timeout=0.1)
    time.sleep(2.0)
    print(f"Serial connected: {port} @ {baud}")
    return ser


def send_command(ser, state, state_lock, command):
    command = command.strip().upper()

    if ser is None:
        print(f"[PI] serial disabled, skipped: {command}")
        return

    ser.write((command + "\n").encode("ascii"))
    ser.flush()

    with state_lock:
        state["last_command"] = command
        if command in ("FORWARD", "CMD:FORWARD", "CMD:FWD"):
            state["motion_state"] = "MOVING"
        elif command in ("STOP", "CMD:STOP"):
            state["motion_state"] = "STOPPED"

    print("[PI]", command)


def create_detector():
    if hasattr(apriltag, "apriltag"):
        return apriltag.apriltag("tag36h11")

    options = apriltag.DetectorOptions(families="tag36h11")
    return apriltag.Detector(options)


def get_detection_parts(det):
    if isinstance(det, dict):
        corners = det.get("lb-rb-rt-lt")
        center = det.get("center")
        tag_id = det.get("id")
    else:
        corners = det.corners
        center = det.center
        tag_id = det.tag_id

    if corners is None or center is None:
        return None

    corners = np.asarray(corners, dtype=float)
    center = np.asarray(center, dtype=float)

    if corners.shape != (4, 2) or center.shape[0] < 2:
        return None

    return corners, float(center[0]), float(center[1]), int(tag_id)


def side_length(p1, p2):
    return math.hypot(float(p2[0] - p1[0]), float(p2[1] - p1[1]))


def wrap_angle_deg(angle):
    while angle > 180.0:
        angle -= 360.0

    while angle < -180.0:
        angle += 360.0

    return angle


def fold_axis_angle_deg(angle):
    angle = wrap_angle_deg(angle)

    if angle > 90.0:
        angle -= 180.0
    elif angle < -90.0:
        angle += 180.0

    return angle


def compute_tag_measurement(det, frame_width, tag_size_cm):
    parts = get_detection_parts(det)
    if parts is None:
        return None

    corners, cx, cy, tag_id = parts

    lb = corners[0]
    rb = corners[1]
    rt = corners[2]
    lt = corners[3]

    left_mid = (lb + lt) * 0.5
    right_mid = (rb + rt) * 0.5

    tag_x_axis = right_mid - left_mid
    raw_yaw_image_deg = math.degrees(math.atan2(tag_x_axis[1], tag_x_axis[0]))
    yaw_error_deg = fold_axis_angle_deg(raw_yaw_image_deg)

    tag_width_px = 0.5 * (side_length(lb, rb) + side_length(lt, rt))
    pixels_per_cm = tag_width_px / tag_size_cm if tag_size_cm > 0.0 else 0.0

    lateral_offset_px = cx - (frame_width * 0.5)
    lateral_offset_norm = lateral_offset_px / (frame_width * 0.5)

    lateral_offset_cm = None
    if pixels_per_cm > 0.0:
        lateral_offset_cm = lateral_offset_px / pixels_per_cm

    return {
        "tag_id": tag_id,
        "corners": corners,
        "center_x_px": cx,
        "center_y_px": cy,
        "lateral_offset_px": lateral_offset_px,
        "lateral_offset_norm": lateral_offset_norm,
        "lateral_offset_cm": lateral_offset_cm,
        "yaw_error_deg": yaw_error_deg,
        "raw_yaw_image_deg": raw_yaw_image_deg,
        "tag_width_px": tag_width_px,
    }


def draw_measurement(frame, measurement):
    corners = measurement["corners"].astype(int)
    cv2.polylines(frame, [corners], True, (0, 255, 0), 2)

    cx = int(measurement["center_x_px"])
    cy = int(measurement["center_y_px"])
    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    frame_h, frame_w = frame.shape[:2]
    cv2.line(frame, (frame_w // 2, 0), (frame_w // 2, frame_h), (255, 255, 0), 1)

    lateral_cm = measurement["lateral_offset_cm"]
    lateral_text = "NA" if lateral_cm is None else f"{lateral_cm:.2f}cm"

    text_1 = (
        f"ID:{measurement['tag_id']} X:{lateral_text} "
        f"Yaw:{measurement['yaw_error_deg']:.2f}deg"
    )
    text_2 = (
        f"px:{measurement['lateral_offset_px']:.1f} "
        f"norm:{measurement['lateral_offset_norm']:.3f}"
    )

    x = int(corners[0][0])
    y = int(corners[0][1])
    cv2.putText(frame, text_1, (x, max(20, y - 28)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)
    cv2.putText(frame, text_2, (x, max(40, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 0, 0), 2)


def draw_status(frame, mega_snapshot):
    motion = mega_snapshot["motion_state"]
    heading = mega_snapshot["H"]
    control = mega_snapshot["E"]
    correction = mega_snapshot["C"]
    left = mega_snapshot["L"]
    right = mega_snapshot["R"]

    line_1 = "f=FORWARD  s=STOP  q=QUIT"
    line_2 = (
        f"{motion} H:{fmt(heading, 2)} E:{fmt(control, 2)} "
        f"C:{fmt(correction, 2)} L:{fmt(left, 1)} R:{fmt(right, 1)}"
    )

    cv2.putText(frame, line_1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 2)
    cv2.putText(frame, line_2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 255), 2)


def read_terminal_command():
    if not sys.stdin.isatty():
        return None

    readable, _, _ = select.select([sys.stdin], [], [], 0)
    if not readable:
        return None

    return sys.stdin.readline().strip().lower()


def snapshot_mega_state(state, state_lock):
    with state_lock:
        return dict(state)


def build_log_row(now_s, frame_index, tag_count, measurement, mega_snapshot):
    mega_time_s = mega_snapshot["time_s"]
    mega_age_s = None
    if mega_time_s is not None:
        mega_age_s = now_s - mega_time_s

    row = {
        "time_s": fmt(now_s),
        "frame_index": frame_index,
        "tag_detected": 1 if measurement is not None else 0,
        "tag_count": tag_count,
        "tag_id": "",
        "lateral_offset_cm": "",
        "lateral_offset_px": "",
        "lateral_offset_norm": "",
        "yaw_error_deg": "",
        "raw_yaw_image_deg": "",
        "center_x_px": "",
        "center_y_px": "",
        "tag_width_px": "",
        "mega_time_s": fmt(mega_time_s),
        "mega_age_s": fmt(mega_age_s),
        "motion_state": mega_snapshot["motion_state"],
        "last_command": mega_snapshot["last_command"],
        "mega_event": mega_snapshot["event"],
        "mega_heading_error_deg": fmt(mega_snapshot["H"]),
        "mega_control_error_deg": fmt(mega_snapshot["E"]),
        "mega_correction_rpm": fmt(mega_snapshot["C"]),
        "mega_left_rpm": fmt(mega_snapshot["L"]),
        "mega_right_rpm": fmt(mega_snapshot["R"]),
        "mega_last_line": mega_snapshot["last_line"],
    }

    if measurement is None:
        return row

    row.update({
        "tag_id": measurement["tag_id"],
        "lateral_offset_cm": fmt(measurement["lateral_offset_cm"]),
        "lateral_offset_px": fmt(measurement["lateral_offset_px"]),
        "lateral_offset_norm": fmt(measurement["lateral_offset_norm"], 5),
        "yaw_error_deg": fmt(measurement["yaw_error_deg"]),
        "raw_yaw_image_deg": fmt(measurement["raw_yaw_image_deg"]),
        "center_x_px": fmt(measurement["center_x_px"]),
        "center_y_px": fmt(measurement["center_y_px"]),
        "tag_width_px": fmt(measurement["tag_width_px"]),
    })

    return row


def print_periodic_status(measurements, mega_snapshot):
    if measurements:
        best = measurements[0]
        lateral = best["lateral_offset_cm"]
        lateral_text = "NA" if lateral is None else f"{lateral:.2f} cm"
        tag_text = (
            f"TAG:{best['tag_id']} lat={lateral_text} "
            f"yaw={best['yaw_error_deg']:.2f} deg"
        )
    else:
        tag_text = "TAG:none"

    mega_text = (
        f"Mega {mega_snapshot['motion_state']} "
        f"H={fmt(mega_snapshot['H'], 2)} "
        f"E={fmt(mega_snapshot['E'], 2)} "
        f"C={fmt(mega_snapshot['C'], 2)} "
        f"L={fmt(mega_snapshot['L'], 1)} "
        f"R={fmt(mega_snapshot['R'], 1)}"
    )
    print(tag_text, "|", mega_text)


def handle_command(command, ser, state, state_lock):
    if command is None:
        return True

    if command in ("f", "forward"):
        send_command(ser, state, state_lock, "FORWARD")
    elif command in ("s", "stop"):
        send_command(ser, state, state_lock, "STOP")
    elif command in ("q", "quit", "exit"):
        send_command(ser, state, state_lock, "STOP")
        return False
    elif command:
        print("Use f/forward, s/stop, or q/quit.")

    return True


def key_to_command(key):
    if key == ord("f"):
        return "f"
    if key == ord("s"):
        return "s"
    if key == ord("q"):
        return "q"
    return None


def camera_device_path(camera_index):
    return f"/dev/video{camera_index}"


def set_v4l2_control(device, name, value):
    command = ["v4l2-ctl", "-d", device, f"--set-ctrl={name}={value}"]

    try:
        result = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError:
        print("v4l2-ctl not found; skipping camera controls.")
        return False

    if result.returncode != 0:
        message = result.stderr.strip() or result.stdout.strip()
        print(f"Could not set {name}={value}: {message}")
        return False

    return True


def apply_v4l2_controls(args):
    if args.skip_v4l2_controls:
        print("Skipping v4l2 camera controls.")
        return

    device = camera_device_path(args.camera)
    controls = [
        ("power_line_frequency", args.power_line_frequency),
        ("exposure_dynamic_framerate", 0),
        ("auto_exposure", 1),
        ("exposure_time_absolute", args.exposure),
    ]

    if args.gain >= 0:
        controls.append(("gain", args.gain))

    print(f"Applying camera controls on {device}...")
    for name, value in controls:
        set_v4l2_control(device, name, value)


def configure_camera(cap, args):
    if args.backend_fourcc:
        fourcc = cv2.VideoWriter_fourcc(*args.backend_fourcc)
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.target_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)


def get_camera_fourcc(cap):
    value = int(cap.get(cv2.CAP_PROP_FOURCC))
    chars = [
        chr(value & 0xFF),
        chr((value >> 8) & 0xFF),
        chr((value >> 16) & 0xFF),
        chr((value >> 24) & 0xFF),
    ]
    return "".join(chars)


def warmup_camera(cap, seconds, max_failures):
    if seconds <= 0.0:
        return

    print(f"Warming camera for {seconds:.1f}s before logging...")
    end_time = time.time() + seconds
    warmup_frames = 0
    consecutive_failures = 0

    while time.time() < end_time:
        ret, _ = cap.read()
        if not ret:
            consecutive_failures += 1
            print(f"Warmup frame failed ({consecutive_failures}/{max_failures})")
            if consecutive_failures >= max_failures:
                print("Warmup stopped because camera failure limit was reached.")
                break
            time.sleep(0.05)
            continue

        warmup_frames += 1
        consecutive_failures = 0

    print(f"Warmup frames discarded: {warmup_frames}")


def main():
    parser = argparse.ArgumentParser(
        description="Log Raspberry Pi AprilTag pose and Mega gyro/motor telemetry."
    )
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--width", type=int, default=DEFAULT_FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_FRAME_HEIGHT)
    parser.add_argument("--target-fps", type=int, default=DEFAULT_TARGET_FPS)
    parser.add_argument("--backend-fourcc", default=DEFAULT_CAMERA_FOURCC,
                        help="Camera pixel format, usually MJPG or YUYV. Use empty string to skip.")
    parser.add_argument("--max-camera-failures", type=int, default=10)
    parser.add_argument("--camera-warmup-sec", type=float, default=DEFAULT_CAMERA_WARMUP_SEC)
    parser.add_argument("--exposure", type=int, default=DEFAULT_EXPOSURE)
    parser.add_argument("--gain", type=int, default=DEFAULT_GAIN,
                        help="Camera gain. Use -1 to leave unchanged.")
    parser.add_argument("--power-line-frequency", type=int, default=DEFAULT_POWER_LINE_FREQUENCY,
                        help="1=50 Hz, 2=60 Hz for most UVC cameras.")
    parser.add_argument("--skip-v4l2-controls", action="store_true")
    parser.add_argument("--port", default=DEFAULT_SERIAL_PORT,
                        help="Mega serial port. Use --port none to disable serial.")
    parser.add_argument("--baud", type=int, default=DEFAULT_BAUD)
    parser.add_argument("--tag-size-cm", type=float, default=DEFAULT_TAG_SIZE_CM)
    parser.add_argument("--log", default=None)
    parser.add_argument("--no-display", action="store_true",
                        help="Do not open the OpenCV preview window.")
    parser.add_argument("--print-interval", type=float, default=0.2)
    args = parser.parse_args()

    log_path = args.log
    if log_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = f"pi_tag_mega_log_{stamp}.csv"

    serial_port = None if args.port.lower() == "none" else args.port
    ser = open_serial(serial_port, args.baud)

    apply_v4l2_controls(args)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        if ser is not None:
            ser.close()
        raise RuntimeError(f"Could not open camera index {args.camera}")
    configure_camera(cap, args)
    warmup_camera(cap, args.camera_warmup_sec, args.max_camera_failures)

    detector = create_detector()

    state = make_initial_mega_state()
    state_lock = threading.Lock()
    stop_event = threading.Event()
    start_time = time.time()

    reader = None
    if ser is not None:
        reader = threading.Thread(
            target=serial_reader,
            args=(ser, stop_event, state, state_lock, start_time),
            daemon=True,
        )
        reader.start()

    print("Controls: f=FORWARD, s=STOP, q=STOP+QUIT")
    print("Enter commands in the terminal, or press keys in the camera window.")
    print("Sign convention:")
    print("  lateral_offset_px/cm positive = tag appears right of image center")
    print("  yaw_error_deg is folded to +/-90 deg, so aligned tags are near 0 deg")
    print("  raw_yaw_image_deg keeps the original image angle for debugging")
    print(
        "Camera: "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"@ {cap.get(cv2.CAP_PROP_FPS):.2f} FPS "
        f"FOURCC={get_camera_fourcc(cap)}"
    )
    print(f"Logging to: {log_path}")

    frame_index = 0
    last_print_time = 0.0
    camera_failures = 0

    try:
        with open(log_path, "w", newline="") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=FIELDNAMES)
            writer.writeheader()

            while not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    camera_failures += 1
                    print(
                        "Camera frame read failed "
                        f"({camera_failures}/{args.max_camera_failures})."
                    )
                    if camera_failures >= args.max_camera_failures:
                        print("Camera failure limit reached.")
                        break
                    time.sleep(0.05)
                    continue

                camera_failures = 0

                frame_index += 1
                now_s = time.time() - start_time

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = detector.detect(gray)
                tag_count = len(detections)
                frame_width = frame.shape[1]

                measurements = []
                for det in detections:
                    measurement = compute_tag_measurement(
                        det, frame_width, args.tag_size_cm
                    )
                    if measurement is None:
                        continue
                    measurements.append(measurement)
                    if not args.no_display:
                        draw_measurement(frame, measurement)

                mega_snapshot = snapshot_mega_state(state, state_lock)

                if measurements:
                    for measurement in measurements:
                        writer.writerow(build_log_row(
                            now_s, frame_index, tag_count, measurement, mega_snapshot
                        ))
                else:
                    writer.writerow(build_log_row(
                        now_s, frame_index, tag_count, None, mega_snapshot
                    ))

                if time.time() - last_print_time >= args.print_interval:
                    print_periodic_status(measurements, mega_snapshot)
                    last_print_time = time.time()

                if not args.no_display:
                    draw_status(frame, mega_snapshot)
                    cv2.imshow("Pi Tag + Mega Logger", frame)
                    window_command = key_to_command(cv2.waitKey(1) & 0xFF)
                    if not handle_command(window_command, ser, state, state_lock):
                        break

                terminal_command = read_terminal_command()
                if not handle_command(terminal_command, ser, state, state_lock):
                    break

    finally:
        stop_event.set()
        if ser is not None:
            send_command(ser, state, state_lock, "STOP")
            if reader is not None:
                reader.join(timeout=1.0)
            ser.close()
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Saved log: {log_path}")


if __name__ == "__main__":
    main()
