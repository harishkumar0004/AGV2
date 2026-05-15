#!/usr/bin/env python3
"""
Manual AGV debug control + compact AprilTag/gyro logger.

Terminal commands:
  f 3      move forward 3 cm
  b 2      move backward 2 cm
  r 90     pivot right 90 degrees
  l 45     pivot left 45 degrees
  s        stop
  q        stop and quit

Upload `03_Manual_debug_distance_angle_control.ino` to the Mega before running.
"""

import argparse
import csv
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import cv2

import Pi_tag_pose_mega_logger as base


FIELDNAMES = [
    "time_s",
    "frame_index",
    "user_input",
    "command_sent",
    "tag_detected",
    "tag_id",
    "tag_yaw_deg",
    "tag_center_x_px",
    "tag_center_y_px",
    "tag_lateral_offset_px",
    "tag_center_deviation_cm",
    "gyro_yaw_deg",
    "gyro_deviation_deg",
    "mega_event",
]


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Manual f/b/r/l AGV debug controller with compact tag/gyro logs."
    )
    parser.add_argument("--camera", type=int, default=base.DEFAULT_CAMERA_INDEX)
    parser.add_argument("--port", default=base.DEFAULT_SERIAL_PORT)
    parser.add_argument("--baud", type=int, default=base.DEFAULT_BAUD)
    parser.add_argument("--tag-size-cm", type=float, default=base.DEFAULT_TAG_SIZE_CM)
    parser.add_argument("--width", type=int, default=base.DEFAULT_FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=base.DEFAULT_FRAME_HEIGHT)
    parser.add_argument("--target-fps", type=int, default=base.DEFAULT_TARGET_FPS)
    parser.add_argument("--backend-fourcc", default=base.DEFAULT_CAMERA_FOURCC)
    parser.add_argument("--max-camera-failures", type=int, default=10)
    parser.add_argument("--camera-warmup-sec", type=float, default=base.DEFAULT_CAMERA_WARMUP_SEC)
    parser.add_argument("--exposure", type=int, default=base.DEFAULT_EXPOSURE)
    parser.add_argument("--gain", type=int, default=base.DEFAULT_GAIN)
    parser.add_argument("--power-line-frequency", type=int, default=base.DEFAULT_POWER_LINE_FREQUENCY)
    parser.add_argument("--skip-v4l2-controls", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--output-dir", default="Pi_csv_outputs")
    return parser


def make_camera_args(args):
    return SimpleNamespace(
        camera=args.camera,
        width=args.width,
        height=args.height,
        target_fps=args.target_fps,
        backend_fourcc=args.backend_fourcc,
        skip_v4l2_controls=args.skip_v4l2_controls,
        exposure=args.exposure,
        gain=args.gain,
        power_line_frequency=args.power_line_frequency,
    )


def log_path(output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"pi_manual_debug_control_log_{stamp}.csv"


def select_best_measurement(detections, frame_width, tag_size_cm):
    measurements = []
    for detection in detections:
        measurement = base.compute_tag_measurement(detection, frame_width, tag_size_cm)
        if measurement is not None:
            measurements.append(measurement)

    if not measurements:
        return None

    measurements.sort(key=lambda item: item["tag_width_px"], reverse=True)
    return measurements[0]


def parse_terminal_command(text):
    if text is None:
        return None, None, False

    raw = text.strip().lower()
    if not raw:
        return None, None, False

    if raw in ("s", "stop"):
        return raw, "STOP", False

    if raw in ("q", "quit", "exit"):
        return raw, "STOP", True

    parts = raw.split()
    if len(parts) != 2:
        print("Use: f <cm>, b <cm>, r <deg>, l <deg>, s, q")
        return raw, None, False

    key, value_text = parts
    try:
        value = float(value_text)
    except ValueError:
        print("Value must be numeric.")
        return raw, None, False

    if value <= 0.0:
        print("Value must be greater than zero.")
        return raw, None, False

    if key == "f":
        return raw, f"FWD_CM:{value:.3f}", False
    if key == "b":
        return raw, f"BACK_CM:{value:.3f}", False
    if key == "r":
        return raw, f"TURN_R_DEG:{value:.3f}", False
    if key == "l":
        return raw, f"TURN_L_DEG:{value:.3f}", False

    print("Use: f <cm>, b <cm>, r <deg>, l <deg>, s, q")
    return raw, None, False


def build_row(now_s, frame_index, user_input, command_sent, measurement, mega_snapshot):
    row = {
        "time_s": base.fmt(now_s),
        "frame_index": frame_index,
        "user_input": user_input or "",
        "command_sent": command_sent or "",
        "tag_detected": 1 if measurement is not None else 0,
        "tag_id": "",
        "tag_yaw_deg": "",
        "tag_center_x_px": "",
        "tag_center_y_px": "",
        "tag_lateral_offset_px": "",
        "tag_center_deviation_cm": "",
        "gyro_yaw_deg": base.fmt(mega_snapshot.get("H")),
        "gyro_deviation_deg": base.fmt(mega_snapshot.get("E")),
        "mega_event": mega_snapshot.get("event", ""),
    }

    if measurement is not None:
        row.update({
            "tag_id": measurement["tag_id"],
            "tag_yaw_deg": base.fmt(measurement["yaw_error_deg"]),
            "tag_center_x_px": base.fmt(measurement["center_x_px"]),
            "tag_center_y_px": base.fmt(measurement["center_y_px"]),
            "tag_lateral_offset_px": base.fmt(measurement["lateral_offset_px"]),
            "tag_center_deviation_cm": base.fmt(measurement["lateral_offset_cm"]),
        })

    return row


def send_manual_command(ser, state, state_lock, command):
    base.send_command(ser, state, state_lock, command)


def quiet_serial_reader(ser, stop_event, state, state_lock, start_time):
    while not stop_event.is_set():
        try:
            line = ser.readline().decode("utf-8", errors="replace").strip()
        except base.serial.SerialException as exc:
            line = f"SERIAL_READ_ERROR:{exc}"
            stop_event.set()

        if not line:
            continue

        with state_lock:
            base.update_mega_state_from_line(state, line, start_time)


def main():
    args = build_arg_parser().parse_args()
    output_path = log_path(args.output_dir)

    serial_port = None if args.port.lower() == "none" else args.port
    ser = base.open_serial(serial_port, args.baud)

    camera_args = make_camera_args(args)
    base.apply_v4l2_controls(camera_args)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        if ser is not None:
            ser.close()
        raise RuntimeError(f"Could not open camera index {args.camera}")

    base.configure_camera(cap, camera_args)
    base.warmup_camera(cap, args.camera_warmup_sec, args.max_camera_failures)
    detector = base.create_detector()

    state = base.make_initial_mega_state()
    state_lock = threading.Lock()
    stop_event = threading.Event()
    start_time = time.time()

    reader = None
    if ser is not None:
        reader = threading.Thread(
            target=quiet_serial_reader,
            args=(ser, stop_event, state, state_lock, start_time),
            daemon=True,
        )
        reader.start()

    print("Commands:")
    print("  f 3    forward 3 cm")
    print("  b 2    backward 2 cm")
    print("  r 90   pivot right 90 deg")
    print("  l 45   pivot left 45 deg")
    print("  s      stop")
    print("  q      stop and quit")
    print(
        "Camera: "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"@ {cap.get(cv2.CAP_PROP_FPS):.2f} FPS "
        f"FOURCC={base.get_camera_fourcc(cap)}"
    )
    print(f"Tag size used for cm conversion: {args.tag_size_cm:.2f} cm")
    print(f"Logging to: {output_path}")

    frame_index = 0
    camera_failures = 0
    last_user_input = ""
    last_command_sent = ""

    try:
        with output_path.open("w", newline="") as log_file:
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
                        send_manual_command(ser, state, state_lock, "STOP")
                        break
                    time.sleep(0.05)
                    continue

                camera_failures = 0
                frame_index += 1
                now_s = time.time() - start_time

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = detector.detect(gray)
                measurement = select_best_measurement(
                    detections,
                    frame_width=frame.shape[1],
                    tag_size_cm=args.tag_size_cm,
                )
                mega_snapshot = base.snapshot_mega_state(state, state_lock)

                text = base.read_terminal_command()
                user_input, command, should_quit = parse_terminal_command(text)
                command_sent_this_frame = ""

                if command:
                    send_manual_command(ser, state, state_lock, command)
                    last_user_input = user_input or ""
                    last_command_sent = command
                    command_sent_this_frame = command

                if not args.no_display:
                    if measurement:
                        base.draw_measurement(frame, measurement)
                    base.draw_status(frame, mega_snapshot)
                    cv2.imshow("Manual Debug Control Logger", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("s"), ord("S")):
                        send_manual_command(ser, state, state_lock, "STOP")
                        last_user_input = "s"
                        last_command_sent = "STOP"
                        command_sent_this_frame = "STOP"
                    elif key in (ord("q"), ord("Q"), 27):
                        send_manual_command(ser, state, state_lock, "STOP")
                        last_user_input = "q"
                        last_command_sent = "STOP"
                        command_sent_this_frame = "STOP"
                        should_quit = True

                row = build_row(
                    now_s,
                    frame_index,
                    command_sent_this_frame and last_user_input,
                    command_sent_this_frame,
                    measurement,
                    mega_snapshot,
                )
                writer.writerow(row)

                if should_quit:
                    break

    finally:
        stop_event.set()
        if ser is not None:
            try:
                send_manual_command(ser, state, state_lock, "STOP")
            except Exception:
                pass
            if reader is not None:
                reader.join(timeout=1.0)
            ser.close()
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Saved log: {output_path}")


if __name__ == "__main__":
    main()
