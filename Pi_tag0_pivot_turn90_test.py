#!/usr/bin/env python3
"""
Single-tag pivot-turn test.

Purpose:
  - Detect tag ID 0.
  - Send TURN_RIGHT_90 to the Mega one time.
  - Log tag extraction values, Mega telemetry, and test actions to CSV.

Upload `02_Tag_route_with_90_turns.ino` to the Mega before running this script.
That sketch performs the actual gyro-based pivot turn after it receives TURN_RIGHT_90.
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

TARGET_TAG_ID = 0
TURN_COMMAND = "TURN_RIGHT_90"
TURN_TARGET_DEG = 90.0
DEFAULT_MIN_STABLE_FRAMES = 1
DEFAULT_DEBUG_PRINT_INTERVAL_SEC = 0.20

FIELDNAMES = [
    "test_state",
    "stable_tag_id",
    "stable_count",
    "test_action",
    "target_tag_id",
    "turn_command_sent",
    "turn_target_deg",
] + base.FIELDNAMES


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Detect tag 0 and command one right 90 degree pivot turn."
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
    parser.add_argument("--auto-arm", action="store_true",
                        help="Immediately arm the test instead of waiting for f.")
    parser.add_argument("--min-stable-frames", type=int, default=DEFAULT_MIN_STABLE_FRAMES)
    parser.add_argument("--debug-print-interval-sec", type=float,
                        default=DEFAULT_DEBUG_PRINT_INTERVAL_SEC)
    parser.add_argument("--output-dir", default="Pi_csv_outputs")
    parser.add_argument("--log-every-frame", action="store_true")
    return parser


def command_from_key(key):
    if key in (ord("f"), ord("F")):
        return "ARM"
    if key in (ord("s"), ord("S")):
        return "STOP"
    if key in (ord("q"), ord("Q"), 27):
        return "QUIT"
    return None


def normalize_terminal_command(text):
    cmd = text.strip().lower()
    if cmd in ("f", "arm", "start", "turn"):
        return "ARM"
    if cmd in ("s", "stop"):
        return "STOP"
    if cmd in ("q", "quit", "exit"):
        return "QUIT"
    return None


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


def update_stable_tag(current_tag_id, previous_tag_id, stable_count):
    if current_tag_id is None:
        return None, 0

    if current_tag_id == previous_tag_id:
        return previous_tag_id, stable_count + 1

    return current_tag_id, 1


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
    return out_dir / f"pi_tag0_pivot_turn90_test_{stamp}.csv"


def add_test_fields(row, test_state, stable_tag_id, stable_count, test_action,
                    turn_command_sent):
    row.update({
        "test_state": test_state,
        "stable_tag_id": stable_tag_id,
        "stable_count": stable_count,
        "test_action": test_action,
        "target_tag_id": TARGET_TAG_ID,
        "turn_command_sent": 1 if turn_command_sent else 0,
        "turn_target_deg": base.fmt(TURN_TARGET_DEG, 1),
    })
    return row


def send_test_command(ser, state, state_lock, command):
    base.send_command(ser, state, state_lock, command)


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
            target=base.serial_reader,
            args=(ser, stop_event, state, state_lock, start_time),
            daemon=True,
        )
        reader.start()

    test_state = "ARMED" if args.auto_arm else "WAIT_ARM"
    previous_tag_id = None
    stable_tag_id = None
    stable_count = 0
    last_mega_event = ""
    frame_index = 0
    camera_failures = 0
    last_debug_print_time = 0.0

    print("Controls: f=ARM TEST, s=STOP, q=STOP+QUIT")
    print(f"Target: detect tag {TARGET_TAG_ID}, then send {TURN_COMMAND}")
    print(f"Min stable frames: {args.min_stable_frames}")
    print(
        "Camera: "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"@ {cap.get(cv2.CAP_PROP_FPS):.2f} FPS "
        f"FOURCC={base.get_camera_fourcc(cap)}"
    )
    print(f"Tag size used for cm conversion: {args.tag_size_cm:.2f} cm")
    print(f"Logging to: {output_path}")

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
                        send_test_command(ser, state, state_lock, "STOP")
                        break
                    time.sleep(0.05)
                    continue

                camera_failures = 0
                frame_index += 1
                now_s = time.time() - start_time
                now_wall = time.time()

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = detector.detect(gray)
                measurement = select_best_measurement(
                    detections,
                    frame_width=frame.shape[1],
                    tag_size_cm=args.tag_size_cm,
                )

                current_tag_id = measurement["tag_id"] if measurement else None
                stable_tag_id, stable_count = update_stable_tag(
                    current_tag_id,
                    previous_tag_id,
                    stable_count,
                )
                previous_tag_id = stable_tag_id

                mega_snapshot = base.snapshot_mega_state(state, state_lock)
                test_action = ""
                turn_command_sent = False

                mega_event = mega_snapshot.get("event", "")
                if mega_event and mega_event != last_mega_event:
                    last_mega_event = mega_event
                    if test_state == "TURNING" and mega_event == "EVT:TURN_DONE":
                        test_state = "DONE"
                        test_action = "turn_done"
                    elif test_state == "TURNING" and mega_event == "EVT:TURN_TIMEOUT":
                        test_state = "ERROR"
                        test_action = "turn_timeout"

                if (
                    test_state == "ARMED"
                    and stable_tag_id == TARGET_TAG_ID
                    and stable_count >= args.min_stable_frames
                ):
                    send_test_command(ser, state, state_lock, TURN_COMMAND)
                    test_state = "TURNING"
                    test_action = "tag_0_turn_right_90"
                    turn_command_sent = True

                terminal_text = base.read_terminal_command()
                terminal_command = normalize_terminal_command(terminal_text) if terminal_text else None

                key_command = None
                if not args.no_display:
                    if measurement:
                        base.draw_measurement(frame, measurement)
                    base.draw_status(frame, mega_snapshot)
                    cv2.putText(
                        frame,
                        f"Test:{test_state} stable:{stable_tag_id} count:{stable_count}",
                        (10, 82),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("Tag 0 Pivot Turn 90 Test", frame)
                    key_command = command_from_key(cv2.waitKey(1) & 0xFF)

                requested_command = terminal_command or key_command
                if requested_command == "ARM":
                    previous_tag_id = None
                    stable_tag_id = None
                    stable_count = 0
                    test_state = "ARMED"
                    test_action = "manual_arm"
                    print("[TEST] armed")
                elif requested_command == "STOP":
                    send_test_command(ser, state, state_lock, "STOP")
                    test_state = "STOPPED"
                    test_action = "manual_stop"
                elif requested_command == "QUIT":
                    send_test_command(ser, state, state_lock, "STOP")
                    test_action = "manual_quit_stop"
                    row = base.build_log_row(now_s, frame_index, len(detections), measurement, mega_snapshot)
                    row = add_test_fields(
                        row, test_state, stable_tag_id, stable_count,
                        test_action, turn_command_sent,
                    )
                    writer.writerow(row)
                    break

                if now_wall - last_debug_print_time >= args.debug_print_interval_sec:
                    lat_text = ""
                    yaw_text = ""
                    if measurement is not None:
                        lat_text = base.fmt(measurement.get("lateral_offset_cm"), 2)
                        yaw_text = base.fmt(measurement.get("yaw_error_deg"), 2)
                    print(
                        "[DBG] "
                        f"state={test_state} tag={stable_tag_id} count={stable_count} "
                        f"lat={lat_text} yaw={yaw_text} "
                        f"sent={1 if turn_command_sent else 0} "
                        f"H={base.fmt(mega_snapshot.get('H'), 2)} "
                        f"E={base.fmt(mega_snapshot.get('E'), 2)} "
                        f"C={base.fmt(mega_snapshot.get('C'), 2)} "
                        f"L={base.fmt(mega_snapshot.get('L'), 1)} "
                        f"R={base.fmt(mega_snapshot.get('R'), 1)} "
                        f"event={mega_snapshot.get('event')} "
                        f"action={test_action}"
                    )
                    last_debug_print_time = now_wall

                if measurement or args.log_every_frame or test_action:
                    row = base.build_log_row(now_s, frame_index, len(detections), measurement, mega_snapshot)
                    row = add_test_fields(
                        row, test_state, stable_tag_id, stable_count,
                        test_action, turn_command_sent,
                    )
                    writer.writerow(row)

    finally:
        stop_event.set()
        if ser is not None:
            try:
                send_test_command(ser, state, state_lock, "STOP")
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
