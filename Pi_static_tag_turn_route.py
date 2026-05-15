#!/usr/bin/env python3
"""
Static AprilTag route runner with live tag alignment correction.

Behavior:
  - Between tags, the Mega gyro controller holds the current heading.
  - While a normal path tag is visible, Pi computes TAG_CORR from:
      lateral_offset_cm and yaw_error_deg
    and sends TAG_CORR:<degrees> to the Mega.
  - tag 3 -> right 90 degree turn
  - tag 5 -> right 90 degree turn
  - tag 7 -> stop

Upload `02_Tag_route_with_90_turns.ino` to the Mega before running this script.
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

TURN_TAGS = {
    3: "TURN_RIGHT_90",
    5: "TURN_RIGHT_90",
}
FINAL_TAG_ID = 7
DEFAULT_MIN_STABLE_FRAMES = 1
DEFAULT_TAG_K_LAT_DEG_PER_CM = 1.0
DEFAULT_TAG_K_YAW = 0.25
DEFAULT_TAG_COMMAND_INTERVAL_SEC = 0.08
DEFAULT_DEBUG_PRINT_INTERVAL_SEC = 0.20

ROUTE_FIELDNAMES = [
    "route_state",
    "stable_tag_id",
    "stable_count",
    "route_action",
    "handled_turn_tags",
    "tag_corr_active",
    "tag_corr_sent",
    "tag_corr_raw_deg",
    "tag_corr_cmd_deg",
    "tag_k_lat_deg_per_cm",
    "tag_k_yaw",
] + base.FIELDNAMES


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run static route with tag alignment correction."
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
    parser.add_argument("--min-stable-frames", type=int, default=DEFAULT_MIN_STABLE_FRAMES)
    parser.add_argument("--tag-k-lat", type=float, default=DEFAULT_TAG_K_LAT_DEG_PER_CM,
                        help="Degrees of heading command per cm lateral offset.")
    parser.add_argument("--tag-k-yaw", type=float, default=DEFAULT_TAG_K_YAW,
                        help="Degrees of heading command per degree of tag yaw error.")
    parser.add_argument("--tag-command-interval-sec", type=float,
                        default=DEFAULT_TAG_COMMAND_INTERVAL_SEC)
    parser.add_argument("--debug-print-interval-sec", type=float,
                        default=DEFAULT_DEBUG_PRINT_INTERVAL_SEC)
    parser.add_argument("--output-dir", default="Pi_csv_outputs")
    parser.add_argument("--log-every-frame", action="store_true")
    return parser


def command_from_key(key):
    if key in (ord("f"), ord("F")):
        return "START_ROUTE"
    if key in (ord("s"), ord("S")):
        return "STOP"
    if key in (ord("q"), ord("Q"), 27):
        return "QUIT"
    return None


def normalize_terminal_command(text):
    cmd = text.strip().lower()
    if cmd in ("f", "forward", "start", "route"):
        return "START_ROUTE"
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


def compute_tag_correction(measurement, args):
    lateral_cm = measurement.get("lateral_offset_cm")
    if lateral_cm is None:
        lateral_cm = 0.0

    yaw_error_deg = measurement.get("yaw_error_deg") or 0.0
    raw_cmd = (args.tag_k_lat * lateral_cm) + (args.tag_k_yaw * yaw_error_deg)
    return raw_cmd, raw_cmd


def route_log_path(output_dir):
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return out_dir / f"pi_static_route_log_{stamp}.csv"


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


def send_route_command(ser, state, state_lock, command):
    base.send_command(ser, state, state_lock, command)


def send_tag_correction(ser, state, state_lock, correction_deg):
    command = f"TAG_CORR:{correction_deg:.2f}"

    if ser is None:
        return

    ser.write((command + "\n").encode("ascii"))
    ser.flush()

    with state_lock:
        state["last_command"] = command


def add_route_fields(row, route_state, stable_tag_id, stable_count, route_action,
                     handled_turn_tags, tag_corr_active, tag_corr_sent,
                     tag_corr_raw_deg, tag_corr_cmd_deg, args):
    row.update({
        "route_state": route_state,
        "stable_tag_id": stable_tag_id,
        "stable_count": stable_count,
        "route_action": route_action,
        "handled_turn_tags": ";".join(str(tag) for tag in sorted(handled_turn_tags)),
        "tag_corr_active": 1 if tag_corr_active else 0,
        "tag_corr_sent": 1 if tag_corr_sent else 0,
        "tag_corr_raw_deg": base.fmt(tag_corr_raw_deg),
        "tag_corr_cmd_deg": base.fmt(tag_corr_cmd_deg),
        "tag_k_lat_deg_per_cm": base.fmt(args.tag_k_lat),
        "tag_k_yaw": base.fmt(args.tag_k_yaw),
    })
    return row


def main():
    args = build_arg_parser().parse_args()
    log_path = route_log_path(args.output_dir)

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

    print("Controls: f=START ROUTE, s=STOP, q=STOP+QUIT")
    print("Route: tag 3 -> right 90, tag 5 -> right 90, tag 7 -> stop")
    print("Tag correction:")
    print(f"  raw_cmd = {args.tag_k_lat:.3f} * lateral_cm + {args.tag_k_yaw:.3f} * yaw_deg")
    print("  tag correction command is not degree-limited")
    print("  motor RPM is still limited inside the Mega sketch by INITIAL_RPM/MAX_RPM")
    print("  If correction goes opposite, change --tag-k-lat sign first.")
    print(
        "Camera: "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"@ {cap.get(cv2.CAP_PROP_FPS):.2f} FPS "
        f"FOURCC={base.get_camera_fourcc(cap)}"
    )
    print(f"Tag size used for cm conversion: {args.tag_size_cm:.2f} cm")
    print(f"Logging to: {log_path}")

    route_state = "IDLE"
    handled_turn_tags = set()
    previous_tag_id = None
    stable_tag_id = None
    stable_count = 0
    last_route_tag_action = None
    last_mega_event = ""
    frame_index = 0
    camera_failures = 0
    last_tag_command_time = 0.0
    tag_correction_was_active = False
    last_debug_print_time = 0.0

    try:
        with log_path.open("w", newline="") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=ROUTE_FIELDNAMES)
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
                        send_route_command(ser, state, state_lock, "STOP")
                        break
                    time.sleep(0.05)
                    continue

                camera_failures = 0
                frame_index += 1
                now_s = time.time() - start_time
                now_wall = time.time()
                frame_width = frame.shape[1]

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = detector.detect(gray)
                measurement = select_best_measurement(
                    detections,
                    frame_width=frame_width,
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
                route_action = ""
                tag_corr_raw_deg = None
                tag_corr_cmd_deg = None
                tag_corr_active = False
                tag_corr_sent = False

                mega_event = mega_snapshot.get("event", "")
                if mega_event and mega_event != last_mega_event:
                    last_mega_event = mega_event
                    if route_state == "TURNING" and mega_event == "EVT:TURN_DONE":
                        send_route_command(ser, state, state_lock, "FORWARD")
                        route_state = "MOVING"
                        route_action = "turn_done_forward"
                    elif route_state == "TURNING" and mega_event == "EVT:TURN_TIMEOUT":
                        send_route_command(ser, state, state_lock, "STOP")
                        route_state = "ERROR"
                        route_action = "turn_timeout_stop"

                stable_tag_visible = (
                    route_state == "MOVING"
                    and measurement is not None
                    and stable_count >= args.min_stable_frames
                )

                if stable_tag_visible:
                    tag_action_key = (stable_tag_id, route_state)

                    if stable_tag_id == FINAL_TAG_ID and tag_action_key != last_route_tag_action:
                        send_tag_correction(ser, state, state_lock, 0.0)
                        send_route_command(ser, state, state_lock, "STOP")
                        tag_correction_was_active = False
                        route_state = "DONE"
                        route_action = f"tag_{FINAL_TAG_ID}_stop"
                        last_route_tag_action = tag_action_key

                    elif stable_tag_id in TURN_TAGS and stable_tag_id not in handled_turn_tags:
                        send_tag_correction(ser, state, state_lock, 0.0)
                        turn_command = TURN_TAGS[stable_tag_id]
                        send_route_command(ser, state, state_lock, turn_command)
                        handled_turn_tags.add(stable_tag_id)
                        tag_correction_was_active = False
                        route_state = "TURNING"
                        route_action = f"tag_{stable_tag_id}_{turn_command.lower()}"
                        last_route_tag_action = tag_action_key

                    else:
                        tag_corr_raw_deg, tag_corr_cmd_deg = compute_tag_correction(measurement, args)
                        tag_corr_active = True
                        if now_wall - last_tag_command_time >= args.tag_command_interval_sec:
                            send_tag_correction(ser, state, state_lock, tag_corr_cmd_deg)
                            last_tag_command_time = now_wall
                            tag_corr_sent = True
                            tag_correction_was_active = True
                            route_action = f"tag_{stable_tag_id}_corr"

                        if tag_action_key != last_route_tag_action:
                            print(f"[ROUTE] checkpoint/correction tag {stable_tag_id}")
                            last_route_tag_action = tag_action_key

                elif tag_correction_was_active:
                    send_tag_correction(ser, state, state_lock, 0.0)
                    tag_correction_was_active = False
                    tag_corr_sent = True
                    route_action = "tag_corr_clear"

                terminal_text = base.read_terminal_command()
                terminal_command = normalize_terminal_command(terminal_text) if terminal_text else None

                key_command = None
                if not args.no_display:
                    if measurement:
                        base.draw_measurement(frame, measurement)
                    base.draw_status(frame, mega_snapshot)
                    cv2.putText(
                        frame,
                        f"Route:{route_state} stable:{stable_tag_id} count:{stable_count}",
                        (10, 82),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        frame,
                        f"TagCmd:{base.fmt(tag_corr_cmd_deg, 2)} sent:{1 if tag_corr_sent else 0}",
                        (10, 108),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (255, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow("AGV Static Tag Route", frame)
                    key_command = command_from_key(cv2.waitKey(1) & 0xFF)

                requested_command = terminal_command or key_command
                if requested_command == "START_ROUTE":
                    handled_turn_tags.clear()
                    previous_tag_id = None
                    stable_tag_id = None
                    stable_count = 0
                    last_route_tag_action = None
                    tag_correction_was_active = False
                    send_tag_correction(ser, state, state_lock, 0.0)
                    send_route_command(ser, state, state_lock, "FORWARD")
                    route_state = "MOVING"
                    route_action = "manual_start_forward"
                elif requested_command == "STOP":
                    send_tag_correction(ser, state, state_lock, 0.0)
                    send_route_command(ser, state, state_lock, "STOP")
                    tag_correction_was_active = False
                    route_state = "STOPPED"
                    route_action = "manual_stop"
                elif requested_command == "QUIT":
                    send_tag_correction(ser, state, state_lock, 0.0)
                    send_route_command(ser, state, state_lock, "STOP")
                    route_action = "manual_quit_stop"
                    row = base.build_log_row(now_s, frame_index, len(detections), measurement, mega_snapshot)
                    row = add_route_fields(
                        row, route_state, stable_tag_id, stable_count, route_action,
                        handled_turn_tags, tag_corr_active, tag_corr_sent,
                        tag_corr_raw_deg, tag_corr_cmd_deg, args
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
                        f"state={route_state} tag={stable_tag_id} count={stable_count} "
                        f"lat={lat_text} yaw={yaw_text} "
                        f"raw={base.fmt(tag_corr_raw_deg, 2)} cmd={base.fmt(tag_corr_cmd_deg, 2)} "
                        f"sent={1 if tag_corr_sent else 0} "
                        f"H={base.fmt(mega_snapshot.get('H'), 2)} "
                        f"E={base.fmt(mega_snapshot.get('E'), 2)} "
                        f"T={base.fmt(mega_snapshot.get('T'), 2)} "
                        f"L={base.fmt(mega_snapshot.get('L'), 1)} "
                        f"R={base.fmt(mega_snapshot.get('R'), 1)} "
                        f"action={route_action}"
                    )
                    last_debug_print_time = now_wall

                if measurement or args.log_every_frame or route_action:
                    row = base.build_log_row(now_s, frame_index, len(detections), measurement, mega_snapshot)
                    row = add_route_fields(
                        row, route_state, stable_tag_id, stable_count, route_action,
                        handled_turn_tags, tag_corr_active, tag_corr_sent,
                        tag_corr_raw_deg, tag_corr_cmd_deg, args
                    )
                    writer.writerow(row)

    finally:
        stop_event.set()
        if ser is not None:
            try:
                send_tag_correction(ser, state, state_lock, 0.0)
                send_route_command(ser, state, state_lock, "STOP")
            except Exception:
                pass
            if reader is not None:
                reader.join(timeout=1.0)
            ser.close()
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    print(f"Saved log: {log_path}")


if __name__ == "__main__":
    main()