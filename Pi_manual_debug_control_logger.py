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
import os
import threading
import time
from contextlib import contextmanager
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
    "camera_x_offset_cm",
    "agv_center_deviation_cm",
    "tag_in_gate",
    "tag_gate_dx_px",
    "tag_gate_dy_px",
    "tag_corr_deg",
    "tag_corr_sent",
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
    parser.add_argument("--camera-x-offset-cm", type=float, default=1.5,
                        help=("Signed camera offset from AGV center in cm. "
                              "Positive means camera is mounted to the right of AGV center."))
    parser.add_argument("--pre-turn-forward-cm", type=float, default=2.0,
                        help=("If a tag is visible when a turn is requested, move forward "
                              "this many cm before starting the turn. Use 0 to disable."))
    parser.add_argument("--turn-tag-memory-sec", type=float, default=0.4,
                        help="Treat a tag as recently visible for this long when a turn is requested.")
    parser.add_argument("--post-turn-forward-cm", type=float, default=2.0,
                        help="Automatic forward move after a turn finishes. Use 0 to disable.")
    parser.add_argument("--post-turn-back-cm", type=float, default=0.0,
                        help="Optional backward move after a turn finishes. Default is disabled.")
    parser.add_argument("--gate-width-px", type=int, default=180,
                        help="Centered tag acceptance box width in pixels.")
    parser.add_argument("--gate-height-px", type=int, default=180,
                        help="Centered tag acceptance box height in pixels.")
    parser.add_argument("--tag-k-lat", type=float, default=1.0,
                        help="Heading correction degrees per cm of tag center offset.")
    parser.add_argument("--tag-k-yaw", type=float, default=0.25,
                        help="Heading correction degrees per degree of tag yaw error.")
    parser.add_argument("--tag-command-interval-sec", type=float, default=0.08)
    parser.add_argument("--disable-tag-correction", action="store_true")
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


def compute_gate_info(measurement, frame_width, frame_height, args):
    gate_w = max(1, int(args.gate_width_px))
    gate_h = max(1, int(args.gate_height_px))
    center_x = frame_width * 0.5
    center_y = frame_height * 0.5
    left = center_x - gate_w * 0.5
    right = center_x + gate_w * 0.5
    top = center_y - gate_h * 0.5
    bottom = center_y + gate_h * 0.5

    info = {
        "left": int(left),
        "right": int(right),
        "top": int(top),
        "bottom": int(bottom),
        "inside": False,
        "dx_px": None,
        "dy_px": None,
    }

    if measurement is None:
        return info

    cx = measurement["center_x_px"]
    cy = measurement["center_y_px"]
    info["dx_px"] = cx - center_x
    info["dy_px"] = cy - center_y
    info["inside"] = left <= cx <= right and top <= cy <= bottom
    return info


def agv_center_deviation_cm(measurement, args):
    lateral_cm = measurement.get("lateral_offset_cm")
    if lateral_cm is None:
        lateral_cm = 0.0

    return lateral_cm + args.camera_x_offset_cm


def compute_tag_correction(measurement, args):
    lateral_cm = agv_center_deviation_cm(measurement, args)
    yaw_error_deg = measurement.get("yaw_error_deg") or 0.0
    return (args.tag_k_lat * lateral_cm) + (args.tag_k_yaw * yaw_error_deg)


def draw_gate_box(frame, gate_info):
    color = (0, 180, 255)
    if gate_info.get("inside"):
        color = (0, 220, 0)

    cv2.rectangle(
        frame,
        (gate_info["left"], gate_info["top"]),
        (gate_info["right"], gate_info["bottom"]),
        color,
        2,
    )


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


def build_row(now_s, frame_index, user_input, command_sent, measurement,
              gate_info, tag_corr_deg, tag_corr_sent, mega_snapshot, args):
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
        "camera_x_offset_cm": base.fmt(args.camera_x_offset_cm),
        "agv_center_deviation_cm": "",
        "tag_in_gate": 1 if gate_info.get("inside") else 0,
        "tag_gate_dx_px": base.fmt(gate_info.get("dx_px")),
        "tag_gate_dy_px": base.fmt(gate_info.get("dy_px")),
        "tag_corr_deg": base.fmt(tag_corr_deg),
        "tag_corr_sent": 1 if tag_corr_sent else 0,
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
            "agv_center_deviation_cm": base.fmt(agv_center_deviation_cm(measurement, args)),
        })

    return row


def send_manual_command(ser, state, state_lock, command):
    base.send_command(ser, state, state_lock, command)

    upper_command = command.strip().upper()
    with state_lock:
        if upper_command.startswith(("FWD_CM:", "BACK_CM:")):
            state["motion_state"] = "MOVING"
        elif upper_command.startswith(("TURN_R_DEG:", "TURN_L_DEG:")):
            state["motion_state"] = "TURNING"
        elif upper_command in ("STOP", "CMD:STOP"):
            state["motion_state"] = "STOPPED"


def send_tag_correction(ser, state, state_lock, correction_deg):
    if ser is None:
        return

    command = f"TAG_CORR:{correction_deg:.2f}"
    ser.write((command + "\n").encode("ascii"))
    ser.flush()

    with state_lock:
        state["last_command"] = command



@contextmanager
def suppress_native_stderr():
    saved_stderr_fd = os.dup(2)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 2)
            yield
    finally:
        os.dup2(saved_stderr_fd, 2)
        os.close(saved_stderr_fd)


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
            if line in ("EVT:MOVE_DONE", "EVT:TURN_DONE", "EVT:TURN_TIMEOUT", "ACK:STOP"):
                state["motion_state"] = "STOPPED"
            elif line in ("ACK:FWD_CM", "ACK:BACK_CM"):
                state["motion_state"] = "MOVING"
            elif line in ("ACK:TURN_R_DEG", "ACK:TURN_L_DEG"):
                state["motion_state"] = "TURNING"


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
    with suppress_native_stderr():
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
    print(f"Camera X offset compensation: {args.camera_x_offset_cm:.2f} cm")
    print(f"Pre-turn forward: {args.pre_turn_forward_cm:.2f} cm when tag is visible")
    print(f"Turn tag memory: {args.turn_tag_memory_sec:.2f} sec")
    print(f"Post-turn forward: {args.post_turn_forward_cm:.2f} cm")
    print(f"Post-turn backup: {args.post_turn_back_cm:.2f} cm")
    print(f"Gate box: {args.gate_width_px}x{args.gate_height_px} px at image center")
    print(f"Tag correction: {args.tag_k_lat:.2f}*offset_cm + {args.tag_k_yaw:.2f}*tag_yaw_deg")
    print(f"Logging to: {output_path}")

    frame_index = 0
    camera_failures = 0
    last_user_input = ""
    last_command_sent = ""
    last_tag_command_time = 0.0
    last_tag_seen_time_s = -1.0
    pending_pre_turn_command = ""
    pre_turn_move_time_s = -1.0
    pending_post_turn_forward = False
    pending_post_turn_back = False
    turn_command_time_s = -1.0

    try:
        with output_path.open("w", newline="") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=FIELDNAMES)
            writer.writeheader()

            with suppress_native_stderr():
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
                    gate_info = compute_gate_info(
                        measurement,
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                        args=args,
                    )
                    if measurement is not None:
                        last_tag_seen_time_s = now_s
                    mega_snapshot = base.snapshot_mega_state(state, state_lock)
                    command_sent_this_frame = ""

                    if pending_pre_turn_command:
                        event_time_s = mega_snapshot.get("time_s")
                        if (
                            mega_snapshot.get("event") == "EVT:MOVE_DONE"
                            and event_time_s is not None
                            and event_time_s >= pre_turn_move_time_s
                        ):
                            auto_cmd = pending_pre_turn_command
                            send_manual_command(ser, state, state_lock, auto_cmd)
                            last_user_input = "auto_pre_turn"
                            last_command_sent = auto_cmd
                            command_sent_this_frame = auto_cmd
                            pending_pre_turn_command = ""
                            pending_post_turn_forward = args.post_turn_forward_cm > 0.0
                            pending_post_turn_back = (
                                not pending_post_turn_forward
                                and args.post_turn_back_cm > 0.0
                            )
                            turn_command_time_s = now_s
                        elif (
                            mega_snapshot.get("event") in ("ACK:STOP", "EVT:TURN_TIMEOUT")
                            and event_time_s is not None
                            and event_time_s >= pre_turn_move_time_s
                        ):
                            pending_pre_turn_command = ""
                            pending_post_turn_forward = False
                            pending_post_turn_back = False

                    if pending_post_turn_forward or pending_post_turn_back:
                        event_time_s = mega_snapshot.get("time_s")
                        if (
                            mega_snapshot.get("event") == "EVT:TURN_DONE"
                            and event_time_s is not None
                            and event_time_s >= turn_command_time_s
                        ):
                            if pending_post_turn_forward:
                                auto_cmd = f"FWD_CM:{args.post_turn_forward_cm:.3f}"
                                send_manual_command(ser, state, state_lock, auto_cmd)
                                last_user_input = "auto_post_turn_forward"
                                last_command_sent = auto_cmd
                                command_sent_this_frame = auto_cmd
                            elif pending_post_turn_back:
                                auto_cmd = f"BACK_CM:{args.post_turn_back_cm:.3f}"
                                send_manual_command(ser, state, state_lock, auto_cmd)
                                last_user_input = "auto_post_turn_back"
                                last_command_sent = auto_cmd
                                command_sent_this_frame = auto_cmd
                            pending_post_turn_forward = False
                            pending_post_turn_back = False
                        elif (
                            mega_snapshot.get("event") == "EVT:TURN_TIMEOUT"
                            and event_time_s is not None
                            and event_time_s >= turn_command_time_s
                        ):
                            pending_post_turn_forward = False
                            pending_post_turn_back = False

                    tag_corr_deg = None
                    tag_corr_sent = False
                    if (
                        not args.disable_tag_correction
                        and measurement is not None
                        and gate_info["inside"]
                        and mega_snapshot.get("motion_state") == "MOVING"
                    ):
                        tag_corr_deg = compute_tag_correction(measurement, args)
                        if time.time() - last_tag_command_time >= args.tag_command_interval_sec:
                            send_tag_correction(ser, state, state_lock, tag_corr_deg)
                            last_tag_command_time = time.time()
                            tag_corr_sent = True

                    text = base.read_terminal_command()
                    user_input, command, should_quit = parse_terminal_command(text)

                    if command:
                        upper_command = command.upper()
                        is_turn_command = upper_command.startswith(("TURN_R_DEG:", "TURN_L_DEG:"))
                        has_recent_tag = (
                            measurement is not None
                            or (
                                args.turn_tag_memory_sec > 0.0
                                and last_tag_seen_time_s >= 0.0
                                and now_s - last_tag_seen_time_s <= args.turn_tag_memory_sec
                            )
                        )
                        should_pre_move = (
                            is_turn_command
                            and has_recent_tag
                            and args.pre_turn_forward_cm > 0.0
                        )

                        if should_pre_move:
                            pre_cmd = f"FWD_CM:{args.pre_turn_forward_cm:.3f}"
                            send_manual_command(ser, state, state_lock, pre_cmd)
                            pending_pre_turn_command = command
                            pre_turn_move_time_s = now_s
                            pending_post_turn_forward = False
                            pending_post_turn_back = False
                            last_user_input = user_input or ""
                            last_command_sent = pre_cmd
                            command_sent_this_frame = pre_cmd
                        else:
                            send_manual_command(ser, state, state_lock, command)
                            last_user_input = user_input or ""
                            last_command_sent = command
                            command_sent_this_frame = command

                            if is_turn_command:
                                pending_post_turn_forward = args.post_turn_forward_cm > 0.0
                                pending_post_turn_back = (
                                    not pending_post_turn_forward
                                    and args.post_turn_back_cm > 0.0
                                )
                                turn_command_time_s = now_s
                            elif upper_command == "STOP":
                                pending_pre_turn_command = ""
                                pending_post_turn_forward = False
                                pending_post_turn_back = False

                    if not args.no_display:
                        draw_gate_box(frame, gate_info)
                        if measurement:
                            base.draw_measurement(frame, measurement)
                        base.draw_status(frame, mega_snapshot)
                        cv2.imshow("Manual Debug Control Logger", frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("s"), ord("S")):
                            send_manual_command(ser, state, state_lock, "STOP")
                            pending_pre_turn_command = ""
                            pending_post_turn_forward = False
                            pending_post_turn_back = False
                            last_user_input = "s"
                            last_command_sent = "STOP"
                            command_sent_this_frame = "STOP"
                        elif key in (ord("q"), ord("Q"), 27):
                            send_manual_command(ser, state, state_lock, "STOP")
                            pending_pre_turn_command = ""
                            pending_post_turn_forward = False
                            pending_post_turn_back = False
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
                        gate_info,
                        tag_corr_deg,
                        tag_corr_sent,
                        mega_snapshot,
                        args,
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


