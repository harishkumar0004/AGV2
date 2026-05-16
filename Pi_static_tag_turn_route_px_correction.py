#!/usr/bin/env python3
"""
Static AprilTag route runner with pixel-based live tag alignment correction.

Behavior:
  - Between tags, the Mega gyro controller holds the current heading.
  - Between tags, the Mega gyro controller holds heading.
  - When a normal path tag appears inside the gate, Pi sends TAG_CORR from
    the tag pixel offset. Negative pixel offset commands a right correction,
    which should make left RPM greater than right RPM.
  - tag 3 -> pre-turn forward, right 90 degree pivot, post-turn forward
  - tag 5 -> pre-turn forward, right 90 degree pivot, post-turn forward
  - tag 7 -> stop

Upload `03_Manual_debug_distance_angle_control.ino` to the Mega before running this script.
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

TURN_TAGS = {
    3: "RIGHT",
    5: "RIGHT",
}
PRE_TURN_SLOWDOWN_TAGS = {tag_id - 1: tag_id for tag_id in TURN_TAGS}
FINAL_TAG_ID = 7
DEFAULT_MIN_STABLE_FRAMES = 1
DEFAULT_TAG_K_LAT_DEG_PER_CM = -1.0
DEFAULT_TAG_K_PX_DEG_PER_PX = 0.02
DEFAULT_TAG_K_YAW = 0.0
DEFAULT_TAG_COMMAND_INTERVAL_SEC = 0.08
DEFAULT_TAG_CORRECTION_HOLD_SEC = 0.30
DEFAULT_DEBUG_PRINT_INTERVAL_SEC = 0.0
DEFAULT_CAMERA_X_OFFSET_CM = 0.0
DEFAULT_NORMAL_MOVE_RPM = 20.0
DEFAULT_TURN_APPROACH_RPM = 6.0
DEFAULT_TURN_CENTER_TOLERANCE_PX = 15.0
DEFAULT_TURN_ALIGN_STEP_CM = 1.0
DEFAULT_TURN_ALIGN_TIMEOUT_SEC = 8.0

ROUTE_FIELDNAMES = [
    "route_state",
    "route_active",
    "stable_tag_id",
    "stable_count",
    "route_action",
    "handled_turn_tags",
    "tag_corr_active",
    "tag_corr_sent",
    "tag_corr_raw_deg",
    "tag_corr_cmd_deg",
    "camera_x_offset_cm",
    "agv_center_deviation_cm",
    "tag_in_gate",
    "tag_gate_dx_px",
    "tag_gate_dy_px",
    "tag_k_lat_deg_per_cm",
    "tag_k_px_deg_per_px",
    "tag_k_yaw",
    "pending_turn_tag_id",
    "pending_turn_command",
    "turn_centered",
    "turn_center_tolerance_px",
    "turn_align_step_cm",
    "turn_align_timeout_sec",
    "move_rpm_command",
    "pre_turn_slowdown_active",
    "next_turn_tag_id",
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
    parser.add_argument("--forward-chunk-cm", type=float, default=300.0,
                        help=("Forward distance command used between tags. The Pi sends a new "
                              "chunk if this finishes before the final tag is found."))
    parser.add_argument("--pre-turn-forward-cm", type=float, default=2.0,
                        help="Move forward this many cm after detecting a turn tag, before pivoting.")
    parser.add_argument("--post-turn-forward-cm", type=float, default=2.0,
                        help="Move forward this many cm after the pivot turn finishes.")
    parser.add_argument("--turn-deg", type=float, default=90.0,
                        help="Pivot angle used for turn tags.")
    parser.add_argument("--normal-move-rpm", type=float, default=DEFAULT_NORMAL_MOVE_RPM,
                        help="Mega MOVE_RPM used for normal forward chunks.")
    parser.add_argument("--turn-approach-rpm", type=float, default=DEFAULT_TURN_APPROACH_RPM,
                        help="Mega MOVE_RPM used after the checkpoint before a turn, while centering on a turn tag, and during 2 cm approach moves.")
    parser.add_argument("--disable-pre-turn-slowdown", action="store_true",
                        help="Do not slow at the checkpoint before turn tags. By default, tag 2 slows for tag 3 and tag 4 slows for tag 5.")
    parser.add_argument("--turn-center-tolerance-px", type=float, default=DEFAULT_TURN_CENTER_TOLERANCE_PX,
                        help="Turn tag is considered centered when abs(tag center x - image center x) is below this many pixels.")
    parser.add_argument("--turn-align-step-cm", type=float, default=DEFAULT_TURN_ALIGN_STEP_CM,
                        help="Small slow forward step used while aligning to a turn tag.")
    parser.add_argument("--turn-align-timeout-sec", type=float, default=DEFAULT_TURN_ALIGN_TIMEOUT_SEC,
                        help="Stop with ERROR if turn-tag centering takes longer than this.")
    parser.add_argument("--camera-x-offset-cm", type=float, default=DEFAULT_CAMERA_X_OFFSET_CM,
                        help="Only logged in this pixel-correction variant; not used for correction.")
    parser.add_argument("--tag-k-lat", type=float, default=DEFAULT_TAG_K_LAT_DEG_PER_CM,
                        help="Legacy cm gain, not used in this pixel-correction variant.")
    parser.add_argument("--tag-k-px", type=float, default=DEFAULT_TAG_K_PX_DEG_PER_PX,
                        help=("Degrees of TAG_CORR per pixel offset. Default is negative, so "
                              "negative px sends positive TAG_CORR, producing right RPM > left RPM."))
    parser.add_argument("--tag-k-yaw", type=float, default=DEFAULT_TAG_K_YAW,
                        help=("Degrees of heading command per degree of tag yaw error. Default is 0 "
                              "because turn tags appear near +/-90 deg after a pivot."))
    parser.add_argument("--tag-command-interval-sec", type=float,
                        default=DEFAULT_TAG_COMMAND_INTERVAL_SEC)
    parser.add_argument("--tag-correction-hold-sec", type=float,
                        default=DEFAULT_TAG_CORRECTION_HOLD_SEC,
                        help="Keep the last tag correction alive for this long after losing the tag.")
    parser.add_argument("--disable-straight-tag-correction", action="store_true",
                        help="Disable pixel-based TAG_CORR on normal straight checkpoint tags.")
    parser.add_argument("--gate-width-px", type=int, default=320,
                        help="Centered tag acceptance box width in pixels.")
    parser.add_argument("--gate-height-px", type=int, default=320,
                        help="Centered tag acceptance box height in pixels.")
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

    parts = cmd.split()
    if len(parts) != 2:
        print("Use: f, f <cm>, b <cm>, r <deg>, l <deg>, s, q")
        return None

    key, value_text = parts
    try:
        value = float(value_text)
    except ValueError:
        print("Value must be numeric. Example: f 50")
        return None

    if value <= 0.0:
        print("Value must be greater than zero.")
        return None

    if key in ("f", "forward"):
        return f"FWD_CM:{value:.3f}"
    if key in ("b", "back", "backward"):
        return f"BACK_CM:{value:.3f}"
    if key in ("r", "right"):
        return f"TURN_R_DEG:{value:.3f}"
    if key in ("l", "left"):
        return f"TURN_L_DEG:{value:.3f}"

    print("Use: f, f <cm>, b <cm>, r <deg>, l <deg>, s, q")
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


def update_stable_tag(current_tag_id, previous_tag_id, stable_count):
    if current_tag_id is None:
        return None, 0

    if current_tag_id == previous_tag_id:
        return previous_tag_id, stable_count + 1

    return current_tag_id, 1


def compute_tag_correction(measurement, args):
    lateral_px = measurement.get("lateral_offset_px") or 0.0
    yaw_error_deg = measurement.get("yaw_error_deg") or 0.0

    # Sign convention for this AGV after requested reversal:
    #   negative px -> positive TAG_CORR -> right RPM greater than left RPM
    #   positive px -> negative TAG_CORR -> left RPM greater than right RPM
    raw_cmd = (args.tag_k_px * lateral_px) + (args.tag_k_yaw * yaw_error_deg)
    return raw_cmd, raw_cmd


def agv_center_deviation_cm(measurement, args):
    if measurement is None:
        return None

    lateral_cm = measurement.get("lateral_offset_cm")
    if lateral_cm is None:
        lateral_cm = 0.0

    return lateral_cm + args.camera_x_offset_cm


def turn_command_from_direction(direction, degrees):
    if direction == "LEFT":
        return f"TURN_L_DEG:{degrees:.3f}"

    return f"TURN_R_DEG:{degrees:.3f}"


def is_turn_tag_centered(gate_info, args):
    dx_px = gate_info.get("dx_px")
    if dx_px is None:
        return False

    return abs(dx_px) <= args.turn_center_tolerance_px


def send_move_rpm(ser, state, state_lock, rpm):
    command = f"MOVE_RPM:{rpm:.2f}"
    send_route_command(ser, state, state_lock, command)
    return command


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


def send_route_command(ser, state, state_lock, command):
    base.send_command(ser, state, state_lock, command)

    upper_command = command.strip().upper()
    with state_lock:
        if upper_command.startswith(("FWD_CM:", "BACK_CM:")):
            state["motion_state"] = "MOVING"
        elif upper_command.startswith(("TURN_R_DEG:", "TURN_L_DEG:")):
            state["motion_state"] = "TURNING"
        elif upper_command in ("STOP", "CMD:STOP"):
            state["motion_state"] = "STOPPED"


def route_serial_reader(ser, stop_event, state, state_lock, start_time):
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
            if not line.startswith("TEL:"):
                state["event_time_s"] = state.get("time_s")
                state["event_seq"] = state.get("event_seq", 0) + 1

            if line in ("EVT:MOVE_DONE", "EVT:TURN_DONE", "EVT:TURN_TIMEOUT", "ACK:STOP"):
                state["motion_state"] = "STOPPED"
            elif line in ("ACK:FWD_CM", "ACK:BACK_CM"):
                state["motion_state"] = "MOVING"
            elif line in ("ACK:TURN_R_DEG", "ACK:TURN_L_DEG"):
                state["motion_state"] = "TURNING"

        if not line.startswith("TEL:"):
            print("[MEGA]", line)


def send_tag_correction(ser, state, state_lock, correction_deg):
    command = f"TAG_CORR:{correction_deg:.2f}"

    if ser is None:
        return

    ser.write((command + "\n").encode("ascii"))
    ser.flush()

    with state_lock:
        state["last_command"] = command



def add_route_fields(row, route_state, route_active, stable_tag_id, stable_count, route_action,
                     handled_turn_tags, tag_corr_active, tag_corr_sent,
                     tag_corr_raw_deg, tag_corr_cmd_deg, measurement,
                     gate_info, pending_turn_tag_id, pending_turn_command,
                     turn_centered, move_rpm_command, pre_turn_slowdown_active,
                     next_turn_tag_id, args):
    row.update({
        "route_state": route_state,
        "route_active": 1 if route_active else 0,
        "stable_tag_id": stable_tag_id,
        "stable_count": stable_count,
        "route_action": route_action,
        "handled_turn_tags": ";".join(str(tag) for tag in sorted(handled_turn_tags)),
        "tag_corr_active": 1 if tag_corr_active else 0,
        "tag_corr_sent": 1 if tag_corr_sent else 0,
        "tag_corr_raw_deg": base.fmt(tag_corr_raw_deg),
        "tag_corr_cmd_deg": base.fmt(tag_corr_cmd_deg),
        "camera_x_offset_cm": base.fmt(args.camera_x_offset_cm),
        "agv_center_deviation_cm": base.fmt(agv_center_deviation_cm(measurement, args)),
        "tag_in_gate": 1 if gate_info.get("inside") else 0,
        "tag_gate_dx_px": base.fmt(gate_info.get("dx_px")),
        "tag_gate_dy_px": base.fmt(gate_info.get("dy_px")),
        "tag_k_lat_deg_per_cm": base.fmt(args.tag_k_lat),
        "tag_k_px_deg_per_px": base.fmt(args.tag_k_px),
        "tag_k_yaw": base.fmt(args.tag_k_yaw),
        "pending_turn_tag_id": pending_turn_tag_id if pending_turn_tag_id is not None else "",
        "pending_turn_command": pending_turn_command,
        "turn_centered": 1 if turn_centered else 0,
        "turn_center_tolerance_px": base.fmt(args.turn_center_tolerance_px),
        "turn_align_step_cm": base.fmt(args.turn_align_step_cm),
        "turn_align_timeout_sec": base.fmt(args.turn_align_timeout_sec),
        "move_rpm_command": move_rpm_command,
        "pre_turn_slowdown_active": 1 if pre_turn_slowdown_active else 0,
        "next_turn_tag_id": next_turn_tag_id if next_turn_tag_id is not None else "",
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
            target=route_serial_reader,
            args=(ser, stop_event, state, state_lock, start_time),
            daemon=True,
        )
        reader.start()

    print("Controls: f=START ROUTE, f <cm>=manual forward, b <cm>=manual backward, r <deg>, l <deg>, s=STOP, q=STOP+QUIT")
    print(f"Route: tag 3 -> right {args.turn_deg:.1f}, tag 5 -> right {args.turn_deg:.1f}, tag 7 -> stop")
    print(f"Forward chunk: {args.forward_chunk_cm:.1f} cm")
    print(f"Pre-turn forward: {args.pre_turn_forward_cm:.1f} cm")
    print(f"Post-turn forward: {args.post_turn_forward_cm:.1f} cm")
    print(f"Normal move RPM: {args.normal_move_rpm:.1f}")
    print(f"Turn approach/alignment RPM: {args.turn_approach_rpm:.1f}")
    if not args.disable_pre_turn_slowdown:
        print("Pre-turn slowdown: tag 2 slows before tag 3, tag 4 slows before tag 5")
    print(f"Turn centering: abs(dx) <= {args.turn_center_tolerance_px:.1f} px, step {args.turn_align_step_cm:.1f} cm")
    print(f"Gate box: {args.gate_width_px}x{args.gate_height_px} px at image center")
    if not args.disable_straight_tag_correction:
        print("Straight tag correction: ENABLED with pixel offset")
        print(
            f"  raw_cmd = {args.tag_k_px:.4f} * lateral_px "
            f"+ {args.tag_k_yaw:.3f} * yaw_deg"
        )
        print("  default yaw gain is 0, so post-turn +/-90 deg tag yaw does not create a huge correction")
        print("  negative px -> positive TAG_CORR -> right RPM greater than left RPM")
        print("  tag correction command is not degree-limited")
        print("  motor RPM is still limited inside the Mega sketch by INITIAL_RPM/MAX_RPM")
        print("  If correction goes opposite, change --tag-k-px sign first.")
        print(f"  tag correction hold after losing tag: {args.tag_correction_hold_sec:.2f} sec")
    else:
        print("Straight tag correction: DISABLED. Gyro holds heading between tags.")
        print("  Normal tags are logged/checkpoints only; turn tags still run the 2cm + turn + 2cm sequence.")
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
    route_active = False
    handled_turn_tags = set()
    slowed_for_turn_tags = set()
    previous_tag_id = None
    stable_tag_id = None
    stable_count = 0
    last_route_tag_action = None
    last_mega_event_seq = 0
    frame_index = 0
    camera_failures = 0
    last_tag_command_time = 0.0
    tag_correction_was_active = False
    last_debug_print_time = 0.0
    pending_turn_tag_id = None
    pending_turn_command = ""
    pending_turn_start_time_s = -1.0
    pending_post_turn_forward = False
    route_step_start_time_s = -1.0
    turn_align_start_time_s = -1.0

    try:
        with log_path.open("w", newline="") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=ROUTE_FIELDNAMES)
            writer.writeheader()

            while not stop_event.is_set():
                with suppress_native_stderr():
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
                frame_height = frame.shape[0]

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = detector.detect(gray)
                measurement = select_best_measurement(
                    detections,
                    frame_width=frame_width,
                    tag_size_cm=args.tag_size_cm,
                )
                gate_info = compute_gate_info(
                    measurement,
                    frame_width=frame_width,
                    frame_height=frame_height,
                    args=args,
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
                turn_centered = is_turn_tag_centered(gate_info, args)
                move_rpm_command = ""
                pre_turn_slowdown_active = False
                next_turn_tag_id = None

                mega_event = mega_snapshot.get("event", "")
                mega_event_seq = mega_snapshot.get("event_seq", 0)
                if mega_event and mega_event_seq > last_mega_event_seq:
                    last_mega_event_seq = mega_event_seq
                    event_time_s = mega_snapshot.get("event_time_s", mega_snapshot.get("time_s"))

                    if (
                        route_active
                        and route_state == "ALIGN_TURN_STEP"
                        and mega_event == "EVT:MOVE_DONE"
                        and event_time_s is not None
                        and event_time_s >= route_step_start_time_s
                    ):
                        route_state = "ALIGN_TURN_TAG"
                        route_step_start_time_s = now_s
                        route_action = "turn_align_step_done"

                    elif (
                        route_active
                        and route_state == "PRE_TURN_FORWARD"
                        and mega_event == "EVT:MOVE_DONE"
                        and event_time_s is not None
                        and event_time_s >= route_step_start_time_s
                    ):
                        send_route_command(ser, state, state_lock, pending_turn_command)
                        route_state = "TURNING"
                        pending_turn_start_time_s = now_s
                        route_step_start_time_s = now_s
                        route_action = f"pre_turn_done_{pending_turn_command.lower()}"

                    elif (
                        route_active
                        and route_state == "TURNING"
                        and mega_event == "EVT:TURN_DONE"
                        and event_time_s is not None
                        and event_time_s >= pending_turn_start_time_s
                    ):
                        if args.post_turn_forward_cm > 0.0:
                            post_cmd = f"FWD_CM:{args.post_turn_forward_cm:.3f}"
                            send_route_command(ser, state, state_lock, post_cmd)
                            route_state = "POST_TURN_FORWARD"
                            route_step_start_time_s = now_s
                            route_action = "turn_done_post_forward"
                        else:
                            move_rpm_command = send_move_rpm(ser, state, state_lock, args.normal_move_rpm)
                            next_cmd = f"FWD_CM:{args.forward_chunk_cm:.3f}"
                            send_route_command(ser, state, state_lock, next_cmd)
                            route_state = "MOVING"
                            route_step_start_time_s = now_s
                            pending_turn_tag_id = None
                            pending_turn_command = ""
                            pending_post_turn_forward = False
                            route_action = "turn_done_forward_chunk"

                    elif (
                        route_active
                        and route_state == "POST_TURN_FORWARD"
                        and mega_event == "EVT:MOVE_DONE"
                        and event_time_s is not None
                        and event_time_s >= route_step_start_time_s
                    ):
                        move_rpm_command = send_move_rpm(ser, state, state_lock, args.normal_move_rpm)
                        next_cmd = f"FWD_CM:{args.forward_chunk_cm:.3f}"
                        send_route_command(ser, state, state_lock, next_cmd)
                        route_state = "MOVING"
                        route_step_start_time_s = now_s
                        pending_turn_tag_id = None
                        pending_turn_command = ""
                        pending_post_turn_forward = False
                        route_action = "post_turn_done_forward_chunk"

                    elif (
                        route_active
                        and route_state == "MOVING"
                        and mega_event == "EVT:MOVE_DONE"
                        and event_time_s is not None
                        and event_time_s >= route_step_start_time_s
                    ):
                        move_rpm_command = send_move_rpm(ser, state, state_lock, args.normal_move_rpm)
                        next_cmd = f"FWD_CM:{args.forward_chunk_cm:.3f}"
                        send_route_command(ser, state, state_lock, next_cmd)
                        route_step_start_time_s = now_s
                        route_action = "forward_chunk_restart"

                    elif (
                        not route_active
                        and route_state in ("MOVING", "PRE_TURN_FORWARD", "POST_TURN_FORWARD")
                        and mega_event == "EVT:MOVE_DONE"
                        and event_time_s is not None
                        and event_time_s >= route_step_start_time_s
                    ):
                        route_state = "STOPPED"
                        route_action = "manual_move_done"

                    elif (
                        not route_active
                        and route_state == "TURNING"
                        and mega_event == "EVT:TURN_DONE"
                        and event_time_s is not None
                        and event_time_s >= route_step_start_time_s
                    ):
                        route_state = "STOPPED"
                        route_action = "manual_turn_done"

                    elif route_state == "TURNING" and mega_event == "EVT:TURN_TIMEOUT":
                        send_route_command(ser, state, state_lock, "STOP")
                        route_state = "ERROR"
                        route_active = False
                        pending_turn_tag_id = None
                        pending_turn_command = ""
                        pending_post_turn_forward = False
                        route_action = "turn_timeout_stop"

                tag_visible_in_gate = (
                    measurement is not None
                    and gate_info["inside"]
                    and stable_count >= args.min_stable_frames
                )
                stable_tag_visible = route_state == "MOVING" and tag_visible_in_gate

                if stable_tag_visible:
                    tag_action_key = (stable_tag_id, route_state)

                    if route_active and stable_tag_id == FINAL_TAG_ID and tag_action_key != last_route_tag_action:
                        send_tag_correction(ser, state, state_lock, 0.0)
                        send_route_command(ser, state, state_lock, "STOP")
                        tag_correction_was_active = False
                        route_state = "DONE"
                        route_active = False
                        route_action = f"tag_{FINAL_TAG_ID}_stop"
                        last_route_tag_action = tag_action_key

                    elif route_active and stable_tag_id in TURN_TAGS and stable_tag_id not in handled_turn_tags:
                        send_tag_correction(ser, state, state_lock, 0.0)
                        move_rpm_command = send_move_rpm(ser, state, state_lock, args.turn_approach_rpm)
                        send_route_command(ser, state, state_lock, "STOP")
                        pending_turn_tag_id = stable_tag_id
                        pending_turn_command = turn_command_from_direction(
                            TURN_TAGS[stable_tag_id],
                            args.turn_deg,
                        )
                        route_state = "ALIGN_TURN_TAG"
                        route_step_start_time_s = now_s
                        turn_align_start_time_s = now_s
                        route_action = f"tag_{stable_tag_id}_align_start"
                        tag_correction_was_active = False
                        last_route_tag_action = tag_action_key

                    else:
                        next_turn_tag_id = PRE_TURN_SLOWDOWN_TAGS.get(stable_tag_id)
                        if (
                            route_active
                            and not args.disable_pre_turn_slowdown
                            and next_turn_tag_id is not None
                            and next_turn_tag_id not in handled_turn_tags
                            and next_turn_tag_id not in slowed_for_turn_tags
                        ):
                            move_rpm_command = send_move_rpm(ser, state, state_lock, args.turn_approach_rpm)
                            slowed_for_turn_tags.add(next_turn_tag_id)
                            pre_turn_slowdown_active = True
                            route_action = f"tag_{stable_tag_id}_slow_for_tag_{next_turn_tag_id}"

                        if not args.disable_straight_tag_correction:
                            tag_corr_raw_deg, tag_corr_cmd_deg = compute_tag_correction(measurement, args)
                            tag_corr_active = True
                            force_tag_command = tag_action_key != last_route_tag_action
                            if force_tag_command or now_wall - last_tag_command_time >= args.tag_command_interval_sec:
                                send_tag_correction(ser, state, state_lock, tag_corr_cmd_deg)
                                last_tag_command_time = now_wall
                                tag_corr_sent = True
                                tag_correction_was_active = True
                                if not route_action:
                                    route_action = f"tag_{stable_tag_id}_corr"
                        elif tag_action_key != last_route_tag_action and not route_action:
                            route_action = f"tag_{stable_tag_id}_checkpoint"

                        if tag_action_key != last_route_tag_action:
                            print(f"[ROUTE] checkpoint/correction tag {stable_tag_id}")
                            last_route_tag_action = tag_action_key

                elif route_active and route_state == "ALIGN_TURN_TAG":
                    if (
                        turn_align_start_time_s >= 0.0
                        and now_s - turn_align_start_time_s > args.turn_align_timeout_sec
                    ):
                        send_tag_correction(ser, state, state_lock, 0.0)
                        send_route_command(ser, state, state_lock, "STOP")
                        route_state = "ERROR"
                        route_active = False
                        tag_correction_was_active = False
                        route_action = "turn_align_timeout_stop"
                    elif tag_visible_in_gate and stable_tag_id == pending_turn_tag_id:
                        tag_corr_raw_deg, tag_corr_cmd_deg = compute_tag_correction(measurement, args)
                        tag_corr_active = True

                        if turn_centered:
                            send_tag_correction(ser, state, state_lock, 0.0)
                            tag_corr_sent = True
                            tag_correction_was_active = False
                            handled_turn_tags.add(pending_turn_tag_id)

                            if args.pre_turn_forward_cm > 0.0:
                                pre_turn_cmd = f"FWD_CM:{args.pre_turn_forward_cm:.3f}"
                                send_route_command(ser, state, state_lock, pre_turn_cmd)
                                route_state = "PRE_TURN_FORWARD"
                                route_step_start_time_s = now_s
                                route_action = f"tag_{pending_turn_tag_id}_centered_pre_turn_forward"
                            else:
                                send_route_command(ser, state, state_lock, pending_turn_command)
                                route_state = "TURNING"
                                pending_turn_start_time_s = now_s
                                route_step_start_time_s = now_s
                                route_action = f"tag_{pending_turn_tag_id}_{pending_turn_command.lower()}"
                        else:
                            send_tag_correction(ser, state, state_lock, tag_corr_cmd_deg)
                            last_tag_command_time = now_wall
                            tag_corr_sent = True
                            tag_correction_was_active = True
                            align_step_cm = max(0.1, args.turn_align_step_cm)
                            send_route_command(ser, state, state_lock, f"FWD_CM:{align_step_cm:.3f}")
                            route_state = "ALIGN_TURN_STEP"
                            route_step_start_time_s = now_s
                            route_action = f"tag_{pending_turn_tag_id}_align_step"
                    else:
                        route_action = "turn_align_wait_tag"

                elif (
                    tag_correction_was_active
                    and route_state not in ("ALIGN_TURN_TAG", "ALIGN_TURN_STEP")
                    and now_wall - last_tag_command_time > args.tag_correction_hold_sec
                ):
                    send_tag_correction(ser, state, state_lock, 0.0)
                    tag_correction_was_active = False
                    tag_corr_sent = True
                    route_action = "tag_corr_clear"

                terminal_text = base.read_terminal_command()
                terminal_command = normalize_terminal_command(terminal_text) if terminal_text else None

                key_command = None
                if not args.no_display:
                    draw_gate_box(frame, gate_info)
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
                    route_active = True
                    handled_turn_tags.clear()
                    slowed_for_turn_tags.clear()
                    previous_tag_id = None
                    stable_tag_id = None
                    stable_count = 0
                    current_event_seq = base.snapshot_mega_state(state, state_lock).get("event_seq", 0)
                    last_mega_event_seq = current_event_seq
                    last_route_tag_action = None
                    tag_correction_was_active = False
                    pending_turn_tag_id = None
                    pending_turn_command = ""
                    pending_turn_start_time_s = -1.0
                    pending_post_turn_forward = False
                    turn_align_start_time_s = -1.0
                    send_tag_correction(ser, state, state_lock, 0.0)
                    move_rpm_command = send_move_rpm(ser, state, state_lock, args.normal_move_rpm)
                    start_cmd = f"FWD_CM:{args.forward_chunk_cm:.3f}"
                    send_route_command(ser, state, state_lock, start_cmd)
                    route_state = "MOVING"
                    route_step_start_time_s = now_s
                    route_action = "manual_start_forward"
                elif requested_command == "STOP":
                    route_active = False
                    send_tag_correction(ser, state, state_lock, 0.0)
                    send_route_command(ser, state, state_lock, "STOP")
                    tag_correction_was_active = False
                    pending_turn_tag_id = None
                    pending_turn_command = ""
                    pending_post_turn_forward = False
                    route_state = "STOPPED"
                    route_action = "manual_stop"
                elif requested_command == "QUIT":
                    route_active = False
                    send_tag_correction(ser, state, state_lock, 0.0)
                    send_route_command(ser, state, state_lock, "STOP")
                    pending_post_turn_forward = False
                    route_action = "manual_quit_stop"
                    row = base.build_log_row(now_s, frame_index, len(detections), measurement, mega_snapshot)
                    row = add_route_fields(
                        row, route_state, route_active, stable_tag_id, stable_count, route_action,
                        handled_turn_tags, tag_corr_active, tag_corr_sent,
                        tag_corr_raw_deg, tag_corr_cmd_deg, measurement,
                        gate_info,
                        pending_turn_tag_id, pending_turn_command,
                        turn_centered, move_rpm_command, pre_turn_slowdown_active,
                        next_turn_tag_id, args
                    )
                    writer.writerow(row)
                    break
                elif requested_command and requested_command.startswith(("FWD_CM:", "BACK_CM:", "TURN_R_DEG:", "TURN_L_DEG:")):
                    route_active = False
                    handled_turn_tags.clear()
                    slowed_for_turn_tags.clear()
                    previous_tag_id = None
                    stable_tag_id = None
                    stable_count = 0
                    last_route_tag_action = None
                    pending_turn_tag_id = None
                    pending_turn_command = ""
                    pending_turn_start_time_s = -1.0
                    pending_post_turn_forward = False
                    tag_correction_was_active = False
                    send_tag_correction(ser, state, state_lock, 0.0)
                    if requested_command.startswith(("FWD_CM:", "BACK_CM:")):
                        move_rpm_command = send_move_rpm(ser, state, state_lock, args.normal_move_rpm)
                    send_route_command(ser, state, state_lock, requested_command)
                    route_step_start_time_s = now_s
                    if requested_command.startswith(("FWD_CM:", "BACK_CM:")):
                        route_state = "MOVING"
                    else:
                        route_state = "TURNING"
                    route_action = f"manual_{requested_command.lower().replace(':', '_')}"

                if (
                    args.debug_print_interval_sec > 0.0
                    and now_wall - last_debug_print_time >= args.debug_print_interval_sec
                ):
                    lat_text = ""
                    agv_text = ""
                    yaw_text = ""
                    if measurement is not None:
                        lat_text = base.fmt(measurement.get("lateral_offset_cm"), 2)
                        agv_text = base.fmt(agv_center_deviation_cm(measurement, args), 2)
                        yaw_text = base.fmt(measurement.get("yaw_error_deg"), 2)
                    print(
                        "[DBG] "
                        f"state={route_state} tag={stable_tag_id} count={stable_count} "
                        f"lat={lat_text} agv={agv_text} yaw={yaw_text} "
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
                        row, route_state, route_active, stable_tag_id, stable_count, route_action,
                        handled_turn_tags, tag_corr_active, tag_corr_sent,
                        tag_corr_raw_deg, tag_corr_cmd_deg, measurement,
                        gate_info,
                        pending_turn_tag_id, pending_turn_command,
                        turn_centered, move_rpm_command, pre_turn_slowdown_active,
                        next_turn_tag_id, args
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