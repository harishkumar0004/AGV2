#!/usr/bin/env python3
"""
Camera FPS checker for Raspberry Pi / USB camera.

Measures:
  1. Raw camera capture FPS.
  2. Optional AprilTag detection FPS.

Usage:
  python3 LogiTech_Camera/Pi_camera_fps_check.py
  python3 LogiTech_Camera/Pi_camera_fps_check.py --detect-tags
  python3 LogiTech_Camera/Pi_camera_fps_check.py --camera 0 --width 640 --height 480 --target-fps 60
  python3 LogiTech_Camera/Pi_camera_fps_check.py --no-display --seconds 10
"""

import argparse
import subprocess
import time

import cv2


def create_detector():
    import apriltag

    if hasattr(apriltag, "apriltag"):
        return apriltag.apriltag("tag36h11")

    options = apriltag.DetectorOptions(families="tag36h11")
    return apriltag.Detector(options)


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

    print(f"Warming camera for {seconds:.1f}s...")
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
    parser = argparse.ArgumentParser(description="Check camera FPS and optional AprilTag detection FPS.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--target-fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--detect-tags", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--backend-fourcc", default="MJPG",
                        help="Camera pixel format, usually MJPG or YUYV. Use empty string to skip.")
    parser.add_argument("--max-failures", type=int, default=10)
    parser.add_argument("--camera-warmup-sec", type=float, default=3.0)
    parser.add_argument("--exposure", type=int, default=40)
    parser.add_argument("--gain", type=int, default=64,
                        help="Camera gain. Use -1 to leave unchanged.")
    parser.add_argument("--power-line-frequency", type=int, default=1,
                        help="1=50 Hz, 2=60 Hz for most UVC cameras.")
    parser.add_argument("--skip-v4l2-controls", action="store_true")
    args = parser.parse_args()

    apply_v4l2_controls(args)

    cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    configure_camera(cap, args)
    warmup_camera(cap, args.camera_warmup_sec, args.max_failures)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    reported_fourcc = get_camera_fourcc(cap)

    detector = create_detector() if args.detect_tags else None

    print("Camera opened")
    print(f"Requested: {args.width}x{args.height} @ {args.target_fps} FPS")
    print(f"Actual:    {actual_width}x{actual_height}")
    print(f"Reported camera FPS: {reported_fps:.2f}")
    print(f"Reported FOURCC: {reported_fourcc}")
    print(f"Detect tags: {args.detect_tags}")
    print("Press q in the preview window to stop.")

    start_time = time.time()
    last_print_time = start_time
    frame_count = 0
    detect_count = 0
    detected_tag_frames = 0
    total_detect_time = 0.0
    consecutive_failures = 0

    try:
        while True:
            now = time.time()
            if now - start_time >= args.seconds:
                break

            ret, frame = cap.read()
            if not ret:
                consecutive_failures += 1
                print(f"Frame read failed ({consecutive_failures}/{args.max_failures})")
                if consecutive_failures >= args.max_failures:
                    break
                time.sleep(0.05)
                continue

            consecutive_failures = 0

            frame_count += 1
            tag_count = 0

            if detector is not None:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                t0 = time.time()
                detections = detector.detect(gray)
                detect_time = time.time() - t0

                total_detect_time += detect_time
                detect_count += 1
                tag_count = len(detections)

                if tag_count > 0:
                    detected_tag_frames += 1

                cv2.putText(frame, f"tags:{tag_count}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            elapsed = now - start_time
            raw_fps = frame_count / elapsed if elapsed > 0.0 else 0.0
            detect_fps = detect_count / total_detect_time if total_detect_time > 0.0 else 0.0

            if now - last_print_time >= 1.0:
                if detector is None:
                    print(f"elapsed={elapsed:.1f}s raw_capture_fps={raw_fps:.2f}")
                else:
                    avg_detect_ms = (total_detect_time / detect_count) * 1000.0 if detect_count else 0.0
                    print(
                        f"elapsed={elapsed:.1f}s "
                        f"raw_capture_fps={raw_fps:.2f} "
                        f"tag_detection_fps={detect_fps:.2f} "
                        f"avg_detect_time={avg_detect_ms:.1f}ms "
                        f"tag_frames={detected_tag_frames}"
                    )
                last_print_time = now

            if not args.no_display:
                cv2.putText(frame, f"raw fps:{raw_fps:.1f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Camera FPS Check", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    elapsed = time.time() - start_time
    raw_fps = frame_count / elapsed if elapsed > 0.0 else 0.0
    detect_fps = detect_count / total_detect_time if total_detect_time > 0.0 else 0.0

    print("\nFinal result")
    print(f"frames captured: {frame_count}")
    print(f"elapsed: {elapsed:.2f}s")
    print(f"raw capture FPS: {raw_fps:.2f}")

    if detector is not None:
        avg_detect_ms = (total_detect_time / detect_count) * 1000.0 if detect_count else 0.0
        print(f"AprilTag detect calls: {detect_count}")
        print(f"AprilTag detection FPS: {detect_fps:.2f}")
        print(f"Average detection time: {avg_detect_ms:.1f}ms")
        print(f"Frames where tag was detected: {detected_tag_frames}")


if __name__ == "__main__":
    main()
