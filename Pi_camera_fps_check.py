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
import time

import cv2


def create_detector():
    import apriltag

    if hasattr(apriltag, "apriltag"):
        return apriltag.apriltag("tag36h11")

    options = apriltag.DetectorOptions(families="tag36h11")
    return apriltag.Detector(options)


def main():
    parser = argparse.ArgumentParser(description="Check camera FPS and optional AprilTag detection FPS.")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--target-fps", type=int, default=30)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--detect-tags", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.target_fps)

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    reported_fps = cap.get(cv2.CAP_PROP_FPS)

    detector = create_detector() if args.detect_tags else None

    print("Camera opened")
    print(f"Requested: {args.width}x{args.height} @ {args.target_fps} FPS")
    print(f"Actual:    {actual_width}x{actual_height}")
    print(f"Reported camera FPS: {reported_fps:.2f}")
    print(f"Detect tags: {args.detect_tags}")
    print("Press q in the preview window to stop.")

    start_time = time.time()
    last_print_time = start_time
    frame_count = 0
    detect_count = 0
    detected_tag_frames = 0
    total_detect_time = 0.0

    try:
        while True:
            now = time.time()
            if now - start_time >= args.seconds:
                break

            ret, frame = cap.read()
            if not ret:
                print("Frame read failed")
                break

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
