#!/usr/bin/env python3

import math

import cv2
from picamera2 import Picamera2
from pupil_apriltags import Detector


TAG_FAMILY = "tag36h11"

picam2 = Picamera2()

config = picam2.create_preview_configuration(
    main={
        "size": (640, 480),
        "format": "RGB888",
    }
)

picam2.configure(config)
picam2.start()

detector = Detector(
    families=TAG_FAMILY,
    nthreads=4,
    quad_decimate=2.0,
    quad_sigma=0.0,
    refine_edges=1,
)

print("Press q to quit.")

while True:

    frame = picam2.capture_array()

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    detections = detector.detect(gray)

    for tag in detections:

        corners = tag.corners.astype(int)

        for i in range(4):
            p1 = tuple(corners[i])
            p2 = tuple(corners[(i + 1) % 4])

            cv2.line(frame, p1, p2, (0, 255, 0), 2)

        center = (int(tag.center[0]), int(tag.center[1]))

        cv2.circle(frame, center, 5, (0, 0, 255), -1)

        top_mid = (
            int((corners[0][0] + corners[1][0]) / 2),
            int((corners[0][1] + corners[1][1]) / 2),
        )

        cv2.line(frame, center, top_mid, (255, 0, 0), 2)

        dx = top_mid[0] - center[0]
        dy = center[1] - top_mid[1]

        yaw_deg = math.degrees(math.atan2(dx, dy))

        cv2.putText(
            frame,
            f"ID: {tag.tag_id}",
            (center[0] + 10, center[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            f"Yaw: {yaw_deg:.1f} deg",
            (center[0] + 10, center[1] + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

    cv2.imshow("AprilTag Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

picam2.stop()
cv2.destroyAllWindows()
