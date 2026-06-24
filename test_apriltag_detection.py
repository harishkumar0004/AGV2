import cv2
import numpy as np

from pupil_apriltags import Detector


image = cv2.imread("./captured_images/img1.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

detector = Detector(
    families="tag36h11",
    nthreads=4
)

results = detector.detect(gray)

print(results)
