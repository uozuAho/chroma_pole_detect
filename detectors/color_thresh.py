"""
- thresholding the image in HSV space for the three colors the pole can display
  (green, blue, pink) and unioning the resulting masks
- The mask is cleaned up with morphologic open/close operations to remove noise
- small contours are dropped
- image moments of the surviving mask are computed and the centroid is taken as
  the ratio of the first-order moments to the zero-order moment
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from detectors.detector import Detector
from my_types import DetectCentroidResult

GREEN = ((40, 60, 60), (85, 255, 255))
BLUE = ((95, 60, 60), (130, 255, 255))
PINK = ((140, 60, 60), (179, 255, 255))

CONTOUR_LIMIT_FACTOR = 0.10


class ColorThresh(Detector):
    @property
    def name(self):
        return "color_thresh"

    def find_centroid(self, image_path: str | Path) -> DetectCentroidResult:
        return find_centroid(image_path)

    def process_frame(self, image: np.ndarray) -> DetectCentroidResult:
        return process_frame(image)


def find_centroid(image_path: str | Path) -> DetectCentroidResult:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not read image: {image_path}")
    return process_frame(image)


def process_frame(image: np.ndarray) -> DetectCentroidResult:
    steps: list[tuple[str, np.ndarray]] = [("1. original", image)]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for lower, upper in (GREEN, BLUE, PINK):
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper))

    steps.append(("2. color masks", mask))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    steps.append(("3. morph open", mask))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    steps.append(("4. morph close", mask))

    mask = _drop_small_contours(mask)

    moments = cv2.moments(mask)
    if moments["m00"] == 0:
        centroid = 0, 0
    else:
        centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])

    steps.append(("5. kept contours", mask))

    return centroid, steps


def _drop_small_contours(mask: np.ndarray) -> np.ndarray:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(cv2.contourArea(c) for c in contours)
    limit = CONTOUR_LIMIT_FACTOR * largest
    kept = np.zeros_like(mask)
    for contour in contours:
        if cv2.contourArea(contour) >= limit:
            cv2.drawContours(kept, [contour], -1, 255, thickness=cv2.FILLED)
    return kept
