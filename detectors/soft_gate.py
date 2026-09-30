"""
Weighted moments detector. An attempt on improving hue_gate.

Each pixel gets a membership weight w in [0, 1] — a trapezoid in H, S and V
around each pole colour — and the centroid is the weighted moment
(sum(x*w)/sum(w), sum(y*w)/sum(w)). No pixel is hard-zeroed at a window
boundary, so the detector degrades gracefully towards a red pole or silhouette
and converges on subpixel positions.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from detectors.detector import AdjustableParam, Detector
from my_types import DetectCentroidResult

HUE_GREEN = (40, 85)
HUE_BLUE = (95, 130)
HUE_PINK = (140, 179)

DEFAULT_SAT_FLOOR = 60
DEFAULT_VAL_FLOOR = 60
DEFAULT_HUE_SOFT = 8.0
DEFAULT_CHROMA_SOFT = 20.0

# falloff margin used to route around the circular hue boundary
HUE_CIRCULAR = 179


class SoftGate(Detector):
    def __init__(
        self,
        name="soft_gate",
        sat_floor=DEFAULT_SAT_FLOOR,
        val_floor=DEFAULT_VAL_FLOOR,
        hue_soft=DEFAULT_HUE_SOFT,
        chroma_soft=DEFAULT_CHROMA_SOFT,
    ):
        self._name = name
        self.sat_floor = sat_floor
        self.val_floor = val_floor
        self.hue_soft = hue_soft
        self.chroma_soft = chroma_soft

    @property
    def name(self):
        return self._name

    def find_centroid(self, image_path: str | Path) -> DetectCentroidResult:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"could not read image: {image_path}")
        return process_frame(
            image,
            sat_floor=self.sat_floor,
            val_floor=self.val_floor,
            hue_soft=self.hue_soft,
            chroma_soft=self.chroma_soft,
        )

    def process_frame(self, image: np.ndarray) -> DetectCentroidResult:
        return process_frame(
            image,
            sat_floor=self.sat_floor,
            val_floor=self.val_floor,
            hue_soft=self.hue_soft,
            chroma_soft=self.chroma_soft,
        )

    def adjustable_params(self) -> list[AdjustableParam]:
        return []

    def set_param(self, name: str, value) -> None:
        raise ValueError(f"no adjustable params on {self.name}")

    def get_param(self, name: str):
        raise ValueError(f"no adjustable params on {self.name}")


def _trapezoid(x: np.ndarray, left: float, peak_l: float, peak_r: float, right: float):
    """Soft ramp: 0 outside [left, right], 1 on [peak_l, peak_r], linear edges."""
    y = np.zeros_like(x, dtype=np.float32)
    y[x > right] = 0.0
    mid = (x >= peak_l) & (x <= peak_r)
    y[mid] = 1.0

    left_edge = (x > left) & (x < peak_l)
    y[left_edge] = (x[left_edge] - left) / max(peak_l - left, 1e-6)

    right_edge = (x > peak_r) & (x < right)
    y[right_edge] = (right - x[right_edge]) / max(right - peak_r, 1e-6)
    return y


def _hue_weight(hue: np.ndarray, hue_soft: float) -> np.ndarray:
    """Hue membership as the max trapezoid across all pole colours."""
    weight = np.zeros_like(hue, dtype=np.float32)
    for lower, upper in (HUE_GREEN, HUE_BLUE, HUE_PINK):
        weight = np.maximum(
            weight,
            _trapezoid(hue, lower - hue_soft, lower, upper, upper + hue_soft),
        )
    return weight


def _chroma_weight(channel: np.ndarray, floor: float, soft: float) -> np.ndarray:
    """Saturation/value membership: 1.0 once above the floor, ramping below."""
    return _trapezoid(channel, floor - soft, floor, 255, 255)


def process_frame(
    image: np.ndarray,
    sat_floor: int,
    val_floor: int,
    hue_soft: float,
    chroma_soft: float,
) -> DetectCentroidResult:
    steps: list[tuple[str, np.ndarray]] = [("1. original", image)]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    w = _hue_weight(h, hue_soft)
    steps.append(("2. hue weights", _as_gray(w)))

    w = w * _chroma_weight(s, sat_floor, chroma_soft)
    w = w * _chroma_weight(v, val_floor, chroma_soft)
    steps.append(("3. sat/val weighted", _as_gray(w)))

    total = float(np.sum(w))
    if total == 0.0:
        centroid = 0.0, 0.0
        steps.append(("4. mask (w>0)", _as_gray(w)))
        return centroid, steps

    ys, xs = np.indices(image.shape[:2])
    cx = float(np.sum(xs.astype(np.float32) * w) / total)
    cy = float(np.sum(ys.astype(np.float32) * w) / total)

    steps.append(("4. mask (w>0)", _as_gray(w)))
    return (cx, cy), steps


def _as_gray(weight: np.ndarray) -> np.ndarray:
    scaled = np.clip(weight, 0.0, 1.0) * 255.0
    return scaled.astype(np.uint8)
