"""
Gate the hue test as in hue_gate, then project the gated response down each
column to form a 1-D profile h[x] = sum_y w(x, y).

A vertical pole collapses to a sharp 1-D peak: even a 1-px-wide distant pole
still contributes one count per LED (~8), which no erosion can delete. The band
of columns making up the peak (those above peak_keep * max) is located, the
x-centroid is read from moments of h within that band, and the y-centroid from
row moments within the band.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from detectors.detector import Detector
from detectors.hue_gate import (
    DEFAULT_KEEP_FRAC,
    DEFAULT_SAT_FLOOR,
    DEFAULT_VAL_FLOOR,
    Threshold,
    _as_mask,
    _gate_brightness,
    _hue_mask,
    _label,
)
from my_types import DetectCentroidResult

DEFAULT_PEAK_KEEP = 0.5


class ColumnProject(Detector):
    def __init__(
        self,
        name="column_project",
        threshold=Threshold.FIXED,
        sat_floor=DEFAULT_SAT_FLOOR,
        val_floor=DEFAULT_VAL_FLOOR,
        keep_frac=DEFAULT_KEEP_FRAC,
        peak_keep=DEFAULT_PEAK_KEEP,
    ):
        self._name = name
        self.threshold = threshold
        self.sat_floor = sat_floor
        self.val_floor = val_floor
        self.keep_frac = keep_frac
        self.peak_keep = peak_keep

    @property
    def name(self):
        return self._name

    def find_centroid(self, image_path: str | Path) -> DetectCentroidResult:
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"could not read image: {image_path}")
        return process_frame(
            image,
            threshold=self.threshold,
            sat_floor=self.sat_floor,
            val_floor=self.val_floor,
            keep_frac=self.keep_frac,
            peak_keep=self.peak_keep,
        )

    def process_frame(self, image: np.ndarray) -> DetectCentroidResult:
        return process_frame(
            image,
            threshold=self.threshold,
            sat_floor=self.sat_floor,
            val_floor=self.val_floor,
            keep_frac=self.keep_frac,
            peak_keep=self.peak_keep,
        )


def process_frame(
    image: np.ndarray,
    threshold: Threshold,
    sat_floor: int,
    val_floor: int,
    keep_frac: float,
    peak_keep: float,
) -> DetectCentroidResult:
    steps: list[tuple[str, np.ndarray]] = [("1. original", image)]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    hue_mask = _hue_mask(h)
    steps.append(("2. hue windows", _as_mask(hue_mask)))

    sat_pass = hue_mask & (s >= sat_floor)
    brightness = _gate_brightness(
        sat_pass, v, threshold=threshold, val_floor=val_floor, keep_frac=keep_frac
    )
    steps.append((_label(threshold), _as_mask(brightness)))

    mask = _as_mask(brightness)
    steps.append(("3. gated response", mask))

    height, width = mask.shape
    col_sum = mask.sum(axis=0).astype(np.float64)
    steps.append(("4. column projection", _render_projection(col_sum, height)))

    peak = col_sum.max()
    if peak == 0.0:
        steps.append(("5. peak band", _as_mask(np.zeros(mask.shape, dtype=bool))))
        return (0.0, 0.0), steps

    active = col_sum >= peak_keep * peak
    peak_col = int(col_sum.argmax())
    left = peak_col
    while left > 0 and active[left - 1]:
        left -= 1
    right = peak_col
    while right < width - 1 and active[right + 1]:
        right += 1

    band_mask = np.zeros(mask.shape, dtype=bool)
    band_mask[:, left : right + 1] = True
    steps.append(("5. peak band", _as_mask(band_mask)))

    band_col_sum = col_sum[left : right + 1]
    if band_col_sum.sum() == 0.0:
        return (0.0, 0.0), steps
    x_w = np.arange(left, right + 1, dtype=np.float64)
    cx = float(np.sum(x_w * band_col_sum) / band_col_sum.sum())

    row_sum = mask[:, left : right + 1].sum(axis=1).astype(np.float64)
    total_row = row_sum.sum()
    if total_row == 0.0:
        cy = 0.0
    else:
        y_w = np.arange(height, dtype=np.float64)
        cy = float(np.sum(y_w * row_sum) / total_row)

    return (cx, cy), steps


def _render_projection(col_sum: np.ndarray, height: int) -> np.ndarray:
    """Render the 1-D column profile as vertical bars in a height x W image."""
    img = np.zeros((height, col_sum.shape[0]), dtype=np.uint8)
    peak = col_sum.max()
    if peak <= 0.0:
        return img
    bar_heights = np.rint(col_sum / peak * (height - 1)).astype(int)
    for x, bar in enumerate(bar_heights):
        if bar > 0:
            img[height - bar :, x] = 255
    return img
