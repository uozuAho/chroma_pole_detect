"""
Gate the hue test with saturation and brightness floors.

A pixel passes iff its hue falls in one of the color windows (green, pink, blue)
**and** its saturation is >= sat_floor **and** its value (brightness) passes the
chosen threshold variant:

- FIXED: value >= val_floor
- OTSU: Otsu threshold on the value channel *within* the hue/saturation mask, so
  the brightness cut adapts to the scene when margins tighten.
- BRIGHTEST: keep the brightest keep_frac of the hue/saturation matched pixels,
  again an adaptive brightness cut.

The resulting mask is cleaned up with morphologic open/close, small contours are
dropped, and the centroid is computed from image moments.

Some analysis of this detector's performance:

HueGate(name="hue_gate_bright20", threshold=Threshold.BRIGHTEST, keep_frac=0.2)
is currently the best performer, but doesn't reject the reflection from the
table. This pulls the detected centroid lower than the real one.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

import cv2
import numpy as np

from detectors.detector import Detector
from imgproc import to_grey
from my_types import DetectCentroidResult

HUE_GREEN = (40, 85)
HUE_BLUE = (95, 130)
HUE_PINK = (140, 179)

DEFAULT_SAT_FLOOR = 60
DEFAULT_VAL_FLOOR = 60
DEFAULT_KEEP_FRAC = 0.20


class Threshold(str, Enum):
    """How the brightness gate is chosen once the hue mask is known."""

    FIXED = "fixed"
    OTSU = "otsu"
    BRIGHTEST = "brightest"


class HueGate(Detector):
    def __init__(
        self,
        name="hue_gate",
        threshold=Threshold.FIXED,
        sat_floor=DEFAULT_SAT_FLOOR,
        val_floor=DEFAULT_VAL_FLOOR,
        keep_frac=DEFAULT_KEEP_FRAC,
    ):
        self._name = name
        self.threshold = threshold
        self.sat_floor = sat_floor
        self.val_floor = val_floor
        self.keep_frac = keep_frac

    @property
    def name(self):
        return self._name

    def find_centroid(self, image_path: str | Path) -> DetectCentroidResult:
        return find_centroid(
            image_path,
            threshold=self.threshold,
            sat_floor=self.sat_floor,
            val_floor=self.val_floor,
            keep_frac=self.keep_frac,
        )

    def process_frame(self, image: np.ndarray) -> DetectCentroidResult:
        return process_frame(
            image,
            threshold=self.threshold,
            sat_floor=self.sat_floor,
            val_floor=self.val_floor,
            keep_frac=self.keep_frac,
        )


def find_centroid(
    image_path: str | Path,
    threshold: Threshold,
    sat_floor: int,
    val_floor: int,
    keep_frac: float,
) -> DetectCentroidResult:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"could not read image: {image_path}")
    return process_frame(
        image,
        threshold=threshold,
        sat_floor=sat_floor,
        val_floor=val_floor,
        keep_frac=keep_frac,
    )


def process_frame(
    image: np.ndarray,
    threshold: Threshold,
    sat_floor: int,
    val_floor: int,
    keep_frac: float,
) -> DetectCentroidResult:
    steps: list[tuple[str, np.ndarray]] = [("1. original", image)]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    hue_mask = _hue_mask(h)
    steps.append(("hue mask", hue_mask))
    sat_mask = s >= sat_floor
    steps.append(("sat floor", sat_mask))
    _, v_threshed = cv2.threshold(
        v.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    steps.append(("v thresh", v_threshed))
    hsv_mask = hue_mask & sat_mask & v_threshed.astype(np.bool_)
    steps.append(("h s v mask", hsv_mask))

    sat_pass = hue_mask & (s >= sat_floor)
    brightness = _gate_brightness(
        sat_pass, v, threshold=threshold, val_floor=val_floor, keep_frac=keep_frac
    )
    steps.append((_label(threshold), _as_mask(brightness)))

    mask = _as_mask(brightness)
    steps.append(("3. brightness gate", mask))

    moments = cv2.moments(to_grey(hsv_mask))
    if moments["m00"] == 0:
        centroid = 0, 0
    else:
        centroid = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])

    return centroid, steps


def _label(threshold: Threshold) -> str:
    return {
        Threshold.FIXED: "3. hue + sat/val floors",
        Threshold.OTSU: "3. hue + sat floor + otsu on V",
        Threshold.BRIGHTEST: "3. hue + sat floor + brightest k%",
    }[threshold]


def _hue_mask(hue: np.ndarray) -> np.ndarray:
    """Boolean mask of pixels whose hue falls in one of the color windows."""
    mask = np.zeros(hue.shape, dtype=bool)
    for lower, upper in (HUE_GREEN, HUE_BLUE, HUE_PINK):
        mask |= (hue >= lower) & (hue <= upper)
    return mask


def _gate_brightness(
    masked: np.ndarray,
    value: np.ndarray,
    *,
    threshold: Threshold,
    val_floor: int,
    keep_frac: float,
) -> np.ndarray:
    threshold = _coerce(threshold)
    if threshold is Threshold.FIXED:
        return masked & (value >= val_floor)

    values = value[masked]
    if values.size == 0:
        return masked

    if threshold is Threshold.OTSU:
        otsu_val, _ = cv2.threshold(
            values.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return masked & (value >= int(otsu_val))

    # else we're using BRIGHTEST

    cutoff = np.quantile(values, 1.0 - keep_frac)
    return masked & (value >= cutoff)


def _coerce(threshold: Threshold) -> Threshold:
    return threshold if isinstance(threshold, Threshold) else Threshold(threshold)


def _as_mask(sel: np.ndarray) -> np.ndarray:
    return (sel * np.uint8(255)).astype(np.uint8)
