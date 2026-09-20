"""
Exploit the pole's fixed shape: 8 evenly spaced blobs on a near-vertical line.

Three complementary techniques are combined and each is surfaced as an
intermediate step so its contribution can be inspected:

- vertical blob stack filter: the gated response is collapsed down each column
  to a 1-D profile h[x]; a vertical pole becomes a sharp peak band.
- Hough-style voting over (phase, spacing): the LED rows (peaks of the row
  projection inside the pole band) vote for every spacing consistent with them;
  the spacing that collects the most votes is the LED pitch.
- periodicity check: the autocorrelation of the row projection peaks at the LED
  spacing; a clean peak is a strong confidence signal.
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

LED_COUNT = 8  # the pole is a vertical tower of 8 LEDs

DEFAULT_PEAK_KEEP = 0.5  # column band cut, as in column_project
DEFAULT_ROW_KEEP = 0.30  # LED rows must reach this fraction of the max row sum
DEFAULT_MIN_SPACING = 3
DEFAULT_MAX_SPACING = 60

CANVAS_H = 48


class StructureDetector(Detector):
    def __init__(
        self,
        name="structure",
        threshold=Threshold.OTSU,
        sat_floor=DEFAULT_SAT_FLOOR,
        val_floor=DEFAULT_VAL_FLOOR,
        keep_frac=DEFAULT_KEEP_FRAC,
        peak_keep=DEFAULT_PEAK_KEEP,
        row_keep=DEFAULT_ROW_KEEP,
        min_spacing=DEFAULT_MIN_SPACING,
        max_spacing=DEFAULT_MAX_SPACING,
    ):
        self._name = name
        self.threshold = threshold
        self.sat_floor = sat_floor
        self.val_floor = val_floor
        self.keep_frac = keep_frac
        self.peak_keep = peak_keep
        self.row_keep = row_keep
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing

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
            row_keep=self.row_keep,
            min_spacing=self.min_spacing,
            max_spacing=self.max_spacing,
        )

    def process_frame(self, image: np.ndarray) -> DetectCentroidResult:
        return process_frame(
            image,
            threshold=self.threshold,
            sat_floor=self.sat_floor,
            val_floor=self.val_floor,
            keep_frac=self.keep_frac,
            peak_keep=self.peak_keep,
            row_keep=self.row_keep,
            min_spacing=self.min_spacing,
            max_spacing=self.max_spacing,
        )


def find_centroid(
    image_path: str | Path,
    threshold: Threshold,
    sat_floor: int,
    val_floor: int,
    keep_frac: float,
    peak_keep: float,
    row_keep: float,
    min_spacing: int,
    max_spacing: int,
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
        peak_keep=peak_keep,
        row_keep=row_keep,
        min_spacing=min_spacing,
        max_spacing=max_spacing,
    )


def process_frame(
    image: np.ndarray,
    threshold: Threshold,
    sat_floor: int,
    val_floor: int,
    keep_frac: float,
    peak_keep: float,
    row_keep: float,
    min_spacing: int,
    max_spacing: int,
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

    # ---- 4. vertical blob stack filter: column projection ----
    col_sum = mask.sum(axis=0).astype(np.float64)
    steps.append(("4. column projection", _render_bars(col_sum, col_sum.size, height)))

    peak = col_sum.max()
    if peak == 0.0:
        steps.append(("5. pole band", _as_mask(np.zeros(mask.shape, dtype=bool))))
        steps.append(("6. LED rows", _as_mask(np.zeros(mask.shape, dtype=bool))))
        steps.append(("7. spacing votes", np.zeros((CANVAS_H, 1), dtype=np.uint8)))
        steps.append(("8. autocorrelation", np.zeros((CANVAS_H, 1), dtype=np.uint8)))
        return (0.0, 0.0), steps

    # keep top n% of column sums to determine a 'band' in
    # which a pole is likely to be
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
    steps.append(("5. pole band", _as_mask(band_mask)))

    band_col_sum = col_sum[left : right + 1]
    x_w = np.arange(left, right + 1, dtype=np.float64)
    cx = float(np.sum(x_w * band_col_sum) / band_col_sum.sum())

    # ---- 5. LED rows: peaks of the row projection inside the band ----
    # at this point, `mask` is the h-s-v gated image
    row_proj = mask[:, left : right + 1].sum(axis=1).astype(np.float64)
    led_rows, led_strength = _row_peaks(row_proj, row_keep)
    led_mask = np.zeros(mask.shape, dtype=bool)
    led_mask[led_rows, left : right + 1] = True
    steps.append(("6. LED rows over band", _mask_led(mask, led_mask)))

    # ---- 6. Hough-style voting over spacing ----
    if len(led_rows) < 2:
        cy = _centroid_y(row_proj)
        steps.append(
            ("7. spacing votes", _render_spacing(led_rows, min_spacing, max_spacing))
        )
        steps.append(
            ("8. autocorrelation", _render_ac(row_proj, min_spacing, max_spacing))
        )
        return (cx, cy), steps

    spacing = _vote_spacing(led_rows, min_spacing, max_spacing)
    steps.append(
        ("7. spacing votes", _render_spacing(led_rows, min_spacing, max_spacing))
    )

    y_top = _fit_phase(led_rows, led_strength, spacing)
    cy = max(min(y_top + (LED_COUNT - 1) / 2.0 * spacing, height - 1.0), 0.0)

    # ---- 7. periodicity check: autocorrelation of the row projection ----
    steps.append(("8. autocorrelation", _render_ac(row_proj, min_spacing, max_spacing)))

    return (cx, cy), steps


def _row_peaks(row_proj: np.ndarray, row_keep: float) -> tuple[np.ndarray, np.ndarray]:
    """Rows that are local maxima of the projection and reach row_keep * max."""
    proj_max = row_proj.max()
    thr = row_keep * proj_max if proj_max > 0.0 else 0.0
    left_src = np.concatenate(([0.0], row_proj[:-1]))
    right_src = np.concatenate((row_proj[1:], [0.0]))
    is_peak = (row_proj >= left_src) & (row_proj > right_src) & (row_proj >= thr)
    rows = np.nonzero(is_peak)[0]
    return rows, row_proj[rows]


def _vote_spacing(rows: np.ndarray, min_s: int, max_s: int) -> int:
    """Every pair of LED rows votes for each spacing consistent with it.

    A vertical separation of `gap` fits `gap / div` when the two rows are `div`
    LED-periods apart; each divisor consistent with the known LED_COUNT casts a
    vote weighted 1/div so small, plausible spacings win ties.
    """
    width = int(max_s - min_s) + 1
    votes = np.zeros(width, dtype=np.float64)
    for k in range(1, rows.size):
        for j in range(k):
            gap = int(rows[k] - rows[j])
            for div in range(1, LED_COUNT + 1):
                if gap % div:
                    continue
                s = gap / div
                if min_s <= s <= max_s:
                    votes[round(s) - min_s] += 1.0 / div
    if votes.sum() == 0.0:
        return 0
    return int(votes.argmax()) + min_s


def _fit_phase(rows: np.ndarray, strength: np.ndarray, spacing: int) -> float:
    """Return y_top such that the detected rows sit on y_top + k * spacing."""
    if spacing <= 0:
        return float(np.median(rows))
    ref = rows[int(np.argmax(strength))]
    indices = np.rint((rows - ref) / spacing).astype(int)
    y_top = float(np.median(rows - indices * spacing))
    return y_top


def _centroid_y(row_proj: np.ndarray) -> float:
    total = row_proj.sum()
    if total == 0.0:
        return 0.0
    return float(np.arange(len(row_proj), dtype=np.float64) @ row_proj) / total


def _autocorr(x: np.ndarray) -> np.ndarray:
    x = x - x.mean()
    n = len(x)
    norm = x @ x
    if norm == 0.0:
        return np.zeros(n, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    for lag in range(n):
        out[lag] = x[lag:] @ x[: n - lag]
    return out / norm


def _render_spacing(rows: np.ndarray, min_s: int, max_s: int) -> np.ndarray:
    width = int(max_s - min_s) + 1
    votes = np.zeros(width, dtype=np.float64)
    for i in range(len(rows)):
        for j in range(len(rows)):
            if i == j:
                continue
            gap = int(abs(rows[i] - rows[j]))
            for div in range(1, LED_COUNT + 1):
                if gap % div:
                    continue
                s = gap / div
                if min_s <= s <= max_s:
                    votes[round(s) - min_s] += 1.0 / div
    peak_v = votes.max()
    out = np.zeros((CANVAS_H, width), dtype=np.uint8)
    if peak_v > 0.0:
        bar = np.rint(votes / peak_v * (CANVAS_H - 1)).astype(int)
        for k, h in enumerate(bar):
            if h > 0:
                out[CANVAS_H - h :, k] = 255
    return out


def _render_ac(row_proj: np.ndarray, min_s: int, max_s: int) -> np.ndarray:
    width = min(max_s, len(row_proj) - 1) + 1
    if width <= 1:
        return np.zeros((CANVAS_H, 1), dtype=np.uint8)
    seg = _autocorr(row_proj)[min_s:width]
    out = np.zeros((CANVAS_H, width), dtype=np.uint8)
    peak = seg.max()
    if peak > 0.0:
        bar = np.clip(np.rint(seg / peak * (CANVAS_H - 1)).astype(int), 0, CANVAS_H - 1)
        for lag, h in enumerate(bar):
            if h > 0:
                out[CANVAS_H - h :, min_s + lag] = 255
    return out


def _render_bars(series: np.ndarray, width: int, height: int) -> np.ndarray:
    img = np.zeros((height, width), dtype=np.uint8)
    peak = series.max()
    if peak <= 0.0 or width <= 0 or height <= 0:
        return img
    bar = np.clip(np.rint(series / peak * (height - 1)).astype(int), 0, height - 1)
    for i, h in enumerate(bar):
        if h > 0:
            img[height - h :, i] = 255
    return img


def _mask_led(mask: np.ndarray, led_mask: np.ndarray) -> np.ndarray:
    both = cv2.cvtColor(mask.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    both[led_mask] = (0, 0, 255)
    return both
