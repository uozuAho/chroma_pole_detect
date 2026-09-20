"""
Hue-class comb matched filter over per-colour row projections.

Extends `comb_fit.py` with the beacon's fixed LED colour sequence
B, B, P, G, G, P, G, B** (top to bottom). NOTE: this is too specific:
light poles may have colours in any sequence.

Diagnosis from `comb_fit.py`: errors are almost entirely in y; the pole's
glowing base sits directly below the LED stack, is blue-hued/saturated/bright,
passes the gate and drags the estimate down. The base glow is 100% blue, while
the LEDs cycle the fixed sequence, so a comb scored against per-hue-class row
projections can only be supported on all 8 teeth by the real LED stack; the base
can feed at most the 3 blue teeth.

Score: `score(a, s) = sum_k p_class(k)[a + k*s]` with classes
[B, B, P, G, G, P, G, B], no anti-teeth and no penalty terms (a missing/dim
LED contributes 0 rather than a negative, which keeps dim-LED images safe).
Candidates are verified (>= 6/8 teeth snap to a genuine local hump in the
overall projection with comparable heights, as in `_verify_comb` from
`comb_fit.py`); if none verifies, the detector falls back to the
region-selection `comb_fit` path unchanged.

## Comments on this detector's performance
- Suspiciously similar performance to comb_fit. Is it just doing the same thing?
"""

from __future__ import annotations

import cv2
import numpy as np

from detectors.comb_fit import (
    DEFAULT_MAX_SPACING,
    DEFAULT_MIN_SPACING,
    DEFAULT_PEAK_KEEP,
    LED_COUNT,
    CombFit,
    _render_bars,
    _render_comb,
    _render_window,
    _top_local_maxima,
)
from detectors.comb_fit import (
    process_frame as comb_process_frame,
)
from detectors.hue_gate import (
    DEFAULT_KEEP_FRAC,
    DEFAULT_SAT_FLOOR,
    DEFAULT_VAL_FLOOR,
    HUE_BLUE,
    HUE_GREEN,
    HUE_PINK,
    Threshold,
    _as_mask,
    _gate_brightness,
    _label,
)
from my_types import DetectCentroidResult

# LED colour sequence top-to-bottom: B, B, P, G, G, P, G, B
CLASS_SEQUENCE = ("b", "b", "p", "g", "g", "p", "g", "b")
REVERSED_SEQUENCE = tuple(reversed(CLASS_SEQUENCE))


class ClassComb(CombFit):
    """Class-sequence comb matched filter (see module docstring)."""

    def __init__(
        self,
        name="class_comb",
        threshold=Threshold.BRIGHTEST,
        sat_floor=DEFAULT_SAT_FLOOR,
        val_floor=DEFAULT_VAL_FLOOR,
        keep_frac=DEFAULT_KEEP_FRAC,
        peak_keep=DEFAULT_PEAK_KEEP,
        min_spacing=DEFAULT_MIN_SPACING,
        max_spacing=DEFAULT_MAX_SPACING,
    ):
        super().__init__(
            name=name,
            threshold=threshold,
            sat_floor=sat_floor,
            val_floor=val_floor,
            keep_frac=keep_frac,
            peak_keep=peak_keep,
            min_spacing=min_spacing,
            max_spacing=max_spacing,
        )

    def process_frame(self, image: np.ndarray) -> DetectCentroidResult:
        return class_comb_process_frame(
            image,
            threshold=self.threshold,
            sat_floor=self.sat_floor,
            val_floor=self.val_floor,
            keep_frac=self.keep_frac,
            peak_keep=self.peak_keep,
            min_spacing=self.min_spacing,
            max_spacing=self.max_spacing,
        )


def class_comb_process_frame(
    image: np.ndarray,
    threshold: Threshold,
    sat_floor: int,
    val_floor: int,
    keep_frac: float,
    peak_keep: float,
    min_spacing: int,
    max_spacing: int,
) -> DetectCentroidResult:
    steps: list[tuple[str, np.ndarray]] = [("1. original", image)]

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    class_masks = {
        "g": (h >= HUE_GREEN[0]) & (h <= HUE_GREEN[1]),
        "b": (h >= HUE_BLUE[0]) & (h <= HUE_BLUE[1]),
        "p": (h >= HUE_PINK[0]) & (h <= HUE_PINK[1]),
    }
    hue_mask = class_masks["g"] | class_masks["b"] | class_masks["p"]
    steps.append(("2. hue windows", _as_mask(hue_mask)))

    sat_pass = {c: m & (s >= sat_floor) for c, m in class_masks.items()}
    brightness = _gate_brightness(
        hue_mask & (s >= sat_floor),
        v,
        threshold=threshold,
        val_floor=val_floor,
        keep_frac=keep_frac,
    )
    mask = _as_mask(brightness)
    steps.append((_label(threshold), mask))

    height, width = mask.shape
    gated = brightness.astype(bool)

    # ---- 4. column band around the strongest column ----
    col_sum = gated.sum(axis=0).astype(np.float64)
    steps.append(("4. column projection", _render_bars(col_sum, height)))

    peak = col_sum.max()
    if peak == 0.0:
        steps.append(("5. comb fit", np.zeros((height, 1), dtype=np.uint8)))
        return (0.0, 0.0), steps

    active = col_sum >= peak_keep * peak
    peak_col = int(col_sum.argmax())
    left = peak_col
    while left > 0 and active[left - 1]:
        left -= 1
    right = peak_col
    while right < width - 1 and active[right + 1]:
        right += 1

    # ---- 5. per-class row projections inside the band ----
    proj = gated[:, left : right + 1]
    class_proj: dict[str, np.ndarray] = {}
    for c, cm in sat_pass.items():
        p = (cm[:, left : right + 1] & gated[:, left : right + 1]).sum(axis=1)
        p = p.astype(np.float64)
        pmax = p.max()
        class_proj[c] = p / pmax if pmax > 0 else p

    top, spacing, _sequence, verified = _class_comb_fit(
        class_proj, min_spacing, max_spacing
    )
    overall = proj.sum(axis=1).astype(np.float64)
    steps.append(("5. comb fit", _render_comb(overall, top, spacing)))

    if not verified:
        # Fall back to the comb_fit region-selection path unchanged.
        return comb_process_frame(
            image,
            threshold=threshold,
            sat_floor=sat_floor,
            val_floor=val_floor,
            keep_frac=keep_frac,
            peak_keep=peak_keep,
            min_spacing=min_spacing,
            max_spacing=max_spacing,
        )

    assert top is not None
    cy = top + (LED_COUNT - 1) / 2.0 * spacing

    # ---- 6. x centroid from columns inside the comb window ----
    y0 = max(int(top - 0.5 * spacing), 0)
    y1 = min(int(top + (LED_COUNT - 0.5) * spacing), height)
    window = gated[y0:y1, left : right + 1].sum(axis=0).astype(np.float64)
    if window.sum() == 0.0:
        window = col_sum[left : right + 1]
    cx = float(np.arange(left, right + 1) @ window / window.sum())

    steps.append(("6. comb window", _render_window(image, left, right, y0, y1, cx, cy)))
    return (cx, cy), steps


def _class_comb_fit(
    class_proj: dict[str, np.ndarray], min_s: int, max_s: int
) -> tuple[int | None, int, tuple[str, ...], bool]:
    """Best (top, spacing, sequence, verified) comb against per-class projections.

    Candidates are ranked by the class-sequence score, evaluated for both the
    expected class sequence and its reverse (in case a pole is oriented
    bottom-to-top), and verified strongest-first. `verified` is False when no
    candidate passes verification (caller falls back).
    """
    n = len(next(iter(class_proj.values())))
    overall = sum(class_proj.values(), start=np.zeros(n))

    candidates: list[tuple[float, int, int, tuple[str, ...]]] = []
    for sequence in (CLASS_SEQUENCE, REVERSED_SEQUENCE):
        for s in range(min_s, max_s + 1):
            # Drop out-of-range teeth from the sum (contributions are >= 0, so
            # a clipped comb simply scores lower, never auto-rejected).
            acc = np.zeros(n, dtype=np.float64)
            for k in range(LED_COUNT):
                acc[: n - k * s] += class_proj[sequence[k]][k * s :]
            for idx in _top_local_maxima(acc, 2):
                candidates.append((float(acc[idx]), idx, s, sequence))
    if not candidates:
        return None, 0, CLASS_SEQUENCE, False

    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    for _, top, s, sequence in candidates[:60]:
        if _verify_class_comb(overall, top, s):
            return top, s, sequence, True
    _, top, s, sequence = candidates[0]
    return top, s, sequence, False


def _verify_class_comb(overall: np.ndarray, top: int, s: int) -> bool:
    """At least 6/8 teeth snap to a genuine, comparable hump in the overall projection.

    Teeth falling outside the projection (edge-clipped combs) are skipped;
    6 of the remaining in-range teeth must still snap and be comparable.
    """
    n = len(overall)
    w = max(2, s // 3)
    heights: list[float] = []
    offsets: list[float] = []
    for k in range(LED_COUNT):
        t = top + k * s
        if t < 0 or t >= n:
            continue
        lo, hi = max(t - w, 0), min(t + w, n - 1)
        if lo == hi:  # single-row clip at the image edge: accept, no hump test
            heights.append(float(overall[lo]))
            offsets.append(float(abs(lo - t)))
            continue
        seg = overall[lo : hi + 1]
        j = int(seg.argmax()) + lo
        rises_left = j > lo and overall[j] > overall[j - 1]
        falls_right = j < hi and overall[j] > overall[j + 1]
        if not (rises_left or falls_right):
            return False
        heights.append(float(overall[j]))
        offsets.append(float(abs(j - t)))
    if len(heights) < 6:
        return False
    med = float(np.median(heights))
    if med <= 0.0:
        return False
    good = int(np.sum((np.asarray(offsets) <= w) & (np.asarray(heights) >= 0.4 * med)))
    return good >= LED_COUNT - 2
