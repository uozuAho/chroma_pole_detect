"""
## What was tried in this detector

Common skeleton: hue+sat+brightest-20% gate → column-projection band around the
strongest column → 1-D row projection p[y] within the band → fit an 8-tooth comb
(top a, pitch s) → y = a + 3.5·s; x = column centroid inside the comb window.

1. **Raw comb score with anti-teeth** (Σ p at teeth − Σ p half a pitch below
   each tooth). Broad humps self-cancel. Works: 01 1.0%, 02 2.5%, 03 1.7%, 05
   1.4%. Fails 04/06: the base is a big *saturated solid block, brighter than
   the LEDs*; a comb straddling LEDs+base with a bogus pitch (s≈17–26)
   out-scores the true one.
2. **Per-pitch bandpass** (subtract running mean with window = s) before
   scoring. Kills broad humps, but also attenuates the *wide* LED humps at close
   range: fixed 02 (0.39%), regressed 01 (5.75%) and 03 (10.77%). Rejected.
3. **Candidate generation + verification** (snap each tooth to a local hump;
   reject flat-topped "humps", require 6/8 comparable heights). Alone too weak:
   combs riding a single wide hump or the base's ramp/edge humps also verify;
   and generating candidates from the anti-teeth score misses the true comb when
   bloom fills the gaps (true comb's anti-score is *negative* in image 6).
4. **Solid-run zeroing** (rows where the band is all-255 = base core, zeroed in
   runs ≥5): 04 → 1.60%, 06 still 6.88% (base *ramp* rows just below saturation
   still support teeth; solid runs are fragmented by a few non-solid rows).
5. **Region selection** (split projection into mass regions bridged by ≤2-row
   gaps; keep the region with the most humps; comb inside it only): 01 22.1% ✗,
   02 2.50% ✓, 03 1.53% ✓, 04 2.28% ✓, 05 2.05% ✓, 06 1.38% ✓.

## Comments on this detector's performance
- fails badly at close range - perhaps the comb's teeth are too fine?
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from detectors.detector import AdjustableParam, Detector
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

DEFAULT_PEAK_KEEP = 0.5
DEFAULT_MIN_SPACING = 2
DEFAULT_MAX_SPACING = 80

# A mass region must hold at least this many humps to be trusted as the LED
# stack; LED_COUNT - 2 mirrors the comb verification tolerance.
MIN_STACK_HUMPS = LED_COUNT - 2


class CombFit(Detector):
    def __init__(
        self,
        name="comb_fit",
        threshold=Threshold.BRIGHTEST,
        sat_floor=DEFAULT_SAT_FLOOR,
        val_floor=DEFAULT_VAL_FLOOR,
        keep_frac=DEFAULT_KEEP_FRAC,
        peak_keep=DEFAULT_PEAK_KEEP,
        min_spacing=DEFAULT_MIN_SPACING,
        max_spacing=DEFAULT_MAX_SPACING,
    ):
        self._name = name
        self.threshold = threshold
        self.sat_floor = sat_floor
        self.val_floor = val_floor
        self.keep_frac = keep_frac
        self.peak_keep = peak_keep
        self.min_spacing = min_spacing
        self.max_spacing = max_spacing
        self._set_adjustable_params()

    @property
    def name(self):
        return self._name

    def _set_adjustable_params(self):
        # name: (param, callback)
        self._adj_params = {
            "threshold": (
                AdjustableParam(
                    name="threshold",
                    label="threshold type",
                    kind="enum",
                    choices=("fixed", "otsu", "brightest"),
                ),
                self._adjust_threshold_type,
            ),
            "sat_floor": (
                AdjustableParam(
                    name="sat_floor",
                    label="sat floor",
                    kind="int",
                    min=0,
                    max=255,
                    step=1,
                ),
                self._adjust_sat_floor,
            ),
            "val_floor": (
                AdjustableParam(
                    name="val_floor",
                    label="val floor (fixed)",
                    kind="int",
                    min=0,
                    max=255,
                    step=1,
                ),
                self._adjust_val_floor,
            ),
            "keep_frac": (
                AdjustableParam(
                    name="keep_frac",
                    label="keep brightest k%",
                    kind="int",
                    min=0,
                    max=100,
                    step=1,
                ),
                self._adjust_keep_frac,
            ),
            "peak_keep": (
                AdjustableParam(
                    name="peak_keep",
                    label="keep % of col peak",
                    kind="int",
                    min=0,
                    max=100,
                    step=1,
                ),
                self._adjust_peak_keep,
            ),
            "min_spacing": (
                AdjustableParam(
                    name="min_spacing",
                    label="min LED spacing px",
                    kind="int",
                    min=0,
                    max=100,
                    step=1,
                ),
                self._adjust_min_spacing,
            ),
            "max_spacing": (
                AdjustableParam(
                    name="max_spacing",
                    label="max LED spacing px",
                    kind="int",
                    min=0,
                    max=100,
                    step=1,
                ),
                self._adjust_max_spacing,
            ),
        }

    def _adjust_threshold_type(self, value):
        if value == "fixed":
            self.threshold = Threshold.FIXED
        elif value == "otsu":
            self.threshold = Threshold.OTSU
        elif value == "brightest":
            self.threshold = Threshold.BRIGHTEST
        else:
            print(f"unknown threshold type {value}")

    def _adjust_sat_floor(self, value):
        self.sat_floor = max(0, min(255, int(value)))

    def _adjust_val_floor(self, value):
        self.val_floor = max(0, min(255, int(value)))

    def _adjust_keep_frac(self, value):
        self.keep_frac = max(0, min(100, int(value))) / 100

    def _adjust_peak_keep(self, value):
        self.peak_keep = max(0, min(100, int(value))) / 100

    def _adjust_min_spacing(self, value):
        self.min_spacing = max(0, min(self.max_spacing, int(value)))

    def _adjust_max_spacing(self, value):
        self.max_spacing = max(self.min_spacing, min(100, int(value)))

    def adjustable_params(self) -> list[AdjustableParam]:
        return [x[0] for x in self._adj_params.values()]

    def set_param(self, name: str, value):
        if name not in self._adj_params:
            raise ValueError(f"unknown adjustable param '{name}' for {self.name}")
        callback = self._adj_params[name][1]
        callback(value)

    def get_param(self, name: str):
        if name == "threshold":
            return self.threshold
        elif name == "sat_floor":
            return self.sat_floor
        elif name == "val_floor":
            return self.val_floor
        elif name == "keep_frac":
            return self.keep_frac * 100
        elif name == "peak_keep":
            return self.peak_keep * 100
        elif name == "min_spacing":
            return self.min_spacing
        elif name == "max_spacing":
            return self.max_spacing
        raise ValueError(f"get unknown adjustable param '{name}'")

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
            min_spacing=self.min_spacing,
            max_spacing=self.max_spacing,
        )


def process_frame(
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

    hue_mask = _hue_mask(h)
    steps.append(("2. hue windows", _as_mask(hue_mask)))

    sat_pass = hue_mask & (s >= sat_floor)
    brightness = _gate_brightness(
        sat_pass, v, threshold=threshold, val_floor=val_floor, keep_frac=keep_frac
    )
    mask = _as_mask(brightness)
    steps.append((_label(threshold), mask))

    height, width = mask.shape

    # ---- 4. column band around the strongest column ----
    col_sum = mask.sum(axis=0).astype(np.float64)
    steps.append(("4. column projection", _render_bars(col_sum, height)))

    peak = col_sum.max()
    if peak == 0.0:
        steps.append(("5. comb fit", _blank_canvas(height)))
        return (0.0, 0.0), steps

    active = col_sum >= peak_keep * peak
    peak_col = int(col_sum.argmax())
    left = peak_col
    while left > 0 and active[left - 1]:
        left -= 1
    right = peak_col
    while right < width - 1 and active[right + 1]:
        right += 1

    # ---- 5. comb fit over the row projection of the band ----
    # WOZ: mask = hue-sat-v gated img
    row_proj = mask[:, left : right + 1].sum(axis=1).astype(np.float64)
    # The pole's glowing base sits below the stack, separated by a dark gap;
    # keep only the mass region with the most humps (the 8-LED stack) so the
    # base cannot support comb teeth.
    _keep_pole_region(row_proj)
    top, spacing = _comb_fit(row_proj, min_spacing, max_spacing)
    steps.append(("5. comb fit", _render_comb(row_proj, top, spacing)))

    if top is None or spacing <= 0:
        total = row_proj.sum()
        cy = float(np.arange(height) @ row_proj / total) if total > 0 else 0.0
        cx = float(
            np.arange(width)[left : right + 1] @ col_sum[left : right + 1] / total
        )
        return (cx, cy), steps

    cy = top + (LED_COUNT - 1) / 2.0 * spacing

    # ---- 6. x centroid from columns inside the comb window ----
    y0 = max(int(top - 0.5 * spacing), 0)
    y1 = min(int(top + (LED_COUNT - 0.5) * spacing), height)
    window = mask[y0:y1, left : right + 1].sum(axis=0).astype(np.float64)
    if window.sum() == 0.0:
        window = col_sum[left : right + 1]
    cx = float(np.arange(left, right + 1) @ window / window.sum())

    steps.append(("6. comb window", _render_window(image, left, right, y0, y1, cx, cy)))
    return (cx, cy), steps


def _comb_fit(proj: np.ndarray, min_s: int, max_s: int) -> tuple[int | None, int]:
    n = len(proj)
    pmax = proj.max()
    if pmax <= 0.0:
        return None, 0
    p = proj / pmax

    candidates: list[tuple[float, int, int]] = []  # (score, top, spacing)
    for s in range(min_s, max_s + 1):
        span = (LED_COUNT - 1) * s
        size = n - span - s
        if size <= 0:
            break
        acc = np.zeros(size, dtype=np.float64)
        for k in range(LED_COUNT):
            acc += p[k * s : k * s + acc.size]
            acc -= p[k * s + s // 2 : k * s + s // 2 + acc.size]
        for idx in _top_local_maxima(acc, 2):
            candidates.append((float(acc[idx]), idx, s))
    if not candidates:
        return None, 0

    candidates.sort(reverse=True)
    for _, top, s in candidates[:60]:
        if _verify_comb(p, top, s):
            return top, s
    _, top, s = candidates[0]
    return top, s


def _find_mass_regions(
    proj: np.ndarray, frac: float = 0.12, bridge: int = 2
) -> list[tuple[int, int]]:
    """Split a projection into mass regions as [start, end) index spans.

    A region grows from a row at or above frac * peak, bridging gaps of up to
    `bridge` below-threshold rows; a larger gap starts the next region.
    """
    pmax = proj.max()
    if pmax <= 0.0:
        return []
    n = len(proj)
    thr = frac * pmax
    regions: list[tuple[int, int]] = []
    i = 0
    while i < n:
        if proj[i] < thr:
            i += 1
            continue
        j = i
        last = i
        while j < n:
            if proj[j] >= thr:
                last = j
            elif j - last > bridge:
                break  # gap of bridge+1 below-threshold rows: start a new region
            j += 1
        regions.append((i, last + 1))
        i = j
    return regions


def _hump_ys(proj: np.ndarray, start: int, end: int, pmax: float) -> list[int]:
    """Indices of local-maxima humps (>= 0.3 * peak) within [start, end)."""
    n = len(proj)
    ys: list[int] = []
    for y in range(start, end):
        if proj[y] < 0.3 * pmax:
            continue
        left_ok = y == 0 or proj[y] >= proj[y - 1]
        right_ok = y == n - 1 or proj[y] > proj[y + 1]
        if left_ok and right_ok:
            ys.append(y)
    return ys


def _region_humps(
    proj: np.ndarray, start: int, end: int, pmax: float
) -> tuple[int, float]:
    """(hump count, mass) of a region."""
    return len(_hump_ys(proj, start, end, pmax)), float(proj[start:end].sum())


def _keep_pole_region(
    proj: np.ndarray,
    frac: float = 0.12,
    bridge: int = 2,
    min_humps: int = MIN_STACK_HUMPS,
) -> tuple[int, int] | None:
    """Zero proj outside the mass region holding the most local humps.

    The row projection of the band shows the LED stack and, below it, the
    pole's glowing base as separate mass regions separated by a near-zero
    gap. The stack region contains ~LED_COUNT humps, the base a single broad
    one, so the hump count identifies the stack without knowing the pitch.

    At close range the dark gaps between LEDs are wider than `bridge`, so the
    stack itself splits into one region per LED, each holding only 1-2 humps.
    No region then looks like the stack: the hump criterion is not decisive
    and zeroing the "winner" would keep a single LED (or even the base).
    In that case the projection is left untouched and the comb fit works
    over the whole band.

    Returns the kept [start, end) span, or None if nothing was zeroed.
    """
    regions = _find_mass_regions(proj, frac, bridge)
    if len(regions) <= 1:
        return None
    pmax = proj.max()
    best = max(
        regions,
        key=lambda r: (
            _region_humps(proj, *r, pmax)[0],
            _region_humps(proj, *r, pmax)[1],
            -r[0],
        ),
    )
    if _region_humps(proj, *best, pmax)[0] < min_humps:
        return None  # no region resembles the 8-LED stack: keep everything
    proj[: best[0]] = 0.0
    proj[best[1] :] = 0.0
    return best


def _top_local_maxima(acc: np.ndarray, count: int) -> list[int]:
    """Up to `count` well-separated local maxima of acc, tallest first."""
    found: list[int] = []
    order = np.argsort(acc)[::-1]
    for idx in order:
        if len(found) >= count:
            break
        if all(abs(idx - f) > 2 for f in found) and acc[idx] > 0.0:
            found.append(int(idx))
    return found


def _verify_comb(p: np.ndarray, top: int, s: int) -> bool:
    """True if at least LED_COUNT-2 teeth sit on genuine, comparable humps."""
    n = len(p)
    w = max(2, s // 3)
    heights = np.empty(LED_COUNT)
    offsets = np.empty(LED_COUNT)
    for k in range(LED_COUNT):
        t = top + k * s
        lo, hi = max(t - w, 0), min(t + w, n - 1)
        if lo >= hi:
            return False
        seg = p[lo : hi + 1]
        j = int(seg.argmax()) + lo
        heights[k] = p[j]
        offsets[k] = abs(j - t)
        rises_left = j > lo and p[j] > p[j - 1]
        falls_right = j < hi and p[j] > p[j + 1]
        if not (rises_left or falls_right):
            return False  # apex sits in a flat run: base block, not a hump
    med = float(np.median(heights))
    if med <= 0.0:
        return False
    good = int(np.sum((offsets <= w) & (heights >= 0.4 * med)))
    return good >= LED_COUNT - 2


def _render_bars(series: np.ndarray, height: int) -> np.ndarray:
    img = np.zeros((height, series.size), dtype=np.uint8)
    peak = series.max()
    if peak <= 0.0 or height <= 0:
        return img
    bar = np.clip(np.rint(series / peak * (height - 1)).astype(int), 0, height - 1)
    for i, bh in enumerate(bar):
        if bh > 0:
            img[height - bh :, i] = 255
    return img


def _render_comb(proj: np.ndarray, top: int | None, spacing: int) -> np.ndarray:
    """Row projection as horizontal bars with the fitted teeth overlaid.

    The projection is a function of image row y, so it is drawn with y running
    down the vertical axis and each bar extending horizontally; this keeps the
    render aligned row-for-row with the band image. Each tooth is a grey
    horizontal line at y = top + k * spacing, marking where comb_fit placed
    an LED.
    """
    img = np.ascontiguousarray(np.rot90(_render_bars(proj, proj.size), k=-1))
    if top is None:
        return img
    for k in range(LED_COUNT):
        y = top + k * spacing
        if 0 <= y < img.shape[0]:
            img[y, :] = 128
    return img


def _render_window(
    image: np.ndarray, left: int, right: int, y0: int, y1: int, cx: float, cy: float
) -> np.ndarray:
    vis = image.copy()
    cv2.rectangle(vis, (left, y0), (right, y1), (0, 255, 0), 2)
    cv2.drawMarker(vis, (round(cx), round(cy)), (0, 0, 255), cv2.MARKER_CROSS, 12, 2)
    return vis


def _blank_canvas(height: int) -> np.ndarray:
    return np.zeros((height, 1), dtype=np.uint8)
