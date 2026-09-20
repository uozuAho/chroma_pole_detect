from collections.abc import Iterable

import cv2

from scratchpads.scratch_helper import *

IMGS = sorted(Path("img").glob("*.jpg"))
IMG_PATH = IMGS[0]


def led_hues_mask(hue_channel: np.ndarray):
    hue_green = (40, 85)
    hue_blue = (95, 130)
    hue_pink = (140, 179)

    def hue_mask(hue: np.ndarray, hue_ranges: Iterable[tuple[int, int]]) -> np.ndarray:
        """Boolean mask of pixels whose hue falls in one of the color windows."""
        mask = np.zeros(hue.shape, dtype=bool)
        for lower, upper in hue_ranges:
            mask |= (hue >= lower) & (hue <= upper)
        return mask

    return hue_mask(hue_channel, (hue_green, hue_blue, hue_pink))


def find_single_peak_bounds(vals: np.ndarray, cutoff=0.5):
    """Find the bounding region around a peak of a 1d array
    Returns (left, right) bounds
    """
    assert vals.ndim == 1
    peak_idx = int(col_sum.argmax())
    is_in_col = col_sum >= 0.5 * col_sum[peak_idx]
    left = peak_idx
    while left > 0 and is_in_col[left - 1]:
        left -= 1
    right = peak_idx
    while right < width - 1 and is_in_col[right + 1]:
        right += 1
    return left, right


image = load(IMG_PATH)


# ==========================
# STEP 1: h s v gate, same as hue gate scratch
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

hmask = led_hues_mask(h)
hue_sat_mask = hmask & (s > 60)
# show(hue_sat_mask)

# val, v_threshed = thresh_otsu(v)
v_threshed = keep_top_k_pc(v, 20)
show(v_threshed)

combined_thresh = led_hues_mask(h) & (s >= 60) & v_threshed.astype(np.bool_)

# ==========================

height, width = combined_thresh.shape
col_sum = combined_thresh.sum(axis=0).astype(np.float64)
pole_left, pole_right = find_single_peak_bounds(col_sum)
pole_row_sum = (
    combined_thresh[:, pole_left : pole_right + 1].sum(axis=1).astype(np.float64)
)
pole_bounds_img = draw_vline(draw_vline(to_rgb(combined_thresh), pole_left), pole_right)


# ==========================
# draw the results
# show_many(
#     [
#         (combined_thresh, "h s v masks combined"),
#         (render_1d_vals(col_sum, width, height), "col sum"),
#         (pole_bounds_img, "pole bounds"),
#         (render_1d_vals(pole_row_sum, width, height, horizontal=True), "h sum"),
#     ]
# )


def _keep_pole_region(proj: np.ndarray, frac: float = 0.12, bridge: int = 2) -> None:
    pmax = proj.max()
    if pmax <= 0.0:
        return
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
        while j < n and j - last <= bridge:
            if proj[j] >= thr:
                last = j
            j += 1
        regions.append((i, last + 1))
        i = j
    if len(regions) <= 1:
        return

    def hump_count(start: int, end: int) -> tuple[int, float]:
        count = 0
        mass = 0.0
        for y in range(start, end):
            mass += proj[y]
            if proj[y] < 0.3 * pmax:
                continue
            left_ok = y == 0 or proj[y] >= proj[y - 1]
            right_ok = y == n - 1 or proj[y] > proj[y + 1]
            count += left_ok and right_ok
        return count, mass

    best = max(
        regions,
        key=lambda r: (hump_count(*r)[0], hump_count(*r)[1], -r[0]),
    )
    proj[: best[0]] = 0.0
    proj[best[1] :] = 0.0


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
    LED_COUNT = 8
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


def _comb_fit(proj: np.ndarray, min_s: int, max_s: int) -> tuple[int | None, int]:
    LED_COUNT = 8
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
    LED_COUNT = 8
    img = _render_bars(proj, proj.size)
    if top is None:
        return img
    for k in range(LED_COUNT):
        y = top + k * spacing
        if 0 <= y < img.shape[0]:
            img[y, :] = 128
    return img


# rowcopy = pole_row_sum.copy()

# pole_row_sum
# rowcopy

# _keep_pole_region(rowcopy)

# top, spacing = _comb_fit(rowcopy, 2, 80)


# c = _render_comb(rowcopy, top, spacing)

# c.shape
# c.dtype

# # show(c)

# show_many(
#     [
#         (combined_thresh, "h s v masks combined"),
#         (c, "comb?"),
#     ]
# )
