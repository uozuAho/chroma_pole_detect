from collections.abc import Iterable

import cv2

from scratchpads.scratch_helper import *

IMGS = sorted(Path("img").glob("*.jpg"))
IMG_PATH = IMGS[1]


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


image = load(IMG_PATH)
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv)

# otsu threshold (adaptive, kinda like k-means to find foreground)
val, v_threshed = thresh_otsu(v)

height, width = v_threshed.shape
col_sum = v_threshed.sum(axis=0).astype(np.float64)

combined_thresh = led_hues_mask(h) & (s >= 60) & v_threshed.astype(np.bool_)

show_many(
    [
        (led_hues_mask(h), "LED hue mask"),
        ((s >= 60), "sat mask"),
        (v_threshed, "v otsu thresh"),
        (combined_thresh, "h s v masks combined"),
        (
            drawdot(to_rgb(combined_thresh), moments(bin2grey(combined_thresh))),
            "combined centroid",
        ),
    ]
)
