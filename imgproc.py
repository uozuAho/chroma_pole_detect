import math

import cv2
import numpy as np


def thresh_otsu(img):
    """Returns: otsu threshold value, thresholded image"""
    val, threshed = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return val, threshed


def to_grey(img: np.ndarray):
    if img.dtype == np.bool_:
        return (img * np.uint8(255)).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def bin2grey(sel: np.ndarray) -> np.ndarray:
    """DEPRECATED: use to_grey"""
    return (sel * np.uint8(255)).astype(np.uint8)


def to_rgb(img: np.ndarray):
    if img.dtype == np.bool_:
        return cv2.cvtColor(bin2grey(img), cv2.COLOR_GRAY2RGB)
    elif img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        # defensive copy as other conversions produce new image
        return img.copy()


def keep_top_k_pc(img: np.ndarray, k: float):
    cutoff = np.quantile(img, 1.0 - k / 100)
    return img >= cutoff


def drawdot(
    img: np.ndarray,
    xy: tuple[int | float, int | float],
    colour=(0, 0, 255),
    thickness=2,
):
    """Draws a dot on the given image"""
    xy = (math.floor(xy[0]), math.floor(xy[1]))
    return cv2.circle(img, xy, 2, colour, thickness)


def draw_vline(img: np.ndarray, x: int, colour=(0, 0, 255), thickness=1):
    """Draws a vertical line on the given image"""
    h = img.shape[0]
    return cv2.line(img, (x, 0), (x, h), colour, thickness)


def moments(img: np.ndarray):
    """Returns the weighted x,y moments of the image"""
    moments = cv2.moments(img)
    if moments["m00"] == 0:
        return None, None
    else:
        return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])


def render_1d_vals(
    vals: np.ndarray, width=256, height=256, horizontal=False
) -> np.ndarray:
    """Render a 1d array of values as an image, where bar height/width shows the value"""
    assert vals.ndim == 1
    img = np.zeros((height, width), dtype=np.uint8)
    peak = vals.max()
    sums_normed = vals / peak
    max_size = width if horizontal else height
    sums_scaled = (sums_normed * (max_size - 1)).astype(int)
    if horizontal:
        for idx, val in enumerate(sums_scaled):
            if val > 0:
                img[idx, 0:val] = 255
    else:
        for idx, val in enumerate(sums_scaled):
            if val > 0:
                img[height - val :, idx] = 255
    return img
