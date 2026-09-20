"""
Helpers for doing image processing in a REPL
"""

import atexit
import contextlib
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from imgproc import *

_gui_queue: queue.Queue = queue.Queue()
_gui_thread: threading.Thread | None = None
_SHOW_TIMEOUT = 2.0


def _gui_loop() -> None:
    """
    OpenCV's HighGUI is not thread-safe: every window call (imshow, waitKey,
    getWindowProperty, destroyWindow) must come from the same single thread.
    All window work is therefore funnelled here.
    """

    # name -> whether the window has ever been reported visible
    tracked: dict[str, bool] = {}
    while True:
        try:
            msg = _gui_queue.get(timeout=0.05)
        except queue.Empty:
            msg = None
        if msg is not None:
            kind, name, payload, done, errors = msg
            try:
                if kind == "show":
                    cv2.imshow(name, payload)
                    tracked.setdefault(name, False)
                    cv2.waitKey(1)
                elif kind == "close":
                    tracked.pop(name, None)
                    cv2.destroyWindow(name)
                elif kind == "quit":
                    with contextlib.suppress(cv2.error):
                        cv2.destroyAllWindows()
                    return
            except cv2.error as exc:
                errors.append(exc)
            finally:
                done.set()
            continue
        for name in list(tracked):
            try:
                visible = cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) >= 1
            except cv2.error:
                visible = False
            if visible:
                tracked[name] = True
            elif tracked[name]:
                with contextlib.suppress(cv2.error):
                    cv2.destroyWindow(name)
                del tracked[name]
        if tracked:
            cv2.waitKey(50)
        else:
            time.sleep(0.05)


def _ensure_gui_thread() -> None:
    global _gui_thread
    if _gui_thread is not None and _gui_thread.is_alive():
        return
    _gui_thread = threading.Thread(target=_gui_loop, name="cv-highgui", daemon=True)
    _gui_thread.start()


def _shutdown_gui() -> None:
    if _gui_thread is None or not _gui_thread.is_alive():
        return
    _gui_queue.put(("quit", "", None, threading.Event(), []))
    _gui_thread.join(timeout=_SHOW_TIMEOUT)


atexit.register(_shutdown_gui)


def load(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    assert img is not None
    return img


def show(img: np.ndarray, window_name="img"):
    _ensure_gui_thread()
    clean_img = bin2grey(img) if img.dtype == np.bool_ else img
    done = threading.Event()
    errors: list[cv2.error] = []
    _gui_queue.put(("show", window_name, clean_img.copy(), done, errors))
    done.wait(timeout=_SHOW_TIMEOUT)
    if errors:
        raise errors[0]


def show_many(items, window_name="img", cols=3):
    entries = []
    for item in items:
        if isinstance(item, tuple):
            img, label = item
        else:
            img, label = item, None
        if img.dtype == np.bool_:
            img = cv2.cvtColor(bin2grey(img), cv2.COLOR_GRAY2RGB)
        elif img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = img.copy()
        entries.append((img, label))

    rows = []
    for start in range(0, len(entries), cols):
        row = entries[start : start + cols]
        row_h = max(img.shape[0] for img, _ in row)
        has_label = any(label is not None for _, label in row)
        pad = 28 if has_label else 0
        row_imgs = []
        for img, label in row:
            h, w = img.shape[:2]
            new_w = max(1, round(w * row_h / h))
            if (new_w, row_h) != (w, h):
                interp = cv2.INTER_AREA if row_h < h else cv2.INTER_LINEAR
                img = cv2.resize(img, (new_w, row_h), interpolation=interp)
            if pad:
                padded = np.full((row_h + pad, new_w, 3), 255, dtype=np.uint8)
                padded[pad:] = img
                if label is not None:
                    cv2.putText(
                        padded,
                        str(label),
                        (4, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 0),
                        1,
                        cv2.LINE_AA,
                    )
                img = padded
            img = cv2.copyMakeBorder(
                img, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=(255, 255, 255)
            )
            row_imgs.append(img)
        rows.append(np.hstack(row_imgs))

    max_w = max(row.shape[1] for row in rows)
    rows = [
        np.hstack(
            [row, np.zeros((row.shape[0], max_w - row.shape[1], 3), dtype=row.dtype)]
        )
        if row.shape[1] < max_w
        else row
        for row in rows
    ]
    show(np.vstack(rows), window_name)
