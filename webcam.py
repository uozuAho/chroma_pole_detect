from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

from detectors.class_comb import ClassComb
from detectors.color_thresh import ColorThresh
from detectors.column_proj import ColumnProject
from detectors.comb_fit import CombFit
from detectors.detector import AdjustableParam, Detector
from detectors.hue_gate import HueGate, Threshold
from detectors.soft_gate import SoftGate
from detectors.structure import StructureDetector
from imgproc import to_rgb
from my_types import LabeledImage, PointXy

DETECTORS: list[Detector] = [
    ColorThresh(),
    HueGate(name="hue_gate_fixed", threshold=Threshold.FIXED),
    HueGate(name="hue_gate_otsu", threshold=Threshold.OTSU),
    HueGate(name="hue_gate_bright20", threshold=Threshold.BRIGHTEST, keep_frac=0.2),
    SoftGate(),
    ColumnProject(),
    StructureDetector(),
    CombFit(),
    ClassComb(),
]


def combine(steps: list[LabeledImage], max_cols=4) -> np.ndarray:
    """Combine all images into a single image"""
    height = max(step.shape[0] for _, step in steps)
    canvases = []
    for label, img in steps:
        if img.ndim == 2:
            img = to_rgb(img)
        canvas = np.full((height + 24, img.shape[1], 3), 200, dtype=np.uint8)
        canvas[24:, :, :] = cv2.resize(
            img, (img.shape[1], height), interpolation=cv2.INTER_NEAREST
        )
        cv2.putText(
            canvas,
            label,
            (8, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        canvases.append(canvas)

    rows = []
    ncols = min(len(canvases), max_cols)
    for i in range(0, len(canvases), ncols):
        row = canvases[i : i + ncols]
        width = max(c.shape[1] for c in row)
        rows.append(
            np.hstack(
                [
                    cv2.copyMakeBorder(
                        c, 0, 0, 0, width - c.shape[1], cv2.BORDER_CONSTANT, value=200
                    )
                    for c in row
                ]
            )
        )
    if len(rows) == 1:
        return rows[0]
    width = max(r.shape[1] for r in rows)
    rows = [
        cv2.copyMakeBorder(
            r, 0, 0, 0, width - r.shape[1], cv2.BORDER_CONSTANT, value=200
        )
        if r.shape[1] != width
        else r
        for r in rows
    ]
    return np.vstack(rows)


def find_detector_or_throw(name: str, detectors: list[Detector]):
    detector = next((d for d in detectors if d.name == name), None)
    if detector is None:
        raise SystemExit(
            f"unknown detector '{name}'; available: "
            + ", ".join(d.name for d in detectors)
        )
    return detector


def draw_centroid(img: np.ndarray, centroid: PointXy):
    cx, cy = int(centroid[0]), int(centroid[1])
    cv2.circle(img, (cx, cy), 10, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(img, (cx - 10, cy), (cx + 10, cy), (0, 0, 255), 2, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 10), (cx, cy + 10), (0, 0, 255), 2, cv2.LINE_AA)


def init_cam(device: int):
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"could not open webcam device {device}")
    return cap


RADIO_RADIUS = 8
RADIO_ROW_HEIGHT = 28
RADIO_LEFT_X = 20
RADIO_PANEL_WIDTH = 300


def trackbar_count(param: AdjustableParam) -> int:
    if param.kind == "int":
        return int(param.max - param.min)
    if param.kind == "float":
        return int((param.max - param.min) / param.step)
    if param.kind == "bool":
        return 1
    raise ValueError(f"unknown param kind '{param.kind}'")


def decode_position(param: AdjustableParam, pos: int):
    if param.kind == "int":
        return int(param.min + pos)
    if param.kind == "float":
        return param.min + pos * param.step
    if param.kind == "bool":
        return bool(pos)
    raise ValueError(f"unknown param kind '{param.kind}'")


class RadioPanel:
    """Radio-button group for enum params, drawn HighGUI-style: circles and
    labels rendered on a canvas, with a mouse callback to pick a choice."""

    def __init__(self, detector: Detector, params: list[AdjustableParam]):
        self.detector = detector
        self.params = params
        self.window = ""
        self.rows: list[tuple[str, str, int, int]] = []
        self.canvas = self._render()

    def attach(self, window: str) -> None:
        self.window = window
        cv2.setMouseCallback(window, self.on_mouse)
        self.show()

    def show(self) -> None:
        cv2.imshow(self.window, self.canvas)

    def _render(self) -> np.ndarray:
        n_rows = sum(len(param.choices) + 1 for param in self.params)
        canvas = np.full(
            (n_rows * RADIO_ROW_HEIGHT + 16, RADIO_PANEL_WIDTH, 3), 200, dtype=np.uint8
        )
        self.rows = []
        y = 20
        for param in self.params:
            cv2.putText(
                canvas,
                param.label,
                (RADIO_LEFT_X, y + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            y += RADIO_ROW_HEIGHT
            value = self.detector.get_param(param.name)
            for choice in param.choices:
                cy = y + RADIO_ROW_HEIGHT // 2
                cx = RADIO_LEFT_X + RADIO_RADIUS + 6
                cv2.circle(canvas, (cx, cy), RADIO_RADIUS, (0, 0, 0), 1, cv2.LINE_AA)
                if choice == value:
                    cv2.circle(
                        canvas, (cx, cy), RADIO_RADIUS - 3, (0, 0, 255), -1, cv2.LINE_AA
                    )
                cv2.putText(
                    canvas,
                    choice,
                    (cx + RADIO_RADIUS + 8, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )
                self.rows.append((param.name, choice, cx, cy))
                y += RADIO_ROW_HEIGHT
        return canvas

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        for name, choice, cx, cy in self.rows:
            if (
                cx - RADIO_RADIUS - 4 <= x <= cx + RADIO_RADIUS + 4
                and cy - RADIO_RADIUS - 4 <= y <= cy + RADIO_RADIUS + 4
            ):
                self.detector.set_param(name, choice)
                self.canvas = self._render()
                self.show()
                break


def init_controls(detector: Detector, window: str) -> None:
    params = detector.adjustable_params()
    if not params:
        return
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    enums = [param for param in params if param.kind == "enum"]
    if enums:
        RadioPanel(detector, enums).attach(window)

    for param in params:
        if param.kind == "enum":
            continue
        value = detector.get_param(param.name)
        if param.kind == "float":
            pos = round((value - param.min) / param.step)
        elif param.kind == "bool":
            pos = int(bool(value))
        else:
            pos = int(value - param.min)

        def onchange(pos: int, param=param):
            detector.set_param(param.name, decode_position(param, pos))

        cv2.createTrackbar(param.label, window, pos, trackbar_count(param), onchange)


def check_fonts():
    if not Path(".venv/lib/python3.14/site-packages/cv2/qt/fonts/").exists():
        lines = (
            "HEY: .venv/lib/python3.14/site-packages/cv2/qt/fonts/  is missing",
            "run ./fontfix",
            "This is why trackbar labels are not working in the controls window",
        )
        for x in lines:
            print(f"####  {x:<80} #####")
        print()


def main() -> None:
    check_fonts()
    detector_name = sys.argv[1]
    cam_device = int(sys.argv[2])

    detector = find_detector_or_throw(detector_name, DETECTORS)
    cap = init_cam(cam_device)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_ms = int(1000 / fps)

    cv2.namedWindow("cam", cv2.WINDOW_NORMAL)
    init_controls(detector, "controls")

    print(f"webcam running with detector '{detector.name}'; press 'q' to quit")

    try:
        while True:
            ok, image = cap.read()
            if not ok:
                print("failed to read a frame")
                break

            image = cv2.flip(image, 1)
            centroid, steps = detector.process_frame(image)
            centroid_img = image.copy()
            draw_centroid(centroid_img, centroid)
            steps.append(("centroid", centroid_img))
            cv2.imshow("cam", combine(steps, max_cols=3))

            if cv2.waitKey(frame_ms) & 0xFF in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
