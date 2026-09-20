from __future__ import annotations

import sys

import cv2
import numpy as np

from detectors.color_thresh import ColorThresh
from detectors.hue_gate import HueGate, Threshold
from my_types import LabeledImage

DETECTORS = [
    ColorThresh(),
    HueGate(name="hue_gate_fixed", threshold=Threshold.FIXED),
    HueGate(name="hue_gate_otsu", threshold=Threshold.OTSU),
    HueGate(name="hue_gate_bright20", threshold=Threshold.BRIGHTEST, keep_frac=0.2),
]


def _label_and_tile(steps: list[LabeledImage]) -> np.ndarray:
    """Convert every step to a 3-channel BGR image and lay them out in a grid."""
    height = max(step.shape[0] for _, step in steps)
    canvases = []
    for label, step in steps:
        if step.ndim == 2:
            step = cv2.cvtColor(step, cv2.COLOR_GRAY2BGR)
        canvas = np.full((height + 24, step.shape[1], 3), 200, dtype=np.uint8)
        canvas[24:, :, :] = cv2.resize(
            step, (step.shape[1], height), interpolation=cv2.INTER_NEAREST
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
    ncols = min(len(canvases), 4)
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


def _show_steps(steps: list[LabeledImage]) -> None:
    cv2.imshow("steps", _label_and_tile(steps))


def main() -> None:
    detector_name = sys.argv[1]
    device = int(sys.argv[2])
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        raise SystemExit(f"could not open webcam device {device}")

    detector = next((d for d in DETECTORS if d.name == detector_name), None)
    if detector is None:
        raise SystemExit(
            f"unknown detector '{detector_name}'; available: "
            + ", ".join(d.name for d in DETECTORS)
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_ms = int(1000 / fps)

    cv2.namedWindow("steps", cv2.WINDOW_NORMAL)

    print(f"webcam running with detector '{detector.name}'; press 'q' to quit")
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                print("failed to read a frame")
                break

            centroid, steps = detector.process_frame(image)

            overlay = steps[0][1].copy()
            cv2.putText(
                overlay,
                f"centroid: ({centroid[0]:.1f}, {centroid[1]:.1f})",
                (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cx, cy = int(centroid[0]), int(centroid[1])
            cv2.circle(overlay, (cx, cy), 10, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(overlay, (cx - 10, cy), (cx + 10, cy), (0, 0, 255), 2, cv2.LINE_AA)
            cv2.line(overlay, (cx, cy - 10), (cx, cy + 10), (0, 0, 255), 2, cv2.LINE_AA)
            _show_steps(steps + [("centroid", overlay)])

            if cv2.waitKey(frame_ms) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
