from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from detectors.class_comb import ClassComb
from detectors.color_thresh import ColorThresh
from detectors.column_proj import ColumnProject
from detectors.comb_fit import CombFit
from detectors.hue_gate import HueGate, Threshold
from detectors.soft_gate import SoftGate
from detectors.structure import StructureDetector

PROC_IMG_ROOT = Path("proc_img")
REAL_CENTROIDS_PATH = Path("img/real_centroids.json")

DETECTORS = [
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


def _load_real_centroids() -> dict[str, tuple[float, float]]:
    """Return a {filename: (x, y)} map for the human-labeled centroids."""
    data = json.loads(REAL_CENTROIDS_PATH.read_text())
    return {
        entry["filename"]: (float(entry["x"]), float(entry["y"]))
        for entry in data["centroids"]
    }


def _save_steps(
    detector_name: str,
    steps: list[tuple[str, np.ndarray]],
    image_path: str | Path,
    real: tuple[float, float] | None,
    centroid: tuple[float, float] | None,
) -> None:
    """Save each processing step as a labeled interim image under proc_img/<algorithm>."""
    tiles = []
    for label, image in steps:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = cv2.resize(image, (400, 300))
        cv2.putText(
            image,
            label,
            (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(image)
    if real is not None:
        overlay = _draw_both_centroids(steps[0][1], centroid, real)
        overlay = cv2.resize(overlay, (400, 300))
        cv2.putText(
            overlay,
            "real + detected centroid",
            (8, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        tiles.append(overlay)
    grid = _tile_grid(tiles, cols=3)
    out_dir = PROC_IMG_ROOT / detector_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(image_path).stem}_steps.png"
    cv2.imwrite(str(out_path), grid)


def _tile_grid(tiles: list[np.ndarray], *, cols: int = 3) -> np.ndarray:
    """Arrange 2D color images into a grid row-major (left-to-right, top-to-bottom)."""
    rows = []
    for start in range(0, len(tiles), cols):
        row = tiles[start : start + cols]
        while len(row) < cols:
            row.append(np.zeros_like(tiles[0]))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def _draw_both_centroids(
    image: np.ndarray,
    centroid: tuple[float, float] | None,
    real: tuple[float, float] | None,
) -> np.ndarray:
    overlay = image.copy()
    if real is not None:
        x, y = round(real[0]), round(real[1])
        cv2.circle(overlay, (x, y), 10, (0, 0, 255), 2)
        cv2.putText(
            overlay,
            f"real: ({real[0]:.1f}, {real[1]:.1f})",
            (x + 12, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    if centroid is not None:
        x, y = round(centroid[0]), round(centroid[1])
        cv2.circle(overlay, (x, y), 10, (0, 255, 255), 2)
        cv2.putText(
            overlay,
            f"det: ({centroid[0]:.1f}, {centroid[1]:.1f})",
            (x + 12, y - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run light-pole detectors over a directory of images."
    )
    parser.add_argument("directory", type=Path, help="directory of test images")
    args = parser.parse_args()

    target = args.directory
    real_centroids = _load_real_centroids()
    if not target.is_dir():
        raise SystemExit(f"expected a directory, not: {target}")

    images = sorted(
        path for path in target.iterdir() if path.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise FileNotFoundError(f"no supported images found in: {target}")

    measurements_dir = Path("measurements")
    measurements_dir.mkdir(parents=True, exist_ok=True)

    summary_lines: list[str] = []
    avg_errors: list[tuple[str, float]] = []
    for detector in DETECTORS:
        print(f"running {detector.name}")
        errors_pct: list[float] = []
        errors_px: list[float] = []
        times_ms: list[float] = []
        detail_lines: list[str] = []
        for image_path in images:
            real_centroid = real_centroids[image_path.name]
            start = time.perf_counter()
            detected_centroid, steps = detector.find_centroid(image_path)
            times_ms.append(1000.0 * (time.perf_counter() - start))
            _save_steps(
                detector.name, steps, image_path, real_centroid, detected_centroid
            )

            error_px = (
                (detected_centroid[0] - real_centroid[0]) ** 2
                + (detected_centroid[1] - real_centroid[1]) ** 2
            ) ** 0.5
            h, w = steps[0][1].shape[:2]
            error_pct = 100.0 * error_px / (w * w + h * h) ** 0.5
            errors_pct.append(error_pct)
            errors_px.append(error_px)
            verdict = "PASS" if error_pct < 5.0 else "FAIL"
            detail_lines.append(
                f"{image_path.stem}: "
                f"det=({detected_centroid[0]:.1f}, {detected_centroid[1]:.1f}) "
                f"real=({real_centroid[0]:.1f}, {real_centroid[1]:.1f}) "
                f"err={error_px:.1f}px ({error_pct:.2f}%) {verdict} "
                f"[{times_ms[-1]:.1f} ms]"
            )

        avg = sum(errors_pct) / len(errors_pct)
        avg_errors.append((detector.name, avg))
        under_goal = sum(error < 5.0 for error in errors_pct)
        detail_lines.append(f"images under 5%: {under_goal}/{len(errors_pct)}")
        detail_lines.append(f"avg error % {avg:.2f}")
        detail_lines.append(f"avg error px {sum(errors_px) / len(errors_px):.1f}")
        detail_lines.append(f"min error % {min(errors_pct):.2f}")
        detail_lines.append(f"max error % {max(errors_pct):.2f}")
        detail_lines.append(f"median error % {statistics.median(errors_pct):.2f}")
        if len(errors_pct) > 1:
            detail_lines.append(f"stdev error % {statistics.stdev(errors_pct):.2f}")
        detail_lines.append(f"total time {sum(times_ms):.1f} ms")
        detail_lines.append(f"avg time/image {sum(times_ms) / len(times_ms):.1f} ms")
        (measurements_dir / f"{detector.name}.txt").write_text(
            "\n".join(detail_lines) + "\n"
        )

    ranked = sorted(avg_errors, key=lambda p: p[1])
    summary_lines.append("Detectors ranked by best average error:\n")
    for name, avg in ranked:
        summary_lines.append(f"{name}: {avg:.2f}%")
    (measurements_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n")


if __name__ == "__main__":
    main()
