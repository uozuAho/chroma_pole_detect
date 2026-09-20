from pathlib import Path
from typing import Protocol

import numpy as np

from my_types import DetectCentroidResult


class Detector(Protocol):
    @property
    def name(self): ...

    def find_centroid(self, image_path: str | Path) -> DetectCentroidResult: ...
    def process_frame(self, image: np.ndarray) -> DetectCentroidResult: ...
