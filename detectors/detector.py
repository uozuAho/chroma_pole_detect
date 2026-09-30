from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from my_types import DetectCentroidResult


@dataclass(frozen=True)
class AdjustableParam:
    name: str
    label: str
    kind: str  # "int", "float", "bool" or "enum"
    min: float = 0
    max: float = 1
    step: float = 1
    choices: tuple[str, ...] = ()


class Detector(Protocol):
    @property
    def name(self): ...

    def find_centroid(self, image_path: str | Path) -> DetectCentroidResult: ...
    def process_frame(self, image: np.ndarray) -> DetectCentroidResult: ...
    def adjustable_params(self) -> list[AdjustableParam]: ...
    def set_param(self, name: str, value) -> None: ...
    def get_param(self, name: str): ...
