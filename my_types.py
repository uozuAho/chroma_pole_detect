import numpy as np

type PointXy = tuple[float, float]
type LabeledImage = tuple[str, np.ndarray]

# centroid + interim processing steps
type DetectCentroidResult = tuple[PointXy, list[LabeledImage]]
