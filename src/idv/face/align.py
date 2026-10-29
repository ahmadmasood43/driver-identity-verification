import cv2
import numpy as np

# Five-point template (eyes, nose tip, mouth corners) for 112x112 ArcFace input.
ARCFACE_TEMPLATE = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def similarity_transform(landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """2x3 rotation + uniform scale + translation mapping landmarks onto the template."""
    dst = ARCFACE_TEMPLATE * (size / 112.0)
    m, _ = cv2.estimateAffinePartial2D(np.asarray(landmarks, np.float32), dst, method=cv2.LMEDS)
    if m is None:
        raise ValueError("could not fit a similarity transform to the landmarks")
    return m


def align(image: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    m = similarity_transform(landmarks, size)
    return cv2.warpAffine(image, m, (size, size), borderValue=0.0)
