import numpy as np


def cv_to_std_lab(lab: np.ndarray) -> np.ndarray:
    """
    Convert OpenCV LAB (L: 0-255, a: 0-255, b: 0-255) to standard LAB
    (L: 0-100, a: -128-127, b: -128-127).
    """
    std = np.zeros_like(lab, dtype=np.float32)
    std[..., 0] = lab[..., 0] * 100.0 / 255.0
    std[..., 1] = lab[..., 1] - 128.0
    std[..., 2] = lab[..., 2] - 128.0
    return std
