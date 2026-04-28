"""Multiscale Retinex for shadow-free albedo estimation."""

from collections.abc import Sequence

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def multiscale_retinex(
    image_bgr: np.ndarray,
    scales: Sequence[int] = (15, 80, 250),
) -> np.ndarray:
    """
    Apply Multiscale Retinex (MSR) to estimate intrinsic albedo.

    Operates on the L channel in LAB space, preserving chrominance.

    Args:
        image_bgr: Input BGR uint8 image.
        scales: Gaussian kernel sigmas.

    Returns:
        Albedo estimate in BGR uint8.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float64)
    l_channel = lab[:, :, 0]

    # Avoid log(0)
    l_channel = np.maximum(l_channel, 1.0)

    retinex = np.zeros_like(l_channel)
    for sigma in scales:
        # Gaussian filter with sigma
        blur = gaussian_filter(l_channel, sigma=sigma)
        blur = np.maximum(blur, 1.0)
        retinex += np.log(l_channel) - np.log(blur)

    retinex /= len(scales)

    # Normalize to [0, 255] using percentiles to be robust against outliers
    low, high = np.percentile(retinex, [1, 99])
    if high > low:
        retinex = np.clip(retinex, low, high)
        retinex = (retinex - low) / (high - low) * 255.0
    else:
        retinex = np.full_like(retinex, 128.0)

    lab[:, :, 0] = retinex
    # Clamp chrominance channels to valid range [0, 255] before uint8 cast
    lab[:, :, 1:] = np.clip(lab[:, :, 1:], 0, 255)
    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result
