"""Multiscale Retinex for shadow-free albedo estimation."""

from collections.abc import Sequence

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter  # type: ignore


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

    retinex_log = np.zeros_like(l_channel)
    for sigma in scales:
        # Gaussian filter with sigma
        blur = gaussian_filter(l_channel, sigma=sigma)
        blur = np.maximum(blur, 1.0)
        retinex_log += np.log(l_channel) - np.log(blur)

    retinex_log /= len(scales)

    # Convert from log-albedo back to linear albedo (R = I / L)
    albedo_lin = np.exp(retinex_log)

    # Restore original global mean luminance to maintain image brightness
    target_mean = np.mean(l_channel)
    current_mean = np.mean(albedo_lin)
    if current_mean > 1e-6:
        albedo_lin *= target_mean / current_mean

    # Clamp to valid L range [0, 255]
    lab[:, :, 0] = np.clip(albedo_lin, 0, 255)

    # Preserve chrominance channels (already in lab[:, :, 1:])
    lab[:, :, 1:] = np.clip(lab[:, :, 1:], 0, 255)
    result = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    return result
