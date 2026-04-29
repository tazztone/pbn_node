"""
Utility to augment an image with surface-normal channels for SLIC.

SLIC in skimage treats ALL channels of the input as color features.
By appending normal-derived channels to the RGB (or LAB) image we
cause superpixel boundaries to respect 3D surface discontinuities.

Strategy
--------
Rather than appending raw XYZ (which has linear units), we append the
*angular gradient* of the normal field — a scalar map that is high at
surface creases and low on flat areas. This is more numerically
compatible with LAB color distances than raw XYZ vectors.

Additionally, we optionally append the normal curvature map (divergence
of the normals) as a second channel.
"""

import cv2
import numpy as np


def normal_angular_gradient(normal_map: np.ndarray) -> np.ndarray:
    """
    Compute per-pixel angular gradient magnitude from a normal map.

    Args:
        normal_map: [H, W, 3] float32, unit vectors in [-1, 1]

    Returns:
        [H, W] float32 in [0, 1] — high at surface creases
    """
    # Compute finite differences of normal components
    # Use Sobel to get smooth gradients
    grad = np.zeros(normal_map.shape[:2], dtype=np.float32)
    for c in range(3):
        gx = cv2.Sobel(normal_map[:, :, c], cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(normal_map[:, :, c], cv2.CV_32F, 0, 1, ksize=3)
        grad += gx**2 + gy**2
    grad = np.sqrt(grad)
    # Normalize to [0, 1]
    vmax = grad.max()
    if vmax > 1e-6:
        grad /= vmax
    return grad


def build_normal_feature_channels(
    normal_map: np.ndarray,
    normal_strength: float,
    target_scale: float = 50.0,
) -> np.ndarray:
    """
    Convert a normal map to extra feature channels compatible with LAB scale.

    LAB values are roughly in [0, 100] for L and [-128, 127] for a, b.
    We scale normal features to a similar range so SLIC balances them
    naturally. `target_scale` sets the maximum value of the normal channels
    (50.0 = half the L range — a reasonable starting point).

    Args:
        normal_map: [H, W, 3] float32 unit vectors
        normal_strength: 0–1 multiplier for the final channel weight
        target_scale: max output value for channel compatibility with LAB

    Returns:
        [H, W, 2] float32 — angular gradient + curvature channels
    """
    # Channel 1: angular gradient (crease detector)
    ang_grad = normal_angular_gradient(normal_map)  # [H, W], [0,1]

    # Channel 2: approximate mean curvature via Laplacian of each normal component
    curvature = np.zeros(normal_map.shape[:2], dtype=np.float32)
    for c in range(3):
        curvature += np.abs(cv2.Laplacian(normal_map[:, :, c], cv2.CV_32F, ksize=3))
    curvature_max = curvature.max()
    if curvature_max > 1e-6:
        curvature /= curvature_max  # normalize to [0,1]

    weight = normal_strength * target_scale
    # Channel 1: angular gradient (crease detector)
    ang_grad_weighted = ang_grad * weight
    # Channel 2: mean curvature
    curvature_weighted = curvature * weight

    out = np.stack([ang_grad_weighted, curvature_weighted], axis=2)
    return out.astype(np.float32)


def augment_image_with_normals(
    lab_image: np.ndarray,
    normal_map: np.ndarray,
    normal_strength: float,
) -> np.ndarray:
    """
    Append normal-derived feature channels to a LAB image for SLIC input.

    Args:
        lab_image: [H, W, 3] float32, LAB in OpenCV scale
        normal_map: [H, W, 3] float32 unit normal vectors
        normal_strength: 0–1 influence weight

    Returns:
        [H, W, 5] float32 — [L, a, b, ang_grad, curvature]
    """
    h, w = lab_image.shape[:2]
    # Resize normal_map if needed
    if normal_map.shape[:2] != (h, w):
        normal_map = cv2.resize(normal_map, (w, h), interpolation=cv2.INTER_LINEAR)
        # Re-normalize after resize
        norms = np.linalg.norm(normal_map, axis=2, keepdims=True).clip(min=1e-6)
        normal_map = normal_map / norms

    normal_channels = build_normal_feature_channels(normal_map, normal_strength)
    return np.concatenate([lab_image, normal_channels], axis=2)
