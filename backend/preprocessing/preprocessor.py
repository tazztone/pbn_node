"""
Image preprocessing module implementing bilateral filtering and histogram equalization.
"""

import cv2
import numpy as np


class Preprocessor:
    """
    Implements image preprocessing operations including edge-preserving blur
    and contrast enhancement.
    """

    def __init__(self):
        """Initialize preprocessor with default parameters."""
        # Bilateral filter parameters - strengthened for PBN
        self.bilateral_d = 15
        self.bilateral_sigma_color = 100
        self.bilateral_sigma_space = 100

        # CLAHE parameters
        self.clahe_clip_limit = 2.0
        self.clahe_tile_grid_size = (8, 8)

    def bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply edge-preserving bilateral filter.

        Args:
            image: Input image (BGR or RGB format)

        Returns:
            Filtered image with preserved edges
        """
        return cv2.bilateralFilter(
            image,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space,
        )

    def histogram_equalization(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

        Args:
            image: Input image (BGR or RGB format)

        Returns:
            Image with enhanced contrast
        """
        # Convert to LAB color space for better results
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid_size)
        l_channel_equalized = clahe.apply(l_channel)

        # Merge channels back
        lab_equalized = cv2.merge([l_channel_equalized, a_channel, b_channel])

        # Convert back to BGR
        return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)

    def preprocess(self, image: np.ndarray, use_clahe: bool = False) -> np.ndarray:
        """
        Complete preprocessing pipeline.

        Args:
            image: Input image
            use_clahe: Whether to apply contrast enhancement (usually bad for PBN)

        Returns:
            Preprocessed image (BGR format)
        """
        # Apply bilateral filter
        filtered = self.bilateral_filter(image)

        # Apply histogram equalization only if requested
        if use_clahe:
            filtered = self.histogram_equalization(filtered)

        return filtered
