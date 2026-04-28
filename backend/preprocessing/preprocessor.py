"""
Image preprocessing module implementing bilateral filtering and histogram equalization.
"""


import cv2
import numpy as np

from ..models import ImageMetadata


class Preprocessor:
    """
    Implements image preprocessing operations including edge-preserving blur
    and contrast enhancement.
    """

    def __init__(self):
        """Initialize preprocessor with default parameters."""
        # Bilateral filter parameters
        self.bilateral_d = 9
        self.bilateral_sigma_color = 75
        self.bilateral_sigma_space = 75

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
        clahe = cv2.createCLAHE(
            clipLimit=self.clahe_clip_limit, tileGridSize=self.clahe_tile_grid_size
        )
        l_channel_equalized = clahe.apply(l_channel)

        # Merge channels back
        lab_equalized = cv2.merge([l_channel_equalized, a_channel, b_channel])

        # Convert back to BGR
        return cv2.cvtColor(lab_equalized, cv2.COLOR_LAB2BGR)

    def detect_image_type(self, image: np.ndarray) -> str:
        """
        Detect image type (portrait vs landscape) by aspect ratio.

        Args:
            image: Input image

        Returns:
            "portrait" if height > width, "landscape" otherwise
        """
        height, width = image.shape[:2]
        return "portrait" if height > width else "landscape"

    def preprocess(self, image: np.ndarray) -> tuple[np.ndarray, ImageMetadata]:
        """
        Complete preprocessing pipeline.

        Args:
            image: Input image (BGR format from cv2.imread)

        Returns:
            Tuple of (preprocessed_image, metadata)
        """
        # Extract metadata
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        file_size = image.nbytes
        image_type = self.detect_image_type(image)

        metadata = ImageMetadata(
            width=width,
            height=height,
            channels=channels,
            file_size=file_size,
            image_type=image_type,
        )

        # Apply bilateral filter
        filtered = self.bilateral_filter(image)

        # Apply histogram equalization
        equalized = self.histogram_equalization(filtered)

        return equalized, metadata
