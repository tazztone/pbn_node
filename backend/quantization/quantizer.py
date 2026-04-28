"""
Color quantization module implementing K-means clustering in LAB color space.
"""

from typing import Optional, Tuple

import cv2
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans

from ..models import ColorPalette


class ColorQuantizer:
    """
    Implements color quantization using K-means clustering in perceptually
    accurate LAB color space with automatic k selection.
    """

    def __init__(
        self,
        use_palette_merge: bool = True,
        ciede2000_merge_thresh: float = 8.0,
        use_ciede2000: bool = True,
    ):
        """Initialize quantizer with default parameters."""
        self.min_k = 2
        self.max_k = 40
        self.k_cap = 30
        self.max_iterations = 100
        self.monochrome_variance_threshold = 0.05  # 5%
        self.monochrome_k = 3
        self.use_palette_merge = use_palette_merge
        self.ciede2000_merge_thresh = ciede2000_merge_thresh
        self.use_ciede2000 = use_ciede2000
        self.protection_map = None

    def kmeans_lab(self, image: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Perform K-means clustering in LAB color space with saturation-biased sampling.

        Args:
            image: Input image (BGR format)
            k: Number of clusters

        Returns:
            Tuple of (quantized_image, cluster_centers_lab)
        """
        # Convert BGR to LAB
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Reshape image to 2D array of pixels
        h, w = lab_image.shape[:2]
        pixels = lab_image.reshape(-1, 3).astype(np.float32)

        # Calculate chroma (saturation) in LAB space: sqrt(a^2 + b^2)
        # In OpenCV LAB, a and b are offset by 128
        a = pixels[:, 1] - 128.0
        b = pixels[:, 2] - 128.0
        chroma = np.sqrt(a**2 + b**2)

        # Add a small base weight to prevent dark/gray areas from being ignored entirely
        weights = chroma + 10.0

        if self.protection_map is None:
            final_weights = weights
        else:
            # Flatten protection map and multiply with chroma weights
            pm_flat = self.protection_map.flatten()
            if len(pm_flat) == len(weights):
                final_weights = weights * pm_flat
            else:
                final_weights = weights

        # Perform K-means clustering with k-means++ initialization, passing weights
        kmeans = KMeans(
            n_clusters=k, init="k-means++", max_iter=self.max_iterations, n_init=10, random_state=42
        )
        kmeans.fit(pixels, sample_weight=final_weights)
        centers = kmeans.cluster_centers_

        # Predict labels for all pixels (without weights) to create the quantized image
        labels = kmeans.predict(pixels)

        # Create quantized image
        quantized_pixels = centers[labels]
        quantized_lab = quantized_pixels.reshape(h, w, 3).astype(np.uint8)

        # Convert back to BGR
        quantized_bgr = cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2BGR)

        return quantized_bgr, centers

    def auto_select_k(self, image: np.ndarray) -> int:
        """
        Automatically select optimal k using KneeLocator algorithm.

        Args:
            image: Input image (BGR format)

        Returns:
            Optimal number of clusters (between min_k and k_cap)
        """
        # Check for monochrome first
        if self.detect_monochrome(image):
            return self.monochrome_k

        # Convert to LAB
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        pixels = lab_image.reshape(-1, 3).astype(np.float32)

        # Sample pixels if image is too large (for performance)
        if len(pixels) > 10000:
            indices = np.random.choice(len(pixels), 10000, replace=False)
            pixels = pixels[indices]

        # Calculate inertia for different k values
        k_range = range(self.min_k, min(self.max_k + 1, len(pixels)))
        inertias = []

        for k in k_range:
            kmeans = KMeans(
                n_clusters=k,
                init="k-means++",
                max_iter=self.max_iterations,
                n_init=3,
                random_state=42,
            )
            kmeans.fit(pixels)
            inertias.append(kmeans.inertia_)

        # Use KneeLocator to find elbow point
        try:
            kneedle = KneeLocator(
                list(k_range), inertias, curve="convex", direction="decreasing", online=True
            )

            optimal_k = kneedle.elbow if kneedle.elbow else self.min_k + 5
        except Exception:
            # Fallback to middle of range if KneeLocator fails
            optimal_k = (self.min_k + self.max_k) // 2

        # Cap k at maximum
        optimal_k = min(optimal_k, self.k_cap)

        return optimal_k

    def detect_monochrome(self, image: np.ndarray) -> bool:
        """
        Detect if image is monochrome (variance < 5%).

        Args:
            image: Input image (BGR format)

        Returns:
            True if image is monochrome, False otherwise
        """
        # Convert to LAB
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

        # Calculate variance in a and b channels (color channels)
        a_channel = lab_image[:, :, 1]
        b_channel = lab_image[:, :, 2]

        # Normalize variance to 0-1 range
        a_variance = np.var(a_channel) / (255**2)
        b_variance = np.var(b_channel) / (255**2)

        # Average variance of color channels
        color_variance = (a_variance + b_variance) / 2

        return bool(color_variance < self.monochrome_variance_threshold)

    def quantize(
        self, image: np.ndarray, num_colors: Optional[int] = None
    ) -> Tuple[np.ndarray, ColorPalette]:
        """
        Complete quantization pipeline.

        Args:
            image: Input image (BGR format)
            num_colors: Number of colors (None for auto-detection)

        Returns:
            Tuple of (quantized_image, color_palette)
        """
        # Determine k
        if num_colors is None:
            k = self.auto_select_k(image)
        else:
            # Enforce constraints on manual k
            k = max(self.min_k, min(num_colors, self.k_cap))

        # Perform quantization
        quantized_image, centers_lab = self.kmeans_lab(image, k)

        # Post-KMeans perceptual palette merge
        if self.use_palette_merge and len(centers_lab) > 1:
            import skimage.color

            from ..utils.color import cv_to_std_lab

            std_centers = cv_to_std_lab(centers_lab)

            merged_centers = []
            used = set()

            for i in range(len(std_centers)):
                if i in used:
                    continue

                cluster_group = [i]
                used.add(i)
                current_group_mean = std_centers[i].copy()

                for j in range(i + 1, len(std_centers)):
                    if j in used:
                        continue

                    # Compute CIEDE2000 distance against current group mean
                    dist = skimage.color.deltaE_ciede2000(
                        current_group_mean[np.newaxis, :], std_centers[[j]]
                    )[0]

                    if dist < self.ciede2000_merge_thresh:
                        cluster_group.append(j)
                        used.add(j)
                        # Update group mean
                        # We can average the std_centers directly for the running mean comparison
                        current_group_mean = np.mean(std_centers[cluster_group], axis=0)

                # Average the merged LAB colors in standard LAB, then convert back?
                # Actually, averaging in CV2 LAB is fine since it's a linear mapping.
                merged_lab = np.mean(centers_lab[cluster_group], axis=0)
                merged_centers.append(merged_lab)

            centers_lab = np.array(merged_centers, dtype=np.float32)

            # Re-quantize the image with the merged centers
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
            h, w = lab_image.shape[:2]
            pixels = lab_image.reshape(-1, 3)

            if self.use_ciede2000:
                std_pixels = cv_to_std_lab(pixels)
                # centers_lab is now `merged_centers`, we need to convert to std lab
                std_colors = cv_to_std_lab(centers_lab)

                K = centers_lab.shape[0]
                dists = np.zeros((pixels.shape[0], K), dtype=np.float32)

                for ki in range(K):
                    dists[:, ki] = skimage.color.deltaE_ciede2000(std_pixels, std_colors[[ki]])
            else:
                # Assign pixels to the closest new merged center using standard euclidean distance
                diffs = pixels[:, np.newaxis, :] - centers_lab[np.newaxis, :, :]
                dists = np.sum(diffs**2, axis=2)

            labels = np.argmin(dists, axis=1)

            quantized_pixels = centers_lab[labels]
            quantized_lab = quantized_pixels.reshape(h, w, 3).astype(np.uint8)
            quantized_image = cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2BGR)

            merged_k = len(centers_lab)

        # Convert LAB centers to RGB for hex representation
        centers_lab_uint8 = centers_lab.astype(np.uint8).reshape(1, -1, 3)
        centers_bgr = cv2.cvtColor(centers_lab_uint8, cv2.COLOR_LAB2BGR)
        centers_rgb = cv2.cvtColor(centers_bgr, cv2.COLOR_BGR2RGB)

        # Generate hex colors
        hex_colors = []
        for rgb in centers_rgb[0]:
            hex_color = "#{:02x}{:02x}{:02x}".format(rgb[0], rgb[1], rgb[2])
            hex_colors.append(hex_color)

        palette = ColorPalette(colors=centers_lab, hex_colors=hex_colors, color_count=merged_k if 'merged_k' in locals() else k)

        return quantized_image, palette
