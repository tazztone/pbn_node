"""
Color quantization module implementing K-means clustering in LAB color space.
"""

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
        self.k_cap = 20
        self.max_iterations = 100
        self.monochrome_variance_threshold = 0.05  # 5%
        self.monochrome_k = 3
        self.use_palette_merge = use_palette_merge
        self.ciede2000_merge_thresh = ciede2000_merge_thresh
        self.use_ciede2000 = use_ciede2000
        self.protection_map: np.ndarray | None = None

    def kmeans_lab(self, image: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
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
        self, image: np.ndarray, num_colors: int | None = None
    ) -> tuple[np.ndarray, ColorPalette]:
        """
        Complete quantization pipeline with pairwise merge and budget enforcement.
        """
        # Determine k
        if num_colors is None:
            k = self.auto_select_k(image)
        else:
            k = max(
                self.min_k, num_colors
            )  # Allow manual k to exceed k_cap if explicitly requested

        # Perform initial quantization
        quantized_image, centers_lab = self.kmeans_lab(image, k)

        # Post-KMeans perceptual palette merge
        if self.use_palette_merge and len(centers_lab) > 1:
            import skimage.color

            from ..utils.color import cv_to_std_lab

            # We use pairwise merging: find closest pair, merge, repeat.
            # This is more robust than sequential merging.

            # Target k: if num_colors was provided, we MUST hit it.
            # If auto-selected, we use ciede2000_merge_thresh.
            target_k = num_colors if num_colors is not None else 0
            current_thresh = self.ciede2000_merge_thresh

            while True:
                std_centers = cv_to_std_lab(centers_lab)
                n = len(std_centers)
                if n <= 1 or (target_k > 0 and n <= target_k):
                    break

                # Compute all-pairs distances
                best_dist = float("inf")
                best_pair = (-1, -1)

                for i in range(n):
                    for j in range(i + 1, n):
                        dist = skimage.color.deltaE_ciede2000(
                            std_centers[i][np.newaxis, :], std_centers[j][np.newaxis, :]
                        )[0]
                        if dist < best_dist:
                            best_dist = dist
                            best_pair = (i, j)

                # If we have a target_k, we merge regardless of threshold until we hit it.
                # If no target_k, we only merge if best_dist < threshold.
                if target_k == 0 and best_dist > current_thresh:
                    break

                # Merge best_pair
                i, j = best_pair
                new_center = np.mean(centers_lab[[i, j]], axis=0)

                # Remove i and j, add new_center
                mask = np.ones(n, dtype=bool)
                mask[[i, j]] = False
                centers_lab = np.vstack([centers_lab[mask], new_center[np.newaxis, :]])

            # Re-quantize the image with the merged centers
            lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
            h, w = lab_image.shape[:2]
            pixels = lab_image.reshape(-1, 3)

            if self.use_ciede2000:
                std_pixels = cv_to_std_lab(pixels)
                std_colors = cv_to_std_lab(centers_lab)
                k_total = centers_lab.shape[0]
                dists = np.zeros((pixels.shape[0], k_total), dtype=np.float32)
                for ki in range(k_total):
                    dists[:, ki] = skimage.color.deltaE_ciede2000(std_pixels, std_colors[[ki]])
            else:
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
            hex_color = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            hex_colors.append(hex_color)

        palette = ColorPalette(
            colors=centers_lab,
            hex_colors=hex_colors,
            color_count=merged_k if "merged_k" in locals() else k,
        )

        return quantized_image, palette
