"""
Color quantization module implementing K-means clustering in LAB color space.
"""

import cv2
import numpy as np
import skimage.color
from kneed import KneeLocator
from sklearn.cluster import KMeans

from ..models import ColorPalette, PerceptionInputs


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

    def _blend_with_albedo(
        self, pixels: np.ndarray, albedo: np.ndarray, material_weight: float, h: int, w: int
    ) -> np.ndarray:
        """Helper to blend LAB pixels with albedo pixels."""
        if albedo.shape[:2] != (h, w):
            albedo = cv2.resize(albedo, (w, h), interpolation=cv2.INTER_LINEAR)
        albedo_lab = cv2.cvtColor(albedo, cv2.COLOR_BGR2LAB).astype(np.float32)
        albedo_pixels = albedo_lab.reshape(-1, 3)
        return material_weight * albedo_pixels + (1.0 - material_weight) * pixels

    def _get_otsu_mask(self, image: np.ndarray) -> np.ndarray:
        """Generate a binary mask using Otsu thresholding on the Value channel."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]
        _, mask = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return mask

    def kmeans_lab(
        self,
        image: np.ndarray,
        k: int,
        albedo: np.ndarray | None = None,
        material_weight: float = 0.5,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform K-means clustering in LAB color space with optional albedo blending.

        Args:
            image: Input image (BGR format)
            k: Number of clusters
            albedo: Optional albedo image (BGR format)
            material_weight: Weight for albedo in blending (0.0 to 1.0)

        Returns:
            Tuple of (quantized_image, cluster_centers_lab)
        """
        # Convert BGR to LAB
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        h, w = lab_image.shape[:2]
        pixels = lab_image.reshape(-1, 3)

        if albedo is not None:
            fit_pixels = self._blend_with_albedo(pixels, albedo, material_weight, h, w)
        else:
            fit_pixels = pixels

        # Calculate chroma (saturation) in LAB space for weighting
        a = fit_pixels[:, 1] - 128.0
        b = fit_pixels[:, 2] - 128.0
        chroma = np.sqrt(a**2 + b**2)

        # Add a small base weight
        weights = chroma + 10.0

        if self.protection_map is not None:
            pm_flat = self.protection_map.flatten()
            if len(pm_flat) == len(weights):
                final_weights = weights * pm_flat
            else:
                final_weights = weights
        else:
            final_weights = weights

        # Perform K-means clustering
        kmeans = KMeans(
            n_clusters=k, init="k-means++", max_iter=self.max_iterations, n_init=10, random_state=42
        )
        kmeans.fit(fit_pixels, sample_weight=final_weights)
        centers = kmeans.cluster_centers_

        # Predict labels for all pixels
        labels = kmeans.predict(fit_pixels)

        # Create quantized image
        quantized_pixels = centers[labels]
        quantized_lab = quantized_pixels.reshape(h, w, 3).astype(np.uint8)

        # Convert back to BGR
        quantized_bgr = cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2BGR)

        return quantized_bgr, centers

    def quantize_with_budget(
        self,
        image: np.ndarray,
        num_colors: int,
        segmentation_mask: np.ndarray,
        background_ids: list[int] | None = None,
        subject_priority: float = 2.0,
        albedo: np.ndarray | None = None,
        material_weight: float = 0.5,
    ) -> tuple[np.ndarray, ColorPalette]:
        """
        Perform quantization with semantic budget allocation.
        """
        from ..utils.color import cv_to_std_lab

        if background_ids is None:
            background_ids = [0]

        h, w = image.shape[:2]

        # Ensure mask matches image size
        if segmentation_mask.shape[:2] != (h, w):
            segmentation_mask = cv2.resize(
                segmentation_mask, (w, h), interpolation=cv2.INTER_NEAREST
            )

        # Identify segments
        unique_labels, counts = np.unique(segmentation_mask, return_counts=True)

        # Filter out very small segments (less than 0.5% of image)
        total_pixels = h * w
        valid_indices = counts > (total_pixels * 0.005)
        labels = unique_labels[valid_indices]
        areas = counts[valid_indices]

        if len(labels) <= 1:
            return self.quantize(image, num_colors, perception=None)

        # Calculate weights for each segment
        # Background segments get weight 1.0, others get subject_priority
        weights = np.array(
            [1.0 if label in background_ids else subject_priority for label in labels]
        )

        # Proportional allocation: k_seg proportional to (area * weight)
        effective_areas = areas * weights
        proportions = effective_areas / np.sum(effective_areas)

        k_total = num_colors
        # Minimum 2 colors per segment to ensure detail
        k_allocated = np.maximum(2, np.round(proportions * k_total)).astype(int)

        # Adjust to match k_total exactly
        for _ in range(k_total * 2):  # Bounded loop to prevent infinite risk
            current_sum = np.sum(k_allocated)
            if current_sum == k_total:
                break

            if current_sum > k_total:
                # Reduce from segment with largest k/proportion ratio
                idx = np.argmax(k_allocated / proportions)
                if k_allocated[idx] > 1:
                    k_allocated[idx] -= 1
                else:
                    # Fallback: find segment that IS > 1
                    reducible = np.where(k_allocated > 1)[0]
                    if len(reducible) == 0:
                        break  # Cannot reduce further safely
                    # Pick from reducible segments the one with largest ratio
                    idx_in_reducible = np.argmax(k_allocated[reducible] / proportions[reducible])
                    k_allocated[reducible[idx_in_reducible]] -= 1
            else:
                # Add to segment with smallest k/proportion ratio
                idx = np.argmin(k_allocated / proportions)
                k_allocated[idx] += 1

        # Collect palettes from each segment
        all_centers = []
        for i, label in enumerate(labels):
            seg_mask = segmentation_mask == label
            seg_pixels_bgr = image[seg_mask].reshape(-1, 1, 3)

            seg_albedo = None
            if albedo is not None:
                seg_albedo = albedo[seg_mask].reshape(-1, 1, 3)

            # Quantize segment
            _, seg_centers = self.kmeans_lab(
                seg_pixels_bgr, k_allocated[i], albedo=seg_albedo, material_weight=material_weight
            )
            all_centers.append(seg_centers)

        merged_centers_lab = np.vstack(all_centers)

        # Re-quantize the entire image with the final merged centers
        lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
        pixels = lab_image.reshape(-1, 3)

        if albedo is not None:
            fit_pixels = self._blend_with_albedo(pixels, albedo, material_weight, h, w)
        else:
            fit_pixels = pixels

        if self.use_ciede2000:
            std_pixels = cv_to_std_lab(fit_pixels)
            std_colors = cv_to_std_lab(merged_centers_lab)
            k_final = merged_centers_lab.shape[0]
            dists = np.zeros((pixels.shape[0], k_final), dtype=np.float32)
            for ki in range(k_final):
                dists[:, ki] = skimage.color.deltaE_ciede2000(std_pixels, std_colors[[ki]])
        else:
            diffs = fit_pixels[:, np.newaxis, :] - merged_centers_lab[np.newaxis, :, :]
            dists = np.sum(diffs**2, axis=2)

        labels_final = np.argmin(dists, axis=1)
        quantized_pixels = merged_centers_lab[labels_final]
        quantized_lab = quantized_pixels.reshape(h, w, 3).astype(np.uint8)
        quantized_image = cv2.cvtColor(quantized_lab, cv2.COLOR_LAB2BGR)

        # Generate Palette
        centers_lab_uint8 = merged_centers_lab.astype(np.uint8).reshape(1, -1, 3)
        centers_bgr = cv2.cvtColor(centers_lab_uint8, cv2.COLOR_LAB2BGR)
        centers_rgb = cv2.cvtColor(centers_bgr, cv2.COLOR_BGR2RGB)
        hex_colors = ["#{:02x}{:02x}{:02x}".format(*rgb) for rgb in centers_rgb[0]]

        palette = ColorPalette(
            colors=merged_centers_lab, hex_colors=hex_colors, color_count=len(merged_centers_lab)
        )

        return quantized_image, palette

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
        self,
        image: np.ndarray,
        num_colors: int | None = None,
        perception: PerceptionInputs | None = None,
    ) -> tuple[np.ndarray, ColorPalette]:
        """
        Complete quantization pipeline with optional perception-stack inputs.
        """
        if perception and perception.albedo is not None and perception.edge_influence > 0:
            # Blend original image with albedo weighted by (1 - edge_map * edge_influence)
            # Pixels at strong edges keep more original color; flat areas lean on albedo
            blend = perception.edge_influence
            if perception.lineart is not None:
                # edge_map dims [H,W] → [H,W,1] for broadcasting
                weight_map = perception.lineart[..., np.newaxis] * blend
                input_image = (
                    image.astype(np.float32) * (1 - weight_map)
                    + perception.albedo.astype(np.float32) * weight_map
                )
            else:
                # No lineart, uniform blend
                input_image = (
                    image.astype(np.float32) * (1 - blend)
                    + perception.albedo.astype(np.float32) * blend
                )
            input_image = np.clip(input_image, 0, 255).astype(np.uint8)
        else:
            input_image = image

        # Determine k
        if num_colors is None:
            k = self.auto_select_k(input_image)
        else:
            k = max(self.min_k, num_colors)

        # Use budget splitting if a segmentation mask is provided or requested via auto_mask
        seg_mask = perception.segmentation_mask if perception else None
        if seg_mask is None and perception and perception.use_auto_mask:
            seg_mask = self._get_otsu_mask(input_image)

        if seg_mask is not None and num_colors is not None and num_colors >= 4:
            return self.quantize_with_budget(
                input_image,
                k,
                seg_mask,
                background_ids=perception.background_ids if perception else [0],
                subject_priority=perception.subject_priority if perception else 2.0,
                albedo=perception.albedo if perception else None,
                material_weight=perception.material_weight if perception else 0.5,
            )

        # Perform initial quantization (with optional albedo)
        albedo = perception.albedo if perception else None
        material_weight = perception.material_weight if perception else 0.5

        quantized_image, centers_lab = self.kmeans_lab(
            input_image, k, albedo=albedo, material_weight=material_weight
        )

        # Post-KMeans perceptual palette merge
        if self.use_palette_merge and len(centers_lab) > 1:
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
            lab_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2LAB).astype(np.float32)
            h, w = lab_image.shape[:2]
            pixels = lab_image.reshape(-1, 3)

            # Use albedo for final assignment if present
            if albedo is not None:
                fit_pixels = self._blend_with_albedo(pixels, albedo, material_weight, h, w)
            else:
                fit_pixels = pixels

            if self.use_ciede2000:
                std_pixels = cv_to_std_lab(fit_pixels)
                std_colors = cv_to_std_lab(centers_lab)
                k_total = centers_lab.shape[0]
                dists = np.zeros((pixels.shape[0], k_total), dtype=np.float32)
                for ki in range(k_total):
                    dists[:, ki] = skimage.color.deltaE_ciede2000(std_pixels, std_colors[[ki]])
            else:
                diffs = fit_pixels[:, np.newaxis, :] - centers_lab[np.newaxis, :, :]
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
