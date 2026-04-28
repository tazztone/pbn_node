"""
Vectorization module implementing contour detection, Visvalingam-Whyatt simplification,
and speckle removal.
"""

import heapq
from typing import cast

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ..models import RegionData

try:
    import bezier

    _HAS_BEZIER = True
except ImportError:
    _HAS_BEZIER = False


class Vectorizer:
    """
    Implements vectorization operations including contour detection,
    topology-preserving simplification, and speckle removal.
    """

    def __init__(self, use_bezier_smooth: bool = False):
        """Initialize vectorizer with default parameters.

        Args:
            use_bezier_smooth: Whether to fit cubic Bézier curves to simplified contours
        """
        self.speckle_threshold = 0.001  # 0.1% of total area
        self.use_bezier_smooth = use_bezier_smooth

    def find_contours(self, region_mask: np.ndarray) -> list[np.ndarray]:
        """
        Find contours using OpenCV.

        Args:
            region_mask: Binary mask for a single region

        Returns:
            List of contours as numpy arrays
        """
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return cast(list[np.ndarray], contours)

    def visvalingam_whyatt(self, contour: np.ndarray, tolerance: float) -> np.ndarray:
        """
        Apply Visvalingam-Whyatt algorithm for topology-preserving simplification.

        This algorithm progressively removes points with the smallest effective area
        until the desired tolerance is reached.

        Args:
            contour: Input contour as numpy array of shape (N, 1, 2) or (N, 2)
            tolerance: Minimum area threshold for point removal (in pixels²)

        Returns:
            Simplified contour
        """
        # Reshape contour to (N, 2) if needed
        if len(contour.shape) == 3:
            points = contour.reshape(-1, 2)
        else:
            points = contour.copy()

        # Need at least 3 points for a polygon
        if len(points) < 3:
            return contour

        # Calculate effective area for each point
        # Effective area = area of triangle formed by point and its neighbors
        def calculate_area(p1, p2, p3):
            """Calculate area of triangle formed by three points."""
            return abs(
                0.5 * (p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1]))
            )

        # Create list of points with their effective areas
        n = len(points)
        point_data = []

        for i in range(n):
            prev_idx = (i - 1) % n
            next_idx = (i + 1) % n
            area = calculate_area(points[prev_idx], points[i], points[next_idx])
            point_data.append({"index": i, "area": area, "removed": False})

        # Use min-heap to efficiently find points with smallest area
        heap = [(pd["area"], i) for i, pd in enumerate(point_data)]
        heapq.heapify(heap)

        # Remove points with area below tolerance
        removed_count = 0
        max_removable = n - 3  # Keep at least 3 points

        while heap and removed_count < max_removable:
            area, idx = heapq.heappop(heap)

            # Skip if already removed or area exceeds tolerance
            if point_data[idx]["removed"] or area > tolerance:
                continue

            # Mark as removed
            point_data[idx]["removed"] = True
            removed_count += 1

            # Recalculate areas for neighbors
            # Find previous non-removed point
            prev_idx = idx
            while True:
                prev_idx = (prev_idx - 1) % n
                if not point_data[prev_idx]["removed"]:
                    break

            # Find next non-removed point
            next_idx = idx
            while True:
                next_idx = (next_idx + 1) % n
                if not point_data[next_idx]["removed"]:
                    break

            # Update areas for neighbors
            for neighbor_idx in [prev_idx, next_idx]:
                if point_data[neighbor_idx]["removed"]:
                    continue

                # Find neighbors of neighbor
                nn_prev = neighbor_idx
                while True:
                    nn_prev = (nn_prev - 1) % n
                    if not point_data[nn_prev]["removed"]:
                        break

                nn_next = neighbor_idx
                while True:
                    nn_next = (nn_next + 1) % n
                    if not point_data[nn_next]["removed"]:
                        break

                new_area = calculate_area(points[nn_prev], points[neighbor_idx], points[nn_next])
                point_data[neighbor_idx]["area"] = new_area
                heapq.heappush(heap, (new_area, neighbor_idx))

        # Build simplified contour from remaining points
        simplified_points = [points[i] for i in range(n) if not point_data[i]["removed"]]

        # Convert back to original shape
        if len(contour.shape) == 3:
            return np.array(simplified_points).reshape(-1, 1, 2)
        else:
            return np.array(simplified_points)

    def remove_speckles(
        self, regions: dict[int, Polygon], colors: np.ndarray, threshold: float = 0.001
    ) -> dict[int, Polygon]:
        """
        Remove speckle regions smaller than threshold.

        Merges small regions into their nearest neighbor based on LAB color distance.

        Args:
            regions: Dictionary of region ID to Polygon
            colors: Color centers in LAB space
            threshold: Area threshold as fraction of total area (default 0.1%)

        Returns:
            Dictionary with speckles removed
        """
        # Calculate total area
        total_area = self.calculate_total_area(regions)
        min_area = total_area * threshold

        # Identify speckles
        speckles = []
        large_regions = {}

        for region_id, polygon in regions.items():
            if polygon.area < min_area:
                speckles.append((region_id, polygon))
            else:
                large_regions[region_id] = polygon

        # If no large regions, keep everything
        if not large_regions:
            return regions

        # Merge each speckle into nearest neighbor by color distance
        for speckle_id, speckle_polygon in speckles:
            if speckle_id >= len(colors):
                continue

            speckle_color = colors[speckle_id - 1]  # Assuming 1-indexed regions

            # Find nearest neighbor by LAB color distance
            min_distance = float("inf")
            nearest_id = None

            for region_id in large_regions.keys():
                if region_id >= len(colors):
                    continue

                region_color = colors[region_id - 1]
                distance = float(np.linalg.norm(speckle_color - region_color))

                if distance < min_distance:
                    min_distance = distance
                    nearest_id = region_id

            # Merge speckle into nearest neighbor
            if nearest_id is not None:
                try:
                    # Union the polygons
                    merged = unary_union([large_regions[nearest_id], speckle_polygon])

                    # Ensure we have a single Polygon
                    if merged.geom_type == "Polygon" and merged.is_valid:
                        large_regions[nearest_id] = merged
                    elif merged.geom_type == "MultiPolygon":
                        # Take the largest polygon from the multipolygon
                        largest = max(merged.geoms, key=lambda p: p.area)
                        if largest.is_valid:
                            large_regions[nearest_id] = largest
                except Exception:
                    # If merge fails, just skip this speckle
                    pass

        return large_regions

    def calculate_total_area(self, regions: dict[int, Polygon]) -> float:
        """
        Calculate total area of all regions.

        Args:
            regions: Dictionary of region ID to Polygon

        Returns:
            Total area in square pixels
        """
        return cast(float, sum(polygon.area for polygon in regions.values()))

    def vectorize(self, region_data: RegionData, simplification: float) -> dict[int, Polygon]:
        """
        Complete vectorization pipeline.

        Args:
            region_data: RegionData from segmentation
            simplification: Tolerance for simplification (0.5-2.0 pixels)

        Returns:
            Dictionary of simplified polygons
        """
        # Validate simplification parameter
        if not (0.5 <= simplification <= 2.0):
            raise ValueError("Simplification must be between 0.5 and 2.0")

        simplified_regions = {}

        for region_id, polygon in region_data.regions.items():
            try:
                # Get polygon exterior coordinates
                coords = np.array(polygon.exterior.coords)

                # Apply Visvalingam-Whyatt simplification
                # Convert tolerance from pixels to area (pixels²)
                tolerance_area = simplification**2
                simplified_coords = self.visvalingam_whyatt(coords, tolerance_area)

                # Bézier path smoothing
                if self.use_bezier_smooth and len(simplified_coords) > 3:
                    simplified_coords = self._apply_bezier_smoothing(simplified_coords)

                # Create new polygon
                if len(simplified_coords) >= 3:
                    simplified_polygon = Polygon(simplified_coords)

                    # Validate polygon
                    if simplified_polygon.is_valid and simplified_polygon.area > 0:
                        simplified_regions[region_id] = simplified_polygon
                    else:
                        # Keep original if simplification failed
                        simplified_regions[region_id] = polygon
                else:
                    # Keep original if too few points
                    simplified_regions[region_id] = polygon
            except Exception:
                # Keep original on any error
                simplified_regions[region_id] = polygon

        return simplified_regions

    def _apply_bezier_smoothing(
        self, coords: np.ndarray, num_points_per_curve: int = 5
    ) -> np.ndarray:
        """
        Fits cubic Bézier curves to the given coordinates to smooth out sharp edges.
        """
        if not _HAS_BEZIER:
            return coords

        # Flatten array if needed
        if len(coords.shape) == 3:
            coords = coords.reshape(-1, 2)

        # Needs at least 4 points to do anything meaningful with cubic bezier
        if len(coords) < 4:
            return coords

        smoothed_coords = []

        # Ensure contour points don't just repeat the first at the end for iteration math
        if np.array_equal(coords[0], coords[-1]):
            coords = coords[:-1]

        n = len(coords)
        if n < 4:
            return coords

        # To smoothly close the polygon, we wrap around using modulo.
        # However, to avoid sharp kinks at segments, we should step by 1 and use a
        # continuous spline like Catmull-Rom or B-spline.
        # Since we're using cubic beziers, we can step by 3 around the loop.
        # But stepping by 3 means we only cover multiples of 3.
        # To perfectly tile the closed loop, we can just pad the sequence to a multiple of 3
        # using the front points, then run the bezier on the padded set.

        # Calculate how many points we need to make (n - 1) a multiple of 3.
        # We need (n_padded - 1) % 3 == 0.
        # Number of segments = ceil((n-1)/3).
        # We'll just step by 3 until we cover the loop and wrap back to the start.

        for i in range(0, n, 3):
            # The control points wrap around the polygon naturally
            p0 = coords[i % n]
            p1 = coords[(i + 1) % n]
            p2 = coords[(i + 2) % n]
            p3 = coords[(i + 3) % n]

            nodes = np.asfortranarray(
                [
                    [p0[0], p1[0], p2[0], p3[0]],
                    [p0[1], p1[1], p2[1], p3[1]],
                ]
            )

            try:
                curve = bezier.Curve(nodes, degree=3)
                s_vals = np.linspace(0.0, 1.0, num_points_per_curve)
                points = curve.evaluate_multi(s_vals).T

                # If we are exactly at the end and closing to the start,
                # we don't want to re-add the overlapping start vertex unless we are finishing.
                # Actually, always appending [:-1] gives a seamless closed contour loop.
                smoothed_coords.extend(points[:-1].tolist())
            except Exception:
                smoothed_coords.extend([p0.tolist(), p1.tolist(), p2.tolist()])

        # Close the loop perfectly back to the start vertex
        if len(smoothed_coords) > 0:
            smoothed_coords.append(smoothed_coords[0])

        return np.array(smoothed_coords)
