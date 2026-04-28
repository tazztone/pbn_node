"""
Label placement module implementing Polylabel algorithm for optimal label positioning.
"""

import time
from typing import Dict

import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import polylabel

from ..models import LabelData


class LabelPlacer:
    """
    Implements smart label placement using Polylabel algorithm with fallbacks
    and font size calculation based on inscribed circle radius.
    """

    def __init__(self, label_mode: str = "polylabel"):
        """Initialize label placer with default parameters.

        Args:
            label_mode: "polylabel" or "centroid"
        """
        self.initial_precision = 1.0
        self.min_precision = 0.01
        self.timeout_ms = 100
        self.min_region_area = 100  # pixels²
        self.min_font_size = 8
        self.max_font_size = 24
        self.font_size_factor = 0.6
        self.label_mode = label_mode

    def polylabel_placement(self, polygon: Polygon, precision: float = 1.0) -> Point:
        """
        Find visual center using polylabel algorithm.

        Polylabel finds the pole of inaccessibility - the most distant internal
        point from the polygon outline.

        Args:
            polygon: Input polygon
            precision: Precision for polylabel algorithm (higher = more accurate but slower)

        Returns:
            Point representing optimal label position
        """
        start_time = time.time()
        current_precision = precision

        while current_precision >= self.min_precision:
            try:
                # Check timeout
                elapsed_ms = (time.time() - start_time) * 1000
                if elapsed_ms > self.timeout_ms:
                    # Timeout - fall back to centroid
                    return polygon.centroid

                # Try polylabel with current precision
                label_point = polylabel(polygon, tolerance=current_precision)
                return label_point

            except Exception:
                # If polylabel fails, halve precision and try again
                current_precision /= 2

                if current_precision < self.min_precision:
                    # All attempts failed - fall back to centroid
                    return polygon.centroid

        # Fallback to centroid
        return polygon.centroid

    def calculate_font_size(self, polygon: Polygon) -> int:
        """
        Calculate appropriate font size for region.

        Uses formula: max(8, min(24, inscribed_circle_radius * 0.6))

        Args:
            polygon: Input polygon

        Returns:
            Font size in pixels
        """
        radius = self.inscribed_circle_radius(polygon)
        font_size = int(radius * self.font_size_factor)

        # Clamp to min/max range
        font_size = max(self.min_font_size, min(self.max_font_size, font_size))

        return font_size

    def should_skip_label(self, polygon: Polygon) -> bool:
        """
        Determine if label should be skipped for small regions.

        Args:
            polygon: Input polygon

        Returns:
            True if region is too small for a label
        """
        return bool(polygon.area < self.min_region_area)

    def inscribed_circle_radius(self, polygon: Polygon) -> float:
        """
        Calculate inscribed circle radius.

        Uses polylabel to find the pole of inaccessibility, which gives us
        the center and radius of the largest inscribed circle.

        Args:
            polygon: Input polygon

        Returns:
            Radius of largest inscribed circle in pixels
        """
        try:
            # Polylabel returns the center of the largest inscribed circle
            center = polylabel(polygon, tolerance=1.0)

            # Calculate distance to nearest edge
            # This is the radius of the inscribed circle
            radius = center.distance(polygon.exterior)

            return float(radius)
        except Exception:
            # Fallback: estimate from area
            # For a circle: area = π * r²
            # So r = sqrt(area / π)
            estimated_radius = np.sqrt(polygon.area / np.pi)
            return float(estimated_radius)

    def place_labels(self, regions: Dict[int, Polygon]) -> LabelData:
        """
        Complete label placement pipeline.

        Args:
            regions: Dictionary of region ID to Polygon

        Returns:
            LabelData with positions, font sizes, and skipped regions
        """
        positions = {}
        font_sizes = {}
        skipped_regions = set()

        for region_id, polygon in regions.items():
            # Check if region is too small
            if self.should_skip_label(polygon):
                skipped_regions.add(region_id)
                continue

            # Find optimal label position
            try:
                if self.label_mode == "polylabel":
                    label_position = self.polylabel_placement(polygon, self.initial_precision)
                else:
                    label_position = polygon.centroid

                # Verify position is within polygon
                if not polygon.contains(label_position):
                    # If not contained, use centroid as fallback
                    label_position = polygon.centroid

                positions[region_id] = label_position

                # Calculate font size
                font_size = self.calculate_font_size(polygon)
                font_sizes[region_id] = font_size

            except Exception:
                # Skip this region if placement fails
                skipped_regions.add(region_id)

        return LabelData(
            positions=positions, font_sizes=font_sizes, skipped_regions=skipped_regions
        )
