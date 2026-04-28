"""
Raster renderer for Paint-by-Numbers results.
"""

from typing import Dict

import cv2
import numpy as np
from shapely.geometry import Polygon

from .backend.models import ColorPalette, LabelData


class PBNRenderer:
    """
    Renders PBN regions and labels to a raster image.
    """

    def render(
        self,
        regions: Dict[int, Polygon],
        labels: LabelData,
        palette: ColorPalette,
        width: int,
        height: int,
        mode: str = "colored",
    ) -> np.ndarray:
        """
        Render PBN to numpy array.

        Args:
            regions: Region ID to Polygon
            labels: Label data
            palette: Color palette
            width: Image width
            height: Image height
            mode: "colored" | "outline" | "print_svg"

        Returns:
            Rendered image in BGR format
        """
        # Create blank image
        if mode == "colored":
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # Both "outline" and "print_svg" have white backgrounds
            canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

        # 1. Fill regions
        for region_id, polygon in regions.items():
            # Get color
            if mode == "colored":
                # Map region ID to color (assuming 1-indexed regions)
                color_idx = (region_id - 1) % len(palette.hex_colors)
                hex_color = palette.hex_colors[color_idx].lstrip("#")
                # Convert hex to BGR
                color = tuple(int(hex_color[i : i + 2], 16) for i in (4, 2, 0))
            else:
                color = (255, 255, 255)  # White

            # Convert polygon to contour
            points = np.array(polygon.exterior.coords, dtype=np.int32)
            cv2.fillPoly(canvas, [points], color)

            # 2. Draw outlines for "outline" and "print_svg" modes
            if mode in ("outline", "print_svg"):
                cv2.polylines(canvas, [points], True, (0, 0, 0), 1)

        # 3. Draw labels
        for region_id, point in labels.positions.items():
            if region_id in labels.font_sizes:
                font_size = labels.font_sizes[region_id]
                x, y = int(point.x), int(point.y)

                # Draw text using OpenCV
                # fontScale is roughly font_size / 20.0
                font_scale = font_size / 24.0
                color = (0, 0, 0)

                # Check background color for "colored" mode to ensure text is visible
                if mode == "colored":
                    color_idx = (region_id - 1) % len(palette.hex_colors)
                    # Simple heuristic: if color is dark, use white text
                    hex_color = palette.hex_colors[color_idx].lstrip("#")
                    rgb = tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
                    brightness = sum(rgb) / 3
                    if brightness < 128:
                        color = (255, 255, 255)

                cv2.putText(
                    canvas,
                    str(region_id),
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    color,
                    1,
                    cv2.LINE_AA,
                )

        return canvas
