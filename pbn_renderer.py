"""
Raster renderer for Paint-by-Numbers results.
"""

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
        regions: dict[int, Polygon],
        labels: LabelData,
        palette: ColorPalette,
        width: int,
        height: int,
        mode: str = "colored",
        region_colors: dict[int, int] | None = None,
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
            region_colors: Optional mapping of region ID to color index (0-based)

        Returns:
            Rendered image in BGR format
        """
        is_colored = mode == "colored"
        is_outline_mode = mode in ("outline", "print_svg")

        # Create blank image
        if is_colored:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            # Both "outline" and "print_svg" have white backgrounds
            canvas = np.ones((height, width, 3), dtype=np.uint8) * 255

        # 1. Fill regions
        for region_id, polygon in regions.items():
            if is_colored:
                # Map region ID to color index
                color_idx = (
                    region_colors[region_id]
                    if (region_colors and region_id in region_colors)
                    else (region_id - 1)
                ) % len(palette.hex_colors)

                hex_color = palette.hex_colors[color_idx]
                rgb = self._hex_to_rgb(hex_color)
                color = (rgb[2], rgb[1], rgb[0])  # BGR
            else:
                color = (255, 255, 255)  # White

            # Convert polygon to contour
            points = np.array(polygon.exterior.coords, dtype=np.int32)
            cv2.fillPoly(canvas, [points], color)

            # 2. Draw outlines
            if is_outline_mode:
                cv2.polylines(canvas, [points], True, (0, 0, 0), 1)

        # 3. Draw labels
        for region_id, point in labels.positions.items():
            if region_id in labels.font_sizes:
                font_size = labels.font_sizes[region_id]
                x, y = int(point.x), int(point.y)

                # fontScale is roughly font_size / 24.0
                font_scale = font_size / 24.0
                text_color = (0, 0, 0)

                # Determine the paint number for this region
                if region_colors and region_id in region_colors:
                    color_idx = region_colors[region_id] % len(palette.hex_colors)
                    label_text = str(region_colors[region_id] + 1)
                else:
                    color_idx = (region_id - 1) % len(palette.hex_colors)
                    label_text = str(region_id)

                # Ensure visibility in colored mode
                if is_colored:
                    hex_color = palette.hex_colors[color_idx]
                    rgb = self._hex_to_rgb(hex_color)
                    # Relative luminance: 0.299R + 0.587G + 0.114B
                    luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                    if luminance < 128:
                        text_color = (255, 255, 255)

                cv2.putText(
                    canvas,
                    label_text,
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    1,
                    cv2.LINE_AA,
                )

        return canvas

    @staticmethod
    def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
        """Convert #RRGGBB hex string to (R, G, B) tuple."""
        h = hex_str.lstrip("#")
        vals = [int(h[i : i + 2], 16) for i in (0, 2, 4)]
        return (vals[0], vals[1], vals[2])
