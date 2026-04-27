"""
SVG generation module for creating optimized paint-by-number templates.
"""

from typing import Dict, List, Tuple

from shapely.geometry import Polygon

from ..models import ColorPalette, LabelData


class SVGGenerator:
    """
    Implements SVG generation with coordinate precision, path grouping,
    color palette embedding, and tight bounding box calculation.
    """

    def __init__(self):
        """Initialize SVG generator with default parameters."""
        self.coordinate_precision = 1  # Decimal places
        self.default_stroke_width = 1
        self.default_stroke_color = "#000000"

    def generate_svg(
        self, regions: Dict[int, Polygon], labels: LabelData, colors: ColorPalette
    ) -> str:
        """
        Generate complete SVG content.

        Args:
            regions: Dictionary of region ID to Polygon
            labels: LabelData with positions and font sizes
            colors: ColorPalette with color information

        Returns:
            Complete SVG as string
        """
        # Calculate viewBox
        min_x, min_y, max_x, max_y = self.calculate_viewbox(regions)
        width = max_x - min_x
        height = max_y - min_y

        # Start SVG
        svg_parts = []
        svg_parts.append('<?xml version="1.0" encoding="UTF-8"?>')
        svg_parts.append('<svg xmlns="http://www.w3.org/2000/svg" ')
        svg_parts.append(f'viewBox="{min_x:.{self.coordinate_precision}f} ')
        svg_parts.append(f"{min_y:.{self.coordinate_precision}f} ")
        svg_parts.append(f"{width:.{self.coordinate_precision}f} ")
        svg_parts.append(f'{height:.{self.coordinate_precision}f}" ')
        svg_parts.append(f'width="{width:.{self.coordinate_precision}f}" ')
        svg_parts.append(f'height="{height:.{self.coordinate_precision}f}">')
        svg_parts.append("\n")

        # Embed color palette as comment
        svg_parts.append(self.embed_color_palette(colors))
        svg_parts.append("\n")

        # Group paths by color
        grouped_paths = self.group_paths_by_color(regions, colors)

        # Generate paths for each color group
        for color_hex, region_ids in grouped_paths.items():
            svg_parts.append(f'  <g fill="{color_hex}" stroke="{self.default_stroke_color}" ')
            svg_parts.append(f'stroke-width="{self.default_stroke_width}">\n')

            for region_id in region_ids:
                if region_id in regions:
                    polygon = regions[region_id]
                    path_data = self._polygon_to_path(polygon)
                    svg_parts.append(f'    <path d="{path_data}" />\n')

            svg_parts.append("  </g>\n")

        # Add labels
        svg_parts.append('  <g font-family="Arial, sans-serif" text-anchor="middle" ')
        svg_parts.append('dominant-baseline="middle">\n')

        for region_id, position in labels.positions.items():
            if region_id in labels.font_sizes:
                font_size = labels.font_sizes[region_id]
                x = position.x
                y = position.y

                svg_parts.append(f'    <text x="{x:.{self.coordinate_precision}f}" ')
                svg_parts.append(f'y="{y:.{self.coordinate_precision}f}" ')
                svg_parts.append(f'font-size="{font_size}">{region_id}</text>\n')

        svg_parts.append("  </g>\n")

        # Close SVG
        svg_parts.append("</svg>")

        return "".join(svg_parts)

    def group_paths_by_color(
        self, regions: Dict[int, Polygon], colors: ColorPalette
    ) -> Dict[str, List[int]]:
        """
        Group regions by color for optimization.

        Args:
            regions: Dictionary of region ID to Polygon
            colors: ColorPalette with color information

        Returns:
            Dictionary mapping hex color to list of region IDs
        """
        grouped: Dict[str, List[int]] = {}

        for region_id in regions.keys():
            # Map region ID to color (assuming 1-indexed regions)
            color_idx = (region_id - 1) % len(colors.hex_colors)
            color_hex = colors.hex_colors[color_idx]

            if color_hex not in grouped:
                grouped[color_hex] = []

            grouped[color_hex].append(region_id)

        return grouped

    def embed_color_palette(self, colors: ColorPalette) -> str:
        """
        Generate color palette as SVG comment.

        Args:
            colors: ColorPalette with color information

        Returns:
            SVG comment string with color palette
        """
        comment_parts = ["  <!-- Color Palette:\n"]

        for i, hex_color in enumerate(colors.hex_colors, 1):
            comment_parts.append(f"    {i}: {hex_color}\n")

        comment_parts.append("  -->")

        return "".join(comment_parts)

    def calculate_viewbox(self, regions: Dict[int, Polygon]) -> Tuple[float, float, float, float]:
        """
        Calculate tight bounding box for viewBox.

        Args:
            regions: Dictionary of region ID to Polygon

        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        if not regions:
            return (0.0, 0.0, 100.0, 100.0)

        min_x = float("inf")
        min_y = float("inf")
        max_x = float("-inf")
        max_y = float("-inf")

        for polygon in regions.values():
            bounds = polygon.bounds  # (minx, miny, maxx, maxy)
            min_x = min(min_x, bounds[0])
            min_y = min(min_y, bounds[1])
            max_x = max(max_x, bounds[2])
            max_y = max(max_y, bounds[3])

        # Add small padding
        padding = 2.0
        min_x -= padding
        min_y -= padding
        max_x += padding
        max_y += padding

        return (min_x, min_y, max_x, max_y)

    def _polygon_to_path(self, polygon: Polygon) -> str:
        """
        Convert Shapely polygon to SVG path data.

        Args:
            polygon: Input polygon

        Returns:
            SVG path data string
        """
        coords = list(polygon.exterior.coords)

        if not coords:
            return ""

        # Start path with Move command
        x0, y0 = coords[0][0], coords[0][1]
        path_parts = [f"M {x0:.{self.coordinate_precision}f},{y0:.{self.coordinate_precision}f}"]

        # Add Line commands for remaining points
        for x, y in coords[1:]:
            path_parts.append(
                f" L {x:.{self.coordinate_precision}f},{y:.{self.coordinate_precision}f}"
            )

        # Close path
        path_parts.append(" Z")

        return "".join(path_parts)
