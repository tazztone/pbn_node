"""
SVG generation module for creating optimized paint-by-number templates.
"""

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
        self,
        regions: dict[int, Polygon],
        labels: LabelData,
        colors: ColorPalette,
        region_colors: dict[int, int] | None = None,
        shared_borders: dict[int, list] | None = None,
        use_shared_borders: bool = True,
        print_mode: bool = False,
    ) -> str:
        """
        Generate complete SVG content.

        Args:
            regions: Dictionary of region ID to Polygon
            labels: LabelData with positions and font sizes
            colors: ColorPalette with color information
            region_colors: Mapping of region ID to color index (0-based)
            shared_borders: Dictionary mapping region IDs to lists of shared LineStrings
            use_shared_borders: Whether to use shared borders for stroke
            print_mode: If True, all fills are white and strokes are black

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
        grouped_paths = self.group_paths_by_color(regions, colors, region_colors)

        # Generate paths for each color group

        for color_hex, region_ids in grouped_paths.items():
            fill_color = "#ffffff" if print_mode else color_hex

            if print_mode:
                fill_stroke = (
                    "none" if (use_shared_borders and shared_borders) else self.default_stroke_color
                )
            else:
                fill_stroke = color_hex  # Same as fill to prevent anti-aliasing gaps

            stroke_width_attr = (
                "" if fill_stroke == "none" else f' stroke-width="{self.default_stroke_width}"'
            )

            svg_parts.append(
                f'  <g fill="{fill_color}" stroke="{fill_stroke}"{stroke_width_attr}>\n'
            )

            for region_id in region_ids:
                if region_id in regions:
                    polygon = regions[region_id]
                    path_data = self._polygon_to_path(polygon)
                    svg_parts.append(f'    <path d="{path_data}" />\n')

            svg_parts.append("  </g>\n")

        # Add shared borders if enabled (only in print mode to avoid black lines on colored SVG)
        if use_shared_borders and shared_borders and print_mode:
            svg_parts.append(f'  <g fill="none" stroke="{self.default_stroke_color}" ')
            svg_parts.append(f'stroke-width="{self.default_stroke_width}">\n')

            # Use a set to avoid drawing the same border twice
            drawn_borders = set()

            for region_id, borders in shared_borders.items():
                if region_id not in regions:
                    continue
                for border in borders:
                    # Create a hashable representation of the LineString coordinates
                    coords = tuple(border.coords)
                    # A LineString can be traversed in two directions.
                    # Sorting coordinates lexically scrambles the geometry, which is wrong.
                    # We should check both original and reversed orders.
                    rev = tuple(reversed(coords))
                    canonical_coords = coords if coords <= rev else rev

                    if canonical_coords not in drawn_borders:
                        drawn_borders.add(canonical_coords)
                        path_data = self._linestring_to_path(border)
                        if path_data:
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

                # Use color index + 1 as the label text
                label_text = str(region_id)
                text_color = self.default_stroke_color

                if region_colors and region_id in region_colors:
                    color_idx = region_colors[region_id]
                    label_text = str(color_idx + 1)

                    # For colored mode, ensure text is readable against the background
                    if not print_mode:
                        hex_color = colors.hex_colors[color_idx]
                        h = hex_color.lstrip("#")
                        rgb = [int(h[i : i + 2], 16) for i in (0, 2, 4)]
                        # Relative luminance: 0.299R + 0.587G + 0.114B
                        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                        text_color = "#ffffff" if luminance < 128 else "#000000"

                svg_parts.append(f'    <text x="{x:.{self.coordinate_precision}f}" ')
                svg_parts.append(f'y="{y:.{self.coordinate_precision}f}" ')
                svg_parts.append(f'fill="{text_color}" ')
                svg_parts.append(f'font-size="{font_size}">{label_text}</text>\n')

        svg_parts.append("  </g>\n")

        # Close SVG
        svg_parts.append("</svg>")

        return "".join(svg_parts)

    def group_paths_by_color(
        self,
        regions: dict[int, Polygon],
        colors: ColorPalette,
        region_colors: dict[int, int] | None = None,
    ) -> dict[str, list[int]]:
        """
        Group regions by color for optimization.
        """
        grouped: dict[str, list[int]] = {}

        for region_id in regions.keys():
            # Map region ID to color
            if region_colors and region_id in region_colors:
                color_idx = region_colors[region_id]
            else:
                # Fallback to sequential modulo if no mapping exists
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

    def calculate_viewbox(self, regions: dict[int, Polygon]) -> tuple[float, float, float, float]:
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

    def _linestring_to_path(self, line) -> str:
        """
        Convert Shapely LineString to SVG path data.
        """
        coords = list(line.coords)
        if not coords:
            return ""

        x0, y0 = coords[0][0], coords[0][1]
        path_parts = [f"M {x0:.{self.coordinate_precision}f},{y0:.{self.coordinate_precision}f}"]

        for x, y in coords[1:]:
            path_parts.append(
                f" L {x:.{self.coordinate_precision}f},{y:.{self.coordinate_precision}f}"
            )

        return "".join(path_parts)
