import numpy as np
import pytest
from shapely.geometry import LineString, Point, Polygon

from pbn_node.backend.models import ColorPalette, LabelData
from pbn_node.backend.svg_generation.svg_generator import SVGGenerator


@pytest.fixture
def svg_generator():
    return SVGGenerator()


@pytest.fixture
def mock_colors():
    # Provide fake hex colors and some dummy np array for LAB
    return ColorPalette(
        colors=np.array([[50, 0, 0], [60, 10, -10], [70, 0, 20]]),
        hex_colors=["#ff0000", "#00ff00", "#0000ff"],
        color_count=3,
    )


@pytest.fixture
def mock_regions():
    return {1: Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]), 2: Polygon([(10, 0), (20, 0), (20, 10), (10, 10)])}


@pytest.fixture
def mock_labels():
    return LabelData(positions={1: Point(5, 5), 2: Point(15, 5)}, font_sizes={1: 12, 2: 12}, skipped_regions=set())


@pytest.mark.unit
class TestSVGGenerator:
    def test_initialization(self, svg_generator):
        assert svg_generator.coordinate_precision == 1
        assert svg_generator.default_stroke_width == 1
        assert svg_generator.default_stroke_color == "#000000"

    def test_calculate_viewbox_empty(self, svg_generator):
        min_x, min_y, max_x, max_y = svg_generator.calculate_viewbox({})
        assert (min_x, min_y, max_x, max_y) == (0.0, 0.0, 100.0, 100.0)

    def test_calculate_viewbox_with_regions(self, svg_generator, mock_regions):
        min_x, min_y, max_x, max_y = svg_generator.calculate_viewbox(mock_regions)
        # Bounds are 0,0 to 20,10. Padding is 2.0.
        # min_x = 0 - 2 = -2
        # min_y = 0 - 2 = -2
        # max_x = 20 + 2 = 22
        # max_y = 10 + 2 = 12
        assert min_x == -2.0
        assert min_y == -2.0
        assert max_x == 22.0
        assert max_y == 12.0

    def test_embed_color_palette(self, svg_generator, mock_colors):
        comment = svg_generator.embed_color_palette(mock_colors)
        assert "<!-- Color Palette:" in comment
        assert "1: #ff0000" in comment
        assert "2: #00ff00" in comment
        assert "3: #0000ff" in comment
        assert "-->" in comment

    def test_polygon_to_path(self, svg_generator):
        poly = Polygon([(0, 0), (10, 0), (10, 10)])
        path = svg_generator._polygon_to_path(poly)
        assert path == "M 0.0,0.0 L 10.0,0.0 L 10.0,10.0 L 0.0,0.0 Z"

    def test_polygon_to_path_empty(self, svg_generator):
        # shapely empty polygon doesn't have exterior coords
        # Let's mock a polygon with no exterior coords or handle ValueError
        # In practice, exterior coords might be empty. But we can just use an empty list manually if needed.
        # For an empty Shapely polygon:
        poly = Polygon()
        path = svg_generator._polygon_to_path(poly)
        assert path == ""

    def test_linestring_to_path(self, svg_generator):
        line = LineString([(0, 0), (10, 10), (20, 20)])
        path = svg_generator._linestring_to_path(line)
        assert path == "M 0.0,0.0 L 10.0,10.0 L 20.0,20.0"

    def test_linestring_to_path_empty(self, svg_generator):
        line = LineString()
        path = svg_generator._linestring_to_path(line)
        assert path == ""

    def test_group_paths_by_color(self, svg_generator, mock_regions, mock_colors):
        # 2 regions, mock_colors has 3 colors.
        # Fallback modulo logic:
        # Region 1 -> (1-1)%3 = 0 -> hex #ff0000
        # Region 2 -> (2-1)%3 = 1 -> hex #00ff00
        grouped = svg_generator.group_paths_by_color(mock_regions, mock_colors)
        assert "#ff0000" in grouped
        assert "#00ff00" in grouped
        assert grouped["#ff0000"] == [1]
        assert grouped["#00ff00"] == [2]

    def test_group_paths_by_color_with_mapping(self, svg_generator, mock_regions, mock_colors):
        region_colors = {1: 2, 2: 2}  # Both mapped to color index 2 (#0000ff)
        grouped = svg_generator.group_paths_by_color(mock_regions, mock_colors, region_colors)
        assert len(grouped) == 1
        assert "#0000ff" in grouped
        assert grouped["#0000ff"] == [1, 2]

    def test_generate_svg_colored_mode(self, svg_generator, mock_regions, mock_labels, mock_colors):
        # Default is colored mode (print_mode=False)
        svg = svg_generator.generate_svg(
            regions=mock_regions,
            labels=mock_labels,
            colors=mock_colors,
            region_colors={1: 0, 2: 1},  # 1: #ff0000, 2: #00ff00
            print_mode=False,
        )
        assert "<svg" in svg
        assert "viewBox=" in svg
        assert "<!-- Color Palette:" in svg
        # Check that we have groups with the right colors
        assert 'fill="#ff0000" stroke="#ff0000"' in svg
        assert 'fill="#00ff00" stroke="#00ff00"' in svg
        # Labels are region indices + 1 (so "1" and "2")
        # Color contrast:
        # #ff0000 -> lum: 0.299*255 = 76.2 < 128 -> #ffffff text
        # #00ff00 -> lum: 0.587*255 = 149.7 > 128 -> #000000 text
        assert 'fill="#ffffff" font-size="12">1</text>' in svg
        assert 'fill="#000000" font-size="12">2</text>' in svg

    def test_generate_svg_print_mode(self, svg_generator, mock_regions, mock_labels, mock_colors):
        # Print mode: fill should be #ffffff, stroke should be default or "none" if shared_borders
        svg = svg_generator.generate_svg(
            regions=mock_regions,
            labels=mock_labels,
            colors=mock_colors,
            region_colors={1: 0, 2: 1},
            print_mode=True,
            use_shared_borders=False,  # No shared borders -> stroke should be default "#000000"
        )
        assert 'fill="#ffffff" stroke="#000000"' in svg
        assert 'fill="#000000" font-size="12">1</text>' in svg
        assert 'fill="#000000" font-size="12">2</text>' in svg

    def test_generate_svg_with_shared_borders_print_mode(self, svg_generator, mock_regions, mock_labels, mock_colors):
        shared_borders = {1: [LineString([(10, 0), (10, 10)])], 2: [LineString([(10, 0), (10, 10)])]}
        svg = svg_generator.generate_svg(
            regions=mock_regions,
            labels=mock_labels,
            colors=mock_colors,
            region_colors={1: 0, 2: 1},
            shared_borders=shared_borders,
            use_shared_borders=True,
            print_mode=True,
        )
        # Groups should have no stroke (since shared borders handle it)
        assert 'fill="#ffffff" stroke="none"' in svg
        # The shared border should be drawn separately in a group
        assert f'fill="none" stroke="{svg_generator.default_stroke_color}"' in svg
        assert "M 10.0,0.0 L 10.0,10.0" in svg
