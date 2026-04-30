import numpy as np
import pytest
from shapely.geometry import Point, Polygon

from pbn_node.backend.models import ColorPalette, LabelData
from pbn_node.pbn_renderer import PBNRenderer


class TestPBNRenderer:
    @pytest.fixture
    def renderer(self):
        return PBNRenderer()

    @pytest.fixture
    def palette(self):
        return ColorPalette(
            colors=np.array([[255, 0, 0], [0, 0, 255]]),  # RGB centers
            hex_colors=["#FF0000", "#0000FF"],
            color_count=2,
        )

    def test_fill_uses_region_colors_mapping(self, renderer, palette):
        # Region 1 is Blue (index 1), Region 2 is Red (index 0)
        regions = {
            1: Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            2: Polygon([(10, 0), (20, 0), (20, 10), (10, 10)]),
        }
        region_colors = {1: 1, 2: 0}
        labels = LabelData(positions={}, font_sizes={}, skipped_regions=set())

        canvas = renderer.render(regions, labels, palette, 20, 10, mode="colored", region_colors=region_colors)

        # BGR format: Red is (0, 0, 255), Blue is (255, 0, 0)
        assert np.all(canvas[5, 5] == [255, 0, 0]), "Region 1 should be blue"
        assert np.all(canvas[5, 15] == [0, 0, 255]), "Region 2 should be red"

    def test_label_text_shows_paint_number(self, renderer, palette, monkeypatch):
        # Mock cv2.putText to capture the text being drawn
        captured_text = []

        def mock_put_text(img, text, org, font_face, font_scale, color, thickness, line_type):
            captured_text.append(text)

        import cv2

        monkeypatch.setattr(cv2, "putText", mock_put_text)

        regions = {1: Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])}
        region_colors = {1: 4}  # Color index 4 -> Label "5"
        # We need a palette with at least 5 colors for modulo to work as expected in the test
        palette5 = ColorPalette(colors=np.zeros((5, 3)), hex_colors=["#000000"] * 5, color_count=5)

        labels = LabelData(positions={1: Point(5, 5)}, font_sizes={1: 12}, skipped_regions=set())

        renderer.render(regions, labels, palette5, 10, 10, mode="colored", region_colors=region_colors)

        assert "5" in captured_text, "Label should be '5' (color index 4 + 1)"

    def test_label_contrast_uses_mapped_color(self, renderer, monkeypatch):
        # Mock cv2.putText to capture the color
        captured_colors = []

        def mock_put_text(img, text, org, font_face, font_scale, color, thickness, line_type):
            captured_colors.append(color)

        import cv2

        monkeypatch.setattr(cv2, "putText", mock_put_text)

        # Black color (#000000) should get white text
        palette = ColorPalette(colors=np.zeros((1, 3)), hex_colors=["#000000"], color_count=1)
        regions = {1: Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])}
        region_colors = {1: 0}
        labels = LabelData(positions={1: Point(5, 5)}, font_sizes={1: 12}, skipped_regions=set())

        renderer.render(regions, labels, palette, 10, 10, mode="colored", region_colors=region_colors)

        assert captured_colors[0] == (255, 255, 255), "Dark background should have white text"

    def test_fallback_when_region_colors_is_none(self, renderer, palette):
        regions = {1: Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])}
        labels = LabelData(positions={}, font_sizes={}, skipped_regions=set())

        # Should fallback to (1-1)%2 = 0 (Red)
        canvas = renderer.render(regions, labels, palette, 10, 10, mode="colored", region_colors=None)
        assert np.all(canvas[5, 5] == [0, 0, 255]), "Fallback should use region_id - 1 mapping"

    def test_multiple_regions_same_color(self, renderer, palette, monkeypatch):
        captured_text = []

        def mock_put_text(img, text, org, font_face, font_scale, color, thickness, line_type):
            captured_text.append(text)

        import cv2

        monkeypatch.setattr(cv2, "putText", mock_put_text)

        regions = {
            1: Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),
            2: Polygon([(10, 0), (20, 0), (20, 10), (10, 10)]),
        }
        region_colors = {1: 1, 2: 1}  # Both Blue (Label "2")
        labels = LabelData(
            positions={1: Point(5, 5), 2: Point(15, 5)},
            font_sizes={1: 12, 2: 12},
            skipped_regions=set(),
        )

        renderer.render(regions, labels, palette, 20, 10, mode="colored", region_colors=region_colors)

        assert captured_text == ["2", "2"], "Both regions should have the same label"
