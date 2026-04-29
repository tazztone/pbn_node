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
    def sample_data(self):
        # 100x100 canvas with one square region
        regions = {1: Polygon([(10, 10), (40, 10), (40, 40), (10, 40)])}
        labels = LabelData(positions={1: Point(25, 25)}, font_sizes={1: 20}, skipped_regions=set())
        palette = ColorPalette(
            colors=np.array([[50, 0, 0]]),  # LAB
            hex_colors=["#ff0000"],  # Red
            color_count=1,
        )
        return regions, labels, palette

    def test_render_colored(self, renderer, sample_data):
        regions, labels, palette = sample_data
        img = renderer.render(regions, labels, palette, 100, 100, mode="colored")

        assert img.shape == (100, 100, 3)
        # Check if the region is filled with red (BGR: 0, 0, 255)
        # We check center of the square
        assert np.array_equal(img[25, 25], [0, 0, 255])
        # Background should be black
        assert np.array_equal(img[5, 5], [0, 0, 0])

    def test_render_outline(self, renderer, sample_data):
        regions, labels, palette = sample_data
        img = renderer.render(regions, labels, palette, 100, 100, mode="outline")

        assert img.shape == (100, 100, 3)
        # Background should be white (255, 255, 255)
        assert np.array_equal(img[5, 5], [255, 255, 255])
        # Region fill should be white (outline mode doesn't fill)
        assert np.array_equal(img[25, 25], [255, 255, 255])
        # Border should be black (0, 0, 0)
        assert np.array_equal(img[10, 10], [0, 0, 0])

    def test_hex_to_rgb(self, renderer):
        assert renderer._hex_to_rgb("#ffffff") == (255, 255, 255)
        assert renderer._hex_to_rgb("#000000") == (0, 0, 0)
        assert renderer._hex_to_rgb("#ff0000") == (255, 0, 0)
        assert renderer._hex_to_rgb("#00ff00") == (0, 255, 0)

    def test_luminance_contrast(self, renderer, sample_data):
        # Dark background (black) should have white text label
        regions, labels, _ = sample_data
        dark_palette = ColorPalette(
            colors=np.array([[0, 0, 0]]),
            hex_colors=["#000000"],  # Black
            color_count=1,
        )
        img = renderer.render(regions, labels, dark_palette, 100, 100, mode="colored")

        # The region center is [25, 25].
        # In a black region, if there is a label, there should be some white pixels.
        # labels.positions[1] is at (25, 25).
        # We search in a wider vicinity because cv2.putText anchor is bottom-left.
        vicinity = img[0:60, 0:60]
        has_label_content = np.any(vicinity > 0)
        assert has_label_content

        # Specifically, check that the label is white-ish (>200 in all channels)
        # to account for anti-aliasing
        assert np.any(np.all(vicinity > 200, axis=2))
