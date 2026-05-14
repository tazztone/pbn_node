import pytest
from pbn_node.backend.labeling.label_placer import LabelPlacer
from shapely.geometry import Polygon


@pytest.mark.unit
class TestLabelPlacer:
    def test_should_skip_label_valid_large(self):
        """Test that a large enough polygon is not skipped."""
        placer = LabelPlacer()
        # Create a 20x20 square, area = 400 > 120
        poly = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)])
        assert placer.should_skip_label(poly) is False

    def test_should_skip_label_valid_small(self):
        """Test that a small polygon is skipped."""
        placer = LabelPlacer()
        # Create a 5x5 square, area = 25 < 120
        poly = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        assert placer.should_skip_label(poly) is True

    def test_should_skip_label_edge_case(self):
        """Test the exact boundary of min_region_area."""
        placer = LabelPlacer()
        # min_region_area is 120. Create a 10 x 12 rectangle, area = 120
        poly = Polygon([(0, 0), (10, 0), (10, 12), (0, 12)])
        assert poly.area == 120
        # should_skip_label returns poly.area < self.min_region_area
        # so 120 < 120 is False
        assert placer.should_skip_label(poly) is False

    def test_should_skip_label_invalid(self):
        """Test that an invalid polygon is skipped."""
        placer = LabelPlacer()
        # Self-intersecting polygon is invalid
        poly = Polygon([(0, 0), (10, 10), (0, 10), (10, 0)])
        assert poly.is_valid is False
        assert placer.should_skip_label(poly) is True

    def test_should_skip_label_none(self):
        """Test that None input is skipped."""
        placer = LabelPlacer()
        assert placer.should_skip_label(None) is True

    def test_should_skip_label_empty(self):
        """Test that an empty polygon is skipped."""
        placer = LabelPlacer()
        poly = Polygon()
        assert poly.is_empty is True
        assert placer.should_skip_label(poly) is True
