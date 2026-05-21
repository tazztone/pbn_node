import pytest
from pbn_node.backend.svg_generation.svg_generator import SVGGenerator
from shapely.geometry import Polygon


@pytest.mark.unit
class TestSVGGenerator:
    @pytest.fixture
    def generator(self):
        return SVGGenerator()

    def test_calculate_viewbox_empty(self, generator):
        """Test that an empty regions dictionary returns the default viewBox."""
        viewbox = generator.calculate_viewbox({})
        assert viewbox == (0.0, 0.0, 100.0, 100.0)

    def test_calculate_viewbox_with_polygons(self, generator):
        """Test calculating the viewBox with mock polygons."""
        # Create simple mock polygons
        # Polygon 1 bounds: (10, 10, 20, 20)
        poly1 = Polygon([(10, 10), (20, 10), (20, 20), (10, 20)])
        # Polygon 2 bounds: (30, 30, 40, 40)
        poly2 = Polygon([(30, 30), (40, 30), (40, 40), (30, 40)])

        regions = {1: poly1, 2: poly2}

        # Combined bounds without padding should be: min_x=10, min_y=10, max_x=40, max_y=40
        # The calculate_viewbox function adds a padding of 2.0 to all sides
        # Expected: min_x=8.0, min_y=8.0, max_x=42.0, max_y=42.0
        expected_viewbox = (8.0, 8.0, 42.0, 42.0)

        viewbox = generator.calculate_viewbox(regions)
        assert viewbox == expected_viewbox
