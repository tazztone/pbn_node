import numpy as np
import pytest
from pbn_node.backend.models import RegionData
from pbn_node.backend.vectorization.vectorizer import Vectorizer
from shapely.geometry import Polygon


def test_init():
    """Test Vectorizer initialization."""
    vectorizer = Vectorizer()
    assert not vectorizer.use_bezier_smooth
    assert vectorizer.speckle_threshold == 0.0005

    vectorizer_bezier = Vectorizer(use_bezier_smooth=True)
    assert vectorizer_bezier.use_bezier_smooth


def test_find_contours():
    """Test contour detection on a simple mask."""
    vectorizer = Vectorizer()
    # Create a simple 10x10 mask with a 5x5 square in the middle
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:7, 2:7] = 255

    contours = vectorizer.find_contours(mask)

    assert len(contours) == 1
    # Contour should have points. The exact number depends on opencv version,
    # but for a 5x5 square with CHAIN_APPROX_SIMPLE it's usually 4.
    assert len(contours[0]) >= 4


def test_visvalingam_whyatt():
    """Test topology-preserving simplification."""
    vectorizer = Vectorizer()

    # Create a simple polygon with 5 points, one of which is almost collinear
    # A(0,0), B(10,0), C(10,10), D(5, 9.9), E(0,10)
    # Point D should be removed first as its effective area is very small
    points = np.array([[0, 0], [10, 0], [10, 10], [5, 9.9], [0, 10]], dtype=np.float32)

    # Calculate effective area of D: base is C to E (width 10). Height is 0.1
    # Area = 0.5 * 10 * 0.1 = 0.5

    # Test with very low tolerance - no points removed
    simplified_low = vectorizer.visvalingam_whyatt(points, tolerance=0.1)
    assert len(simplified_low) == 5

    # Test with higher tolerance - point D removed
    simplified_high = vectorizer.visvalingam_whyatt(points, tolerance=1.0)
    assert len(simplified_high) == 4

    # Verify D (5, 9.9) is the point removed
    remaining_points = {(p[0], p[1]) for p in simplified_high}
    assert (5.0, 9.9) not in remaining_points


def test_calculate_total_area():
    """Test total area calculation."""
    vectorizer = Vectorizer()

    # Create two regions with known areas
    regions = {
        1: Polygon([(0, 0), (10, 0), (10, 10), (0, 10)]),  # Area: 100
        2: Polygon([(0, 0), (5, 0), (5, 5), (0, 5)]),  # Area: 25
    }

    total_area = vectorizer.calculate_total_area(regions)
    assert total_area == 125.0


def test_remove_speckles():
    """Test speckle removal functionality."""
    vectorizer = Vectorizer()

    # Create a large region and a small speckle
    # Make them touch so unary_union can merge them successfully
    large_poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)])  # Area: 10000
    speckle_poly = Polygon([(100, 100), (101, 100), (101, 101), (100, 101)])  # Area: 1

    regions = {
        1: large_poly,
        2: speckle_poly,
    }

    region_colors = {
        1: 0,  # Map to color index 0
        2: 1,  # Map to color index 1
    }

    # Color centers in LAB space
    # Make them somewhat close so merging happens
    colors = np.array(
        [
            [50, 0, 0],  # Color 0
            [55, 0, 0],  # Color 1
        ],
        dtype=np.float32,
    )

    # Threshold of 0.01 (1%) means area < 100 is a speckle
    cleaned_regions, updated_colors = vectorizer.remove_speckles(regions, region_colors, colors, threshold=0.01)

    # Region 2 (speckle) should be merged into Region 1
    assert 2 not in updated_colors
    assert len(cleaned_regions) == 1
    assert 1 in cleaned_regions

    # When shapes only touch at a single point, unary_union creates a MultiPolygon.
    # The `remove_speckles` logic specifically handles this:
    # "elif merged.geom_type == 'MultiPolygon': Take the largest polygon from the multipolygon"
    # So the speckle is discarded from the result, keeping area the same, but it is removed from region_colors.
    assert cleaned_regions[1].area == large_poly.area


def test_vectorize():
    """Test full vectorization pipeline."""
    vectorizer = Vectorizer()

    # Create a region with some unnecessary points
    polygon = Polygon([(0, 0), (10, 0), (10, 10), (5, 9.9), (0, 10)])

    # Create RegionData
    region_data = RegionData(
        regions={1: polygon}, region_colors={1: 0}, segmented_matrix=np.zeros((20, 20), dtype=np.int32)
    )

    # Run vectorization with high enough simplification to remove (5, 9.9)
    # The simplification parameter gets squared and scaled, so we need a large enough value
    simplified_regions = vectorizer.vectorize(region_data, simplification=2.0)

    assert 1 in simplified_regions

    # Simplification might not trigger if total area is too small,
    # but let's test it at least returns valid polygons
    result_poly = simplified_regions[1]
    assert result_poly.is_valid
    assert result_poly.geom_type == "Polygon"


def test_vectorize_invalid_simplification():
    """Test vectorization with invalid parameters."""
    vectorizer = Vectorizer()
    region_data = RegionData(regions={}, region_colors={}, segmented_matrix=np.zeros((10, 10)))

    with pytest.raises(ValueError, match="Simplification must be at least 0.0"):
        vectorizer.vectorize(region_data, simplification=-1.0)


def test_bezier_smoothing():
    """Test cubic Bezier smoothing."""
    # Ensure bezier is available for this test to be meaningful
    try:
        import bezier  # noqa

        has_bezier = True
    except ImportError:
        has_bezier = False

    if not has_bezier:
        pytest.skip("bezier module not installed")

    vectorizer = Vectorizer(use_bezier_smooth=True)

    # A simple square
    coords = np.array([[0, 0], [10, 0], [10, 10], [0, 10]], dtype=np.float32)

    # Apply smoothing
    smoothed = vectorizer._apply_bezier_smoothing(coords, num_points_per_curve=5)

    # The result should have more points than the input (smoothing adds points)
    # Because we step by 3 and pad, it generates multiple curves
    assert len(smoothed) > len(coords)

    # Ensure shape is correct
    assert smoothed.shape[1] == 2
