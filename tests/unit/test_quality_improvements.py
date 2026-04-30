import numpy as np
import pytest
from shapely.geometry import Polygon

from pbn_node.backend.labeling.label_placer import LabelPlacer
from pbn_node.backend.segmentation.segmenter import RegionSegmenter


@pytest.mark.unit
def test_lineart_preserves_boundaries():
    """Verify that lineart prevents majority vote from overwriting boundary pixels."""
    # Create a 20x20 image with a vertical split
    mat = np.ones((20, 20), dtype=np.int32)
    mat[:, 10:] = 2

    # Case: Strong lineart at the boundary (column 10)
    lineart = np.zeros((20, 20), dtype=np.float32)
    lineart[:, 10] = 1.0  # Strong edge

    seg = RegionSegmenter(edge_weight_map=lineart, lineart_strength=1.0)
    # The smoothing logic now explicitly preserves pixels where lineart > 0.5
    smoothed = seg._smooth_pbnify_vectorized(mat.copy())

    # Boundary at column 10 should be exactly color 2 as in original mat
    assert np.all(smoothed[:, 10] == 2)
    assert np.all(smoothed[:, 9] == 1)


@pytest.mark.unit
def test_thin_cleanup_veto_on_edge():
    """Verify that thin regions survive if they are on a lineart edge."""
    # Create a thin 2px horizontal stripe
    mat = np.ones((20, 20), dtype=np.int32)
    mat[10:12, :] = 2

    # Case 1: No lineart - 2px stripe < 5px min_width, should be merged
    seg_no_line = RegionSegmenter(use_thin_cleanup=True, min_region_width=5)
    cleaned_no_line = seg_no_line._thin_region_cleanup(mat.copy(), 5)
    assert np.all(cleaned_no_line == 1)

    # Case 2: Lineart on the stripe
    lineart = np.zeros((20, 20), dtype=np.float32)
    lineart[10:12, :] = 1.0

    seg_with_line = RegionSegmenter(use_thin_cleanup=True, min_region_width=5, edge_weight_map=lineart)
    cleaned_with_line = seg_with_line._thin_region_cleanup(mat.copy(), 5)
    assert 2 in cleaned_with_line  # Stripe survived veto


@pytest.mark.unit
def test_label_avoids_lineart_edge_nudge():
    """Verify that labels are nudged away from lineart edges."""
    # Create a U-shaped polygon where the center (50, 50) is INSIDE but we'll block it
    # Arms at x=0-20 and x=80-100, connected by bottom at y=80-100
    poly = Polygon([(0, 0), (20, 0), (20, 80), (80, 80), (80, 0), (100, 0), (100, 100), (0, 100)])
    regions = {1: poly}

    # Polylabel for this shape would land in one of the arms or the bottom.
    # Let's say it lands at (10, 50). We'll block that area.
    lineart = np.zeros((100, 100), dtype=np.float32)

    # Block the entire left arm
    lineart[:, 0:30] = 1.0

    placer = LabelPlacer(label_mode="polylabel", lineart=lineart)
    label_data = placer.place_labels(regions)
    pos = label_data.positions[1]

    # Position should have been nudged out of the left arm (x < 30)
    assert pos.x >= 30
