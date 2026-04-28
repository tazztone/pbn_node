import numpy as np
import pytest
from shapely.geometry import Polygon

from pbn_node.backend.labeling.label_placer import LabelPlacer
from pbn_node.backend.preprocessing.retinex import multiscale_retinex
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

    seg_with_line = RegionSegmenter(
        use_thin_cleanup=True, min_region_width=5, edge_weight_map=lineart
    )
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


@pytest.mark.unit
def test_auto_albedo_reduces_lighting_bias():
    """Verify that MSR Retinex reduces lighting bias relative to local detail."""
    # Create a 100x100 image with:
    # 1. A slow lighting gradient (y=0: 100, y=100: 200)
    # 2. A sharp color change (x=0-50: Red, x=50-100: Green)
    y, x = np.mgrid[0:100, 0:100]
    lighting = (y / 100.0 * 100 + 100).astype(np.float32) / 255.0

    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Left half: Red (0, 0, 200) BGR
    image[:, :50, 2] = (200 * lighting[:, :50]).astype(np.uint8)
    # Right half: Green (0, 200, 0) BGR
    image[:, 50:, 1] = (200 * lighting[:, 50:]).astype(np.uint8)

    albedo = multiscale_retinex(image)

    # In the input, the Red color at top-left (y=0) is darker than Red at bottom-left (y=100)
    # In the albedo, they should be more similar.
    c_top = albedo[5, 25, 2].astype(np.float32)
    c_bot = albedo[95, 25, 2].astype(np.float32)

    in_top = image[5, 25, 2].astype(np.float32)
    in_bot = image[95, 25, 2].astype(np.float32)

    in_diff = abs(in_bot - in_top)
    out_diff = abs(c_bot - c_top)

    # Out diff should be smaller than in diff (gradient was suppressed)
    assert out_diff < in_diff
