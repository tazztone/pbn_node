import numpy as np
import pytest

from pbn_node.backend.segmentation.segmenter import RegionSegmenter


@pytest.mark.unit
def test_lineart_strength_effectiveness():
    """Verify that changing lineart_strength actually changes the segmentation output."""
    # Create a 20x20 image with a vertical split
    # Color 1 on left, Color 2 on right
    mat = np.ones((20, 20), dtype=np.int32)
    mat[:, 10:] = 2

    # Create a lineart map with a medium-strength edge at the boundary
    # If the threshold is hardcoded to 0.8, a 0.6 edge will be ignored regardless of strength.
    lineart = np.zeros((20, 20), dtype=np.float32)
    lineart[:, 10] = 0.6  # Medium edge

    # Case 1: lineart_strength=0.1 -> threshold approx 0.94 -> 0.6 < 0.94 -> IGNORED
    seg_weak = RegionSegmenter(edge_weight_map=lineart, lineart_strength=0.1)
    # This run is just to initialize the segmenter state if needed,
    # but we'll focus on the unbalanced_mat cases below.
    _ = seg_weak._smooth_pbnify_vectorized(mat.copy())

    # Case 2: lineart_strength=1.0 -> threshold 0.4 -> 0.6 > 0.4 -> PROTECTED
    seg_strong = RegionSegmenter(edge_weight_map=lineart, lineart_strength=1.0)
    _ = seg_strong._smooth_pbnify_vectorized(mat.copy())

    # In the weak case, smoothing (9x9 majority) might overwrite the boundary
    # if it feels like it, but more importantly, the 'Phase 1 refinement'
    # override should NOT trigger for 0.6 if threshold > 0.6.
    # In the strong case, the refinement SHOULD trigger and keep column 10 as color 2.

    # Let's force a scenario where smoothing WOULD overwrite if not protected.
    # Fill most of the image with Color 1, except for column 10 which is Color 2.
    unbalanced_mat = np.ones((20, 20), dtype=np.int32)
    unbalanced_mat[:, 10] = 2

    # Majority filter in 9x9 window will see mostly 1s and overwrite the 2s at col 10
    # UNLESS refinement protects it.

    res_weak = seg_weak._smooth_pbnify_vectorized(unbalanced_mat.copy())
    res_strong = seg_strong._smooth_pbnify_vectorized(unbalanced_mat.copy())

    # Weak strength (0.1) -> threshold ~0.94. edge 0.6 < 0.94 -> No protection.
    # Majority filter overwrites col 10.
    assert np.all(res_weak[:, 10] == 1)

    # Strong strength (1.0) -> threshold 0.4. edge 0.6 > 0.4 -> Protected.
    # Refinement restores col 10.
    assert np.all(res_strong[:, 10] == 2)
