import numpy as np
import pytest
import skimage.segmentation

from pbn_node.backend.preprocessing.normal_features import augment_image_with_normals


@pytest.mark.unit
def test_normal_guided_slic_crease_preservation():
    """Verify that normal-augmented SLIC preserves edges where colors are similar."""
    # Create an image with two adjacent regions of the SAME color (Gray)
    # but DIFFERENT normals (Flat vs Angled)
    h, w = 100, 100
    image_lab = np.full((h, w, 3), [50, 128, 128], dtype=np.float32)  # Neutral gray in LAB

    # Create a normal map: Left half is flat (0,0,1), Right half is angled (0.7,0,0.7)
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[:, :50] = [0, 0, 1.0]  # Z-up
    normals[:, 50:] = [0.707, 0, 0.707]  # Angled 45 deg

    # Without normals, SLIC should see a uniform image and produce random/uniform segments
    skimage.segmentation.slic(
        image_lab, n_segments=2, compactness=1, start_label=1, channel_axis=-1
    )
    # Probability of the split being at x=50 is low

    # With normals
    augmented = augment_image_with_normals(image_lab, normals, normal_strength=1.0)
    print(f"Augmented shape: {augmented.shape}, max values: {augmented.max(axis=(0, 1))}")
    segments_with_norm = skimage.segmentation.slic(
        augmented, n_segments=10, compactness=0.1, start_label=1, channel_axis=-1
    )
    print(f"Unique segments: {np.unique(segments_with_norm)}")

    # Check that segments are mostly contained within one half (don't cross the crease)
    for seg_id in np.unique(segments_with_norm):
        mask = segments_with_norm == seg_id
        left_count = np.sum(mask[:, :50])
        right_count = np.sum(mask[:, 50:])
        total = left_count + right_count
        # Each segment should be >90% on one side
        assert max(left_count, right_count) / total > 0.9, f"Segment {seg_id} crosses crease!"


if __name__ == "__main__":
    test_normal_guided_slic_crease_preservation()
