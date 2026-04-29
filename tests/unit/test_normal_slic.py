import numpy as np
import pytest
import skimage.segmentation

from pbn_node.backend.preprocessing.normal_features import augment_image_with_normals


@pytest.mark.unit
def test_normal_guided_slic_crease_preservation():
    """Verify that normal-augmented SLIC preserves edges where colors are similar."""
    # Create an image with two adjacent regions of the SAME color (Gray)
    # but DIFFERENT normals (Flat vs Angled)
    h, w = 200, 200
    image_lab = np.full((h, w, 3), [50, 128, 128], dtype=np.float32)  # Neutral gray in LAB

    # Create a normal map: Left half is flat (0,0,1), Right half is angled (0.7,0,0.7)
    normals = np.zeros((h, w, 3), dtype=np.float32)
    normals[:, :100] = [0, 0, 1.0]  # Z-up
    normals[:, 100:] = [0.707, 0, 0.707]  # Angled 45 deg

    # With normals
    augmented = augment_image_with_normals(image_lab, normals, normal_strength=1.0)
    segments_with_norm = skimage.segmentation.slic(
        augmented, n_segments=4, compactness=0.1, start_label=1, channel_axis=-1
    )

    # Check that segments are mostly contained within one half (don't cross the crease)
    for seg_id in np.unique(segments_with_norm):
        mask = segments_with_norm == seg_id
        left_count = np.sum(mask[:, :100])
        right_count = np.sum(mask[:, 100:])
        total = left_count + right_count
        # Each segment should be >80% on one side
        assert (
            max(left_count, right_count) / total > 0.8
        ), f"Segment {seg_id} crosses crease ({max(left_count, right_count) / total:.2f})!"


if __name__ == "__main__":
    test_normal_guided_slic_crease_preservation()
