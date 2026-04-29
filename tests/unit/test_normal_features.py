import numpy as np
import pytest

from pbn_node.backend.preprocessing.normal_features import augment_image_with_normals


@pytest.mark.unit
def test_augment_image_with_normals_shape():
    """Verify that the augmentation produces a 5-channel image."""
    h, w = 100, 100
    lab_image = np.zeros((h, w, 3), dtype=np.float32)
    normal_map = np.zeros((h, w, 3), dtype=np.float32)
    normal_map[:, :, 2] = 1.0  # Unit Z

    strength = 0.5
    augmented = augment_image_with_normals(lab_image, normal_map, strength)

    assert augmented.shape == (h, w, 5)
    assert augmented.dtype == np.float32


@pytest.mark.unit
def test_augment_image_with_normals_weighting():
    """Verify that the normal strength correctly weights the channels."""
    h, w = 10, 10
    lab_image = np.ones((h, w, 3), dtype=np.float32) * 50.0
    normal_map = np.ones((h, w, 3), dtype=np.float32)
    # Normals are already normalized in the helper

    strength = 1.0
    augmented = augment_image_with_normals(lab_image, normal_map, strength)

    # Standard LAB values are L: 0-100, a/b: -128-127
    # Normals are -1 to 1.
    # The helper scales normals by strength * 100.0 (or similar) to match LAB magnitude.
    # Let's check the implementation logic.

    # If strength is 0.5, normals should have half the influence of strength 1.0.
    augmented_half = augment_image_with_normals(lab_image, normal_map, 0.5)

    # Normal channels are indices 3 and 4 (X and Y gradients)
    assert np.allclose(augmented[:, :, 3:5], augmented_half[:, :, 3:5] * 2.0)


@pytest.mark.unit
def test_augment_image_with_normals_resizing():
    """Verify that the normal map is resized if it doesn't match the image."""
    lab_image = np.zeros((100, 100, 3), dtype=np.float32)
    normal_map = np.zeros((50, 50, 3), dtype=np.float32)

    augmented = augment_image_with_normals(lab_image, normal_map, 0.5)
    assert augmented.shape == (100, 100, 5)
