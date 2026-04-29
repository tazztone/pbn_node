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
    # Create a gradient normal map so there are non-zero angular gradients
    normal_map = np.zeros((h, w, 3), dtype=np.float32)
    # X-gradient: normals rotate from Z to X
    for x in range(w):
        angle = (x / w) * np.pi / 2
        normal_map[:, x, 0] = np.sin(angle)
        normal_map[:, x, 2] = np.cos(angle)

    strength = 1.0
    augmented_full = augment_image_with_normals(lab_image, normal_map, strength)

    strength = 0.5
    augmented_half = augment_image_with_normals(lab_image, normal_map, strength)

    # Normal channels are indices 3 and 4
    # Verify that higher strength leads to higher magnitude in normal channels
    full_mag = np.mean(np.abs(augmented_full[:, :, 3:5]))
    half_mag = np.mean(np.abs(augmented_half[:, :, 3:5]))
    assert full_mag > half_mag

    # Verify strength 0 leads to near-zero normal influence
    augmented_zero = augment_image_with_normals(lab_image, normal_map, 0.0)
    assert np.allclose(augmented_zero[:, :, 3:5], 0.0)


@pytest.mark.unit
def test_augment_image_with_normals_resizing():
    """Verify that the normal map is resized if it doesn't match the image."""
    lab_image = np.zeros((100, 100, 3), dtype=np.float32)
    normal_map = np.zeros((50, 50, 3), dtype=np.float32)

    augmented = augment_image_with_normals(lab_image, normal_map, 0.5)
    assert augmented.shape == (100, 100, 5)
