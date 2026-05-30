import numpy as np
import pytest

from pbn_node.backend.utils.color import cv_to_std_lab


@pytest.mark.unit
def test_cv_to_std_lab_conversion():
    # Test specific known values
    # L: 0 -> 0.0, 255 -> 100.0, 127 -> 127 * 100 / 255 = 49.8039
    # a: 0 -> -128.0, 255 -> 127.0, 128 -> 0.0
    # b: 0 -> -128.0, 255 -> 127.0, 128 -> 0.0

    cv_lab = np.array([
        [0, 0, 0],
        [255, 255, 255],
        [127, 128, 128],
    ], dtype=np.uint8)

    expected_std_lab = np.array([
        [0.0, -128.0, -128.0],
        [100.0, 127.0, 127.0],
        [127 * 100.0 / 255.0, 0.0, 0.0],
    ], dtype=np.float32)

    result = cv_to_std_lab(cv_lab)

    assert result.dtype == np.float32
    np.testing.assert_allclose(result, expected_std_lab, atol=1e-5)


@pytest.mark.unit
def test_cv_to_std_lab_shapes():
    # Test with 2D array (e.g. just a list of pixels)
    lab_2d = np.zeros((10, 3), dtype=np.uint8)
    res_2d = cv_to_std_lab(lab_2d)
    assert res_2d.shape == (10, 3)

    # Test with 3D array (e.g. an image)
    lab_3d = np.zeros((10, 10, 3), dtype=np.uint8)
    res_3d = cv_to_std_lab(lab_3d)
    assert res_3d.shape == (10, 10, 3)
