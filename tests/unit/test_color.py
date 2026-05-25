import numpy as np
import pytest
from pbn_node.backend.utils.color import cv_to_std_lab

def test_cv_to_std_lab_basic():
    # OpenCV LAB ranges: L 0-255, a 0-255, b 0-255
    # Standard LAB ranges: L 0-100, a -128-127, b -128-127

    cv_lab = np.array([[[128, 128, 128]]], dtype=np.uint8)
    std_lab = cv_to_std_lab(cv_lab)

    # 128 * 100 / 255 = 50.196
    # 128 - 128 = 0
    # 128 - 128 = 0

    assert np.allclose(std_lab[0, 0, 0], 50.196, atol=1e-3)
    assert np.allclose(std_lab[0, 0, 1], 0)
    assert np.allclose(std_lab[0, 0, 2], 0)
