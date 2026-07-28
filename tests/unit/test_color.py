import numpy as np
import pytest
from pbn_node.backend.utils.color import cv_to_std_lab


@pytest.mark.unit
def test_cv_to_std_lab_black():
    # cv LAB black: L=0, a=128, b=128
    lab = np.array([[[0, 128, 128]]], dtype=np.uint8)
    std = cv_to_std_lab(lab)
    assert np.allclose(std, np.array([[[0.0, 0.0, 0.0]]], dtype=np.float32))


@pytest.mark.unit
def test_cv_to_std_lab_white():
    # cv LAB white: L=255, a=128, b=128
    lab = np.array([[[255, 128, 128]]], dtype=np.uint8)
    std = cv_to_std_lab(lab)
    assert np.allclose(std, np.array([[[100.0, 0.0, 0.0]]], dtype=np.float32))


@pytest.mark.unit
def test_cv_to_std_lab_extremes():
    # Extremes for a and b
    # cv LAB 0 -> -128, 255 -> 127
    lab = np.array([[[255, 255, 255]], [[0, 0, 0]]], dtype=np.uint8)
    std = cv_to_std_lab(lab)
    assert np.allclose(std[0], np.array([[100.0, 127.0, 127.0]], dtype=np.float32))
    assert np.allclose(std[1], np.array([[0.0, -128.0, -128.0]], dtype=np.float32))


@pytest.mark.unit
def test_cv_to_std_lab_shape():
    # test larger shape
    lab = np.zeros((10, 10, 3), dtype=np.uint8)
    std = cv_to_std_lab(lab)
    assert std.shape == (10, 10, 3)


@pytest.mark.unit
def test_cv_to_std_lab_type():
    # Output type should be float32
    lab = np.zeros((5, 5, 3), dtype=np.uint8)
    std = cv_to_std_lab(lab)
    assert std.dtype == np.float32


@pytest.mark.unit
def test_cv_to_std_lab_invalid_type():
    with pytest.raises((TypeError, AttributeError)):
        cv_to_std_lab(None)
    with pytest.raises((TypeError, AttributeError)):
        cv_to_std_lab([[[1, 2, 3]]])


@pytest.mark.unit
def test_cv_to_std_lab_missing_channels():
    lab = np.zeros((5, 5, 2), dtype=np.uint8)
    with pytest.raises(IndexError):
        cv_to_std_lab(lab)


@pytest.mark.unit
def test_cv_to_std_lab_empty():
    lab = np.array([], dtype=np.uint8).reshape(0, 0, 3)
    std = cv_to_std_lab(lab)
    assert std.shape == (0, 0, 3)
    assert std.size == 0
