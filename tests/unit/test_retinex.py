import cv2
import numpy as np
import pytest

from pbn_node.backend.preprocessing.retinex import multiscale_retinex


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


@pytest.mark.unit
def test_retinex_chrominance_stability():
    """Verify that Retinex normalization doesn't corrupt chrominance channels."""
    # Create a bright saturated image
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    # Bright Blue: (200, 50, 50) BGR
    image[:, :, 0] = 200
    image[:, :, 1] = 50
    image[:, :, 2] = 50

    albedo = multiscale_retinex(image)

    # Check that A and B channels in LAB space are preserved within reasonable range
    lab_in = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_out = cv2.cvtColor(albedo, cv2.COLOR_BGR2LAB)

    # A and B channels are 1 and 2
    # They should not have major shifts (>10 uint8 levels)
    shift_a = np.mean(np.abs(lab_in[:, :, 1].astype(np.int16) - lab_out[:, :, 1].astype(np.int16)))
    shift_b = np.mean(np.abs(lab_in[:, :, 2].astype(np.int16) - lab_out[:, :, 2].astype(np.int16)))

    assert shift_a < 10
    assert shift_b < 10
