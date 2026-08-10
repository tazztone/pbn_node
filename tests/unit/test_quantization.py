from unittest.mock import patch

import numpy as np
import pytest
from pbn_node.backend.quantization.quantizer import ColorQuantizer


@pytest.mark.unit
def test_detect_monochrome():
    quantizer = ColorQuantizer()

    # Clearly monochrome
    img_gray = np.ones((100, 100, 3), dtype=np.uint8) * 128
    assert quantizer.detect_monochrome(img_gray) is True

    # Pure white
    img_white = np.ones((100, 100, 3), dtype=np.uint8) * 255
    assert quantizer.detect_monochrome(img_white) is True

    # Pure black
    img_black = np.zeros((100, 100, 3), dtype=np.uint8)
    assert quantizer.detect_monochrome(img_black) is True

    # Clearly colored
    img_color = np.zeros((100, 100, 3), dtype=np.uint8)
    img_color[:50, :50] = [255, 0, 0]
    img_color[50:, 50:] = [0, 255, 0]
    assert quantizer.detect_monochrome(img_color) is False


@pytest.mark.unit
def test_quantize_fixed_k(sample_image_np):
    quantizer = ColorQuantizer()
    k = 4
    quantized, palette = quantizer.quantize(sample_image_np, num_colors=k)

    assert palette.color_count == k
    assert len(palette.hex_colors) == k
    assert quantized.shape == sample_image_np.shape


@pytest.mark.unit
def test_auto_select_k_exception(sample_image_np):
    quantizer = ColorQuantizer()

    # Bypass monochrome check to reach KneeLocator
    quantizer.detect_monochrome = lambda img: False

    # Mock KneeLocator to raise an Exception
    with patch("pbn_node.backend.quantization.quantizer.KneeLocator", side_effect=Exception("Test Exception")):
        k = quantizer.auto_select_k(sample_image_np)

    expected_k = min((quantizer.min_k + quantizer.max_k) // 2, quantizer.k_cap)
    assert k == expected_k


@pytest.mark.unit
def test_quantize_auto_k(sample_image_np):
    quantizer = ColorQuantizer()
    quantized, palette = quantizer.quantize(sample_image_np, num_colors=None)

    assert palette.color_count >= quantizer.min_k
    assert palette.color_count <= quantizer.k_cap
