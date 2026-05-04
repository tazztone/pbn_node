import numpy as np

from pbn_node.backend.models import PerceptionInputs
from pbn_node.backend.quantization.quantizer import ColorQuantizer


def test_quantizer_albedo_resize_guard():
    """
    Regression test for Bug 2: Quantizer inline blend missing resize guard.
    Verifies that the quantizer can handle albedo with different dimensions.
    """
    quantizer = ColorQuantizer()

    # Create a 64x64 image
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    # Create a 32x32 albedo map
    albedo = np.ones((32, 32, 3), dtype=np.uint8) * 128

    # Perception inputs with albedo and edge influence
    perception = PerceptionInputs(albedo=albedo, edge_influence=0.5)

    # This should NOT crash now
    quantized, palette = quantizer.quantize(image, num_colors=8, perception=perception)

    assert quantized.shape == (64, 64, 3)
    assert palette.color_count <= 8


def test_quantizer_lineart_resize_guard():
    """
    Regression test for Bug 2: Quantizer inline blend missing resize guard for lineart.
    """
    quantizer = ColorQuantizer()

    # Create a 64x64 image
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    # Create a 32x32 albedo and lineart map
    albedo = np.ones((32, 32, 3), dtype=np.uint8) * 128
    lineart = np.ones((32, 32), dtype=np.float32) * 0.5

    # Perception inputs with albedo and lineart
    perception = PerceptionInputs(albedo=albedo, lineart=lineart, edge_influence=0.5)

    # This should NOT crash now
    quantized, palette = quantizer.quantize(image, num_colors=8, perception=perception)

    assert quantized.shape == (64, 64, 3)
