import numpy as np
import pytest

from pbn_node.backend.models import PerceptionInputs
from pbn_node.backend.quantization.quantizer import ColorQuantizer


@pytest.mark.unit
def test_budget_allocation_k_sum():
    """Verify that allocated colors sum exactly to the requested k."""
    quantizer = ColorQuantizer()
    h, w = 100, 100
    # Use non-uniform image to avoid ConvergenceWarning
    image = np.random.RandomState(42).randint(0, 255, (h, w, 3), dtype=np.uint8)

    # Create a mask with 3 segments
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:30, :] = 1
    mask[30:60, :] = 2
    mask[60:, :] = 3

    num_colors = 10
    perception = PerceptionInputs(segmentation_mask=mask)

    _, palette = quantizer.quantize(image, num_colors=num_colors, perception=perception)

    assert palette.color_count == num_colors


@pytest.mark.unit
def test_budget_allocation_min_k():
    """Verify that small segments get at least 2 colors (min_k)."""
    quantizer = ColorQuantizer()
    h, w = 100, 100
    image = np.random.RandomState(42).randint(0, 255, (h, w, 3), dtype=np.uint8)

    # Create a mask with one huge segment and one tiny segment
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:-1, :] = 1  # 9900 pixels
    mask[-1, :] = 2  # 100 pixels (1%)

    num_colors = 10
    perception = PerceptionInputs(segmentation_mask=mask)

    _, palette = quantizer.quantize(image, num_colors=num_colors, perception=perception)
    assert palette.color_count == num_colors


@pytest.mark.unit
def test_albedo_guided_quantization_shift():
    """Verify that albedo influence changes the palette."""
    quantizer = ColorQuantizer()
    h, w = 100, 100

    # Original image is mostly dark
    image = np.random.RandomState(42).randint(0, 50, (h, w, 3), dtype=np.uint8)

    # Albedo is mostly bright red
    albedo = np.random.RandomState(43).randint(200, 255, (h, w, 3), dtype=np.uint8)
    albedo[:, :, 0] = 0  # Remove blue
    albedo[:, :, 1] = 0  # Remove green (Red in BGR)

    num_colors = 2

    # Case 1: material_weight = 0 (only original image)
    p1 = PerceptionInputs(albedo=albedo, material_weight=0.0)
    _, pal1 = quantizer.quantize(image, num_colors=num_colors, perception=p1)

    # Case 2: material_weight = 1 (only albedo)
    p2 = PerceptionInputs(albedo=albedo, material_weight=1.0)
    _, pal2 = quantizer.quantize(image, num_colors=num_colors, perception=p2)

    # The colors should be different
    assert not np.allclose(pal1.colors[0], pal2.colors[0], atol=10)
