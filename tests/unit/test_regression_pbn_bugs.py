import numpy as np

from pbn_node.backend.models import PerceptionInputs
from pbn_node.backend.preprocessing.normal_features import augment_image_with_normals
from pbn_node.backend.quantization.quantizer import ColorQuantizer


def test_normal_map_upscale_no_grid_artifacts():
    """
    Regression test for Bug 1: Normal map upscaling causing grid artifacts.
    Verifies that features are computed at native resolution before upscaling.
    """
    # Create a small 4x4 normal map with a single surface crease
    # Top half: [0, 0, 1], Bottom half: [0, 1, 0]
    small_normals = np.zeros((4, 4, 3), dtype=np.float32)
    small_normals[:2, :, 2] = 1.0  # Blue (Z)
    small_normals[2:, :, 1] = 1.0  # Green (Y)

    # Target resolution 64x64
    h, w = 64, 64
    lab_image = np.zeros((h, w, 3), dtype=np.float32)

    # Augment with normals
    strength = 1.0
    augmented = augment_image_with_normals(lab_image, small_normals, strength)

    # Get the angular gradient channel (index 3)
    ang_grad = augmented[:, :, 3]

    # If Sobel was computed AFTER upscaling, it would detect the grid lines
    # of the 4x4 blocks. If computed BEFORE, it should only have ONE major
    # horizontal line at the center (the actual crease).

    # Check vertical center line (where the 4x4 blocks meet)
    # The crease is between row 1 and 2 of the 4x4 grid.
    crease_region = ang_grad[30:34, :]
    assert (
        np.max(crease_region) > 25.0
    ), f"Crease not detected strongly enough: {np.max(crease_region)}"

    # Check other areas that would have grid artifacts if upscaling happened first
    # Specifically, check rows that are far from the crease in the 4x4 grid
    artifact_region_1 = ang_grad[8, :]
    artifact_region_2 = ang_grad[56, :]

    # Threshold 2.0 is small compared to 50.0 (4%), allows for minor edge effects
    assert (
        np.max(artifact_region_1) < 2.0
    ), f"Grid artifact detected at row 8: {np.max(artifact_region_1)}"
    assert (
        np.max(artifact_region_2) < 2.0
    ), f"Grid artifact detected at row 56: {np.max(artifact_region_2)}"


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
