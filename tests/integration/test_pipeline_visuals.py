import os

import cv2
import numpy as np
import pytest

from pbn_node.backend.models import PerceptionInputs, ProcessingParameters
from pbn_node.pbn_pipeline import ImageProcessor


@pytest.mark.integration
def test_pipeline_perception_impact():
    """
    Integration test to verify that lineart and normal maps actually
    influence the pipeline output.
    """
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    example_dir = os.path.join(base_path, "example_inputs")

    img_path = os.path.join(example_dir, "boat.webp")
    lineart_path = os.path.join(example_dir, "boat_lineart.webp")
    normals_path = os.path.join(example_dir, "boat_normals.webp")

    if not os.path.exists(img_path):
        pytest.skip(f"Example input not found at {img_path}")

    img = cv2.imread(img_path)

    pipeline = ImageProcessor()
    params_vanilla = ProcessingParameters(num_colors=8, use_slic=True, slic_n_segments=500, slic_compactness=10.0)

    # 1. Run Vanilla
    res_vanilla = pipeline.process_array(img, params_vanilla)
    img_vanilla = res_vanilla.quantized

    # 2. Run with Lineart
    if os.path.exists(lineart_path):
        lineart = cv2.imread(lineart_path, cv2.IMREAD_GRAYSCALE)
        lineart = cv2.resize(lineart, (img.shape[1], img.shape[0]))
        lineart = lineart.astype(np.float32) / 255.0

        perception_lineart = PerceptionInputs(lineart=lineart, lineart_strength=1.0, edge_influence=0.8)
        params_lineart = ProcessingParameters(
            num_colors=8,
            use_slic=True,
            slic_n_segments=500,
            slic_compactness=10.0,
            perception=perception_lineart,
        )
        res_lineart = pipeline.process_array(img, params_lineart)
        img_lineart = res_lineart.quantized

        diff = cv2.absdiff(img_vanilla, img_lineart)
        mean_diff = np.mean(diff)
        print(f"Mean pixel diff (vanilla vs lineart): {mean_diff}")

        # Assert that lineart actually changed the output
        assert mean_diff > 0.1, "Lineart had no influence on the output!"

    # 3. Run with Normals
    if os.path.exists(normals_path):
        normals = cv2.imread(normals_path)
        normals = cv2.resize(normals, (img.shape[1], img.shape[0]))
        normals = (normals.astype(np.float32) / 255.0) * 2.0 - 1.0

        perception_normals = PerceptionInputs(normal_map=normals, normal_strength=0.8)
        params_normals = ProcessingParameters(
            num_colors=8,
            use_slic=True,
            slic_n_segments=500,
            slic_compactness=10.0,
            perception=perception_normals,
        )
        res_normals = pipeline.process_array(img, params_normals)
        img_normals = res_normals.quantized

        diff_norm = cv2.absdiff(img_vanilla, img_normals)
        mean_diff_norm = np.mean(diff_norm)
        print(f"Mean pixel diff (vanilla vs normals): {mean_diff_norm}")

        # Assert that normals actually changed the output
        assert mean_diff_norm > 0.1, "Normals had no influence on the output!"


if __name__ == "__main__":
    # Allow running standalone
    try:
        test_pipeline_perception_impact()
        print("Test PASSED")
    except Exception as e:
        print(f"Test FAILED: {e}")
