import os

import cv2
import numpy as np
from pbn_node.backend.models import ProcessingParameters
from pbn_node.pbn_pipeline import ImageProcessor


def test_debug_preview():
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    img_path = os.path.join(base_path, "example_inputs/boat.webp")
    img = cv2.imread(img_path)

    pipeline = ImageProcessor()
    params = ProcessingParameters(num_colors=8)
    res = pipeline.process_array(img, params)

    print(f"\nSegmented matrix shape: {res.quantized.shape}")
    print(f"Max pixel value in quantized: {np.max(res.quantized)}")

    if hasattr(res, "segmented_matrix") and res.segmented_matrix is not None:
        print(f"Segmented matrix shape: {res.segmented_matrix.shape}")
        print(f"Max region ID: {np.max(res.segmented_matrix)}")
        print(f"Min region ID: {np.min(res.segmented_matrix)}")
        unique_ids = np.unique(res.segmented_matrix)
        print(f"Unique region IDs count: {len(unique_ids)}")
        print(f"Region colors count: {len(res.region_colors)}")
    else:
        # Check if it's in RegionData inside res
        print("res.segmented_matrix not found")
