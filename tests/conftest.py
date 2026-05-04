import os
import sys

import numpy as np
import pytest
import torch

# Add the parent of the custom node directory to path
# This allows us to import as 'pbn_node.module' which supports relative imports
node_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(node_dir))

# Add the tests directory to path for mock import
tests_dir = os.path.dirname(__file__)
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

# --- MOCKING SETUP ---
from mock_comfyui import install_comfyui_mocks  # noqa: E402

install_comfyui_mocks()

# --- FIXTURES ---


@pytest.fixture
def sample_image_np():
    """Create a 128x128 RGB numpy image."""
    img = np.zeros((128, 128, 3), dtype=np.uint8)
    # Add some colored rectangles
    img[10:50, 10:50] = [255, 0, 0]  # Red
    img[60:110, 60:110] = [0, 255, 0]  # Green
    img[10:50, 60:110] = [0, 0, 255]  # Blue
    return img


@pytest.fixture
def sample_image_tensor():
    """Create a [1, 128, 128, 3] RGB torch tensor."""
    img = np.zeros((1, 128, 128, 3), dtype=np.float32)
    img[0, 10:50, 10:50] = [1.0, 0, 0]  # Red
    img[0, 60:110, 60:110] = [0, 1.0, 0]  # Green
    img[0, 10:50, 60:110] = [0, 0, 1.0]  # Blue
    return torch.from_numpy(img)
