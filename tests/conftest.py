import os
import sys
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

# Add the parent of the custom node directory to path
# This allows us to import as 'pbn_node.module' which supports relative imports
node_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(node_dir))

# --- MOCKING SETUP ---

# Mock folder_paths
mock_folder_paths = MagicMock()
# Use a real temp directory for mocks so file operations don't fail by default

mock_folder_paths.get_temp_directory.return_value = tempfile.gettempdir()
if "folder_paths" not in sys.modules:
    sys.modules["folder_paths"] = mock_folder_paths

# Mock comfy
if "comfy" not in sys.modules:
    sys.modules["comfy"] = MagicMock()

# --- Mock comfy_api for V3 nodes ---


class MockComfyNode:
    pass


class MockSchema:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


class MockNodeOutput(tuple):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls, args)
        instance.ui = kwargs.get("ui")  # type: ignore
        return instance


class MockInputType:
    @staticmethod
    def Input(name, *args, **kwargs):  # noqa: N802
        m = MagicMock(name=name)
        m.name = name
        return m

    @staticmethod
    def Output(**kwargs):  # noqa: N802
        return MagicMock()


class MockHidden:
    @staticmethod
    def Input(name, *args, **kwargs):  # noqa: N802
        m = MagicMock(name=name)
        m.name = name
        return m


class MockUI:
    @staticmethod
    def PreviewImage(*args, **kwargs):  # noqa: N802
        m = MagicMock()
        m.values = [{"filename": "test.png", "subfolder": "", "type": "temp"}]
        return m


# Build mock io module
mock_io = MagicMock()
mock_io.ComfyNode = MockComfyNode
mock_io.Schema = MockSchema
mock_io.NodeOutput = MockNodeOutput
mock_io.String = MockInputType
mock_io.Int = MockInputType
mock_io.Float = MockInputType
mock_io.Boolean = MockInputType
mock_io.Combo = MockInputType
mock_io.Image = MockInputType
mock_io.Audio = MockInputType
mock_io.Hidden = MockHidden

# Build mock ui module
mock_ui = MagicMock()
mock_ui.PreviewImage = MockUI.PreviewImage

# Build mock comfy_api module structure
mock_comfy_api = MagicMock()
mock_comfy_api_latest = MagicMock()
mock_comfy_api_latest.io = mock_io
mock_comfy_api_latest.ui = mock_ui


class MockComfyAPISync:
    def __init__(self):
        self.execution = MagicMock()


mock_comfy_api_latest.ComfyAPISync = MockComfyAPISync

if "comfy_api" not in sys.modules:
    sys.modules["comfy_api"] = mock_comfy_api
if "comfy_api.latest" not in sys.modules:
    sys.modules["comfy_api.latest"] = mock_comfy_api_latest
if "comfy_api.latest.io" not in sys.modules:
    sys.modules["comfy_api.latest.io"] = mock_io
if "comfy_api.latest.ui" not in sys.modules:
    sys.modules["comfy_api.latest.ui"] = mock_ui

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
