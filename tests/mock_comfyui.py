import sys
import tempfile
from unittest.mock import MagicMock


def install_comfyui_mocks():
    """
    Idempotently installs mocks for ComfyUI modules.
    Useful for standalone scripts and pytest runs.
    """

    # Mock folder_paths
    if "folder_paths" not in sys.modules:
        mock_folder_paths = MagicMock()
        mock_folder_paths.get_temp_directory.return_value = tempfile.gettempdir()
        sys.modules["folder_paths"] = mock_folder_paths

    # Mock comfy
    if "comfy" not in sys.modules:
        sys.modules["comfy"] = MagicMock()

    # --- Mock comfy_api for V3 nodes ---
    if "comfy_api" not in sys.modules or "comfy_api.latest" not in sys.modules:

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

        sys.modules["comfy_api"] = mock_comfy_api
        sys.modules["comfy_api.latest"] = mock_comfy_api_latest
        sys.modules["comfy_api.latest.io"] = mock_io
        sys.modules["comfy_api.latest.ui"] = mock_ui
