import sys
from pathlib import Path

# 1. Setup paths: add project root and parent directory to sys.path
QUALITY_DIR = Path(__file__).parent.resolve()
ROOT_DIR = Path(__file__).parents[2].resolve()
PARENT_DIR = ROOT_DIR.parent.resolve()
TESTS_DIR = ROOT_DIR / "tests"

for path in [ROOT_DIR, PARENT_DIR, TESTS_DIR]:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# 2. Idempotently install ComfyUI mocks for standalone script execution
from mock_comfyui import install_comfyui_mocks  # noqa: E402

install_comfyui_mocks()

# 3. Standardize common directories
EXAMPLE_DIR = ROOT_DIR / "example_inputs"
OUT_DIR = QUALITY_DIR / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)
