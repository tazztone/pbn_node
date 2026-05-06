import sys
from pathlib import Path

# Standardize common directories as pure constants (no import-time side effects)
QUALITY_DIR = Path(__file__).parent.resolve()
ROOT_DIR = Path(__file__).parents[2].resolve()
PARENT_DIR = ROOT_DIR.parent.resolve()
TESTS_DIR = ROOT_DIR / "tests"
EXAMPLE_DIR = ROOT_DIR / "example_inputs"
OUT_DIR = QUALITY_DIR / "output"


def setup(setup_mocks: bool = True):
    """
    Explicitly initializes paths and ComfyUI mocks for standalone script execution.
    """
    # 1. Setup paths: add project root, its parent, and tests directories to sys.path
    for path in [ROOT_DIR, PARENT_DIR, TESTS_DIR]:
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    # 2. Idempotently create the quality output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Idempotently install ComfyUI mocks if requested
    if setup_mocks:
        from mock_comfyui import install_comfyui_mocks  # noqa: E402

        install_comfyui_mocks()
