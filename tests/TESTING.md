# Testing guide for pbn_node

The testing guide covers running and adding tests for the Paint By Number custom
nodes. The testing suite lets you verify node logic and pipeline performance
without requiring a live ComfyUI server.

## Quick start

To run the tests, you must use the ComfyUI virtual environment to ensure all
dependencies like `torch` and `shapely` are available.

<!-- prettier-ignore -->
> [!IMPORTANT]
> Always use the exact command format below. Don't use `python` or `pytest`
> directly from your shell.

```bash
# Execute from the pbn_node directory
cd custom_nodes/pbn_node

# Run all tests using the ComfyUI venv (relative to pbn_node)
../../venv/bin/python tests/run_tests.py

# Run with verbose output
../../venv/bin/python tests/run_tests.py -v

# Run only unit tests
../../venv/bin/python tests/run_tests.py unit/
```

---

## Test structure

The project organizes tests into unit and integration categories to balance
execution speed and system-wide verification.

```text
tests/
├── conftest.py          # Centralized ComfyUI and API mocking
├── pytest.ini           # Local configuration for path isolation
├── run_tests.py         # Wrapper script for environment setup
├── unit/                # Fast tests for individual backend modules
│   ├── test_pbn_node.py
│   ├── test_pbn_renderer.py
│   ├── test_preprocessing.py
│   ├── test_quantization.py
│   └── test_segmentation.py
└── integration/         # Full workflow tests
    ├── test_pipeline.py # Orchestrator verification
    └── test_nodes.py    # ComfyUI node class verification
```

---

## Test categories

We use unit tests for isolated logic and integration tests for end-to-end
pipeline verification.

### Unit tests (20 tests)

Fast tests that validate individual backend components without ComfyUI
dependencies.

| Test | Description |
|------|-------------|
| `test_detect_image_type` | Aspect ratio detection |
| `test_preprocess_metadata` | Metadata extraction accuracy |
| `test_preprocess_output_shape`| Image dimensions preservation |
| `test_detect_monochrome` | Variance-based color detection |
| `test_quantize_fixed_k` | Color reduction with manual k |
| `test_quantize_auto_k` | Automatic k selection (elbow method) |
| `test_build_adjacency_graph` | Region connectivity mapping |
| `test_segment_pipeline` | Full segmentation data structure |
| `test_direct_color_segmentation` | Segmentation without watershed |
| `test_execute_returns_svg_preview` | Node UI payload verification |
| `test_svg_filename_determinism` | Hash-based filename consistency |
| `test_budget_allocation_k_sum` | Perception budget summation |
| `test_budget_allocation_min_k` | Minimum k per segment check |
| `test_albedo_guided_quantization_shift` | Albedo influence verification |
| `test_normal_guided_slic_crease_preservation` | Normal map geometric alignment |
| `test_fill_uses_region_colors_mapping` | Renderer color mapping accuracy |
| `test_label_text_shows_paint_number` | Label text vs paint number check |
| `test_label_contrast_uses_mapped_color` | Contrast visibility per region |
| `test_fallback_when_region_colors_is_none` | Renderer backward compatibility |
| `test_multiple_regions_same_color` | Label consistency across islands |

### Integration tests (4 tests)

Tests that verify the full processing pipeline and the ComfyUI node class.

| Test | Description |
|------|-------------|
| `test_image_processor_full_run` | Full array-to-SVG processing flow |
| `test_image_processor_batch_simulation` | Processor reuse and consistency |
| `test_pbn_node_execute` | V3 Node execution and output packing |
| `test_pbn_node_output_modes` | Result rendering styles (colored, outline) |

---

## Test markers

You can run tests by their defined markers to focus on specific scopes.

```bash
# Run by marker
pytest -m unit           # Fast component tests
pytest -m integration    # Slower workflow tests
```

---

## How it works

The test infrastructure overcomes several challenges common in ComfyUI node
development.

### Problems solved

Standard test runners often fail with ComfyUI nodes because:

1.  **Parent `pytest.ini` interference**: ComfyUI's root config restricts pytest
    to specific directories.
2.  **Module dependencies**: Nodes import `folder_paths` and `comfy` which
    don't exist in a standalone test environment.

### Solution

1.  **`tests/conftest.py`**: Mocks ComfyUI modules (`folder_paths`, `comfy`,
    `comfy_api`) at module load time (before test collection).
2.  **`tests/pytest.ini`**: Local config with empty `pythonpath =` to override
    parent settings.
3.  **`tests/run_tests.py`**: Wrapper script that changes to the `tests/`
    directory before execution.

---

## Adding new tests

Follow these patterns when adding new tests to maintain consistency.

### Unit tests

1. Create a test file in `tests/unit/test_*.py`.
2. Add the `@pytest.mark.unit` marker.
3. Import from the `pbn_node` package.

```python
import pytest
from pbn_node.backend.preprocessing.preprocessor import Preprocessor


@pytest.mark.unit
def test_my_function():
    processor = Preprocessor()
    # assert ...
```

### Integration tests

1. Create a test file in `tests/integration/test_*.py`.
2. Add the `@pytest.mark.integration` marker.
3. Import node classes directly (conftest provides mocks).

```python
import pytest
from pbn_node.pbn_node import PaintByNumberNode

@pytest.mark.integration
def test_my_node(sample_image_tensor):
    node = PaintByNumberNode()
    result = node.execute(image=sample_image_tensor, ...)
    assert result is not None
```

---

## Troubleshooting

Use this table to resolve common testing issues.

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: folder_paths` | Mocks not loading - run via `run_tests.py` |
| "No tests found" in IDE | Run from `tests/` directory or use `run_tests.py` |
| `ModuleNotFoundError` (pytest) | You are using the wrong python. Use the ComfyUI venv. |
| `ImportError` (relative) | Ensure imports use the `pbn_node.module` format. |
| Parent `pytest.ini` override | Local `tests/pytest.ini` must have `pythonpath =` (empty). |
