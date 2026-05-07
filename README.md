# ComfyUI Paint by Number

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://python.org)
[![ComfyUI](https://img.shields.io/badge/ComfyUI-custom%20node-orange)](https://github.com/comfyanonymous/ComfyUI)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Lint](https://img.shields.io/badge/lint-ruff-purple)](https://docs.astral.sh/ruff/)

A ComfyUI custom node that transforms digital images into high-quality, printable paint-by-number templates. Generates vector-aligned color regions with accurate numbering directly inside your ComfyUI workflows.

## Tech Stack

| Layer | Technology |
|---|---|
| Runtime | Python 3.10+ |
| ComfyUI API | V3 Node API |
| Color quantization | scikit-learn (K-Means / auto-detect) |
| Image processing | OpenCV, NumPy (vectorized majority filter) |
| SVG output | Custom SVG renderer with inline vector preview |
| Shadow removal | Retinex-based auto-albedo |
| Linting | Ruff |
| Type checking | Mypy |
| Testing | pytest |
| Dependency management | uv + pip |

## Architecture

```
pbn_node/
├── __init__.py             # ComfyUI node registration
├── node.py                 # Main V3 node class — inputs, outputs, execute()
├── pipeline/
│   ├── quantizer.py        # Color quantization + auto-color detection
│   ├── smoother.py         # Vectorized majority filter (fast region smoothing)
│   ├── labeler.py          # Region numbering and paint label assignment
│   ├── albedo.py           # Retinex-based shadow removal (auto-albedo)
│   └── renderer.py         # SVG + raster output renderer
├── tests/
│   ├── run_tests.py        # Test runner wrapper (required for mock loading)
│   └── TESTING.md          # Testing infrastructure docs
└── requirements.txt
```

The pipeline is fully decoupled: each stage (quantize → smooth → label → render) is independently testable. A quality feedback loop uses computational geometry and edge fidelity metrics to catch regressions in artistic output.

## Node Inputs

| Input | Type | Description |
|---|---|---|
| `image` | IMAGE | Input image tensor |
| `segmentation` | IMAGE (optional) | Mask/segmentation map — proportional color budget allocation |
| `lineart` | IMAGE (optional) | Edge map (Canny, HED) — prevents color bleeding across boundaries |
| `num_colors` | INT | Color clusters (0 = auto-detect) |
| `simplification` | FLOAT | Contour simplification factor (0.5–2.0) |
| `output_mode` | ENUM | `colored` / `outline` / `quantized` |
| `preset` | ENUM | `fast` / `balanced` / `portrait` / `custom` |
| `subject_priority` | FLOAT | Color budget multiplier for non-background segments |
| `material_weight` | FLOAT | Albedo influence over original photo during quantization |
| `edge_influence` | FLOAT | Lineart bias on color quantization boundaries |
| `use_auto_albedo` | BOOL | Auto-estimate shadow-free albedo when no external map is provided |

## Node Outputs

| Output | Type | Description |
|---|---|---|
| `IMAGE` | IMAGE | Rendered raster template |
| `SVG` | STRING | Raw SVG data (also previewed inline on the node) |
| `COLOR_COUNT` | INT | Total unique colors in the final template |

## Installation

**Requirements:** Python 3.10+

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/tazztone/pbn_node
pip install -r pbn_node/requirements.txt
# Restart ComfyUI — node appears under Image > Process > Paint By Number
```

## Development

```bash
# Lint & format
uvx ruff check .
uvx ruff format .

# Type check
uvx mypy .

# Tests (must use wrapper script)
../../venv/bin/python tests/run_tests.py

# Pre-commit hooks (auto-runs all checks)
pip install pre-commit && pre-commit install
```

See [tests/TESTING.md](tests/TESTING.md) for testing infrastructure details.

## License

MIT — see [LICENSE](LICENSE).
