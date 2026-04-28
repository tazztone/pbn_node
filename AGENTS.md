# AGENTS.md

## Purpose
- Maintain the ComfyUI PBN node using V3 API and modular backend.
- Use 'comfyui-node-*' development skills for all node-related tasks.

## Allowed without asking
- Read files and run tests/linting.
- Fix linting or type errors.
- Update documentation to reflect code changes.

## Ask first
- Install new dependencies.
- Significant architectural changes.

## Commands
- Test: `../../venv/bin/python tests/run_tests.py`
- Lint: `uvx ruff check .`
- Format: `uvx ruff format .`
- Typecheck: `uvx mypy .`
- Pre-commit: `pre-commit run --all-files`

## Non-obvious conventions
- **Mocks**: Always use `run_tests.py`; it loads ComfyUI mocks in `tests/conftest.py`.
- **Imports**: In tests, use package-level: `from pbn_node.backend...`.

## Canonical examples
- Node Class: [pbn_node.py](file:///home/tazztone/Applications/Data/Packages/ComfyUI/custom_nodes/pbn_node/pbn_node.py)
- Backend Logic: [quantizer.py](file:///home/tazztone/Applications/Data/Packages/ComfyUI/custom_nodes/pbn_node/backend/quantization/quantizer.py)
- Unit Test: [test_quantization.py](file:///home/tazztone/Applications/Data/Packages/ComfyUI/custom_nodes/pbn_node/tests/unit/test_quantization.py)

## Done criteria
- **Docs updated**: README.md, ROADMAP.md, and TESTING.md reflect the changes (if needed).
