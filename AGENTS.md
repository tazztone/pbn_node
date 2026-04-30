# AGENTS.md

## Purpose
- Maintain the ComfyUI PBN node using V3 API and modular backend.
- Use 'comfyui-node-*' development skills for all node-related tasks.

## Commands
- Test: `../../venv/bin/python tests/run_tests.py`

## Non-obvious conventions
- **Mocks**: Always use `run_tests.py`; it loads ComfyUI mocks in `tests/conftest.py`.
- **Imports**: In tests, use package-level: `from pbn_node.backend...`.

## Done criteria
- **Docs updated**: README.md, ROADMAP.md, and TESTING.md reflect the changes (if needed).
