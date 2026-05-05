# PBN Quality Pipeline

This directory contains tools and reports for evaluating the Paint-By-Number generation quality.

## Quick Iteration & Sweep Workflow

Use `quick_iteration.py` to rapidly test settings or explore the parameter space.

```bash
# Single iteration
python tests/quality/quick_iteration.py --colors 32 --smoothing 13 --simplification 0.5 --min-region-size 200 --tag "golden"

# Parameter sweep (Smoothing x Simplification)
# Generates a visual grid (sweep_<tag>.png) for rapid comparison.
python tests/quality/quick_iteration.py --sweep --tag "grid_v1"
```

## Coverage Metrics

We track two distinct coverage metrics to separate geometric truth from visual perception:

| Metric | What it measures | Purpose |
|---|---|---|
| `fill_coverage` | Raw polygon geometry area. | **Geometric Health**. Catches topological drift and simplification over-removal. |
| `render_coverage`| Final renderer output (incl. 2px shared borders). | **Visual Solidity**. Validates that the final image has no visible gaps ("grout" effect). |

### Interpreting Failures
- **`fill_coverage` drops, `render_coverage` stays high**: The renderer is successfully masking topological drift. This is acceptable for production as long as the drift isn't so wide that it affects print registration.
- **Both drop**: The pipeline is genuinely degrading. Simplification is too aggressive or the segmenter is losing regions.

## Human Review

While metrics catch regressions, final artistic approval happens in the HTML report:
1. Run `python tests/quality/render_report.py`.
2. Open `tests/quality/output/report.html` in your browser.
This is the default entry point for reviewing side-by-side comparisons, Canny edge fidelity, and LLM-based visual critiques.

---

## Session Findings (2026-05-05)

### 1. The "Topological Drift" Discovery
*   **Discovery**: Even with `simplification=0.0`, geometric gaps existed between regions (~0.2%).
*   **Root Cause**: Independent Visvalingam-Whyatt simplification causes shared boundaries to diverge. One region's edge loses a vertex that its neighbor keeps, creating "empty space" gaps.
*   **Resolution**: Standardized the preview to use `PBNRenderer` which draws `shared_borders` with `thickness=2`. This "grout" seals the drift gaps visually.

### 2. Parameter Sweep Insights
*   **Visual Parity**: Standardizing the preview to use `PBNRenderer` ensures the diagnostic "Gap Map" matches the final deliverable 1:1.
*   **Simplification Limits**: At `simplification >= 1.0`, the topological drift exceeds the 2px renderer seal, leading to significant coverage drops (70-80%). Sticking to `0.5` is critical for high-fidelity templates.

## Known Issues

### 1. High Simplification Drift
At simplification levels > 0.5, the renderer's 2px seal is no longer sufficient to bridge the gap between drifting polygons.

### 2. Speck Ratio Maintenance
Without explicit `min_region_size`, the speck ratio (tiny fragments) can climb as high as 90%. Always run with `--min-region-size 200` for production-grade "paintability."

## Iteration Log
Full metrics history is maintained in [output/iteration_log.csv](output/iteration_log.csv).
Combined Gap Map grids for sweeps are saved as `sweep_<tag>.png`.
