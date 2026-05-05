# PBN Quality Pipeline

This directory contains tools and reports for evaluating the Paint-By-Number generation quality.

## Quick Iteration Tool

Use `quick_iteration.py` to rapidly test different pipeline settings on a sample image.

```bash
# Single iteration
python tests/quality/quick_iteration.py --colors 32 --smoothing 13 --simplification 0.5 --min-region-size 200 --tag "golden"

# Parameter sweep (Smoothing x Simplification)
python tests/quality/quick_iteration.py --sweep --tag "grid_v1"
```

## Session Findings (2026-05-05)

### 1. The "Topological Drift" Discovery
*   **Discovery**: Even with `simplification=0.0`, geometric gaps existed between regions (~0.2%).
*   **Root Cause**: Independent Visvalingam-Whyatt simplification (even at 0.1 tolerance) causes shared boundaries to diverge. One region's edge loses a vertex that its neighbor keeps, creating "empty space" gaps.
*   **Resolution**: Standardized the preview to use `PBNRenderer` which draws `shared_borders` with `thickness=2`. This "grout" seals the drift gaps visually.

### 2. Dual Metrics Strategy
*   **Fill_Geo**: Measures raw polygon coverage (detects topological drift).
*   **Render_Coverage**: Measures the final renderer output (validates visual solidness).
*   **Result**: At `simpl=0.5`, we hit **99.99% Render Coverage** despite ~0.3% geometric drift.

### 3. Parameter Sweep Insights
*   **Visual Parity**: Standardizing the preview to use `PBNRenderer` ensures the diagnostic "Gap Map" matches the final deliverable 1:1.
*   **Simplification Limits**: At `simplification >= 1.0`, the topological drift exceeds the 2px renderer seal, leading to significant coverage drops (70-80%). Sticking to `0.5` is critical for high-fidelity templates.

## Known Issues

### 1. High Simplification Drift
At simplification levels > 0.5, the renderer's 2px seal is no longer sufficient to bridge the gap between drifting polygons.
*   **Action**: Investigate geometric `buffer(0.1)` on polygons or a truly topological simplification engine if high simplification is required.

### 2. Speck Ratio Maintenance
Without explicit `min_region_size`, the speck ratio (tiny fragments) can climb as high as 90%.
*   **Recommendation**: Always run with `--min-region-size 200` for production-grade "paintability."

## Iteration Log
Full metrics history is maintained in [output/iteration_log.csv](output/iteration_log.csv).
Combined Gap Map grids for sweeps are saved as `sweep_<tag>.png`.
