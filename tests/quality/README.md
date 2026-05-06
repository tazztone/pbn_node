# PBN Quality Pipeline

This directory contains tools and reports for evaluating the Paint-By-Number generation quality.

## Quick Iteration, Sweep & Comparison Workflows

### 1. Single Iteration & Parameter Sweeps
Use `quick_iteration.py` to rapidly test settings or explore the parameter space.

```bash
# Single iteration with Golden settings
python tests/quality/quick_iteration.py --colors 32 --smoothing 13 --simplification 0.5 --min-region-size 200 --tag "golden"

# Parameter sweep (Smoothing x Simplification)
# Generates a visual grid (sweep_<tag>.png) for rapid comparison.
python tests/quality/quick_iteration.py --sweep --tag "grid_v1"
```

### 2. Side-By-Side Preset Comparison
Use `compare_presets.py` to generate a unified horizontal side-by-side comparison image demonstrating the aesthetic leap between the legacy defaults and our optimized golden preset:

```bash
python tests/quality/compare_presets.py
```
- **Output**: Generates a labeled high-resolution image at `tests/quality/output/comparison.png` showing:
  1. **Original Source Image**
  2. **Legacy Defaults**: Low-detail, rigid shapes with `simpl=1.0`, washed-out Retinex albedo, and jagged pixel edges.
  3. **Golden Candidate**: Crisp details with `simpl=0.5`, rich natural colors, 13px majority smoothing, and flowing Bézier lines.

---

## Coverage Metrics

We track two distinct coverage metrics to separate geometric truth from visual perception:

| Metric | What it measures | Purpose |
|---|---|---|
| `fill_coverage` | Raw polygon geometry area. | **Geometric Health**. Catches topological drift and simplification over-removal. |
| `render_coverage`| Final renderer output (incl. 2px shared borders). | **Visual Solidity**. Validates that the final image has no visible gaps ("grout" effect). |

### Interpreting Failures
- **`fill_coverage` drops, `render_coverage` stays high**: The renderer is successfully masking topological drift. This is acceptable for production as long as the drift isn't so wide that it affects print registration.
- **Both drop**: The pipeline is genuinely degrading. Simplification is too aggressive or the segmenter is losing regions.

---

## Human Review & Visual Report Cards

While metrics catch regressions, final artistic approval happens in the HTML report:
1. Run `python tests/quality/render_report.py`.
2. Open `tests/quality/output/report.html` in your browser.
This is the default entry point for reviewing side-by-side comparisons, Canny edge fidelity, and LLM-based visual critiques.

---

## Session Findings (2026-05-06)

### 1. The "Topological Drift" Discovery
*   **Discovery**: Even with `simplification=0.0`, geometric gaps existed between regions (~0.2%).
*   **Root Cause**: Independent Visvalingam-Whyatt simplification causes shared boundaries to diverge. One region's edge loses a vertex that its neighbor keeps, creating "empty space" gaps.
*   **Resolution**: Standardized the preview to use `PBNRenderer` which draws `shared_borders` with `thickness=2`. This "grout" seals the drift gaps visually.

### 2. Spurious "Random Lines" Spike Resolution
*   **Discovery**: Long, straight, or diagonal line spikes crossed across random regions in the output.
*   **Root Cause**: When two regions touched in multiple disconnected locations (e.g., sky bands touching on both the left and right sides of a foreground subject), the coordinates were appended to a flat list in raster scan order. Generating a single `LineString` from this list connected the disconnected points sequentially, drawing lines directly across the image.
*   **Resolution**: Grouped border points into **8-connected contiguous components** using NetworkX, and created a separate `LineString` for each contiguous segment. Points in each segment are sorted using a greedy nearest-neighbor walk to produce continuous, correct boundary paths.

### 3. Label Clutter & Overlap Elimination
*   **Discovery**: Detailed areas (faces, hands) became cluttered with overlapping, illegible tiny labels, and labels near canvas edges were cut off.
*   **Root Cause**: Every tiny island speck down to 10 pixels² received its own label, and labels were placed close to each other without margins or collision checks.
*   **Resolution**:
    - Increased `min_region_area` for labeling from `16` to `120` px² and discarded labeling for tiny islands under `80` px² entirely.
    - Implemented a **25px Minimum Distance Guard** between placed labels to skip placing overlapping labels in high-detail areas.
    - Added **Canvas Edge Margins** based on font size to push border numbers safely inward.

---

## Iteration Log
Full metrics history is maintained in [output/iteration_log.csv](output/iteration_log.csv).
Combined Gap Map grids for sweeps are saved as `sweep_<tag>.png`.
Unified preset comparisons are saved as `comparison.png`.
