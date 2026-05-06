# Paint-by-number quality pipeline

This directory contains tools and reports for evaluating the Paint-By-Number
generation quality.

## Quick iteration, sweep, and comparison workflows

You can use the provided Python scripts to run isolated parameter sweeps and
analyze results without booting the full ComfyUI interface.

### 1. Single iteration and parameter sweeps

Use `quick_iteration.py` to rapidly test settings or explore the parameter
space.

```bash
# Run a single iteration with the calibrated golden settings
python tests/quality/quick_iteration.py --colors 32 --smoothing 13 --simplification 0.5 --min-region-size 200 --tag "golden"

# Run a parameter sweep (Smoothing x Simplification) to generate a visual grid (sweep_<tag>.png)
python tests/quality/quick_iteration.py --sweep --tag "grid_v1"
```

---

## Coverage metrics

We track two distinct coverage metrics to separate geometric truth from visual
perception:

| Metric | What it measures | Purpose |
|---|---|---|
| `fill_coverage` | Raw polygon geometry area. | **Geometric health**. Catches topological drift and simplification over-removal. |
| `render_coverage`| Final renderer output (including 2px shared borders). | **Visual solidity**. Validates that the final image has no visible gaps ("grout" effect). |

### Interpreting failures

<!-- prettier-ignore -->
> [!NOTE]
> - **`fill_coverage` drops, `render_coverage` stays high**: The renderer is
> successfully masking topological drift. This is acceptable for production
> as long as the drift isn't so wide that it affects print registration.
> - **Both drop**: The pipeline is genuinely degrading. Simplification is too
> aggressive or the segmenter is losing regions.

---

## Human review and visual report cards

While metrics catch regressions, final artistic approval happens in the HTML
report:

1. Run `python tests/quality/render_report.py`.
2. Open `tests/quality/output/report.html` in your browser.

This is the default entry point for reviewing side-by-side comparisons, Canny
edge fidelity, and AI-based visual critiques.

---

## Session findings (May 6, 2026)

### 1. The topological drift discovery

- **Discovery**: Even with `simplification=0.0`, geometric gaps existed between
  regions (approximately 0.2%).
- **Root cause**: Independent Visvalingam-Whyatt simplification causes shared
  boundaries to diverge. One region's edge loses a vertex that its neighbor
  keeps, creating "empty space" gaps.
- **Resolution**: Standardized the preview to use `PBNRenderer` which draws
  `shared_borders` with a thickness of 2. This "grout" seals the drift gaps
  visually.

### 2. Spurious random lines spike resolution

- **Discovery**: Long, straight, or diagonal line spikes crossed across random
  regions in the output.
- **Root cause**: When two regions touched in multiple disconnected locations
  (for example, sky bands touching on both the left and right sides of a
  foreground subject), the coordinates were appended to a flat list in raster
  scan order. Generating a single `LineString` from this list connected the
  disconnected points sequentially, drawing lines directly across the image.
- **Resolution**: Grouped border points into 8-connected contiguous components
  using NetworkX, and created a separate `LineString` for each contiguous
  segment. Points in each segment are sorted using a greedy nearest-neighbor
  walk to produce continuous, correct boundary paths.

### 3. Label clutter and overlap elimination

- **Discovery**: Detailed areas (faces, hands) became cluttered with
  overlapping, illegible tiny labels, and labels near canvas edges were cut off.
- **Root cause**: Every tiny island speck down to 10 pixels² received its own
  label, and labels were placed close to each other without margins or collision
  checks.
- **Resolution**:
  - Increased `min_region_area` for labeling from 16 to 120 pixels² and
    discarded labeling for tiny islands under 80 pixels² entirely.
  - Implemented a 25px minimum distance guard between placed labels to skip
    placing overlapping labels in high-detail areas.
  - Added canvas edge margins based on font size to push border numbers safely
    inward.

---

## Iteration log

- Full metrics history is maintained in `output/iteration_log.csv`.
- Combined gap map grids for sweeps are saved as `sweep_<tag>.png`.
