# PBN Quality Pipeline

This directory contains tools and reports for evaluating the Paint-By-Number generation quality.

## Quick Iteration Tool

Use `quick_iteration.py` to rapidly test different pipeline settings on a sample image.

```bash
python tests/quality/quick_iteration.py --colors 32 --smoothing 13 --simplification 0.5 --min-region-size 200 --tag "golden"
```

## Session Findings (2026-05-05)

### 1. The "Simplification Gap" Problem
*   **Discovery**: Setting `simplification` to the default `1.0` caused a massive drop in coverage (Fill ~94.9%).
*   **Root Cause**: At `1.0`, many small or thin regions collapse during Visvalingam-Whyatt simplification to fewer than 3 points and are deleted, leaving "black holes" in the output.
*   **Resolution**: Always use `simplification = 0.5` for high-fidelity templates.

### 2. Complexity vs. Smoothing
*   **Discovery**: `smoothing_kernel_size` is the primary knob for controlling region count.
*   **Findings**:
    *   `smooth=9` (baseline): ~924 regions (too complex for 32 colors).
    *   `smooth=13`: ~718 regions (Sweet spot for high detail).
    *   `smooth=17`: ~578 regions (Good for "painterly" look).
    *   `smooth=21`: ~471 regions (Starting to lose facial features).

### 3. The Speck Crisis
*   **Discovery**: The "Speck Ratio" (regions < 0.1% area) was as high as 87% with default settings.
*   **Root Cause**: The hardcoded segmenter merge threshold (~20px) was too low, allowing thousands of tiny noise fragments to survive.
*   **Resolution**: Exposing `min_region_size` as a parameter. Setting it to `150–200` dropped the speck ratio to ~60%, significantly improving template "paintability" without losing key### Phase 3: The Other Direction (0.0 Simplification)
**Goal**: Reach 100% fill by disabling all simplification and testing low smoothing.
**Result**: 99.61% Fill (at 0.0 simpl). Large clusters of missing pixels identified.

#### Roadblock: The "Largest Contour" Fallacy
We discovered that the segmenter was only taking the `max(contours)` for each region. In complex images, smoothing often creates "islands" (disconnected parts of the same color). Dropping these islands was the primary cause of the persistent ~0.6% gaps.

#### Roadblock: Island ID Collision
A naive fix to assign unique IDs to islands caused a collision with existing region IDs, leading to a drop in coverage (89%).

**Current Status**: Implementing a robust island-splitting logic with collision-safe IDs.

## Known Issues

### 1. Persistent 1-Pixel Gaps
Even with `simplification=0.5`, the coverage is ~99.4% instead of 100%. Tiny slivers (viewable in the "Gap Map") persist.
*   **Hypothesis**: Floating-point rounding during polygon rasterization in the preview generator.
*   **Action**: Investigate "sealing" the gaps by drawing a 1px matching border or increasing rasterization precision.

### 2. Boundary Visualization Flaws
The "Boundaries" diagnostic panel currently uses Canny edge detection on a grayscale version of the result.
*   **Issue**: Edges between regions of similar luminance are not detected (missing green lines).
*   **Action**: Update the iteration script to draw boundaries directly from the polygon data.

## Iteration Log
Full metrics history is maintained in [output/iteration_log.csv](output/iteration_log.csv).
